# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream

# Copyright 2025 Xinhua Chen, Duke CEI Center
# 
# This file has been modified by Xinhua Chen, Duke CEI Center. 
# Changes include:
# 1. Integrated Diffusion Scratchpad (DPad) for efficient inference.
# 2. Added full support for semi-autoregressive decoding.

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)
from .sampler import GaussianSampler, Sampler, SSMSampler, StreamingDLLMSampler, UniformSampler

logger = logging.get_logger(__name__)

def get_num_transfer_tokens(mask_indices, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_indices.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_indices.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def suffix_dropout(x, sampler: Sampler, block_end):
    q_indices = torch.arange(block_end, device=x.device).unsqueeze(0).expand(x.shape[0],-1)
    suffix_indices = sampler.sample(torch.arange(block_end, x.shape[1], device=x.device)).unsqueeze(0).expand(x.shape[0],-1)
    
    q_indices = torch.cat([q_indices, suffix_indices], dim=-1)
    k_indices = q_indices.clone()

    assert q_indices.max() < x.shape[1]
    return q_indices, k_indices


def _clamp01(v):
    return float(max(0.0, min(1.0, float(v))))


def _get_model_input_embeddings(model):
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    raise AttributeError("model has no get_input_embeddings() method")


def _select_suffix_soft_seq_positions(qv, block_end, local_window, non_local_only=False):
    suffix_seq_pos = torch.nonzero(qv >= int(block_end), as_tuple=False).squeeze(-1).to(torch.long)
    if suffix_seq_pos.numel() == 0:
        return suffix_seq_pos
    if bool(non_local_only):
        suffix_abs = qv[suffix_seq_pos].to(torch.long)
        keep = suffix_abs >= int(block_end) + int(local_window)
        suffix_seq_pos = suffix_seq_pos[keep]
    return suffix_seq_pos


def get_topk_soft_embedding(logits, embedding_weight, topk):
    if logits is None or logits.ndim != 2:
        raise ValueError("logits must be [N, V]")
    vocab = int(logits.shape[-1])
    k = max(1, min(int(topk), vocab))
    topk_vals, topk_ids = torch.topk(logits, k=k, dim=-1)
    topk_probs = F.softmax(topk_vals.to(torch.float32), dim=-1)
    topk_emb = embedding_weight[topk_ids]
    soft_emb = (topk_emb.to(torch.float32) * topk_probs.unsqueeze(-1)).sum(dim=1)
    return soft_emb.to(dtype=embedding_weight.dtype), topk_ids, topk_probs


def build_suffix_soft_state(mask_embedding, topk_soft_embedding, alpha):
    a = _clamp01(alpha)
    mask_vec = mask_embedding.unsqueeze(0).to(device=topk_soft_embedding.device, dtype=topk_soft_embedding.dtype)
    return (1.0 - a) * mask_vec + a * topk_soft_embedding


def apply_current_block_warm_start(
    inputs_embeds,
    q_indices,
    current_block_positions,
    suffix_soft_states,
    suffix_soft_valid,
    mask_embedding,
    beta,
):
    if inputs_embeds is None or current_block_positions.numel() == 0:
        return inputs_embeds
    qv = q_indices[0] if q_indices.dim() == 2 else q_indices
    pos_map = {int(p): i for i, p in enumerate(qv.to(torch.long).tolist())}
    valid_abs = []
    valid_seq = []
    for p in current_block_positions.to(torch.long).tolist():
        if p < 0 or p >= int(suffix_soft_valid.shape[0]):
            continue
        if not bool(suffix_soft_valid[p].item()):
            continue
        si = pos_map.get(int(p), None)
        if si is None:
            continue
        valid_abs.append(int(p))
        valid_seq.append(int(si))
    if len(valid_abs) == 0:
        return inputs_embeds
    abs_t = torch.tensor(valid_abs, dtype=torch.long, device=inputs_embeds.device)
    seq_t = torch.tensor(valid_seq, dtype=torch.long, device=inputs_embeds.device)
    b = _clamp01(beta)
    warm = suffix_soft_states[abs_t].to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    mask_vec = mask_embedding.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).unsqueeze(0)
    init_vec = (1.0 - b) * mask_vec + b * warm
    out = inputs_embeds
    out[:, seq_t, :] = init_vec.unsqueeze(0)
    return out


def prepare_suffix_soft_inputs(
    input_ids,
    embedding_layer,
    q_indices,
    block_end,
    local_window,
    non_local_only,
    suffix_soft_states,
    suffix_soft_valid,
    current_block_positions=None,
    mask_embedding=None,
    warm_start_beta=0.5,
):
    inputs_embeds = embedding_layer(input_ids)
    qv = q_indices[0] if q_indices.dim() == 2 else q_indices
    suffix_seq_pos = _select_suffix_soft_seq_positions(
        qv=qv,
        block_end=block_end,
        local_window=local_window,
        non_local_only=non_local_only,
    )
    suffix_abs = qv[suffix_seq_pos].to(torch.long)

    if suffix_abs.numel() > 0:
        known = suffix_soft_valid[suffix_abs.to(device=suffix_soft_valid.device)]
        if bool(known.any().item()):
            known_abs = suffix_abs[known].to(dtype=torch.long, device=suffix_soft_states.device)
            known_seq = suffix_seq_pos[known].to(dtype=torch.long, device=inputs_embeds.device)
            inputs_embeds[:, known_seq, :] = suffix_soft_states[known_abs].to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            ).unsqueeze(0)

    if current_block_positions is not None:
        inputs_embeds = apply_current_block_warm_start(
            inputs_embeds=inputs_embeds,
            q_indices=q_indices,
            current_block_positions=current_block_positions,
            suffix_soft_states=suffix_soft_states,
            suffix_soft_valid=suffix_soft_valid,
            mask_embedding=mask_embedding,
            beta=warm_start_beta,
        )

    return inputs_embeds, suffix_seq_pos, suffix_abs


def update_suffix_soft_cache(
    logits,
    suffix_seq_pos,
    suffix_abs,
    embedding_weight,
    mask_embedding,
    topk,
    alpha,
    suffix_soft_states,
    suffix_soft_valid,
):
    if suffix_seq_pos.numel() == 0:
        return
    suffix_logits = logits[0, suffix_seq_pos, :]
    soft_emb, _, _ = get_topk_soft_embedding(
        logits=suffix_logits,
        embedding_weight=embedding_weight,
        topk=topk,
    )
    soft_state = build_suffix_soft_state(
        mask_embedding=mask_embedding,
        topk_soft_embedding=soft_emb,
        alpha=alpha,
    )
    cache_abs = suffix_abs.to(dtype=torch.long, device=suffix_soft_states.device)
    suffix_soft_states[cache_abs] = soft_state.to(
        device=suffix_soft_states.device,
        dtype=suffix_soft_states.dtype,
    )
    suffix_soft_valid[cache_abs] = True

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.
        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        dropout: Optional[str] = 'null',
        sigma: Optional[float] = None,
        scale: Optional[float] = None,
        preserved_tokens: Optional[int] = None,
        window: Optional[int] = None,
        local_window: Optional[int] = 128,
        use_suffix_soft_state: Optional[bool] = False,
        suffix_soft_topk: Optional[int] = 5,
        suffix_soft_alpha: Optional[float] = 0.5,
        current_warm_start_beta: Optional[float] = 0.5,
        suffix_soft_non_local_only: Optional[bool] = False,
        early_termination: Optional[bool] = True,
        eos: Optional[int] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )
        threshold = kwargs.get("threshold", 0.9)
        block_length = kwargs.get("block_length", 32)
        dual_cache = kwargs.get("dual_cache", False)
        use_cache = kwargs.get("use_cache", False)
        if bool(use_suffix_soft_state) and str(dropout).strip().lower() == "null":
            raise ValueError("use_suffix_soft_state currently requires dropout != 'null' in dream path.")
        
        if dropout == 'null':
            if use_cache:
                result, nfe = self._sample_cache_baseline(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    threshold=threshold,
                    block_length=block_length,
                    dual_cache=dual_cache,
                    early_termination=early_termination,
                    eos=eos
                )
            else:
                result, nfe = self._sample_baseline(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    threshold=threshold,
                    block_length=block_length,
                    early_termination=early_termination,
                    eos=eos
                )
        else:
            if use_cache:
                result, nfe = self._sample_cache(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    threshold=threshold,
                    block_length=block_length,
                    dual_cache=dual_cache,
                    dropout=dropout,
                    sigma=sigma,
                    scale=scale,
                    preserved_tokens=preserved_tokens,
                    window=window,
                    local_window=local_window,
                    use_suffix_soft_state=use_suffix_soft_state,
                    suffix_soft_topk=suffix_soft_topk,
                    suffix_soft_alpha=suffix_soft_alpha,
                    current_warm_start_beta=current_warm_start_beta,
                    suffix_soft_non_local_only=suffix_soft_non_local_only,
                    early_termination=early_termination,
                    eos=eos
                )
            else:
                result, nfe = self._sample(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    threshold=threshold,
                    block_length=block_length,
                    dropout=dropout,
                    sigma=sigma,
                    scale=scale,
                    preserved_tokens=preserved_tokens,
                    window=window,
                    local_window=local_window,
                    use_suffix_soft_state=use_suffix_soft_state,
                    suffix_soft_topk=suffix_soft_topk,
                    suffix_soft_alpha=suffix_soft_alpha,
                    current_warm_start_beta=current_warm_start_beta,
                    suffix_soft_non_local_only=suffix_soft_non_local_only,
                    early_termination=early_termination,
                    eos=eos
                )
        return result, nfe

    def _sample_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: Optional[float] = 0.9,
        block_length: Optional[int] = 32,
        dual_cache: bool = False,
        dropout: Optional[str] = 'null',
        sigma: Optional[float] = None,
        scale: Optional[float] = None,
        preserved_tokens: Optional[int] = None,
        window: Optional[int] = None,
        local_window: Optional[int] = 128,
        use_suffix_soft_state: Optional[bool] = False,
        suffix_soft_topk: Optional[int] = 5,
        suffix_soft_alpha: Optional[float] = 0.5,
        current_warm_start_beta: Optional[float] = 0.5,
        suffix_soft_non_local_only: Optional[bool] = False,
        early_termination: Optional[bool] = True,
        eos: Optional[int] = None,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        
        # Handle block configuration
        if block_length is None:
            block_length = gen_length  # Default: single block (original behavior)
        
        assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks
        timesteps = torch.linspace(1, generation_config.eps, steps_per_block + 1, device=x.device)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # Initialize cache for the prompt
        past_key_values = None

        if dropout == 'gaussian':
            sampler = GaussianSampler(length=gen_length, sigma=sigma, scale=scale, window=window)
        elif dropout == 'uniform':
            sampler = UniformSampler(length=gen_length, number=preserved_tokens, window=window)
        elif dropout == 'ssm':
            sampler = SSMSampler(
                length=gen_length,
                window=window,
                local_window=local_window,
                block_size=block_length,
            )
        elif dropout == 'streaming_dllm':
            sampler = StreamingDLLMSampler(
                length=gen_length,
                local_window=local_window,
            )
        else:
            raise ValueError(f"dropout {dropout} not recognized")
        
        seq_len = x.shape[1]
        nfe = 0

        use_suffix_soft_state = bool(use_suffix_soft_state)
        suffix_soft_topk = max(1, int(suffix_soft_topk))
        suffix_soft_alpha = _clamp01(suffix_soft_alpha)
        current_warm_start_beta = _clamp01(current_warm_start_beta)
        suffix_soft_non_local_only = bool(suffix_soft_non_local_only)

        emb_layer = None
        emb_weight = None
        mask_embed = None
        suffix_soft_state_cache = None
        suffix_soft_valid = None
        if use_suffix_soft_state:
            emb_layer = _get_model_input_embeddings(self)
            emb_weight = emb_layer.weight
            mask_embed = emb_weight[int(mask_token_id)].detach()
            suffix_soft_state_cache = torch.zeros(
                (int(seq_len), int(emb_weight.shape[-1])),
                device=emb_weight.device,
                dtype=emb_weight.dtype,
            )
            suffix_soft_valid = torch.zeros((int(seq_len),), device=emb_weight.device, dtype=torch.bool)

        # Process each block
        for num_block in range(num_blocks):
            
            current_block_start = input_ids.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length

            q_indices, k_indices = suffix_dropout(x, sampler, current_block_end)
            # q_indices: [:block_end] + [preserved_masks]
            # Since all the tokens following current block are masks, there is no need to use indices to get them.
            # This operation is basically equivalent to x_pruned = x.gather(1, q_indices), except that slicing will not create a copy of x.
            x_pruned = x[:, :q_indices.shape[1]]

            model_inputs_embeds = None
            suffix_seq_pos = None
            suffix_abs = None
            if use_suffix_soft_state:
                current_positions = None
                if num_block > 0:
                    current_positions = torch.arange(
                        current_block_start,
                        current_block_end,
                        device=x_pruned.device,
                        dtype=torch.long,
                    )
                model_inputs_embeds, suffix_seq_pos, suffix_abs = prepare_suffix_soft_inputs(
                    input_ids=x_pruned,
                    embedding_layer=emb_layer,
                    q_indices=q_indices,
                    block_end=current_block_end,
                    local_window=local_window,
                    non_local_only=suffix_soft_non_local_only,
                    suffix_soft_states=suffix_soft_state_cache,
                    suffix_soft_valid=suffix_soft_valid,
                    current_block_positions=current_positions,
                    mask_embedding=mask_embed,
                    warm_start_beta=current_warm_start_beta,
                )

                # Dual cache retains the selected suffix KV for the whole block.
                # Probe once to build soft states before creating that cache.
                if dual_cache and suffix_seq_pos.numel() > 0:
                    probe_output = self(
                        input_ids=None,
                        attention_mask=attention_mask,
                        position_ids=tok_idx,
                        inputs_embeds=model_inputs_embeds,
                        use_cache=False,
                        q_indices=q_indices,
                        k_indices=k_indices,
                        ori_seq_len=seq_len,
                        update_rope=True,
                    )
                    probe_logits = torch.cat(
                        [probe_output.logits[:, :1], probe_output.logits[:, :-1]], dim=1
                    )
                    update_suffix_soft_cache(
                        logits=probe_logits,
                        suffix_seq_pos=suffix_seq_pos,
                        suffix_abs=suffix_abs,
                        embedding_weight=emb_weight,
                        mask_embedding=mask_embed,
                        topk=suffix_soft_topk,
                        alpha=suffix_soft_alpha,
                        suffix_soft_states=suffix_soft_state_cache,
                        suffix_soft_valid=suffix_soft_valid,
                    )
                    nfe += 1
                    model_inputs_embeds, suffix_seq_pos, suffix_abs = prepare_suffix_soft_inputs(
                        input_ids=x_pruned,
                        embedding_layer=emb_layer,
                        q_indices=q_indices,
                        block_end=current_block_end,
                        local_window=local_window,
                        non_local_only=suffix_soft_non_local_only,
                        suffix_soft_states=suffix_soft_state_cache,
                        suffix_soft_valid=suffix_soft_valid,
                        current_block_positions=current_positions,
                        mask_embedding=mask_embed,
                        warm_start_beta=current_warm_start_beta,
                    )

            # Build the block cache. In dual-cache mode it includes the mixed
            # selected suffix; in prefix-cache mode that suffix is recomputed.
            model_output = self(
                input_ids=None if model_inputs_embeds is not None else x_pruned,
                attention_mask=attention_mask,
                position_ids=tok_idx,
                inputs_embeds=model_inputs_embeds,
                use_cache=True,
                q_indices=q_indices,
                k_indices=k_indices,
                ori_seq_len=seq_len,
                update_rope=True,
            )
            past_key_values = model_output.past_key_values
            logits = model_output.logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            if use_suffix_soft_state and not dual_cache:
                update_suffix_soft_cache(
                    logits=logits,
                    suffix_seq_pos=suffix_seq_pos,
                    suffix_abs=suffix_abs,
                    embedding_weight=emb_weight,
                    mask_embedding=mask_embed,
                    topk=suffix_soft_topk,
                    alpha=suffix_soft_alpha,
                    suffix_soft_states=suffix_soft_state_cache,
                    suffix_soft_valid=suffix_soft_valid,
                )
            confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            x_pruned[:, current_block_start] = x0[:, current_block_start]
            
            # Extract only previous block cache
            if not dual_cache:
                new_past_key_values = []
                for i in range(len(past_key_values)):
                    new_past_key_values.append(())
                    for j in range(len(past_key_values[i])):
                        new_past_key_values[i] += (past_key_values[i][j][:, :current_block_start, :],)
                past_key_values = new_past_key_values
                q_indices = q_indices[:,current_block_start:]
            else:
                replace_position = torch.zeros_like(x, dtype=torch.bool)
                replace_position[:, current_block_start:current_block_end] = 1
                q_indices = q_indices[:, current_block_start: current_block_end]

            i = 1
            while True:
                # Use cache for generation
                if dual_cache:
                    mask_indices = (x_pruned[:, current_block_start:current_block_end] == mask_token_id)
                else:
                    mask_indices = (x_pruned[:, current_block_start:] == mask_token_id)
                
                # Prepare attention mask for cached generation
                if attention_mask != "full":
                    # Adjust attention mask for current position
                    current_attention_mask = attention_mask[:, :, :, current_block_start:]
                    assert 0
                else:
                    current_attention_mask = attention_mask
                # print("here!", x_pruned[:, current_block_start:])

                if dual_cache:
                    model_output = self(x_pruned[:, current_block_start:current_block_end], current_attention_mask, 
                                    tok_idx[:, current_block_start:current_block_end] if tok_idx is not None else None, 
                                    past_key_values=past_key_values, use_cache=True, dual_cache=dual_cache, replace_position=replace_position, q_indices=q_indices, k_indices=k_indices, ori_seq_len=seq_len, update_rope=(i==1))
                else:
                    prefix_input_ids = x_pruned[:, current_block_start:]
                    model_inputs_embeds = None
                    if use_suffix_soft_state:
                        model_inputs_embeds, suffix_seq_pos, suffix_abs = prepare_suffix_soft_inputs(
                            input_ids=prefix_input_ids,
                            embedding_layer=emb_layer,
                            q_indices=q_indices,
                            block_end=current_block_end,
                            local_window=local_window,
                            non_local_only=suffix_soft_non_local_only,
                            suffix_soft_states=suffix_soft_state_cache,
                            suffix_soft_valid=suffix_soft_valid,
                        )
                    model_output = self(
                        input_ids=None if model_inputs_embeds is not None else prefix_input_ids,
                        attention_mask=current_attention_mask,
                        position_ids=tok_idx[:, current_block_start:] if tok_idx is not None else None,
                        inputs_embeds=model_inputs_embeds,
                        past_key_values=past_key_values,
                        use_cache=True,
                        q_indices=q_indices,
                        k_indices=k_indices,
                        ori_seq_len=seq_len,
                        update_rope=(i == 1),
                    )
                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
                if use_suffix_soft_state and not dual_cache:
                    update_suffix_soft_cache(
                        logits=logits,
                        suffix_seq_pos=suffix_seq_pos,
                        suffix_abs=suffix_abs,
                        embedding_weight=emb_weight,
                        mask_embedding=mask_embed,
                        topk=suffix_soft_topk,
                        alpha=suffix_soft_alpha,
                        suffix_soft_states=suffix_soft_state_cache,
                        suffix_soft_valid=suffix_soft_valid,
                    )
                if alg == 'confidence_threshold':
                    mask_logits = logits[mask_indices]

                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    
                    if dual_cache:
                        x_ = torch.zeros_like(x_pruned[:, current_block_start:current_block_end], device=self.device, dtype=torch.long) + mask_token_id
                        full_confidence = torch.full_like(x_pruned[:, current_block_start:current_block_end], -torch.inf, device=self.device, dtype=logits.dtype)
                    else:
                        x_ = torch.zeros_like(x_pruned[:, current_block_start:], device=self.device, dtype=torch.long) + mask_token_id
                        full_confidence = torch.full_like(x_pruned[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)
                    
                    x_[mask_indices] = x0.clone()
                    full_confidence[mask_indices] = confidence
                    full_confidence[:, block_length:] = -torch.inf
                    # print("confi: ", full_confidence)
                    
                    current_transfer_tokens = (x_pruned[:, current_block_start:current_block_end] == mask_token_id).sum()
                    
                    selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
                    transfer_index = torch.zeros_like(x_, device=x.device, dtype=torch.bool)
                    
                    select_index = select_index.to(x.device)
                    transfer_index[0, select_index[0]] = True
                    for k in range(1, current_transfer_tokens):
                        if selected_confidence[0, k] < threshold:
                            transfer_index[0, select_index[0, k]] = False
                    # print("x: ", x_[transfer_index], flush=True)
                    # assert 0, x_[transfer_index]
                    if dual_cache:
                        x_pruned[:, current_block_start:current_block_end][transfer_index] = x_[transfer_index]
                    else:
                        x_pruned[:, current_block_start:][transfer_index] = x_[transfer_index]
                    
                else:
                    if i == steps_per_block:
                        break
                    t = timesteps[i]
                    s = timesteps[i + 1]
                    mask_indices[:, block_length:] = False
                    mask_logits = logits[mask_indices]
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                    num_mask_token = mask_indices.sum() / mask_indices.shape[0]
                    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps_per_block - 1 else int(num_mask_token)
                    if dual_cache:
                        full_confidence = torch.full_like(x_pruned[:, current_block_start:current_block_end], -torch.inf, device=self.device, dtype=logits.dtype)
                    else:
                        full_confidence = torch.full_like(x_pruned[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_indices] = confidence
                    full_confidence[:, block_length:] = -torch.inf
                    
                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                        if dual_cache:
                            x_ = torch.zeros_like(x_pruned[:, current_block_start:current_block_end], device=self.device, dtype=torch.long) + mask_token_id
                        else:
                            x_ = torch.zeros_like(x_pruned[:, current_block_start:], device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_indices] = x0.clone()
                        row_indices = torch.arange(x_pruned.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                        if dual_cache:
                            x_pruned[:, current_block_start:current_block_end][row_indices,transfer_index] = x_[row_indices,transfer_index]
                        else:
                            x_pruned[:, current_block_start:][row_indices,transfer_index] = x_[row_indices,transfer_index]
                i += 1
                nfe += 1

                if (x_pruned[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                    # Early Termination
                    if early_termination is True and (x_pruned[:, current_block_start:current_block_end] == eos).any():
                        x[:, current_block_end: ] = eos
                        if return_dict_in_generate:
                            return DreamModelOutput(
                                sequences=x,
                                history=histories,
                            ), nfe
                        else:
                            return x, nfe
                    break

        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            ), nfe
        else:
            return x, nfe
        
    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: Optional[float] = None,
        block_length: Optional[int] = 32,
        dropout: Optional[str] = 'null',
        sigma: Optional[float] = None,
        scale: Optional[float] = None,
        preserved_tokens: Optional[int] = None,
        window: Optional[int] = None,
        local_window: Optional[int] = 128,
        use_suffix_soft_state: Optional[bool] = False,
        suffix_soft_topk: Optional[int] = 5,
        suffix_soft_alpha: Optional[float] = 0.5,
        current_warm_start_beta: Optional[float] = 0.5,
        suffix_soft_non_local_only: Optional[bool] = False,
        early_termination: Optional[bool] = True,
        eos: Optional[int] = None
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp

        histories = [] if (return_dict_in_generate and output_history) else None
        use_suffix_soft_state = bool(use_suffix_soft_state)
        suffix_soft_topk = max(1, int(suffix_soft_topk))
        suffix_soft_alpha = _clamp01(suffix_soft_alpha)
        current_warm_start_beta = _clamp01(current_warm_start_beta)
        suffix_soft_non_local_only = bool(suffix_soft_non_local_only)

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        
        # Handle block configuration
        if block_length is None:
            block_length = gen_length  # Default: single block (original behavior)
        
        assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length
        
        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks
        timesteps = torch.linspace(1, generation_config.eps, steps_per_block + 1, device=x.device)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        if dropout == 'gaussian':
            sampler = GaussianSampler(length=gen_length, sigma=sigma, scale=scale, window=window)
        elif dropout == 'uniform':
            sampler = UniformSampler(length=gen_length, number=preserved_tokens, window=window)
        elif dropout == 'ssm':
            sampler = SSMSampler(
                length=gen_length,
                window=window,
                local_window=local_window,
                block_size=block_length,
            )
        elif dropout == 'streaming_dllm':
            sampler = StreamingDLLMSampler(
                length=gen_length,
                local_window=local_window,
            )
        else:
            raise ValueError(f"dropout {dropout} not recognized")

        nfe = 0
        emb_layer = None
        emb_weight = None
        mask_embed = None
        suffix_soft_state_cache = None
        suffix_soft_valid = None
        if use_suffix_soft_state:
            emb_layer = _get_model_input_embeddings(self)
            emb_weight = emb_layer.weight
            mask_embed = emb_weight[int(mask_token_id)].detach()
            suffix_soft_state_cache = torch.zeros(
                (int(x.shape[1]), int(emb_weight.shape[-1])),
                device=emb_weight.device,
                dtype=emb_weight.dtype,
            )
            suffix_soft_valid = torch.zeros((int(x.shape[1]),), device=emb_weight.device, dtype=torch.bool)

        # Initialize cache for the prompt
        # past_key_values = None

        # Process each block
        for num_block in range(num_blocks):
            
            current_block_start = input_ids.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length

            q_indices, k_indices = suffix_dropout(x, sampler, current_block_end)
            seq_len = x.shape[1]
            print("pruned k: ", x.shape[1] - k_indices.shape[1])
            # q_indices: [:block_end] + [preserved_masks]
            # Since all the tokens following current block are masks, there is no need to use indices to get them.
            # This operation is basically equivalent to x_pruned = x.gather(1, q_indices), except that slicing will not create a copy of x.
            x_pruned = x[:, :q_indices.shape[1]]

            i = 1
            while True:
                mask_indices = (x_pruned == mask_token_id)
                mask_indices[:, current_block_end:] = False
                
                # Prepare attention mask for cached generation
                if attention_mask != "full":
                    # Adjust attention mask for current position
                    current_attention_mask = attention_mask[:, :, :, current_block_start:]
                else:
                    current_attention_mask = attention_mask

                model_inputs_embeds = None
                if use_suffix_soft_state:
                    qv = q_indices[0] if q_indices.dim() == 2 else q_indices
                    model_inputs_embeds = emb_layer(x_pruned)
                    suffix_seq_pos = _select_suffix_soft_seq_positions(
                        qv=qv,
                        block_end=current_block_end,
                        local_window=local_window,
                        non_local_only=suffix_soft_non_local_only,
                    )
                    if suffix_seq_pos.numel() > 0:
                        suffix_abs = qv[suffix_seq_pos].to(torch.long)
                        known = suffix_soft_valid[suffix_abs.to(device=suffix_soft_valid.device)]
                        if bool(known.any().item()):
                            known_abs = suffix_abs[known].to(dtype=torch.long, device=suffix_soft_state_cache.device)
                            known_seq = suffix_seq_pos[known].to(dtype=torch.long, device=model_inputs_embeds.device)
                            model_inputs_embeds[:, known_seq, :] = suffix_soft_state_cache[known_abs].to(
                                device=model_inputs_embeds.device,
                                dtype=model_inputs_embeds.dtype,
                            ).unsqueeze(0)
                    if num_block > 0 and i == 1:
                        current_positions = torch.arange(
                            current_block_start, current_block_end, device=model_inputs_embeds.device, dtype=torch.long
                        )
                        model_inputs_embeds = apply_current_block_warm_start(
                            inputs_embeds=model_inputs_embeds,
                            q_indices=q_indices,
                            current_block_positions=current_positions,
                            suffix_soft_states=suffix_soft_state_cache,
                            suffix_soft_valid=suffix_soft_valid,
                            mask_embedding=mask_embed,
                            beta=current_warm_start_beta,
                        )

                model_output = self(
                    input_ids=None if model_inputs_embeds is not None else x_pruned,
                    attention_mask=current_attention_mask,
                    position_ids=tok_idx if tok_idx is not None else None,
                    inputs_embeds=model_inputs_embeds,
                    ori_seq_len=seq_len,
                    q_indices=q_indices,
                    k_indices=k_indices,
                    update_rope=(i==1),
                )
                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
                if use_suffix_soft_state:
                    qv = q_indices[0] if q_indices.dim() == 2 else q_indices
                    suffix_seq_pos = _select_suffix_soft_seq_positions(
                        qv=qv,
                        block_end=current_block_end,
                        local_window=local_window,
                        non_local_only=suffix_soft_non_local_only,
                    )
                    if suffix_seq_pos.numel() > 0:
                        suffix_abs = qv[suffix_seq_pos].to(torch.long)
                        suffix_logits = logits[0, suffix_seq_pos, :]
                        soft_emb, _, _ = get_topk_soft_embedding(
                            logits=suffix_logits,
                            embedding_weight=emb_weight,
                            topk=suffix_soft_topk,
                        )
                        soft_state = build_suffix_soft_state(
                            mask_embedding=mask_embed,
                            topk_soft_embedding=soft_emb,
                            alpha=suffix_soft_alpha,
                        )
                        cache_abs = suffix_abs.to(dtype=torch.long, device=suffix_soft_state_cache.device)
                        suffix_soft_state_cache[cache_abs] = soft_state.to(
                            device=suffix_soft_state_cache.device,
                            dtype=suffix_soft_state_cache.dtype,
                        )
                        suffix_soft_valid[cache_abs] = True
                if alg == 'confidence_threshold':
                    mask_logits = logits[mask_indices]
                
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    
                    x_ = torch.zeros_like(x_pruned, device=self.device, dtype=torch.long) + mask_token_id
                    full_confidence = torch.full_like(x_pruned, -torch.inf, device=self.device, dtype=logits.dtype)
                    
                    x_[mask_indices] = x0.clone()
                    full_confidence[mask_indices] = confidence
                    full_confidence[:, current_block_end:] = -torch.inf
                    
                    current_transfer_tokens = (x_pruned[:, current_block_start:current_block_end] == mask_token_id).sum()
                    
                    selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
                    transfer_index = torch.zeros_like(x_, device=x.device, dtype=torch.bool)
                    
                    select_index = select_index.to(x.device)
                    transfer_index[0, select_index[0]] = True
                    for k in range(1, current_transfer_tokens):
                        if selected_confidence[0, k] < threshold:
                            transfer_index[0, select_index[0, k]] = False
                    x_pruned[transfer_index] = x_[transfer_index]
                else:
                    if i == steps_per_block:
                        break
                    t = timesteps[i]
                    s = timesteps[i + 1]
                    mask_logits = logits[mask_indices]
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                    num_mask_token = mask_indices.sum() / mask_indices.shape[0]
                    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps_per_block - 1 else int(num_mask_token)
                    full_confidence = torch.full_like(x_pruned, -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_indices] = confidence
                    full_confidence[:, current_block_end:] = -torch.inf
                    
                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                        x_ = torch.zeros_like(x_pruned, device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_indices] = x0.clone()
                        row_indices = torch.arange(x_pruned.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                        x_pruned[row_indices, transfer_index] = x_[row_indices, transfer_index]
                i += 1
                nfe += 1

                if (x_pruned[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                    print(f"decoded block {num_block} with {i} steps")
                    # Early Termination
                    if early_termination is True and (x_pruned[:, current_block_start:current_block_end] == eos).any():
                        x[:, current_block_end: ] = eos
                        if return_dict_in_generate:
                            return DreamModelOutput(
                                sequences=x,
                                history=histories,
                            ), nfe
                        else:
                            return x, nfe
                    break

        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            ), nfe
        else:
            return x, nfe
        
    def _sample_cache_baseline(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: Optional[float] = 0.9,
        block_length: Optional[int] = 32,
        dual_cache: bool = False,
        early_termination: Optional[bool] = True,
        eos: Optional[int] = None
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        '''
        Original Fast-dLLM Implementation
        '''
        # init values
        
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        
        # Handle block configuration
        if block_length is None:
            block_length = gen_length  # Default: single block (original behavior)
        
        assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length
        
        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks
        timesteps = torch.linspace(1, generation_config.eps, steps_per_block + 1, device=x.device)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # Initialize cache for the prompt
        past_key_values = None
        nfe = 0

        # Process each block
        for num_block in range(num_blocks):
            
            current_block_start = input_ids.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length

            # update cache
            model_output = self(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            logits = model_output.logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            x[:, current_block_start] = x0[:, current_block_start]
            
            # Extract only previous block cache
            if not dual_cache:
                new_past_key_values = []
                for i in range(len(past_key_values)):
                    new_past_key_values.append(())
                    for j in range(len(past_key_values[i])):
                        new_past_key_values[i] += (past_key_values[i][j][:, :current_block_start, :],)
                past_key_values = new_past_key_values
            else:
                replace_position = torch.zeros_like(x, dtype=torch.bool)
                replace_position[:, current_block_start:current_block_end] = 1
                
            i = 1
            while True:
                # Use cache for generation
                if dual_cache:
                    mask_index = (x[:, current_block_start:current_block_end] == mask_token_id)
                else:
                    mask_index = (x[:, current_block_start:] == mask_token_id)
                
                # Prepare attention mask for cached generation
                if attention_mask != "full":
                    # Adjust attention mask for current position
                    current_attention_mask = attention_mask[:, :, :, current_block_start:]
                else:
                    current_attention_mask = attention_mask
                # print("here!", x[:, current_block_start:])
                if dual_cache:
                    model_output = self(x[:, current_block_start:current_block_end], current_attention_mask, 
                                    tok_idx[:, current_block_start:current_block_end] if tok_idx is not None else None, 
                                    past_key_values=past_key_values, use_cache=True, dual_cache=dual_cache, replace_position=replace_position)
                else:
                    model_output = self(x[:, current_block_start:], current_attention_mask, 
                                    tok_idx[:, current_block_start:] if tok_idx is not None else None, 
                                    past_key_values=past_key_values, use_cache=True)
                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
                if alg == 'confidence_threshold':
                    mask_logits = logits[mask_index]
                
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    
                    if dual_cache:
                        x_ = torch.zeros_like(x[:, current_block_start:current_block_end], device=self.device, dtype=torch.long) + mask_token_id
                        full_confidence = torch.full_like(x[:, current_block_start:current_block_end], -torch.inf, device=self.device, dtype=logits.dtype)
                    else:
                        x_ = torch.zeros_like(x[:, current_block_start:], device=self.device, dtype=torch.long) + mask_token_id
                        full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)
                    
                    x_[mask_index] = x0.clone()
                    full_confidence[mask_index] = confidence
                    full_confidence[:, block_length:] = -torch.inf

                    # print("confi: ", full_confidence)
                    
                    current_transfer_tokens = (x[:, current_block_start:current_block_end] == mask_token_id).sum()
                    
                    selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
                    transfer_index = torch.zeros_like(x_, device=x.device, dtype=torch.bool)
                    
                    select_index = select_index.to(x.device)
                    transfer_index[0, select_index[0]] = True
                    for k in range(1, current_transfer_tokens):
                        if selected_confidence[0, k] < threshold:
                            transfer_index[0, select_index[0, k]] = False
                    if dual_cache:
                        x[:, current_block_start:current_block_end][transfer_index] = x_[transfer_index]
                    else:
                        x[:, current_block_start:][transfer_index] = x_[transfer_index]
                    # print("x: ", x_[transfer_index], flush=True)
                    # assert 0, x_[transfer_index]
                else:
                    if i == steps_per_block:
                        break
                    t = timesteps[i]
                    s = timesteps[i + 1]
                    mask_index[:, block_length:] = False
                    mask_logits = logits[mask_index]
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                    num_mask_token = mask_index.sum() / mask_index.shape[0]
                    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps_per_block - 1 else int(num_mask_token)
                    if dual_cache:
                        full_confidence = torch.full_like(x[:, current_block_start:current_block_end], -torch.inf, device=self.device, dtype=logits.dtype)
                    else:
                        full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index] = confidence
                    full_confidence[:, block_length:] = -torch.inf
                    
                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
                        if dual_cache:
                            x_ = torch.zeros_like(x[:, current_block_start:current_block_end], device=self.device, dtype=torch.long) + mask_token_id
                        else:
                            x_ = torch.zeros_like(x[:, current_block_start:], device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_index] = x0.clone()
                        row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                        if dual_cache:
                            x[:, current_block_start:current_block_end][row_indices,transfer_index] = x_[row_indices,transfer_index]
                        else:
                            x[:, current_block_start:][row_indices,transfer_index] = x_[row_indices,transfer_index]
                i += 1
                nfe += 1

                if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                    if early_termination is True and (x[:, current_block_start:current_block_end] == eos).any():
                        x[:, current_block_end: ] = eos
                        if return_dict_in_generate:
                            return DreamModelOutput(
                                sequences=x,
                                history=histories,
                            ), nfe
                        else:
                            return x, nfe
                    break

        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            ), nfe
        else:
            return x, nfe
        

    def _sample_baseline(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: Optional[float] = None,
        block_length: Optional[int] = 32,
        early_termination: Optional[bool] = True,
        eos: Optional[int] = None
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        '''
        Semi-Autoregressive Dream
        '''
        # init values

        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        
        # Handle block configuration
        if block_length is None:
            block_length = gen_length  # Default: single block (original behavior)
        
        assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length
        
        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks
        timesteps = torch.linspace(1, generation_config.eps, steps_per_block + 1, device=x.device)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # Initialize cache for the prompt
        # past_key_values = None

        # Process each block
        nfe = 0
        for num_block in range(num_blocks):
            
            current_block_start = input_ids.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length
  
            i = 1
            while True:
                # mask_index = (x == mask_token_id)
                mask_index = (x == mask_token_id)
                mask_index[:, current_block_end:] = False
                
                # Prepare attention mask for cached generation
                if attention_mask != "full":
                    # Adjust attention mask for current position
                    current_attention_mask = attention_mask[:, :, :, current_block_start:]
                else:
                    current_attention_mask = attention_mask

                model_output = self(x, current_attention_mask, 
                                    tok_idx if tok_idx is not None else None)
                
                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
                if alg == 'confidence_threshold':
                    mask_logits = logits[mask_index]
                
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                    
                    x_[mask_index] = x0.clone()
                    full_confidence[mask_index] = confidence
                    full_confidence[:, current_block_end:] = -torch.inf
                    
                    current_transfer_tokens = (x[:, current_block_start:current_block_end] == mask_token_id).sum()
                    
                    selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
                    transfer_index = torch.zeros_like(x_, device=x.device, dtype=torch.bool)
                    
                    select_index = select_index.to(x.device)
                    transfer_index[0, select_index[0]] = True
                    for k in range(1, current_transfer_tokens):
                        if selected_confidence[0, k] < threshold:
                            transfer_index[0, select_index[0, k]] = False
                    x[transfer_index] = x_[transfer_index]
                elif alg == 'entropy':
                    if i == steps_per_block:
                        break
                    t = timesteps[i]
                    s = timesteps[i + 1]
                    mask_index[:, current_block_end:] = False
                    mask_logits = logits[mask_index]
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                    num_mask_token = mask_index.sum() / mask_index.shape[0]
                    number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps_per_block - 1 else int(num_mask_token)
                    # if dual_cache:
                    #     full_confidence = torch.full_like(x[:, current_block_start:current_block_end], -torch.inf, device=self.device, dtype=logits.dtype)
                    # else:
                    #     full_confidence = torch.full_like(x[:, current_block_start:], -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
                    full_confidence[mask_index] = confidence
                    full_confidence[:, current_block_end:] = -torch.inf
                    
                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)

                        x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                        x_[mask_index] = x0.clone()
                        row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                        x[row_indices,transfer_index] = x_[row_indices,transfer_index]

                i += 1
                nfe += 1

                if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                    if early_termination is True and (x[:, current_block_start:current_block_end] == eos).any():
                        x[:, current_block_end: ] = eos
                        if return_dict_in_generate:
                            return DreamModelOutput(
                                sequences=x,
                                history=histories,
                            ), nfe
                        else:
                            return x, nfe
                    break

        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            ), nfe
        else:
            return x, nfe
