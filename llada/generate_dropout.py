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
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

# Copyright 2025 Xinhua Chen, Duke CEI Center
# 
# This file has been modified by Xinhua Chen, Duke CEI Center. Changes include:
# 1. Integrated Diffusion Scratchpad (DPad) for efficient inference.



import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
from sampler import (
    Sampler,
    GaussianSampler,
    UniformSampler,
    SSMSampler,
)
torch.set_printoptions(threshold=np.inf)


def _record_suffix_stats(stats, kept_suffix_tokens, available_suffix_tokens):
    if stats is None:
        return
    stats["suffix_block_samples"] = stats.get("suffix_block_samples", 0) + 1
    stats["kept_suffix_tokens_total"] = stats.get("kept_suffix_tokens_total", 0) + int(kept_suffix_tokens)
    stats["available_suffix_tokens_total"] = stats.get("available_suffix_tokens_total", 0) + int(available_suffix_tokens)


def _get_model_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _get_model_input_embeddings(model):
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    if hasattr(model, "module") and hasattr(model.module, "get_input_embeddings"):
        return model.module.get_input_embeddings()
    raise AttributeError("model has no get_input_embeddings() method")


def _clamp01(v):
    return float(max(0.0, min(1.0, float(v))))


def _dedup_sorted_positions(abs_pos, seq_pos):
    if abs_pos is None or abs_pos.numel() == 0:
        return abs_pos, seq_pos
    keep = torch.ones_like(abs_pos, dtype=torch.bool)
    if abs_pos.numel() > 1:
        keep[1:] = abs_pos[1:] != abs_pos[:-1]
    return abs_pos[keep], seq_pos[keep]


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
        raise ValueError("logits must be a 2D tensor [N, V]")
    vocab = int(logits.shape[-1])
    k = max(1, min(int(topk), vocab))
    topk_vals, topk_ids = torch.topk(logits, k=k, dim=-1)
    topk_probs = F.softmax(topk_vals.to(torch.float32), dim=-1)
    topk_emb = embedding_weight[topk_ids]
    soft_emb = (topk_emb.to(torch.float32) * topk_probs.unsqueeze(-1)).sum(dim=1)
    return soft_emb.to(dtype=embedding_weight.dtype), topk_ids, topk_probs


def build_suffix_soft_state(mask_embedding, topk_soft_embedding, alpha):
    a = _clamp01(alpha)
    if topk_soft_embedding is None or topk_soft_embedding.numel() == 0:
        return topk_soft_embedding
    if mask_embedding.ndim == 1:
        mask_vec = mask_embedding.unsqueeze(0)
    else:
        mask_vec = mask_embedding
    mask_vec = mask_vec.to(device=topk_soft_embedding.device, dtype=topk_soft_embedding.dtype)
    return (1.0 - a) * mask_vec + a * topk_soft_embedding


def apply_current_block_warm_start(
    inputs_embeds,
    q_ref,
    current_block_positions,
    suffix_soft_states,
    suffix_soft_valid,
    # mask_embedding,
    # beta,
):
    if inputs_embeds is None:
        return inputs_embeds, 0
    if current_block_positions is None or current_block_positions.numel() == 0:
        return inputs_embeds, 0
    if suffix_soft_states is None or suffix_soft_valid is None:
        return inputs_embeds, 0

    qv = q_ref[0] if q_ref.dim() == 2 else q_ref
    pos_map = {int(p): i for i, p in enumerate(qv.to(torch.long).tolist())}

    valid_abs = []
    valid_seq_pos = []
    for p in current_block_positions.to(torch.long).tolist():
        if p < 0 or p >= int(suffix_soft_valid.shape[0]):
            continue
        if not bool(suffix_soft_valid[p].item()):
            continue
        seq_pos = pos_map.get(int(p), None)
        if seq_pos is None:
            continue
        valid_abs.append(int(p))
        valid_seq_pos.append(int(seq_pos))

    if len(valid_abs) == 0:
        return inputs_embeds, 0

    abs_t = torch.tensor(valid_abs, dtype=torch.long, device=inputs_embeds.device)
    seq_t = torch.tensor(valid_seq_pos, dtype=torch.long, device=inputs_embeds.device)

    # b = _clamp01(beta)
    warm_vec = suffix_soft_states[abs_t].to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    # mask_vec = mask_embedding.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device).unsqueeze(0)
    # init_vec = (1.0 - b) * mask_vec + b * warm_vec
    # inputs_embeds[:, seq_t, :] = init_vec.unsqueeze(0)
    # The warm-start blend coefficient is fixed at 1.0: use the cached
    # suffix soft state directly when the token enters the current block.
    inputs_embeds[:, seq_t, :] = warm_vec.unsqueeze(0)
    return inputs_embeds, int(abs_t.numel())

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


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

def get_transfer_index(logits, temperature, remasking, mask_indices, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_indices, x0, x)
    confidence = torch.where(mask_indices, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_indices.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def suffix_dropout(x, sampler: Sampler, block_end):
    q_indices = torch.arange(block_end, device=x.device).unsqueeze(0).expand(x.shape[0],-1)
    suffix_indices = sampler.sample(torch.arange(block_end, x.shape[1], device=x.device)).unsqueeze(0).expand(x.shape[0],-1)
    
    q_indices = torch.cat([q_indices, suffix_indices], dim=-1)
    k_indices = q_indices.clone()

    assert q_indices.max() < x.shape[1]
    return q_indices, k_indices


def build_sampler(dropout, gen_length, block_length, sigma, scale, preserved_tokens, window, local_window):
    if dropout == "gaussian":
        return GaussianSampler(length=gen_length, sigma=sigma, scale=scale, window=window)
    if dropout == "uniform":
        return UniformSampler(length=gen_length, number=preserved_tokens, window=window)

    ssm_modes = {
        "ssm": {},
        "ssm_local_none": {"local_window_mode": "none"},
        "ssm_mid_none": {"local_middle_block_mode": "none"},
        "ssm_mid_end_only": {"local_middle_block_mode": "end_only"},
        "ssm_mid_start_end": {"local_middle_block_mode": "start_end"},
        "ssm_mid_mid_only": {"local_middle_block_mode": "mid_only"},
        "ssm_last_start_middle_end": {"local_last_block_mode": "start_middle_end"},
        "ssm_last_end_only": {"local_last_block_mode": "end_only"},
        "ssm_last_start_only": {"local_last_block_mode": "start_only"},
        "ssm_last_none": {"local_last_block_mode": "none"},
    }
    if dropout not in ssm_modes:
        raise ValueError(f"dropout {dropout} not recognized")

    return SSMSampler(
        length=gen_length,
        window=window,
        local_window=local_window,
        num_anchors=16,
        block_size=block_length,
        **ssm_modes[dropout],
    )

@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, eos_id=126081, threshold=None, 
             dropout='null', sigma=None, scale=None, preserved_tokens=0, window=None, early_termination=True,
             local_window=128,
             use_suffix_soft_state=False, suffix_soft_topk=5, suffix_soft_alpha=0.5,
             suffix_soft_non_local_only=False, stats=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(_get_model_device(model))
    x[:, :prompt.shape[1]] = prompt.clone()
    seq_len = x.shape[1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0

    sampler = build_sampler(
        dropout, gen_length, block_length, sigma, scale, preserved_tokens, window, local_window
    )

    use_suffix_soft_state = bool(use_suffix_soft_state)
    suffix_soft_topk = max(1, int(suffix_soft_topk))
    suffix_soft_alpha = _clamp01(suffix_soft_alpha)
    # current_warm_start_beta = _clamp01(current_warm_start_beta)
    suffix_soft_non_local_only = bool(suffix_soft_non_local_only)

    emb_layer = None
    emb_weight = None
    mask_embed = None
    suffix_soft_state_cache = None
    suffix_soft_valid = None
    if use_suffix_soft_state:
        emb_layer = _get_model_input_embeddings(model)
        emb_weight = emb_layer.weight
        mask_embed = emb_weight[int(mask_id)].detach()
        suffix_soft_state_cache = torch.zeros(
            (int(seq_len), int(emb_weight.shape[-1])),
            device=emb_weight.device,
            dtype=emb_weight.dtype,
        )
        suffix_soft_valid = torch.zeros((int(seq_len),), device=emb_weight.device, dtype=torch.bool)
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block+1) * block_length
        block_mask_indices = (x[:, block_start: block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_indices, steps)

        q_indices, k_indices = suffix_dropout(x, sampler, block_end)
        # q_indices: [:block_end] + [preserved_masks]
        # Since all the tokens following current block are masks, there is no need to use indices to get them.
        # This operation is basically equivalent to x_pruned = x.gather(1, q_indices), except that slicing will not create a copy of x.
        x_pruned = x[:,:q_indices.shape[1]]
        q_ref = q_indices
        available_suffix_tokens = max(0, x.shape[1] - block_end)
        kept_suffix_tokens = max(0, q_ref.shape[1] - block_end)
        _record_suffix_stats(stats, kept_suffix_tokens, available_suffix_tokens)

        i = 0
        while True:
            nfe += 1
            model_inputs_embeds = None
            if use_suffix_soft_state:
                model_inputs_embeds = emb_layer(x_pruned)
                qv = q_ref[0] if q_ref.dim() == 2 else q_ref
                suffix_seq_pos = _select_suffix_soft_seq_positions(
                    qv=qv,
                    block_end=block_end,
                    local_window=local_window,
                    non_local_only=suffix_soft_non_local_only,
                )
                if suffix_seq_pos.numel() > 0:
                    suffix_abs = qv[suffix_seq_pos].to(torch.long)
                    suffix_abs, suffix_seq_pos = _dedup_sorted_positions(suffix_abs, suffix_seq_pos)
                    known_mask = suffix_soft_valid[suffix_abs.to(device=suffix_soft_valid.device)]
                    if bool(known_mask.any().item()):
                        known_abs = suffix_abs[known_mask].to(dtype=torch.long, device=suffix_soft_state_cache.device)
                        known_seq_pos = suffix_seq_pos[known_mask].to(dtype=torch.long, device=model_inputs_embeds.device)
                        model_inputs_embeds[:, known_seq_pos, :] = suffix_soft_state_cache[known_abs].to(
                            device=model_inputs_embeds.device,
                            dtype=model_inputs_embeds.dtype,
                        ).unsqueeze(0)

                if num_block > 0 and i == 0:
                    current_block_positions = torch.arange(block_start, block_end, device=model_inputs_embeds.device, dtype=torch.long)
                    model_inputs_embeds, _ = apply_current_block_warm_start(
                        inputs_embeds=model_inputs_embeds,
                        q_ref=q_ref,
                        current_block_positions=current_block_positions,
                        suffix_soft_states=suffix_soft_state_cache,
                        suffix_soft_valid=suffix_soft_valid,
                    )

            logits = model(
                x_pruned,
                inputs_embeds=model_inputs_embeds,
                q_indices=q_indices,
                k_indices=k_indices,
                seq_len=seq_len,
                update_rope=(i==0),
            ).logits

            if use_suffix_soft_state:
                qv = q_ref[0] if q_ref.dim() == 2 else q_ref
                suffix_seq_pos = _select_suffix_soft_seq_positions(
                    qv=qv,
                    block_end=block_end,
                    local_window=local_window,
                    non_local_only=suffix_soft_non_local_only,
                )
                if suffix_seq_pos.numel() > 0:
                    suffix_abs = qv[suffix_seq_pos].to(torch.long)
                    suffix_abs, suffix_seq_pos = _dedup_sorted_positions(suffix_abs, suffix_seq_pos)
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
            mask_indices = (x_pruned == mask_id)
            mask_indices[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
     
            x0, transfer_index = get_transfer_index(logits, 
                                                    temperature, 
                                                    remasking, 
                                                    mask_indices, 
                                                    x_pruned, 
                                                    num_transfer_tokens[:, i] if threshold is None else None, 
                                                    threshold=threshold)                    
          
            x_pruned[transfer_index] = x0[transfer_index]
  
            i += 1
            if (x_pruned[:, block_start: block_end] == mask_id).sum() == 0:
                # print(f"decoded block {num_block} with {i} steps")
                if early_termination is True:
                    if (x_pruned[:, block_start:block_end] == eos_id).any():
                        x[:, block_end: ] = eos_id
                        return x, nfe
                break

    return x, nfe


@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, eos_id=126081, threshold=None, 
             dropout='null', sigma=None, scale=None, preserved_tokens=0, window=None, early_termination=True,
             local_window=128,
             use_suffix_soft_state=False, suffix_soft_topk=5, suffix_soft_alpha=0.5,
             suffix_soft_non_local_only=False, stats=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(_get_model_device(model))
    x[:, :prompt.shape[1]] = prompt.clone()
    seq_len = x.shape[1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    sampler = build_sampler(
        dropout, gen_length, block_length, sigma, scale, preserved_tokens, window, local_window
    )

    use_suffix_soft_state = bool(use_suffix_soft_state)
    suffix_soft_topk = max(1, int(suffix_soft_topk))
    suffix_soft_alpha = _clamp01(suffix_soft_alpha)
    # current_warm_start_beta = _clamp01(current_warm_start_beta)
    suffix_soft_non_local_only = bool(suffix_soft_non_local_only)
    emb_layer = None
    emb_weight = None
    mask_embed = None
    suffix_soft_state_cache = None
    suffix_soft_valid = None
    if use_suffix_soft_state:
        emb_layer = _get_model_input_embeddings(model)
        emb_weight = emb_layer.weight
        mask_embed = emb_weight[int(mask_id)].detach()
        suffix_soft_state_cache = torch.zeros(
            (int(seq_len), int(emb_weight.shape[-1])),
            device=emb_weight.device,
            dtype=emb_weight.dtype,
        )
        suffix_soft_valid = torch.zeros((int(seq_len),), device=emb_weight.device, dtype=torch.bool)
            
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length

        block_mask_indices = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_indices, steps)

        q_indices, k_indices = suffix_dropout(x, sampler, block_end)
        # q_indices: [:block_end] + [preserved_masks]
        # Since all the tokens following current block are masks, there is no need to use indices to get them.
        # This operation is basically equivalent to x_pruned = x.gather(1, q_indices), except that slicing will not create a copy of x.
        x_pruned = x[:,:q_indices.shape[1]]
        q_ref = q_indices
        available_suffix_tokens = max(0, x.shape[1] - block_end)
        kept_suffix_tokens = max(0, q_ref.shape[1] - block_end)
        _record_suffix_stats(stats, kept_suffix_tokens, available_suffix_tokens)

        qv = q_ref[0] if q_ref.dim() == 2 else q_ref

        model_inputs_embeds = None
        if use_suffix_soft_state:
            model_inputs_embeds = emb_layer(x_pruned)
            suffix_seq_pos = _select_suffix_soft_seq_positions(
                qv=qv,
                block_end=block_end,
                local_window=local_window,
                non_local_only=suffix_soft_non_local_only,
            )
            if suffix_seq_pos.numel() > 0:
                suffix_abs = qv[suffix_seq_pos].to(torch.long)
                suffix_abs, suffix_seq_pos = _dedup_sorted_positions(suffix_abs, suffix_seq_pos)
                known_mask = suffix_soft_valid[suffix_abs.to(device=suffix_soft_valid.device)]
                if bool(known_mask.any().item()):
                    known_abs = suffix_abs[known_mask].to(dtype=torch.long, device=suffix_soft_state_cache.device)
                    known_seq_pos = suffix_seq_pos[known_mask].to(dtype=torch.long, device=model_inputs_embeds.device)
                    model_inputs_embeds[:, known_seq_pos, :] = suffix_soft_state_cache[known_abs].to(
                        device=model_inputs_embeds.device,
                        dtype=model_inputs_embeds.dtype,
                    ).unsqueeze(0)

            if num_block > 0:
                current_block_positions = torch.arange(block_start, block_end, device=model_inputs_embeds.device, dtype=torch.long)
                model_inputs_embeds, _ = apply_current_block_warm_start(
                    inputs_embeds=model_inputs_embeds,
                    q_ref=q_ref,
                    current_block_positions=current_block_positions,
                    suffix_soft_states=suffix_soft_state_cache,
                    suffix_soft_valid=suffix_soft_valid,
                    mask_embedding=mask_embed,
                    beta=current_warm_start_beta,
                )

        output = model(
            x_pruned,
            inputs_embeds=model_inputs_embeds,
            use_cache=True,
            q_indices=q_indices,
            k_indices=k_indices,
            seq_len=seq_len,
            update_rope=True,
        )
        past_key_values = output.past_key_values
        logits = output.logits

        if use_suffix_soft_state:
            suffix_seq_pos = _select_suffix_soft_seq_positions(
                qv=qv,
                block_end=block_end,
                local_window=local_window,
                non_local_only=suffix_soft_non_local_only,
            )
            if suffix_seq_pos.numel() > 0:
                suffix_abs = qv[suffix_seq_pos].to(torch.long)
                suffix_abs, suffix_seq_pos = _dedup_sorted_positions(suffix_abs, suffix_seq_pos)
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
        mask_indices = (x_pruned == mask_id)
        mask_indices[:, block_end:] = 0

        i = 0
        x0, transfer_index = get_transfer_index(logits, 
                                                temperature, 
                                                remasking, 
                                                mask_indices, 
                                                x_pruned, 
                                                num_transfer_tokens[:, i] if threshold is None else None, 
                                                threshold=threshold)
        
        x_pruned[transfer_index] = x0[transfer_index]

        q_indices = q_indices[:,block_start:]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1

        while True:
            if (x_pruned[:, block_start:block_end] == mask_id).sum() == 0:
                # print(f"decoded block {num_block} with {i} steps")
                if early_termination is True:
                    if (x_pruned[:, block_start:block_end] == eos_id).any():
                        x[:, block_end: ] = eos_id
                        return x, nfe
                break
            nfe += 1
    
            model_inputs_embeds = None
            if use_suffix_soft_state:
                q_ref_step = q_indices
                qv_step = q_ref_step[0] if q_ref_step.dim() == 2 else q_ref_step
                model_inputs_embeds = emb_layer(x_pruned[:, block_start:])
                suffix_seq_pos = _select_suffix_soft_seq_positions(
                    qv=qv_step,
                    block_end=block_end,
                    local_window=local_window,
                    non_local_only=suffix_soft_non_local_only,
                )
                if suffix_seq_pos.numel() > 0:
                    suffix_abs = qv_step[suffix_seq_pos].to(torch.long)
                    suffix_abs, suffix_seq_pos = _dedup_sorted_positions(suffix_abs, suffix_seq_pos)
                    known_mask = suffix_soft_valid[suffix_abs.to(device=suffix_soft_valid.device)]
                    if bool(known_mask.any().item()):
                        known_abs = suffix_abs[known_mask].to(dtype=torch.long, device=suffix_soft_state_cache.device)
                        known_seq_pos = suffix_seq_pos[known_mask].to(dtype=torch.long, device=model_inputs_embeds.device)
                        model_inputs_embeds[:, known_seq_pos, :] = suffix_soft_state_cache[known_abs].to(
                            device=model_inputs_embeds.device,
                            dtype=model_inputs_embeds.dtype,
                        ).unsqueeze(0)

            logits = model(
                x_pruned[:, block_start:],
                inputs_embeds=model_inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                q_indices=q_indices,
                k_indices=k_indices,
                seq_len=seq_len,
                update_rope=(i==1),
            ).logits

            if use_suffix_soft_state:
                q_ref_step = q_indices
                qv_step = q_ref_step[0] if q_ref_step.dim() == 2 else q_ref_step
                suffix_seq_pos = _select_suffix_soft_seq_positions(
                    qv=qv_step,
                    block_end=block_end,
                    local_window=local_window,
                    non_local_only=suffix_soft_non_local_only,
                )
                if suffix_seq_pos.numel() > 0:
                    suffix_abs = qv_step[suffix_seq_pos].to(torch.long)
                    suffix_abs, suffix_seq_pos = _dedup_sorted_positions(suffix_abs, suffix_seq_pos)
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

            mask_indices = (x_pruned[:, block_start:] == mask_id)
            mask_indices[:, block_length:] = 0
            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_indices, x_pruned[:, block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold=threshold)

            x_pruned[:, block_start:][transfer_index] = x0[transfer_index]

            i += 1
    
    return x, nfe


@ torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, eos_id=126081, threshold=None, 
            dropout='null', sigma=None, scale=None, preserved_tokens=0, window=None, early_termination=True,
            local_window=128,
            use_suffix_soft_state=False, suffix_soft_topk=5, suffix_soft_alpha=0.5,
            suffix_soft_non_local_only=False, stats=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(_get_model_device(model))
    x[:, :prompt.shape[1]] = prompt.clone()
    seq_len = x.shape[1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  

    sampler = build_sampler(
        dropout, gen_length, block_length, sigma, scale, preserved_tokens, window, local_window
    )

    use_suffix_soft_state = bool(use_suffix_soft_state)
    suffix_soft_topk = max(1, int(suffix_soft_topk))
    suffix_soft_alpha = _clamp01(suffix_soft_alpha)
    # current_warm_start_beta = _clamp01(current_warm_start_beta)
    suffix_soft_non_local_only = bool(suffix_soft_non_local_only)

    emb_layer = None
    emb_weight = None
    mask_embed = None
    suffix_soft_state_cache = None
    suffix_soft_valid = None
    if use_suffix_soft_state:
        emb_layer = _get_model_input_embeddings(model)
        emb_weight = emb_layer.weight
        mask_embed = emb_weight[int(mask_id)].detach()
        suffix_soft_state_cache = torch.zeros(
            (int(seq_len), int(emb_weight.shape[-1])),
            device=emb_weight.device,
            dtype=emb_weight.dtype,
        )
        suffix_soft_valid = torch.zeros((int(seq_len),), device=emb_weight.device, dtype=torch.bool)
        
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = block_start + block_length

        block_mask_indices = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_indices, steps)

        q_indices, k_indices = suffix_dropout(x, sampler, block_end)
        # q_indices: [:block_end] + [preserved_masks]
        # Since all the tokens following current block are masks, there is no need to use indices to get them.
        # This operation is basically equivalent to x_pruned = x.gather(1, q_indices), except that slicing will not create a copy of x.
        x_pruned = x[:,:q_indices.shape[1]]
        q_ref = q_indices
        available_suffix_tokens = max(0, x.shape[1] - block_end)
        kept_suffix_tokens = max(0, q_ref.shape[1] - block_end)
        _record_suffix_stats(stats, kept_suffix_tokens, available_suffix_tokens)

        model_inputs_embeds = None
        if use_suffix_soft_state:
            qv = q_ref[0] if q_ref.dim() == 2 else q_ref
            suffix_seq_pos = _select_suffix_soft_seq_positions(
                qv=qv,
                block_end=block_end,
                local_window=local_window,
                non_local_only=suffix_soft_non_local_only,
            )
            suffix_abs = torch.tensor([], dtype=torch.long, device=qv.device)
            if suffix_seq_pos.numel() > 0:
                suffix_abs = qv[suffix_seq_pos].to(torch.long)
                suffix_abs, suffix_seq_pos = _dedup_sorted_positions(suffix_abs, suffix_seq_pos)

            def prepare_inputs_embeds():
                inputs_embeds = emb_layer(x_pruned)
                if suffix_abs.numel() > 0:
                    known_mask = suffix_soft_valid[suffix_abs.to(device=suffix_soft_valid.device)]
                    if bool(known_mask.any().item()):
                        known_abs = suffix_abs[known_mask].to(dtype=torch.long, device=suffix_soft_state_cache.device)
                        known_seq_pos = suffix_seq_pos[known_mask].to(dtype=torch.long, device=inputs_embeds.device)
                        inputs_embeds[:, known_seq_pos, :] = suffix_soft_state_cache[known_abs].to(
                            device=inputs_embeds.device,
                            dtype=inputs_embeds.dtype,
                        ).unsqueeze(0)

                if num_block > 0:
                    current_block_positions = torch.arange(
                        block_start, block_end, device=inputs_embeds.device, dtype=torch.long
                    )
                    inputs_embeds, _ = apply_current_block_warm_start(
                        inputs_embeds=inputs_embeds,
                        q_ref=q_ref,
                        current_block_positions=current_block_positions,
                        suffix_soft_states=suffix_soft_state_cache,
                        suffix_soft_valid=suffix_soft_valid,
                    )
                return inputs_embeds

            model_inputs_embeds = prepare_inputs_embeds()

            # Probe the selected future suffix once, then cache the mixed
            # representation for all remaining denoising steps in this block.
            if suffix_seq_pos.numel() > 0:
                probe_logits = model(
                    x_pruned,
                    inputs_embeds=model_inputs_embeds,
                    use_cache=False,
                    q_indices=q_indices,
                    k_indices=k_indices,
                    seq_len=seq_len,
                    update_rope=True,
                ).logits
                nfe += 1

                suffix_logits = probe_logits[0, suffix_seq_pos, :]
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
                model_inputs_embeds = prepare_inputs_embeds()

        # Build one mixed KV cache for this block. Subsequent denoising steps
        # only replace current-block entries and reuse the cached suffix.
        output = model(
            x_pruned,
            inputs_embeds=model_inputs_embeds,
            use_cache=True,
            q_indices=q_indices,
            k_indices=k_indices,
            seq_len=seq_len,
            update_rope=True,
        )
        past_key_values = output.past_key_values
        logits = output.logits
        mask_indices = (x_pruned == mask_id)
        mask_indices[:, block_end:] = 0

        i = 0
        x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_indices, x_pruned, num_transfer_tokens[:, i] if threshold is None else None, threshold=threshold)
        x_pruned[transfer_index] = x0[transfer_index]

        q_indices = q_indices[:,block_start:block_end]

        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x_pruned, dtype=torch.bool)
        replace_position[:, block_start:block_end] = 1
        
        while True:
            if (x_pruned[:, block_start:block_end] == mask_id).sum() == 0:
                # print(f"decoded block {num_block} with {i} steps")
                if early_termination is True:
                    if (x_pruned[:, block_start:block_end] == eos_id).any():
                        x[:, block_end: ] = eos_id
                        return x, nfe
                break

            nfe += 1
   
            logits = model(x_pruned[:, block_start: block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position, q_indices=q_indices, k_indices=k_indices, seq_len=seq_len, update_rope=(i==1)).logits
            mask_indices = (x_pruned[:, block_start: block_end] == mask_id)
 
            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_indices, x_pruned[:, block_start: block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold=threshold)
            x_pruned[:, block_start: block_end][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe
