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
# - [Briefly describe the changes you made]

import torch
import numpy as np
import torch.nn.functional as F
import json
import os
from collections import defaultdict
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM

def _record_suffix_stats(stats, kept_suffix_tokens, available_suffix_tokens):
    if stats is None:
        return
    stats["suffix_block_samples"] = stats.get("suffix_block_samples", 0) + 1
    stats["kept_suffix_tokens_total"] = stats.get("kept_suffix_tokens_total", 0) + int(kept_suffix_tokens)
    stats["available_suffix_tokens_total"] = stats.get("available_suffix_tokens_total", 0) + int(available_suffix_tokens)


def _record_suffix_bucket_stats(stats, block_end, full_seq_len):
    if stats is None:
        return

    available_suffix_tokens = max(0, int(full_seq_len) - int(block_end))
    if available_suffix_tokens <= 0:
        return

    # Vanilla keeps all suffix tokens, so kept == available in each bucket.
    b1 = (available_suffix_tokens + 2) // 3
    b2 = (2 * available_suffix_tokens + 2) // 3

    avail_near = b1
    avail_mid = max(0, b2 - b1)
    avail_far = max(0, available_suffix_tokens - b2)

    stats["suffix_kept_near_total"] = stats.get("suffix_kept_near_total", 0) + int(avail_near)
    stats["suffix_kept_mid_total"] = stats.get("suffix_kept_mid_total", 0) + int(avail_mid)
    stats["suffix_kept_far_total"] = stats.get("suffix_kept_far_total", 0) + int(avail_far)

    stats["suffix_available_near_total"] = stats.get("suffix_available_near_total", 0) + int(avail_near)
    stats["suffix_available_mid_total"] = stats.get("suffix_available_mid_total", 0) + int(avail_mid)
    stats["suffix_available_far_total"] = stats.get("suffix_available_far_total", 0) + int(avail_far)


def _phase_name(step_idx, total_steps):
    if total_steps <= 1:
        return "mid"
    ratio = float(step_idx) / float(max(total_steps - 1, 1))
    if ratio < (1.0 / 3.0):
        return "early"
    if ratio < (2.0 / 3.0):
        return "mid"
    return "late"


def _record_uncertainty_stats(stats, logits, mask_index, step_idx, total_steps, enabled=False):
    if stats is None or (not enabled):
        return

    masked_logits = logits[mask_index]
    if masked_logits.numel() == 0:
        return

    probs = F.softmax(masked_logits.to(torch.float32), dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]

    count = int(entropy.numel())
    entropy_sum = float(entropy.sum().item())
    margin_sum = float(margin.sum().item())

    stats["uncertainty_mask_count"] = stats.get("uncertainty_mask_count", 0) + count
    stats["uncertainty_entropy_sum"] = stats.get("uncertainty_entropy_sum", 0.0) + entropy_sum
    stats["uncertainty_margin_sum"] = stats.get("uncertainty_margin_sum", 0.0) + margin_sum

    phase = _phase_name(step_idx=step_idx, total_steps=total_steps)
    stats[f"uncertainty_{phase}_count"] = stats.get(f"uncertainty_{phase}_count", 0) + count
    stats[f"uncertainty_{phase}_entropy_sum"] = stats.get(f"uncertainty_{phase}_entropy_sum", 0.0) + entropy_sum
    stats[f"uncertainty_{phase}_margin_sum"] = stats.get(f"uncertainty_{phase}_margin_sum", 0.0) + margin_sum


def _update_mean_stats(stats, prefix, val, count):
    stats[f"{prefix}_sum"] = stats.get(f"{prefix}_sum", 0.0) + float(val)
    stats[f"{prefix}_count"] = stats.get(f"{prefix}_count", 0) + int(count)


def _finalize_suffix_top1_first_block_stats(stats):
    if stats is None:
        return
    c = int(stats.get("suffix_top1_first_block_obs_count", 0))
    if c <= 0:
        return

    s = float(stats.get("suffix_top1_first_block_sum", 0.0))
    s2 = float(stats.get("suffix_top1_first_block_sq_sum", 0.0))
    mean = s / float(c)
    var = max(0.0, (s2 / float(c)) - mean * mean)
    stats["suffix_top1_first_block_mean"] = float(mean)
    stats["suffix_top1_first_block_std"] = float(np.sqrt(var))

    interval_keys = [
        "suffix_top1_first_block_count_lt_0p3",
        "suffix_top1_first_block_count_0p3_0p5",
        "suffix_top1_first_block_count_0p5_0p7",
        "suffix_top1_first_block_count_ge_0p7",
    ]
    for k in interval_keys:
        v = int(stats.get(k, 0))
        stats[f"{k}_ratio"] = float(v / float(c))

    for p in ["start", "middle", "end"]:
        ps = float(stats.get(f"suffix_top1_pos_{p}_sum", 0.0))
        pc = int(stats.get(f"suffix_top1_pos_{p}_count", 0))
        if pc > 0:
            stats[f"suffix_top1_pos_{p}_mean"] = float(ps / float(pc))

    for d in ["n1", "n2", "n3p"]:
        ds = float(stats.get(f"suffix_top1_dist_{d}_sum", 0.0))
        dc = int(stats.get(f"suffix_top1_dist_{d}_count", 0))
        if dc > 0:
            stats[f"suffix_top1_dist_{d}_mean"] = float(ds / float(dc))

    start_c = int(stats.get("suffix_top1_start_token_count", 0))
    other_c = int(stats.get("suffix_top1_nonstart_token_count", 0))
    if start_c > 0:
        stats["suffix_top1_start_token_mean"] = float(
            stats.get("suffix_top1_start_token_sum", 0.0) / float(start_c)
        )
    if other_c > 0:
        stats["suffix_top1_nonstart_token_mean"] = float(
            stats.get("suffix_top1_nonstart_token_sum", 0.0) / float(other_c)
        )


def _record_suffix_top1_first_block_stats(
    stats,
    logits,
    block_end,
    block_length,
    step_idx,
    enabled=False,
):
    if stats is None or (not enabled):
        return
    if logits is None or logits.ndim != 3:
        return

    seq_len = int(logits.shape[1])
    block_end = int(block_end)
    if block_end >= seq_len:
        return

    bl = max(1, int(block_length))
    suffix_logits = logits[0, block_end:, :]
    if suffix_logits.numel() == 0:
        return

    top1 = F.softmax(suffix_logits.to(torch.float32), dim=-1).amax(dim=-1)
    if top1.numel() == 0:
        return
    v = top1.detach().cpu().numpy().astype(np.float64, copy=False)
    suffix_len = int(v.shape[0])

    stats["suffix_top1_first_block_enabled"] = True
    stats["suffix_top1_first_block_steps_captured"] = stats.get("suffix_top1_first_block_steps_captured", 0) + 1
    stats["suffix_top1_first_block_last_step"] = int(step_idx)
    if "suffix_top1_first_block_token_count" not in stats:
        stats["suffix_top1_first_block_token_count"] = int(suffix_len)

    obs_count = int(v.size)
    obs_sum = float(v.sum())
    obs_sq_sum = float(np.square(v).sum())
    obs_min = float(v.min())
    obs_max = float(v.max())

    stats["suffix_top1_first_block_obs_count"] = stats.get("suffix_top1_first_block_obs_count", 0) + obs_count
    stats["suffix_top1_first_block_sum"] = stats.get("suffix_top1_first_block_sum", 0.0) + obs_sum
    stats["suffix_top1_first_block_sq_sum"] = stats.get("suffix_top1_first_block_sq_sum", 0.0) + obs_sq_sum
    if "suffix_top1_first_block_min" not in stats:
        stats["suffix_top1_first_block_min"] = obs_min
        stats["suffix_top1_first_block_max"] = obs_max
    else:
        stats["suffix_top1_first_block_min"] = float(min(float(stats["suffix_top1_first_block_min"]), obs_min))
        stats["suffix_top1_first_block_max"] = float(max(float(stats["suffix_top1_first_block_max"]), obs_max))

    stats["suffix_top1_first_block_count_lt_0p3"] = stats.get("suffix_top1_first_block_count_lt_0p3", 0) + int((v < 0.3).sum())
    stats["suffix_top1_first_block_count_0p3_0p5"] = stats.get("suffix_top1_first_block_count_0p3_0p5", 0) + int(((v >= 0.3) & (v < 0.5)).sum())
    stats["suffix_top1_first_block_count_0p5_0p7"] = stats.get("suffix_top1_first_block_count_0p5_0p7", 0) + int(((v >= 0.5) & (v < 0.7)).sum())
    stats["suffix_top1_first_block_count_ge_0p7"] = stats.get("suffix_top1_first_block_count_ge_0p7", 0) + int((v >= 0.7).sum())

    hist_bins = int(stats.get("suffix_top1_hist_num_bins", 20))
    hist_bins = max(4, hist_bins)
    hist_edges = np.linspace(0.0, 1.0, hist_bins + 1, dtype=np.float64)
    hist_counts, _ = np.histogram(v, bins=hist_edges)
    if "suffix_top1_hist_counts" not in stats:
        stats["suffix_top1_hist_counts"] = [0] * hist_bins
        stats["suffix_top1_hist_edges"] = [float(x) for x in hist_edges.tolist()]
        stats["suffix_top1_hist_num_bins"] = int(hist_bins)
    cur_hist = stats.get("suffix_top1_hist_counts", [0] * hist_bins)
    if len(cur_hist) != hist_bins:
        cur_hist = [0] * hist_bins
    stats["suffix_top1_hist_counts"] = [int(cur_hist[i]) + int(hist_counts[i]) for i in range(hist_bins)]

    n_rel_blocks = int((suffix_len + bl - 1) // bl)
    for rel_b in range(1, n_rel_blocks + 1):
        lo = int((rel_b - 1) * bl)
        hi = int(min(rel_b * bl, suffix_len))
        seg = v[lo:hi]
        if seg.size <= 0:
            continue

        _update_mean_stats(stats, f"suffix_top1_rel_block_{rel_b}", float(seg.sum()), int(seg.size))

        if rel_b == 1:
            dist_key = "n1"
        elif rel_b == 2:
            dist_key = "n2"
        else:
            dist_key = "n3p"
        _update_mean_stats(stats, f"suffix_top1_dist_{dist_key}", float(seg.sum()), int(seg.size))

        seg_len = int(seg.size)
        start_v = float(seg[0])
        mid_v = float(seg[(seg_len - 1) // 2])
        end_v = float(seg[-1])
        _update_mean_stats(stats, "suffix_top1_pos_start", start_v, 1)
        _update_mean_stats(stats, "suffix_top1_pos_middle", mid_v, 1)
        _update_mean_stats(stats, "suffix_top1_pos_end", end_v, 1)

        stats["suffix_top1_start_token_sum"] = stats.get("suffix_top1_start_token_sum", 0.0) + start_v
        stats["suffix_top1_start_token_count"] = stats.get("suffix_top1_start_token_count", 0) + 1
        if seg_len > 1:
            nonstart = seg[1:]
            stats["suffix_top1_nonstart_token_sum"] = (
                stats.get("suffix_top1_nonstart_token_sum", 0.0) + float(nonstart.sum())
            )
            stats["suffix_top1_nonstart_token_count"] = (
                stats.get("suffix_top1_nonstart_token_count", 0) + int(nonstart.size)
            )

    _finalize_suffix_top1_first_block_stats(stats)


def _resolve_figure4a_capture_spec(prompt_len, seq_len, gen_length, block_length, figure4a_block_id, figure4a_c):
    num_blocks = gen_length // block_length

    if int(figure4a_block_id) >= 0:
        block_id = int(figure4a_block_id)
        if block_id >= num_blocks:
            return None
        query_start = prompt_len + block_id * block_length
        query_end = min(query_start + block_length, seq_len)
        if query_start >= seq_len or query_end <= query_start:
            return None
        return {
            'mode': 'block_id',
            'target_block_idx': block_id,
            'query_start': int(query_start),
            'query_end': int(query_end),
            'suffix_start': int(query_end),
            'figure4a_block_id': int(block_id),
            'figure4a_c': -1,
        }

    c = int(figure4a_c)
    if c < 0 or c >= seq_len:
        return None

    if c < prompt_len:
        target_block_idx = 0
    else:
        target_block_idx = (c - prompt_len) // block_length
        if target_block_idx >= num_blocks:
            return None

    # c-mode: query from token c to the end of the block where c belongs.
    block_start = (c // block_length) * block_length
    block_end = min(block_start + block_length, seq_len)
    query_start = c
    query_end = max(c + 1, block_end)
    query_end = min(query_end, seq_len)
    if query_end <= query_start:
        return None

    return {
        'mode': 'c',
        'target_block_idx': int(target_block_idx),
        'query_start': int(query_start),
        'query_end': int(query_end),
        'suffix_start': int(query_end),
        'figure4a_block_id': -1,
        'figure4a_c': int(c),
    }


def _extract_figure4a_layer_scores(attentions, query_start, query_end, suffix_start):
    if attentions is None:
        return None

    layer_scores = []
    for attn in attentions:
        if attn is None:
            continue
        if attn.ndim != 4:
            continue

        lq = int(attn.shape[-2])
        lk = int(attn.shape[-1])
        q_lo = int(max(0, query_start))
        q_hi = int(min(query_end, lq))
        k_lo = int(min(max(0, suffix_start), lk))

        if q_hi <= q_lo or k_lo >= lk:
            continue

        # [H, B, S] -> average over head/query to get [S]
        x = attn[0, :, q_lo:q_hi, k_lo:].to(torch.float32)
        if x.numel() == 0:
            continue
        layer_scores.append(x.mean(dim=(0, 1)).detach().cpu().numpy())

    if not layer_scores:
        return None

    min_suffix_len = min(int(v.shape[0]) for v in layer_scores)
    if min_suffix_len <= 0:
        return None
    if any(int(v.shape[0]) != min_suffix_len for v in layer_scores):
        layer_scores = [v[:min_suffix_len] for v in layer_scores]

    return np.stack(layer_scores, axis=0).astype(np.float32, copy=False)


def _extract_figure4a_suffix_block_metrics(attentions, query_start, query_end, suffix_start, block_size):
    if attentions is None:
        return None
    if int(block_size) <= 0:
        return None

    layer_scores = []
    block_score_list = []
    block_entropy_list = []
    block_first_list = []
    block_last_list = []
    block_delta_list = []

    for attn in attentions:
        if attn is None or attn.ndim != 4:
            continue

        lq = int(attn.shape[-2])
        lk = int(attn.shape[-1])
        q_lo = int(max(0, query_start))
        q_hi = int(min(query_end, lq))
        k_lo = int(min(max(0, suffix_start), lk))
        if q_hi <= q_lo or k_lo >= lk:
            continue

        # [H, Q, S] -> [S]
        x = attn[0, :, q_lo:q_hi, k_lo:].to(torch.float32)
        if x.numel() == 0:
            continue
        token_scores = x.mean(dim=(0, 1)).detach().cpu().numpy().astype(np.float32, copy=False)
        if token_scores.size == 0:
            continue
        layer_scores.append(token_scores)

        n_suffix = int(token_scores.shape[0])
        n_blocks = int((n_suffix + block_size - 1) // block_size)

        bs = np.full((n_blocks,), np.nan, dtype=np.float32)
        be = np.full((n_blocks,), np.nan, dtype=np.float32)
        bf = np.full((n_blocks,), np.nan, dtype=np.float32)
        bl = np.full((n_blocks,), np.nan, dtype=np.float32)
        bd = np.full((n_blocks,), np.nan, dtype=np.float32)

        for b in range(n_blocks):
            lo = b * int(block_size)
            hi = min((b + 1) * int(block_size), n_suffix)
            vals = token_scores[lo:hi]
            if vals.size == 0:
                continue

            score = float(np.sum(vals))
            mass = max(score, 1e-12)
            probs = vals / mass
            ent = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())
            if vals.size > 1:
                ent = ent / float(np.log(vals.size))
            else:
                ent = 0.0

            first = float(vals[0])
            last = float(vals[-1])

            bs[b] = score
            be[b] = float(ent)
            bf[b] = first
            bl[b] = last
            bd[b] = first - last

        block_score_list.append(bs)
        block_entropy_list.append(be)
        block_first_list.append(bf)
        block_last_list.append(bl)
        block_delta_list.append(bd)

    if not layer_scores:
        return None

    min_suffix_len = min(int(v.shape[0]) for v in layer_scores)
    layer_scores = [v[:min_suffix_len] for v in layer_scores]
    layer_scores_arr = np.stack(layer_scores, axis=0).astype(np.float32, copy=False)

    max_blocks = max(int(v.shape[0]) for v in block_score_list)
    n_layers = len(block_score_list)

    def _stack_block_arrays(arr_list):
        out = np.full((n_layers, max_blocks), np.nan, dtype=np.float32)
        for li, arr in enumerate(arr_list):
            out[li, :arr.shape[0]] = arr
        return out

    return {
        'layer_scores': layer_scores_arr,
        'block_score': _stack_block_arrays(block_score_list),
        'block_entropy': _stack_block_arrays(block_entropy_list),
        'block_first_score': _stack_block_arrays(block_first_list),
        'block_last_score': _stack_block_arrays(block_last_list),
        'block_first_last_delta': _stack_block_arrays(block_delta_list),
        'num_suffix_blocks': int(max_blocks),
        'suffix_len': int(min_suffix_len),
    }


def _build_observation_target_positions(block_end, block_length, seq_len, k):
    k = max(1, int(k))
    if block_end >= seq_len:
        return []
    next_block_end = min(int(block_end) + int(block_length), int(seq_len))
    if next_block_end <= int(block_end):
        return []
    return list(range(int(block_end), min(int(block_end) + k, next_block_end)))


def _record_observation_target_uncertainty(stats, logits, target_positions, step_idx):
    if stats is None or len(target_positions) == 0:
        return
    idx = torch.tensor(target_positions, device=logits.device, dtype=torch.long)
    tgt_logits = logits[0, idx, :]
    if tgt_logits.numel() == 0:
        return
    probs = F.softmax(tgt_logits.to(torch.float32), dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    confidence = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1]

    c = int(entropy.numel())
    e_sum = float(entropy.sum().item())
    conf_sum = float(confidence.sum().item())
    margin_sum = float(margin.sum().item())

    stats["obs_target_count_total"] = stats.get("obs_target_count_total", 0) + c
    stats["obs_target_entropy_sum_total"] = stats.get("obs_target_entropy_sum_total", 0.0) + e_sum
    stats["obs_target_confidence_sum_total"] = stats.get("obs_target_confidence_sum_total", 0.0) + conf_sum
    stats["obs_target_margin_sum_total"] = stats.get("obs_target_margin_sum_total", 0.0) + margin_sum

    stats[f"obs_target_count_step_{step_idx}"] = stats.get(f"obs_target_count_step_{step_idx}", 0) + c
    stats[f"obs_target_entropy_sum_step_{step_idx}"] = stats.get(f"obs_target_entropy_sum_step_{step_idx}", 0.0) + e_sum
    stats[f"obs_target_confidence_sum_step_{step_idx}"] = stats.get(f"obs_target_confidence_sum_step_{step_idx}", 0.0) + conf_sum
    stats[f"obs_target_margin_sum_step_{step_idx}"] = stats.get(f"obs_target_margin_sum_step_{step_idx}", 0.0) + margin_sum

    # Per-target-slot logging (target slot t means the t-th token in block n+1).
    # This keeps step-wise records aligned with "block n+1 first-k token" requirement.
    for t in range(c):
        ent_t = float(entropy[t].item())
        conf_t = float(confidence[t].item())
        mar_t = float(margin[t].item())
        abs_pos_t = int(target_positions[t])

        stats[f"obs_target_t{t}_count_total"] = stats.get(f"obs_target_t{t}_count_total", 0) + 1
        stats[f"obs_target_t{t}_entropy_sum_total"] = stats.get(f"obs_target_t{t}_entropy_sum_total", 0.0) + ent_t
        stats[f"obs_target_t{t}_confidence_sum_total"] = stats.get(f"obs_target_t{t}_confidence_sum_total", 0.0) + conf_t
        stats[f"obs_target_t{t}_margin_sum_total"] = stats.get(f"obs_target_t{t}_margin_sum_total", 0.0) + mar_t
        stats[f"obs_target_t{t}_abs_pos_sum_total"] = stats.get(f"obs_target_t{t}_abs_pos_sum_total", 0.0) + float(abs_pos_t)

        stats[f"obs_target_t{t}_count_step_{step_idx}"] = stats.get(f"obs_target_t{t}_count_step_{step_idx}", 0) + 1
        stats[f"obs_target_t{t}_entropy_sum_step_{step_idx}"] = stats.get(f"obs_target_t{t}_entropy_sum_step_{step_idx}", 0.0) + ent_t
        stats[f"obs_target_t{t}_confidence_sum_step_{step_idx}"] = stats.get(f"obs_target_t{t}_confidence_sum_step_{step_idx}", 0.0) + conf_t
        stats[f"obs_target_t{t}_margin_sum_step_{step_idx}"] = stats.get(f"obs_target_t{t}_margin_sum_step_{step_idx}", 0.0) + mar_t
        stats[f"obs_target_t{t}_abs_pos_sum_step_{step_idx}"] = stats.get(f"obs_target_t{t}_abs_pos_sum_step_{step_idx}", 0.0) + float(abs_pos_t)


def _slice_key_scores_for_observation(attentions, target_positions, suffix_start):
    if attentions is None or len(target_positions) == 0:
        return None
    per_layer = []
    for attn in attentions:
        if attn is None or attn.ndim != 4:
            continue
        lq = int(attn.shape[-2])
        lk = int(attn.shape[-1])
        valid_q = [int(p) for p in target_positions if 0 <= int(p) < lq]
        if len(valid_q) == 0:
            continue
        k_lo = int(min(max(0, int(suffix_start)), lk))
        if k_lo >= lk:
            continue
        q_idx = torch.tensor(valid_q, device=attn.device, dtype=torch.long)
        x = attn[0, :, q_idx, k_lo:].to(torch.float32)  # [H, Q, S]
        if x.numel() == 0:
            continue
        per_layer.append(x.mean(dim=(0, 1)))  # [S]
    if len(per_layer) == 0:
        return None
    min_len = min(int(v.shape[0]) for v in per_layer)
    if min_len <= 0:
        return None
    per_layer = [v[:min_len] for v in per_layer]
    return torch.stack(per_layer, dim=0).mean(dim=0)  # [S]


def _pick_pos_indices_in_block(block_len, pos_name, k, rng):
    if block_len <= 0:
        return []
    k = min(max(1, int(k)), int(block_len))
    if pos_name == "start":
        return list(range(0, k))
    if pos_name == "end":
        return list(range(int(block_len) - k, int(block_len)))
    if pos_name == "middle":
        center = (int(block_len) - 1) // 2
        half = k // 2
        lo = max(0, center - half)
        hi = lo + k
        if hi > int(block_len):
            hi = int(block_len)
            lo = hi - k
        return list(range(lo, hi))
    # random
    pool = list(range(int(block_len)))
    if k >= len(pool):
        return pool
    sel = rng.choice(np.asarray(pool, dtype=np.int64), size=k, replace=False)
    return sorted(int(v) for v in sel.tolist())


def _record_observation_attention_stats(
    stats,
    attentions,
    target_positions,
    block_end,
    seq_len,
    block_length,
    k,
    step_idx,
    seed,
):
    if stats is None or attentions is None or len(attentions) == 0:
        return
    key_scores = _slice_key_scores_for_observation(
        attentions=attentions,
        target_positions=target_positions,
        suffix_start=block_end,
    )
    if key_scores is None:
        return

    suffix_len = int(min(int(seq_len) - int(block_end), int(key_scores.shape[0])))
    if suffix_len <= 0:
        return
    key_scores = key_scores[:suffix_len]
    total_mass = float(key_scores.sum().item())
    if total_mass <= 0.0:
        return

    n_blocks = int((suffix_len + int(block_length) - 1) // int(block_length))
    if n_blocks <= 0:
        return

    pos_names = ["start", "middle", "end", "random"]
    pos_mass = {p: 0.0 for p in pos_names}
    pos_count = {p: 0 for p in pos_names}
    rng = np.random.default_rng(int(seed) + int(step_idx) * 1000003 + int(block_end))

    for b in range(n_blocks):
        lo = b * int(block_length)
        hi = min((b + 1) * int(block_length), suffix_len)
        blen = int(hi - lo)
        if blen <= 0:
            continue
        for p in pos_names:
            rel_idx = _pick_pos_indices_in_block(block_len=blen, pos_name=p, k=k, rng=rng)
            if len(rel_idx) == 0:
                continue
            abs_idx = torch.tensor([lo + int(r) for r in rel_idx], device=key_scores.device, dtype=torch.long)
            pos_mass[p] += float(key_scores[abs_idx].sum().item())
            pos_count[p] += int(len(rel_idx))

    stats["obs_suffix_total_mass_sum_total"] = stats.get("obs_suffix_total_mass_sum_total", 0.0) + float(total_mass)
    stats[f"obs_suffix_total_mass_sum_step_{step_idx}"] = stats.get(f"obs_suffix_total_mass_sum_step_{step_idx}", 0.0) + float(total_mass)
    stats[f"obs_suffix_block_count_step_{step_idx}"] = stats.get(f"obs_suffix_block_count_step_{step_idx}", 0) + int(n_blocks)

    for p in pos_names:
        mass = float(pos_mass[p])
        count = int(pos_count[p])
        ratio = mass / max(total_mass, 1e-12)
        routing = ratio
        stats[f"obs_pos_{p}_count_total"] = stats.get(f"obs_pos_{p}_count_total", 0) + count
        stats[f"obs_attn_mass_{p}_sum_total"] = stats.get(f"obs_attn_mass_{p}_sum_total", 0.0) + mass
        stats[f"obs_attn_ratio_{p}_sum_total"] = stats.get(f"obs_attn_ratio_{p}_sum_total", 0.0) + ratio
        stats[f"obs_routing_{p}_sum_total"] = stats.get(f"obs_routing_{p}_sum_total", 0.0) + routing

        stats[f"obs_pos_{p}_count_step_{step_idx}"] = stats.get(f"obs_pos_{p}_count_step_{step_idx}", 0) + count
        stats[f"obs_attn_mass_{p}_sum_step_{step_idx}"] = stats.get(f"obs_attn_mass_{p}_sum_step_{step_idx}", 0.0) + mass
        stats[f"obs_attn_ratio_{p}_sum_step_{step_idx}"] = stats.get(f"obs_attn_ratio_{p}_sum_step_{step_idx}", 0.0) + ratio
        stats[f"obs_routing_{p}_sum_step_{step_idx}"] = stats.get(f"obs_routing_{p}_sum_step_{step_idx}", 0.0) + routing


def _build_intervention_ablation_positions(seq_len, block_end, block_length, budget, mode, seed, step_idx):
    """
    Build absolute suffix positions to be virtually masked/ablated under the same budget.
    Modes: mask_start, mask_end, mask_random, mask_start_end
    """
    suffix_start = int(block_end)
    suffix_end = int(seq_len)
    suffix_len = max(0, suffix_end - suffix_start)
    k = max(0, min(int(budget), suffix_len))
    if k == 0 or suffix_len == 0:
        return []

    # Future block boundaries in suffix region.
    bs = max(1, int(block_length))
    blocks = []
    cur = suffix_start
    while cur < suffix_end:
        b_lo = cur
        b_hi = min(cur + bs, suffix_end)  # exclusive
        blocks.append((b_lo, b_hi))
        cur = b_hi

    starts = [b[0] for b in blocks if b[0] < b[1]]
    ends = [b[1] - 1 for b in blocks if b[0] < b[1]]
    all_suffix = list(range(suffix_start, suffix_end))

    if mode == "mask_start":
        base = starts if len(starts) > 0 else all_suffix
        out = base[:k]
    elif mode == "mask_end":
        base = ends if len(ends) > 0 else all_suffix
        out = base[-k:] if len(base) >= k else base
    elif mode == "mask_start_end":
        k_start = k // 2
        k_end = k - k_start
        left = (starts[:k_start] if len(starts) > 0 else all_suffix[:k_start])
        right = (ends[-k_end:] if len(ends) > 0 else all_suffix[-k_end:])
        out = left + right
    elif mode == "mask_random":
        rng = np.random.default_rng(int(seed) + 104729 * int(step_idx) + 1009 * int(block_end))
        if k >= len(all_suffix):
            out = all_suffix
        else:
            picked = rng.choice(np.asarray(all_suffix, dtype=np.int64), size=k, replace=False)
            out = sorted(int(v) for v in picked.tolist())
    else:
        raise ValueError(f"Unknown intervention mode: {mode}")

    # Dedup and trim to the same budget.
    out = sorted(set(int(v) for v in out if suffix_start <= int(v) < suffix_end))
    if len(out) > k:
        out = out[:k]
    return out


def _record_observation_intervention_target_uncertainty(
    stats, logits, target_positions, step_idx, mode, q_abs_positions=None
):
    if stats is None or len(target_positions) == 0:
        return
    if q_abs_positions is not None:
        pos_map = {int(p): i for i, p in enumerate(q_abs_positions.to(torch.long).tolist())}
        local_idx = [pos_map[int(p)] for p in target_positions if int(p) in pos_map]
        if len(local_idx) == 0:
            return
        idx = torch.tensor(local_idx, device=logits.device, dtype=torch.long)
    else:
        idx = torch.tensor(target_positions, device=logits.device, dtype=torch.long)
    tgt_logits = logits[0, idx, :]
    if tgt_logits.numel() == 0:
        return
    probs = F.softmax(tgt_logits.to(torch.float32), dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    confidence = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1]

    c = int(entropy.numel())
    e_sum = float(entropy.sum().item())
    conf_sum = float(confidence.sum().item())
    margin_sum = float(margin.sum().item())

    pfx = f"obs_intv_{mode}_target"
    stats[f"{pfx}_count_total"] = stats.get(f"{pfx}_count_total", 0) + c
    stats[f"{pfx}_entropy_sum_total"] = stats.get(f"{pfx}_entropy_sum_total", 0.0) + e_sum
    stats[f"{pfx}_confidence_sum_total"] = stats.get(f"{pfx}_confidence_sum_total", 0.0) + conf_sum
    stats[f"{pfx}_margin_sum_total"] = stats.get(f"{pfx}_margin_sum_total", 0.0) + margin_sum

    stats[f"{pfx}_count_step_{step_idx}"] = stats.get(f"{pfx}_count_step_{step_idx}", 0) + c
    stats[f"{pfx}_entropy_sum_step_{step_idx}"] = stats.get(f"{pfx}_entropy_sum_step_{step_idx}", 0.0) + e_sum
    stats[f"{pfx}_confidence_sum_step_{step_idx}"] = stats.get(f"{pfx}_confidence_sum_step_{step_idx}", 0.0) + conf_sum
    stats[f"{pfx}_margin_sum_step_{step_idx}"] = stats.get(f"{pfx}_margin_sum_step_{step_idx}", 0.0) + margin_sum


def _record_attention_rollout_stats(
    stats,
    attentions,
    block_start,
    block_end,
    seq_len,
    block_length,
    step_idx,
    residual_alpha=0.5,
    rollout_matrix=None,
):
    """
    Attention rollout for structural routing analysis.
    Report Q(current block)->Start and Q->NonStart path strengths.
    """
    if stats is None or attentions is None or len(attentions) == 0:
        return

    # Build rollout matrix R = Π_l (alpha*I + (1-alpha)*A_l)
    a0 = attentions[0]
    if a0 is None or a0.ndim != 4:
        return
    seq_n = int(a0.shape[-1])
    if seq_n <= 0:
        return
    device = a0.device
    dtype = torch.float32
    I = torch.eye(seq_n, device=device, dtype=dtype)
    R = I.clone()

    alpha = float(max(0.0, min(1.0, residual_alpha)))
    eps = 1e-8
    for attn in attentions:
        if attn is None or attn.ndim != 4:
            continue
        A = attn[0].to(dtype).mean(dim=0)  # [S, S]
        if A.shape[0] != seq_n or A.shape[1] != seq_n:
            continue
        A_hat = alpha * I + (1.0 - alpha) * A
        A_hat = A_hat / A_hat.sum(dim=-1, keepdim=True).clamp_min(eps)
        R = R @ A_hat

    q_lo = int(max(0, block_start))
    q_hi = int(min(block_end, seq_n))
    s_lo = int(min(max(0, block_end), seq_n))
    s_hi = int(min(seq_len, seq_n))
    if q_hi <= q_lo or s_hi <= s_lo:
        return

    q_idx = torch.arange(q_lo, q_hi, device=device, dtype=torch.long)
    suffix_idx = torch.arange(s_lo, s_hi, device=device, dtype=torch.long)
    if suffix_idx.numel() == 0:
        return

    bs = max(1, int(block_length))
    start_tokens = torch.arange(s_lo, s_hi, bs, device=device, dtype=torch.long)
    start_tokens = start_tokens[(start_tokens >= s_lo) & (start_tokens < s_hi)]
    if start_tokens.numel() == 0:
        start_tokens = suffix_idx[:1]

    nonstart_mask = ~torch.isin(suffix_idx, start_tokens)
    nonstart_tokens = suffix_idx[nonstart_mask]
    if nonstart_tokens.numel() == 0:
        nonstart_tokens = start_tokens

    q_to_start = float(R[q_idx][:, start_tokens].mean().item())
    q_to_nonstart = float(R[q_idx][:, nonstart_tokens].mean().item())
    ratio = q_to_start / max(q_to_nonstart, eps)
    gap = q_to_start - q_to_nonstart

    stats["rollout_samples_total"] = stats.get("rollout_samples_total", 0) + 1
    stats["rollout_q_to_start_sum_total"] = stats.get("rollout_q_to_start_sum_total", 0.0) + q_to_start
    stats["rollout_q_to_nonstart_sum_total"] = stats.get("rollout_q_to_nonstart_sum_total", 0.0) + q_to_nonstart
    stats["rollout_ratio_sum_total"] = stats.get("rollout_ratio_sum_total", 0.0) + ratio
    stats["rollout_gap_sum_total"] = stats.get("rollout_gap_sum_total", 0.0) + gap

    stats[f"rollout_samples_step_{step_idx}"] = stats.get(f"rollout_samples_step_{step_idx}", 0) + 1
    stats[f"rollout_q_to_start_sum_step_{step_idx}"] = stats.get(f"rollout_q_to_start_sum_step_{step_idx}", 0.0) + q_to_start
    stats[f"rollout_q_to_nonstart_sum_step_{step_idx}"] = stats.get(f"rollout_q_to_nonstart_sum_step_{step_idx}", 0.0) + q_to_nonstart
    stats[f"rollout_ratio_sum_step_{step_idx}"] = stats.get(f"rollout_ratio_sum_step_{step_idx}", 0.0) + ratio
    stats[f"rollout_gap_sum_step_{step_idx}"] = stats.get(f"rollout_gap_sum_step_{step_idx}", 0.0) + gap


def _compute_rollout_matrix(attentions, residual_alpha=0.5):
    if attentions is None or len(attentions) == 0:
        return None
    a0 = attentions[0]
    if a0 is None or a0.ndim != 4:
        return None
    seq_n = int(a0.shape[-1])
    if seq_n <= 0:
        return None
    device = a0.device
    dtype = torch.float32
    I = torch.eye(seq_n, device=device, dtype=dtype)
    R = I.clone()
    eps = 1e-8
    alpha = float(max(0.0, min(1.0, residual_alpha)))
    for attn in attentions:
        if attn is None or attn.ndim != 4:
            continue
        A = attn[0].to(dtype).mean(dim=0)
        if A.shape[0] != seq_n or A.shape[1] != seq_n:
            continue
        A_hat = alpha * I + (1.0 - alpha) * A
        A_hat = A_hat / A_hat.sum(dim=-1, keepdim=True).clamp_min(eps)
        R = R @ A_hat
    return R


def _record_rollout_offset_profile_stats(
    stats,
    rollout_matrix,
    block_start,
    block_end,
    seq_len,
    block_length,
    step_idx,
):
    """
    For all suffix future blocks, record Q(current)->token rollout scores by (block_id, offset).
    """
    if stats is None or rollout_matrix is None:
        return
    seq_n = int(rollout_matrix.shape[-1])
    q_lo = int(max(0, block_start))
    q_hi = int(min(block_end, seq_n))
    s_lo = int(min(max(0, block_end), seq_n))
    s_hi = int(min(seq_len, seq_n))
    if q_hi <= q_lo or s_hi <= s_lo:
        return

    q_idx = torch.arange(q_lo, q_hi, device=rollout_matrix.device, dtype=torch.long)
    bs = max(1, int(block_length))

    stats["rollout_offset_profile_samples_total"] = stats.get("rollout_offset_profile_samples_total", 0) + 1
    stats[f"rollout_offset_profile_samples_step_{step_idx}"] = stats.get(f"rollout_offset_profile_samples_step_{step_idx}", 0) + 1

    for tok in range(s_lo, s_hi):
        rel = tok - s_lo
        block_id = int(rel // bs)
        offset = int(rel % bs)
        score = float(rollout_matrix[q_idx, tok].mean().item())
        stats[f"rollout_offset_sum_step_{step_idx}_block_{block_id}_offset_{offset}"] = (
            stats.get(f"rollout_offset_sum_step_{step_idx}_block_{block_id}_offset_{offset}", 0.0) + score
        )
        stats[f"rollout_offset_count_step_{step_idx}_block_{block_id}_offset_{offset}"] = (
            stats.get(f"rollout_offset_count_step_{step_idx}_block_{block_id}_offset_{offset}", 0) + 1
        )

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


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def _probe_a_normalize_config(probe_a):
    cfg = {
        "enable": False,
        "mode": "global_shift",  # global_shift | intra_block_shuffle | boundary_shift
        "delta": 1,
        "shuffle_type": "local_swap",  # local_swap | full_shuffle
        "boundary_variant": "start_plus1",  # start_plus1 | start_to_middle | nonstart_to_start
        "seed": 1234,
        "target_layers": [-1],  # -1 means all layers
        "collect_metrics": True,
        "collect_next_block_metrics": True,
        "topk_match_k": 8,
    }
    if isinstance(probe_a, dict):
        cfg.update(probe_a)

    cfg["enable"] = bool(cfg.get("enable", False))
    cfg["mode"] = str(cfg.get("mode", "global_shift")).strip().lower()
    cfg["shuffle_type"] = str(cfg.get("shuffle_type", "local_swap")).strip().lower()
    cfg["boundary_variant"] = str(cfg.get("boundary_variant", "start_plus1")).strip().lower()
    cfg["delta"] = int(cfg.get("delta", 1))
    cfg["seed"] = int(cfg.get("seed", 1234))
    cfg["collect_metrics"] = bool(cfg.get("collect_metrics", True))
    cfg["collect_next_block_metrics"] = bool(cfg.get("collect_next_block_metrics", True))
    cfg["topk_match_k"] = int(cfg.get("topk_match_k", 8))

    target_layers = cfg.get("target_layers", [-1])
    if isinstance(target_layers, int):
        target_layers = [target_layers]
    elif not isinstance(target_layers, (list, tuple)):
        target_layers = [-1]
    cfg["target_layers"] = [int(v) for v in target_layers]
    return cfg


def _probe_a_get_num_layers(model):
    n_layers = 0
    try:
        n_layers = int(getattr(model.model.config, "n_layers", 0))
    except Exception:
        n_layers = 0
    if n_layers <= 0:
        try:
            n_layers = int(len(model.model.transformer.blocks))
        except Exception:
            n_layers = 0
    return max(0, int(n_layers))


def _probe_a_target_layer_set(cfg, n_layers):
    target_layers = cfg.get("target_layers", [-1])
    if any(int(v) < 0 for v in target_layers):
        return set(range(n_layers))
    out = set()
    for v in target_layers:
        iv = int(v)
        if 0 <= iv < n_layers:
            out.add(iv)
    return out


def _probe_a_build_suffix_mapping(full_seq_len, suffix_start, cfg, block_idx, step_idx, layer_idx):
    suffix_start = int(suffix_start)
    full_seq_len = int(full_seq_len)
    if suffix_start >= full_seq_len:
        return None

    mode = cfg["mode"]
    delta = int(cfg["delta"])
    seed = int(cfg["seed"])
    shuffle_type = cfg["shuffle_type"]
    boundary_variant = cfg["boundary_variant"]

    suffix_positions = torch.arange(suffix_start, full_seq_len, dtype=torch.long)
    mapped = suffix_positions.clone()
    suffix_len = int(mapped.numel())

    if mode == "global_shift":
        mapped = mapped + delta
    elif mode == "intra_block_shuffle":
        if suffix_len >= 2:
            if shuffle_type == "full_shuffle":
                rng = np.random.default_rng(seed + 1000003 * int(block_idx) + 10007 * int(step_idx) + int(layer_idx))
                perm = rng.permutation(suffix_len)
                mapped = mapped[torch.from_numpy(perm).to(dtype=torch.long)]
            else:
                # local_swap: swap neighbor pairs (0,1), (2,3), ...
                mapped = mapped.clone()
                for i in range(0, suffix_len - 1, 2):
                    a = mapped[i].item()
                    b = mapped[i + 1].item()
                    mapped[i] = b
                    mapped[i + 1] = a
    elif mode == "boundary_shift":
        if suffix_len > 0:
            mapped = mapped.clone()
            if boundary_variant == "start_to_middle":
                mapped[0] = suffix_positions[suffix_len // 2]
            elif boundary_variant == "nonstart_to_start":
                if suffix_len > 1:
                    mapped[1] = suffix_positions[0]
            else:
                # start_plus1
                mapped[0] = mapped[0] + 1
    else:
        return None

    mapped = torch.clamp(mapped, min=0)
    return mapped


def _probe_a_apply_mapping(abs_positions, full_seq_len, suffix_start, mapped_suffix):
    if mapped_suffix is None:
        return abs_positions.clone()

    full_seq_len = int(full_seq_len)
    suffix_start = int(suffix_start)
    out = abs_positions.clone()
    if mapped_suffix.device != out.device or mapped_suffix.dtype != torch.long:
        mapped_suffix = mapped_suffix.to(device=out.device, dtype=torch.long)
    mask = (out >= suffix_start) & (out < full_seq_len)
    if mask.any():
        offsets = (out[mask] - suffix_start).to(torch.long)
        out[mask] = mapped_suffix[offsets]
    out = torch.clamp(out, min=0)
    return out


def _probe_a_prepare_indices_for_call(
    base_q_abs,
    base_k_abs,
    full_seq_len,
    suffix_start,
    cfg,
    n_layers,
    block_idx,
    step_idx,
):
    target_layers = _probe_a_target_layer_set(cfg, n_layers)
    if len(target_layers) == 0:
        return base_q_abs, base_k_abs, int(full_seq_len)

    all_layers = len(target_layers) == int(n_layers)
    if all_layers:
        mapped = _probe_a_build_suffix_mapping(
            full_seq_len=full_seq_len,
            suffix_start=suffix_start,
            cfg=cfg,
            block_idx=block_idx,
            step_idx=step_idx,
            layer_idx=-1,
        )
        q_out = _probe_a_apply_mapping(base_q_abs, full_seq_len, suffix_start, mapped)
        k_out = _probe_a_apply_mapping(base_k_abs, full_seq_len, suffix_start, mapped)
        rope_seq_len = int(max(int(q_out.max().item()), int(k_out.max().item())) + 1)
        rope_seq_len = max(rope_seq_len, int(full_seq_len))
        return q_out, k_out, rope_seq_len

    q_list = []
    k_list = []
    rope_max = int(full_seq_len) - 1
    for layer_idx in range(n_layers):
        if layer_idx in target_layers:
            mapped = _probe_a_build_suffix_mapping(
                full_seq_len=full_seq_len,
                suffix_start=suffix_start,
                cfg=cfg,
                block_idx=block_idx,
                step_idx=step_idx,
                layer_idx=layer_idx,
            )
            q_i = _probe_a_apply_mapping(base_q_abs, full_seq_len, suffix_start, mapped)
            k_i = _probe_a_apply_mapping(base_k_abs, full_seq_len, suffix_start, mapped)
        else:
            q_i = base_q_abs.clone()
            k_i = base_k_abs.clone()
        q_list.append(q_i)
        k_list.append(k_i)
        rope_max = max(rope_max, int(q_i.max().item()), int(k_i.max().item()))

    rope_seq_len = int(rope_max + 1)
    rope_seq_len = max(rope_seq_len, int(full_seq_len))
    return q_list, k_list, rope_seq_len


def _probe_a_update_layer_metrics(layer_acc, attentions_base, attentions_probe, q_lo, q_hi, mode, delta):
    if attentions_base is None or attentions_probe is None:
        return

    denom = float(delta) if (str(mode) == "global_shift" and int(delta) != 0) else 1.0
    n_layers = min(len(attentions_base), len(attentions_probe))
    for layer_id in range(n_layers):
        attn_b = attentions_base[layer_id]
        attn_p = attentions_probe[layer_id]
        if attn_b is None or attn_p is None:
            continue
        if attn_b.ndim != 4 or attn_p.ndim != 4:
            continue
        q_lo_i = int(max(0, q_lo))
        q_hi_i = int(min(int(attn_b.shape[-2]), int(q_hi)))
        if q_hi_i <= q_lo_i:
            continue

        b = attn_b[0, :, q_lo_i:q_hi_i, :].to(torch.float32)
        p = attn_p[0, :, q_lo_i:q_hi_i, :].to(torch.float32)
        if b.numel() == 0 or p.numel() == 0:
            continue

        b_tok = b.mean(dim=(0, 1))
        p_tok = p.mean(dim=(0, 1))
        pos = torch.arange(b_tok.shape[0], device=b_tok.device, dtype=torch.float32)
        b_den = b_tok.sum().clamp_min(1e-12)
        p_den = p_tok.sum().clamp_min(1e-12)
        center_b = float((b_tok * pos).sum().item() / b_den.item())
        center_p = float((p_tok * pos).sum().item() / p_den.item())
        adr = (center_p - center_b) / denom
        delta_e = float(torch.mean(torch.abs(p - b)).item())

        rec = layer_acc.setdefault(int(layer_id), {"adr_sum": 0.0, "de_sum": 0.0, "count": 0})
        rec["adr_sum"] += float(adr)
        rec["de_sum"] += float(delta_e)
        rec["count"] += 1


def _probe_a_record_block_step_metrics(block_step_metrics, logits_probe, logits_base, token_lo, token_hi, step_idx, topk_match_k):
    if step_idx < 0 or step_idx > 2:
        return
    lo = int(max(0, token_lo))
    hi = int(min(int(logits_probe.shape[1]), int(token_hi)))
    if hi <= lo:
        return

    lp = logits_probe[:, lo:hi, :].to(torch.float32)
    probs = F.softmax(lp, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
    confidence = probs.max(dim=-1).values.mean()
    block_step_metrics["entropy"].append(float(entropy.item()))
    block_step_metrics["confidence"].append(float(confidence.item()))

    if logits_base is not None:
        lb = logits_base[:, lo:hi, :]
        top_p = torch.argmax(lp, dim=-1).reshape(-1)
        top_b = torch.argmax(lb, dim=-1).reshape(-1)
        k = int(min(int(topk_match_k), int(top_p.numel()), int(top_b.numel())))
        if k > 0:
            match = (top_p[:k] == top_b[:k]).to(torch.float32).mean()
            block_step_metrics["top1_match"].append(float(match.item()))


def _probe_a_finalize_logs(stats, block_layer_metrics, block_step_metrics, cfg):
    if stats is None:
        return
    raw_mode = str(cfg.get("mode", "global_shift"))
    mode_map = {
        "global_shift": "shift",
        "intra_block_shuffle": "shuffle",
        "boundary_shift": "boundary",
    }
    mode = mode_map.get(raw_mode, raw_mode)
    delta = int(cfg.get("delta", 0))

    logs = []
    layer_adr = defaultdict(list)
    n_blocks = len(block_layer_metrics)
    for b in range(n_blocks):
        next_b = b + 1
        if next_b < len(block_step_metrics):
            ent_list = block_step_metrics[next_b].get("entropy", [])
            conf_list = block_step_metrics[next_b].get("confidence", [])
            match_list = block_step_metrics[next_b].get("top1_match", [])
            next_entropy = float(np.mean(ent_list)) if len(ent_list) > 0 else float("nan")
            next_conf = float(np.mean(conf_list)) if len(conf_list) > 0 else float("nan")
            next_match = float(np.mean(match_list)) if len(match_list) > 0 else float("nan")
        else:
            next_entropy = float("nan")
            next_conf = float("nan")
            next_match = float("nan")

        for layer_id, rec in block_layer_metrics[b].items():
            if rec["count"] <= 0:
                continue
            adr = float(rec["adr_sum"] / rec["count"])
            de = float(rec["de_sum"] / rec["count"])
            logs.append(
                {
                    "block_id": int(b),
                    "layer_id": int(layer_id),
                    "mode": mode,
                    "delta": int(delta),
                    "ADR": adr,
                    "deltaE": de,
                    "ΔE": de,
                    "next_block_entropy": next_entropy,
                    "next_block_confidence": next_conf,
                    "next_block_top1_match": next_match,
                }
            )
            layer_adr[int(layer_id)].append(adr)

    stats["probe_a_layer_logs"] = logs
    stats["probe_a_layer_adr_curve"] = {
        int(k): float(np.mean(v)) for k, v in layer_adr.items() if len(v) > 0
    }


def _probe_b_normalize_config(probe_b):
    cfg = {
        "enable": False,
        "mode": "normal",  # normal | frozen | removed
        "collect_metrics": True,
        "verify_removed_leak": True,
        "leak_tol": 1e-12,
    }
    if isinstance(probe_b, dict):
        cfg.update(probe_b)
    cfg["enable"] = bool(cfg.get("enable", False))
    cfg["mode"] = str(cfg.get("mode", "normal")).strip().lower()
    if cfg["mode"] not in {"normal", "frozen", "removed"}:
        cfg["mode"] = "normal"
    cfg["collect_metrics"] = bool(cfg.get("collect_metrics", True))
    cfg["verify_removed_leak"] = bool(cfg.get("verify_removed_leak", True))
    try:
        cfg["leak_tol"] = float(cfg.get("leak_tol", 1e-12))
    except Exception:
        cfg["leak_tol"] = 1e-12
    return cfg


def _probe_b_init_step_metrics():
    return [
        {
            "entropy": [],
            "confidence": [],
            "token_acc": [],
            "token_nll": [],
            "attention_mass": [],
            "attention_start_ratio": [],
            "removed_leak_mass": [],
            "removed_leak_max_prob": [],
        }
        for _ in range(3)
    ]


def _probe_b_record_step_metrics(step_metrics, logits, x_tokens, token_lo, token_hi, step_idx, mask_id):
    if step_metrics is None:
        return
    if step_idx < 0 or step_idx > 2:
        return
    lo = int(max(0, token_lo))
    hi = int(min(int(logits.shape[1]), int(token_hi)))
    if hi <= lo:
        return
    rec = step_metrics[int(step_idx)]

    lp = logits[:, lo:hi, :].to(torch.float32)
    probs = F.softmax(lp, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
    confidence = probs.max(dim=-1).values.mean()
    rec["entropy"].append(float(entropy.item()))
    rec["confidence"].append(float(confidence.item()))

    if x_tokens is None:
        return
    tgt = x_tokens[:, lo:hi]
    valid = (tgt != int(mask_id))
    if valid.any():
        logp = F.log_softmax(lp, dim=-1)
        nll = -torch.gather(logp, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
        rec["token_nll"].append(float(nll[valid].mean().item()))
        pred = torch.argmax(lp, dim=-1)
        acc = (pred[valid] == tgt[valid]).to(torch.float32).mean()
        rec["token_acc"].append(float(acc.item()))


def _probe_b_record_attention_metrics(step_metrics, attentions, q_lo, q_hi, suffix_lo, suffix_hi, step_idx):
    if step_metrics is None or attentions is None:
        return
    if step_idx < 0 or step_idx > 2:
        return
    rec = step_metrics[int(step_idx)]
    mass_vals = []
    ratio_vals = []

    for attn in attentions:
        if attn is None or attn.ndim != 4:
            continue
        lq = int(attn.shape[-2])
        lk = int(attn.shape[-1])
        q0 = int(max(0, q_lo))
        q1 = int(min(int(q_hi), lq))
        k0 = int(max(0, suffix_lo))
        k1 = int(min(int(suffix_hi), lk))
        if q1 <= q0 or k1 <= k0:
            continue

        cur = attn[0, :, q0:q1, :].to(torch.float32)
        suffix_mass = cur[:, :, k0:k1].sum(dim=-1).mean()
        start_mass = cur[:, :, k0:k0 + 1].sum(dim=-1).mean()
        total = float(suffix_mass.item())
        if np.isfinite(total):
            mass_vals.append(total)
            if total > 0.0:
                ratio_vals.append(float(start_mass.item()) / max(total, 1e-12))

    if len(mass_vals) > 0:
        rec["attention_mass"].append(float(np.mean(mass_vals)))
    if len(ratio_vals) > 0:
        rec["attention_start_ratio"].append(float(np.mean(ratio_vals)))


def _probe_b_record_removed_leak_metrics(step_metrics, attentions, q_lo, q_hi, suffix_lo, suffix_hi, step_idx):
    if step_metrics is None or attentions is None:
        return
    if step_idx < 0 or step_idx > 2:
        return
    rec = step_metrics[int(step_idx)]
    mass_vals = []
    max_vals = []

    for attn in attentions:
        if attn is None or attn.ndim != 4:
            continue
        lq = int(attn.shape[-2])
        lk = int(attn.shape[-1])
        q0 = int(max(0, q_lo))
        q1 = int(min(int(q_hi), lq))
        k0 = int(max(0, suffix_lo))
        k1 = int(min(int(suffix_hi), lk))
        if q1 <= q0 or k1 <= k0:
            continue
        leak = attn[0, :, q0:q1, k0:k1].to(torch.float32)
        if leak.numel() == 0:
            continue
        mass_vals.append(float(leak.sum(dim=-1).mean().item()))
        max_vals.append(float(leak.max().item()))

    if len(mass_vals) > 0:
        rec["removed_leak_mass"].append(float(np.mean(mass_vals)))
    if len(max_vals) > 0:
        rec["removed_leak_max_prob"].append(float(np.max(max_vals)))


def _probe_b_finalize_logs(stats, step_metrics, cfg):
    if stats is None or step_metrics is None:
        return

    mode = str(cfg.get("mode", "normal")).strip().lower()
    leak_tol = float(cfg.get("leak_tol", 1e-12))

    def _mean_or_nan(vals):
        if vals is None or len(vals) == 0:
            return float("nan")
        return float(np.mean(vals))

    def _max_or_nan(vals):
        if vals is None or len(vals) == 0:
            return float("nan")
        return float(np.max(vals))

    sample_log = {"mode": mode}
    for s in range(3):
        rec = step_metrics[s]
        idx = s + 1
        sample_log[f"current_block_entropy_step{idx}"] = _mean_or_nan(rec.get("entropy", []))
        sample_log[f"current_block_confidence_step{idx}"] = _mean_or_nan(rec.get("confidence", []))
        sample_log[f"current_block_token_acc_step{idx}"] = _mean_or_nan(rec.get("token_acc", []))
        sample_log[f"current_block_token_nll_step{idx}"] = _mean_or_nan(rec.get("token_nll", []))

    if mode == "removed":
        sample_log["attention_mass"] = None
        sample_log["attention_start_ratio"] = None
        sample_log["attention_status"] = "N/A"
        sample_log["removed_leak_tol"] = float(leak_tol)
        pass_flags = []
        for s in range(3):
            rec = step_metrics[s]
            idx = s + 1
            leak_mass = _mean_or_nan(rec.get("removed_leak_mass", []))
            leak_max = _max_or_nan(rec.get("removed_leak_max_prob", []))
            sample_log[f"removed_leak_mass_step{idx}"] = leak_mass
            sample_log[f"removed_leak_max_prob_step{idx}"] = leak_max
            step_pass = None
            if np.isfinite(leak_mass) and np.isfinite(leak_max):
                step_pass = bool((leak_mass <= leak_tol) and (leak_max <= leak_tol))
                pass_flags.append(step_pass)
            sample_log[f"removed_leak_pass_step{idx}"] = step_pass
        sample_log["removed_leak_hard_check_pass"] = bool(len(pass_flags) == 3 and all(pass_flags))
    else:
        all_mass = []
        all_ratio = []
        for rec in step_metrics:
            all_mass.extend(rec.get("attention_mass", []))
            all_ratio.extend(rec.get("attention_start_ratio", []))
        sample_log["attention_mass"] = _mean_or_nan(all_mass)
        sample_log["attention_start_ratio"] = _mean_or_nan(all_ratio)
        sample_log["attention_status"] = "ok"

    stats["probe_b_sample_log"] = sample_log


def _wsr_probe_normalize_config(trajectory_probe):
    cfg = {
        "enable": False,
        "sample_id": -1,
        "output_jsonl_path": "",
        "block_length": 128,
        "random_k": 8,
        "random_seed": 1234,
        "local_window": -1,
        "eps": 1e-8,
        "mask_id": 126336,
        "enable_layer_lm_head": False,
        "record_final_generated_token_prob": False,
        "tag_high_low_drift": False,
        "drift_quantile": 0.75,
    }
    if isinstance(trajectory_probe, dict):
        cfg.update(trajectory_probe)

    cfg["enable"] = bool(cfg.get("enable", False))
    cfg["sample_id"] = int(cfg.get("sample_id", -1))
    cfg["output_jsonl_path"] = str(cfg.get("output_jsonl_path", "") or "")
    cfg["block_length"] = max(1, int(cfg.get("block_length", 128)))
    cfg["random_k"] = max(0, int(cfg.get("random_k", 8)))
    cfg["random_seed"] = int(cfg.get("random_seed", 1234))
    cfg["local_window"] = int(cfg.get("local_window", -1))
    cfg["eps"] = float(max(1e-12, cfg.get("eps", 1e-8)))
    cfg["mask_id"] = int(cfg.get("mask_id", 126336))
    cfg["enable_layer_lm_head"] = bool(cfg.get("enable_layer_lm_head", False))
    cfg["record_final_generated_token_prob"] = bool(cfg.get("record_final_generated_token_prob", False))
    cfg["tag_high_low_drift"] = bool(cfg.get("tag_high_low_drift", False))
    dq = float(cfg.get("drift_quantile", 0.75))
    cfg["drift_quantile"] = float(min(0.99, max(0.5, dq)))
    return cfg


class _WriteStoreReadTrajectoryProbeRuntime:
    def __init__(self, model, cfg, stats=None):
        self.model = model
        self.cfg = cfg
        self.stats = stats
        self.eps = float(cfg["eps"])
        self.block_length_default = int(cfg["block_length"])
        self.random_k = int(cfg["random_k"])
        self.random_seed = int(cfg["random_seed"])
        self.local_window_cfg = int(cfg["local_window"])
        self.mask_id = int(cfg["mask_id"])
        self.enable_layer_lm_head = bool(cfg["enable_layer_lm_head"])
        self.record_final_generated_token_prob = bool(cfg["record_final_generated_token_prob"])
        self.tag_high_low_drift = bool(cfg["tag_high_low_drift"])
        self.drift_quantile = float(cfg["drift_quantile"])
        self.output_jsonl_path = str(cfg["output_jsonl_path"])
        self.default_sample_id = int(cfg.get("sample_id", -1))

        self.handles = []
        self.capture_enabled = False
        self.ctx = None
        self.prev_layer_hidden = None
        self.pending_layer_rec = None
        self.current_step_hidden = {}
        self.current_step_top1 = {}
        self.prev_step_hidden = {}
        self.prev_step_top1 = {}

        self._ln_f = None
        self._out_emb = None
        if self.enable_layer_lm_head:
            try:
                self._ln_f = self.model.model.transformer.ln_f
                self._out_emb = self.model.get_output_embeddings()
            except Exception:
                self._ln_f = None
                self._out_emb = None
                self.enable_layer_lm_head = False

        self.fp = None
        if self.output_jsonl_path.strip() != "":
            out_dir = os.path.dirname(self.output_jsonl_path)
            if out_dir != "":
                os.makedirs(out_dir, exist_ok=True)
            self.fp = open(self.output_jsonl_path, "a", encoding="utf-8", buffering=1)

        blocks = []
        try:
            blocks = list(self.model.model.transformer.blocks)
        except Exception:
            blocks = []
        for layer_id, block in enumerate(blocks):
            h = block.register_forward_hook(self._make_block_hook(layer_id))
            self.handles.append(h)

        if self.stats is not None:
            self.stats["wsr_probe_enabled"] = True
            self.stats["wsr_probe_output_jsonl_path"] = self.output_jsonl_path
            self.stats["wsr_probe_block_length"] = int(self.block_length_default)
            self.stats["wsr_probe_random_k"] = int(self.random_k)
            self.stats["wsr_probe_local_window"] = int(self.local_window_cfg)
            self.stats["wsr_probe_layer_lm_head"] = bool(self.enable_layer_lm_head)

    def close(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        if self.fp is not None:
            try:
                self.fp.close()
            except Exception:
                pass
            self.fp = None

    def set_step_context(
        self,
        sample_id,
        denoising_step,
        block_id,
        block_start,
        block_end,
        seq_len,
        block_length,
        x_tokens,
    ):
        seq_len_i = int(seq_len)
        block_end_i = int(block_end)
        suffix_len = max(0, seq_len_i - block_end_i)
        if suffix_len <= 0:
            self.ctx = {
                "active_suffix": False,
                "sample_id": int(sample_id),
                "denoising_step": int(denoising_step),
                "block_id": int(block_id),
            }
            return

        bl = max(1, int(block_length))
        offsets = np.arange(suffix_len, dtype=np.int64)
        suffix_positions = offsets + block_end_i
        block_distance = (offsets // bl) + 1
        offset_in_block = offsets % bl
        block_end_offset = np.minimum((((offsets // bl) + 1) * bl) - 1, suffix_len - 1)
        is_start = offset_in_block == 0
        is_end = offsets == block_end_offset
        is_middle = ~(is_start | is_end)

        local_window = int(self.local_window_cfg)
        if local_window <= 0:
            local_window = int(bl)
        is_local = offsets < int(local_window)

        random_mask = np.zeros((suffix_len,), dtype=bool)
        rk = int(min(self.random_k, suffix_len))
        if rk > 0:
            sid = int(sample_id)
            seed = int(
                self.random_seed
                + 1000003 * max(0, sid)
                + 10007 * int(block_id)
                + 97 * int(denoising_step)
            )
            rng = np.random.default_rng(seed)
            if rk >= suffix_len:
                random_mask[:] = True
            else:
                ridx = rng.choice(np.arange(suffix_len, dtype=np.int64), size=rk, replace=False)
                random_mask[ridx] = True

        token_ids = None
        try:
            if x_tokens is not None and x_tokens.ndim == 2 and int(x_tokens.shape[1]) >= int(seq_len_i):
                token_ids = (
                    x_tokens[0, block_end_i:seq_len_i]
                    .detach()
                    .to("cpu", dtype=torch.long)
                    .numpy()
                    .astype(np.int64, copy=False)
                )
        except Exception:
            token_ids = None

        self.ctx = {
            "active_suffix": True,
            "sample_id": int(sample_id),
            "denoising_step": int(denoising_step),
            "block_id": int(block_id),
            "block_start": int(block_start),
            "block_end": int(block_end_i),
            "seq_len": int(seq_len_i),
            "block_length": int(bl),
            "suffix_len": int(suffix_len),
            "suffix_positions": suffix_positions,
            "block_distance": block_distance,
            "is_start": is_start,
            "is_middle": is_middle,
            "is_end": is_end,
            "is_local": is_local,
            "random_mask": random_mask,
            "token_ids": token_ids,
        }

    def begin_forward(self):
        if self.ctx is None or (not bool(self.ctx.get("active_suffix", False))):
            self.capture_enabled = False
            return
        self.capture_enabled = True
        self.prev_layer_hidden = None
        self.pending_layer_rec = None
        self.current_step_hidden = {}
        self.current_step_top1 = {}

    def end_forward(self):
        if not self.capture_enabled:
            return
        if self.pending_layer_rec is not None:
            self._flush_layer_records(self.pending_layer_rec, store_stability=None)
            self.pending_layer_rec = None
        self.prev_step_hidden = dict(self.current_step_hidden)
        if self.enable_layer_lm_head:
            self.prev_step_top1 = dict(self.current_step_top1)
        self.current_step_hidden = {}
        self.current_step_top1 = {}
        self.prev_layer_hidden = None
        self.capture_enabled = False

    def _cosine(self, a, b):
        num = (a * b).sum(dim=-1)
        den = torch.linalg.vector_norm(a, ord=2, dim=-1) * torch.linalg.vector_norm(b, ord=2, dim=-1)
        den = den + float(self.eps)
        return num / den

    def _write_record(self, rec):
        if self.fp is None:
            return
        self.fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _layer_top1_stats(self, suffix_hidden, layer_id):
        top1_conf_np = None
        top1_id_np = None
        top1_stable_np = None
        final_prob_np = None

        if (not self.enable_layer_lm_head) or (self._out_emb is None):
            return top1_conf_np, top1_id_np, top1_stable_np, final_prob_np

        try:
            h = suffix_hidden.to(torch.float32)
            if self._ln_f is not None:
                h = self._ln_f(h)
            if hasattr(self._out_emb, "weight"):
                weight = self._out_emb.weight
                logits = F.linear(h, weight.to(dtype=torch.float32))
            else:
                return top1_conf_np, top1_id_np, top1_stable_np, final_prob_np

            lse = torch.logsumexp(logits, dim=-1)
            top1_id = torch.argmax(logits, dim=-1)
            top1_logit = torch.gather(logits, dim=-1, index=top1_id.unsqueeze(-1)).squeeze(-1)
            top1_conf = torch.exp(top1_logit - lse)

            top1_conf_np = (
                top1_conf.detach().to("cpu", dtype=torch.float32).numpy().astype(np.float32, copy=False)
            )
            top1_id_np = top1_id.detach().to("cpu", dtype=torch.long).numpy().astype(np.int64, copy=False)

            prev_top1 = self.prev_step_top1.get(int(layer_id), None)
            if prev_top1 is not None and prev_top1.shape == top1_id_np.shape:
                top1_stable_np = (top1_id_np == prev_top1)

            if self.record_final_generated_token_prob:
                tok_ids = self.ctx.get("token_ids", None) if self.ctx is not None else None
                if tok_ids is not None and tok_ids.ndim == 1 and tok_ids.shape[0] == int(logits.shape[1]):
                    ids = torch.from_numpy(tok_ids).to(device=logits.device, dtype=torch.long).unsqueeze(0)
                    ids = ids.expand(logits.shape[0], -1)
                    valid = ids != int(self.mask_id)
                    tok_logit = torch.gather(logits, dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
                    tok_prob = torch.exp(tok_logit - lse)
                    tok_prob = torch.where(valid, tok_prob, torch.full_like(tok_prob, float("nan")))
                    final_prob_np = tok_prob.detach().to("cpu", dtype=torch.float32).numpy().astype(np.float32, copy=False)
        except Exception:
            return None, None, None, None

        return top1_conf_np, top1_id_np, top1_stable_np, final_prob_np

    def _flush_layer_records(self, layer_rec, store_stability):
        if self.ctx is None or (not bool(self.ctx.get("active_suffix", False))):
            return

        s_len = int(self.ctx["suffix_len"])
        if s_len <= 0:
            return

        sample_id = int(self.ctx.get("sample_id", self.default_sample_id))
        step_id = int(self.ctx["denoising_step"])
        block_id = int(self.ctx["block_id"])
        layer_id = int(layer_rec["layer_id"])
        suffix_positions = self.ctx["suffix_positions"]
        block_distance = self.ctx["block_distance"]
        is_start = self.ctx["is_start"]
        is_middle = self.ctx["is_middle"]
        is_end = self.ctx["is_end"]
        is_local = self.ctx["is_local"]
        random_mask = self.ctx["random_mask"]

        write_delta = layer_rec.get("write_delta", None)
        write_delta_norm = layer_rec.get("write_delta_norm", None)
        step_drift = layer_rec.get("step_drift", None)
        top1_stable = layer_rec.get("top1_stable", None)
        top1_conf = layer_rec.get("top1_confidence", None)
        top1_id = layer_rec.get("top1_id", None)
        final_prob = layer_rec.get("final_generated_token_prob", None)

        drift_hi = None
        drift_lo = None
        if self.tag_high_low_drift and step_drift is not None:
            try:
                dv = np.asarray(step_drift, dtype=np.float32).reshape(-1)
                dv = dv[np.isfinite(dv)]
                if dv.size > 0:
                    q = float(self.drift_quantile)
                    drift_hi = float(np.quantile(dv, q))
                    drift_lo = float(np.quantile(dv, 1.0 - q))
            except Exception:
                drift_hi = None
                drift_lo = None

        for j in range(s_len):
            pos_type = "future_block_middle_token"
            if bool(is_start[j]):
                pos_type = "future_block_start_token"
            elif bool(is_end[j]):
                pos_type = "future_block_end_token"
            elif bool(is_middle[j]):
                pos_type = "future_block_middle_token"

            tags = [pos_type]
            if not bool(is_start[j]):
                tags.append("non_start_suffix_token")
            if bool(is_local[j]):
                tags.append("local_suffix_token")
            if bool(random_mask[j]):
                tags.append("random_suffix_token")

            cur_drift = None
            if step_drift is not None:
                try:
                    v = float(step_drift[0, j])
                    cur_drift = v if np.isfinite(v) else None
                except Exception:
                    cur_drift = None
            if cur_drift is not None and drift_hi is not None and drift_lo is not None:
                if cur_drift >= drift_hi:
                    tags.append("high_drift_suffix")
                elif cur_drift <= drift_lo:
                    tags.append("low_drift_suffix")

            cur_write = None
            if write_delta is not None:
                try:
                    v = float(write_delta[0, j])
                    cur_write = v if np.isfinite(v) else None
                except Exception:
                    cur_write = None
            cur_write_norm = None
            if write_delta_norm is not None:
                try:
                    v = float(write_delta_norm[0, j])
                    cur_write_norm = v if np.isfinite(v) else None
                except Exception:
                    cur_write_norm = None

            cur_store = None
            if store_stability is not None:
                try:
                    v = float(store_stability[0, j])
                    cur_store = v if np.isfinite(v) else None
                except Exception:
                    cur_store = None

            cur_top1_stable = None
            if top1_stable is not None:
                try:
                    cur_top1_stable = bool(top1_stable[0, j])
                except Exception:
                    cur_top1_stable = None
            cur_top1_conf = None
            if top1_conf is not None:
                try:
                    v = float(top1_conf[0, j])
                    cur_top1_conf = v if np.isfinite(v) else None
                except Exception:
                    cur_top1_conf = None
            cur_top1_id = None
            if top1_id is not None:
                try:
                    cur_top1_id = int(top1_id[0, j])
                except Exception:
                    cur_top1_id = None
            cur_final_prob = None
            if final_prob is not None:
                try:
                    v = float(final_prob[0, j])
                    cur_final_prob = v if np.isfinite(v) else None
                except Exception:
                    cur_final_prob = None

            for tag_idx, tag in enumerate(tags):
                out = {
                    "sample_id": int(sample_id),
                    "denoising_step": int(step_id),
                    "block_id": int(block_id),
                    "layer_id": int(layer_id),
                    "token_position": int(suffix_positions[j]),
                    "token_type": str(tag),
                    "is_primary_type": bool(tag_idx == 0),
                    "block_distance": int(block_distance[j]),
                    "write_delta": cur_write,
                    "write_delta_norm": cur_write_norm,
                    "store_stability": cur_store,
                    "step_drift": cur_drift,
                    "top1_stable": cur_top1_stable,
                    "top1_confidence": cur_top1_conf,
                    "top1_id": cur_top1_id,
                }
                if self.record_final_generated_token_prob:
                    out["final_generated_token_prob"] = cur_final_prob
                self._write_record(out)

    def _make_block_hook(self, layer_id):
        def _hook(module, inputs, output):
            if (not self.capture_enabled) or self.ctx is None or (not bool(self.ctx.get("active_suffix", False))):
                return
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden is None or hidden.ndim != 3:
                return

            block_end = int(self.ctx["block_end"])
            if block_end >= int(hidden.shape[1]):
                return

            suffix_hidden = hidden[:, block_end:, :]
            if suffix_hidden.numel() == 0:
                return
            curr_cpu = suffix_hidden.detach().to("cpu", dtype=torch.float32)

            write_delta_np = None
            write_delta_norm_np = None
            store_prev_np = None

            if self.prev_layer_hidden is not None and self.prev_layer_hidden.shape == curr_cpu.shape:
                diff = curr_cpu - self.prev_layer_hidden
                wd = torch.linalg.vector_norm(diff, ord=2, dim=-1)
                pn = torch.linalg.vector_norm(self.prev_layer_hidden, ord=2, dim=-1)
                wdn = wd / (pn + float(self.eps))
                write_delta_np = wd.numpy().astype(np.float32, copy=False)
                write_delta_norm_np = wdn.numpy().astype(np.float32, copy=False)

                store_prev = self._cosine(self.prev_layer_hidden, curr_cpu)
                store_prev_np = store_prev.numpy().astype(np.float32, copy=False)

            step_drift_np = None
            prev_step_h = self.prev_step_hidden.get(int(layer_id), None)
            if prev_step_h is not None and prev_step_h.shape == curr_cpu.shape:
                prev_f = prev_step_h.to(torch.float32)
                drift = 1.0 - self._cosine(curr_cpu, prev_f)
                step_drift_np = drift.numpy().astype(np.float32, copy=False)

            top1_conf_np, top1_id_np, top1_stable_np, final_prob_np = self._layer_top1_stats(
                suffix_hidden=suffix_hidden,
                layer_id=layer_id,
            )

            if self.pending_layer_rec is not None:
                self._flush_layer_records(self.pending_layer_rec, store_stability=store_prev_np)

            self.pending_layer_rec = {
                "layer_id": int(layer_id),
                "write_delta": write_delta_np,
                "write_delta_norm": write_delta_norm_np,
                "step_drift": step_drift_np,
                "top1_stable": top1_stable_np,
                "top1_confidence": top1_conf_np,
                "top1_id": top1_id_np,
                "final_generated_token_prob": final_prob_np,
            }
            self.prev_layer_hidden = curr_cpu
            self.current_step_hidden[int(layer_id)] = curr_cpu.to(torch.float16)
            if top1_id_np is not None:
                self.current_step_top1[int(layer_id)] = top1_id_np.astype(np.int64, copy=False)

        return _hook


def _model_forward_with_wsr_runtime(runtime, model, kwargs):
    if runtime is None:
        return model(**kwargs)
    runtime.begin_forward()
    out = model(**kwargs)
    runtime.end_forward()
    return out


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, eos_id=126081, threshold=None, factor=None, early_termination=False, stats=None,
             collect_hypothesis_stats=False, collect_figure4a=False, figure4a_block_id=-1, figure4a_c=-1,
             figure4a_exit_after_capture=False, collect_observation=False, observation_k=3, observation_seed=1234,
             collect_observation_intervention=False, observation_intervention_k=8,
             collect_attention_rollout=False, rollout_residual_alpha=0.5,
             collect_rollout_offset_profile=False, probe_a=None, probe_b=None,
             collect_suffix_top1_first_block=False, suffix_top1_hist_bins=20,
             trajectory_probe=None):
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
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()


    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    figure4a_capture_spec = None
    if collect_figure4a:
        figure4a_capture_spec = _resolve_figure4a_capture_spec(
            prompt_len=int(prompt.shape[1]),
            seq_len=int(x.shape[1]),
            gen_length=int(gen_length),
            block_length=int(block_length),
            figure4a_block_id=int(figure4a_block_id),
            figure4a_c=int(figure4a_c),
        )
        if stats is not None:
            stats['figure4a_requested'] = True
            stats['figure4a_block_id'] = int(figure4a_block_id)
            stats['figure4a_c'] = int(figure4a_c)
            if figure4a_capture_spec is None:
                stats['figure4a_status'] = 'invalid_capture_spec'

    figure4a_captured = False
    if collect_observation and stats is not None:
        stats["observation_requested"] = True
        stats["observation_k"] = int(observation_k)
        stats["observation_seed"] = int(observation_seed)
    if collect_observation_intervention and stats is not None:
        stats["observation_intervention_requested"] = True
        stats["observation_intervention_k"] = int(observation_intervention_k)
    if collect_attention_rollout and stats is not None:
        stats["attention_rollout_requested"] = True
        stats["rollout_residual_alpha"] = float(rollout_residual_alpha)
    if collect_rollout_offset_profile and stats is not None:
        stats["rollout_offset_profile_requested"] = True
        stats["rollout_residual_alpha"] = float(rollout_residual_alpha)
    if collect_suffix_top1_first_block and stats is not None:
        stats["suffix_top1_first_block_requested"] = True
        stats["suffix_top1_hist_num_bins"] = int(max(4, int(suffix_top1_hist_bins)))

    probe_cfg = _probe_a_normalize_config(probe_a)
    probe_enabled = bool(probe_cfg.get("enable", False))
    probe_collect_metrics = bool(probe_enabled and probe_cfg.get("collect_metrics", True) and (stats is not None))
    probe_collect_next = bool(probe_enabled and probe_cfg.get("collect_next_block_metrics", True) and (stats is not None))
    probe_n_layers = _probe_a_get_num_layers(model) if probe_enabled else 0

    probe_b_cfg = _probe_b_normalize_config(probe_b)
    probe_b_enabled = bool(probe_b_cfg.get("enable", False))
    probe_b_collect_metrics = bool(probe_b_enabled and probe_b_cfg.get("collect_metrics", True) and (stats is not None))
    probe_b_mode = str(probe_b_cfg.get("mode", "normal"))
    probe_b_step_metrics = _probe_b_init_step_metrics() if probe_b_collect_metrics else None

    probe_layer_metrics_by_block = None
    probe_step_metrics_by_block = None
    if probe_collect_metrics:
        probe_layer_metrics_by_block = [dict() for _ in range(num_blocks)]
    if probe_collect_next:
        probe_step_metrics_by_block = [
            {"entropy": [], "confidence": [], "top1_match": []} for _ in range(num_blocks)
        ]

    if probe_enabled and stats is not None:
        stats["probe_a_enabled"] = True
        stats["probe_a_mode"] = str(probe_cfg.get("mode", "global_shift"))
        stats["probe_a_delta"] = int(probe_cfg.get("delta", 0))
        stats["probe_a_shuffle_type"] = str(probe_cfg.get("shuffle_type", "local_swap"))
        stats["probe_a_boundary_variant"] = str(probe_cfg.get("boundary_variant", "start_plus1"))
        stats["probe_a_target_layers"] = list(probe_cfg.get("target_layers", [-1]))
    if probe_b_enabled and stats is not None:
        stats["probe_b_enabled"] = True
        stats["probe_b_mode"] = str(probe_b_mode)

    trajectory_cfg = _wsr_probe_normalize_config(trajectory_probe)
    trajectory_runtime = None
    if bool(trajectory_cfg.get("enable", False)):
        trajectory_cfg["mask_id"] = int(mask_id)
        trajectory_cfg["block_length"] = int(block_length)
        trajectory_runtime = _WriteStoreReadTrajectoryProbeRuntime(
            model=model,
            cfg=trajectory_cfg,
            stats=stats,
        )

    nfe = 0
    try:
        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            block_start = prompt.shape[1] + num_block * block_length
            block_end = block_start + block_length
            available_suffix_tokens = max(0, x.shape[1] - block_end)
            _record_suffix_stats(stats, available_suffix_tokens, available_suffix_tokens)
            _record_suffix_bucket_stats(stats, block_end, x.shape[1])
            i = 0
            while True:
                nfe += 1
                mask_index = (x == mask_id)
                need_figure4a_attn = bool(
                    collect_figure4a
                    and figure4a_capture_spec is not None
                    and (not figure4a_captured)
                    and (num_block == int(figure4a_capture_spec['target_block_idx']))
                    and (i == 0)
                )
                need_observation_attn = bool(collect_observation and (stats is not None))
                need_rollout_attn = bool((collect_attention_rollout or collect_rollout_offset_profile) and (stats is not None))

                need_probe_attn = bool(probe_collect_metrics)
                need_probe_b_attn = bool(
                    probe_b_collect_metrics
                    and (i <= 2)
                    and (
                        (probe_b_mode in {"normal", "frozen"})
                        or (probe_b_mode == "removed" and bool(probe_b_cfg.get("verify_removed_leak", True)))
                    )
                )
                need_non_probe_b_attn = bool(
                    need_figure4a_attn or need_observation_attn or need_rollout_attn or need_probe_attn
                )
                output_attentions_flag = bool(need_non_probe_b_attn or need_probe_b_attn)

                if trajectory_runtime is not None:
                    trajectory_runtime.set_step_context(
                        sample_id=int(trajectory_cfg.get("sample_id", -1)),
                        denoising_step=int(i),
                        block_id=int(num_block),
                        block_start=int(block_start),
                        block_end=int(block_end),
                        seq_len=int(x.shape[1]),
                        block_length=int(block_length),
                        x_tokens=x,
                    )

                probe_b_kwargs = {}
                if probe_b_enabled:
                    probe_b_kwargs = {
                        "probe_b_mode": str(probe_b_mode),
                        "probe_b_current_start": int(block_start),
                        "probe_b_current_end": int(block_end),
                        "probe_b_suffix_start": int(block_end),
                    }

                baseline_logits = None
                probe_b_attentions = None
                if probe_enabled:
                    full_seq_len = int(x.shape[1])
                    base_q_abs = torch.arange(0, full_seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
                    base_k_abs = torch.arange(0, full_seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
                    probe_q_arg, probe_k_arg, rope_seq_len = _probe_a_prepare_indices_for_call(
                        base_q_abs=base_q_abs,
                        base_k_abs=base_k_abs,
                        full_seq_len=full_seq_len,
                        suffix_start=block_end,
                        cfg=probe_cfg,
                        n_layers=probe_n_layers,
                        block_idx=num_block,
                        step_idx=i,
                    )
                    # Keep generation path numerically aligned with vanilla: logits come from
                    # output_attentions=False path. Attention maps (if needed) are collected
                    # in an extra forward pass for analysis only.
                    model_out = _model_forward_with_wsr_runtime(
                        runtime=trajectory_runtime,
                        model=model,
                        kwargs={
                            "input_ids": x,
                            "output_attentions": False,
                            "q_indices": probe_q_arg,
                            "k_indices": probe_k_arg,
                            "seq_len": rope_seq_len,
                            "update_rope": (i == 0),
                            **probe_b_kwargs,
                        },
                    )
                    logits = model_out.logits

                    if output_attentions_flag:
                        model_out = model(
                            x,
                            output_attentions=True,
                            q_indices=probe_q_arg,
                            k_indices=probe_k_arg,
                            seq_len=rope_seq_len,
                            update_rope=False,
                            **probe_b_kwargs,
                        )
                        probe_b_attentions = model_out.attentions

                    need_baseline = bool(probe_collect_metrics or (probe_collect_next and i <= 2))
                    if need_baseline:
                        if probe_collect_next and i <= 2:
                            baseline_out_logits = model(
                                x,
                                output_attentions=False,
                                q_indices=base_q_abs,
                                k_indices=base_k_abs,
                                seq_len=full_seq_len,
                                update_rope=False,
                                **probe_b_kwargs,
                            )
                            baseline_logits = baseline_out_logits.logits
                        if probe_collect_metrics:
                            baseline_out = model(
                                x,
                                output_attentions=True,
                                q_indices=base_q_abs,
                                k_indices=base_k_abs,
                                seq_len=full_seq_len,
                                update_rope=False,
                                **probe_b_kwargs,
                            )
                            _probe_a_update_layer_metrics(
                                layer_acc=probe_layer_metrics_by_block[num_block],
                                attentions_base=baseline_out.attentions,
                                attentions_probe=model_out.attentions,
                                q_lo=block_start,
                                q_hi=block_end,
                                mode=probe_cfg.get("mode", "global_shift"),
                                delta=probe_cfg.get("delta", 0),
                            )
                    if probe_collect_next and i <= 2:
                        _probe_a_record_block_step_metrics(
                            block_step_metrics=probe_step_metrics_by_block[num_block],
                            logits_probe=logits,
                            logits_base=baseline_logits,
                            token_lo=block_start,
                            token_hi=block_end,
                            step_idx=i,
                            topk_match_k=int(probe_cfg.get("topk_match_k", 8)),
                        )
                else:
                    if probe_b_enabled:
                        model_out = _model_forward_with_wsr_runtime(
                            runtime=trajectory_runtime,
                            model=model,
                            kwargs={
                                "input_ids": x,
                                "output_attentions": need_non_probe_b_attn,
                                **probe_b_kwargs,
                            },
                        )
                        logits = model_out.logits
                        if need_probe_b_attn and (not need_non_probe_b_attn):
                            probe_b_out = model(
                                x,
                                output_attentions=True,
                                **probe_b_kwargs,
                            )
                            probe_b_attentions = probe_b_out.attentions
                        else:
                            probe_b_attentions = model_out.attentions
                    else:
                        model_out = _model_forward_with_wsr_runtime(
                            runtime=trajectory_runtime,
                            model=model,
                            kwargs={
                                "input_ids": x,
                                "output_attentions": output_attentions_flag,
                            },
                        )
                        probe_b_attentions = model_out.attentions
                    logits = model_out.logits

                if probe_b_collect_metrics and i <= 2:
                    _probe_b_record_step_metrics(
                        step_metrics=probe_b_step_metrics,
                        logits=logits,
                        x_tokens=x,
                        token_lo=block_start,
                        token_hi=block_end,
                        step_idx=i,
                        mask_id=mask_id,
                    )
                    if probe_b_mode in {"normal", "frozen"}:
                        _probe_b_record_attention_metrics(
                            step_metrics=probe_b_step_metrics,
                            attentions=probe_b_attentions,
                            q_lo=block_start,
                            q_hi=block_end,
                            suffix_lo=block_end,
                            suffix_hi=int(x.shape[1]),
                            step_idx=i,
                        )
                    elif probe_b_mode == "removed" and bool(probe_b_cfg.get("verify_removed_leak", True)):
                        _probe_b_record_removed_leak_metrics(
                            step_metrics=probe_b_step_metrics,
                            attentions=probe_b_attentions,
                            q_lo=block_start,
                            q_hi=block_end,
                            suffix_lo=block_end,
                            suffix_hi=int(x.shape[1]),
                            step_idx=i,
                        )

                if collect_observation and (stats is not None):
                    target_positions = _build_observation_target_positions(
                        block_end=block_end,
                        block_length=block_length,
                        seq_len=x.shape[1],
                        k=observation_k,
                    )
                    _record_observation_target_uncertainty(
                        stats=stats,
                        logits=logits,
                        target_positions=target_positions,
                        step_idx=i,
                    )
                    _record_observation_attention_stats(
                        stats=stats,
                        attentions=model_out.attentions,
                        target_positions=target_positions,
                        block_end=block_end,
                        seq_len=x.shape[1],
                        block_length=block_length,
                        k=observation_k,
                        step_idx=i,
                        seed=observation_seed + num_block * 10007,
                    )

                if collect_attention_rollout and (stats is not None):
                    _record_attention_rollout_stats(
                        stats=stats,
                        attentions=model_out.attentions,
                        block_start=block_start,
                        block_end=block_end,
                        seq_len=x.shape[1],
                        block_length=block_length,
                        step_idx=i,
                        residual_alpha=rollout_residual_alpha,
                    )

                if collect_rollout_offset_profile and (stats is not None):
                    rollout_matrix = _compute_rollout_matrix(
                        model_out.attentions,
                        residual_alpha=rollout_residual_alpha,
                    )
                    _record_rollout_offset_profile_stats(
                        stats=stats,
                        rollout_matrix=rollout_matrix,
                        block_start=block_start,
                        block_end=block_end,
                        seq_len=x.shape[1],
                        block_length=block_length,
                        step_idx=i,
                    )

                if collect_observation_intervention and (stats is not None):
                    target_positions_intv = _build_observation_target_positions(
                        block_end=block_end,
                        block_length=block_length,
                        seq_len=x.shape[1],
                        k=observation_k,
                    )
                    for mode in ["mask_start", "mask_end", "mask_random", "mask_start_end"]:
                        ablate_abs = _build_intervention_ablation_positions(
                            seq_len=x.shape[1],
                            block_end=block_end,
                            block_length=block_length,
                            budget=observation_intervention_k,
                            mode=mode,
                            seed=observation_seed + num_block * 10007,
                            step_idx=i,
                        )
                        if len(ablate_abs) == 0:
                            continue
                        # Keep target query tokens observable under intervention.
                        ablate_set = set(int(v) for v in ablate_abs)
                        for t in target_positions_intv:
                            if int(t) in ablate_set:
                                ablate_set.remove(int(t))
                        if len(ablate_set) == 0:
                            continue

                        keep = torch.ones((x.shape[1],), dtype=torch.bool, device=x.device)
                        keep[torch.tensor(sorted(ablate_set), dtype=torch.long, device=x.device)] = False
                        keep_idx_1d = torch.nonzero(keep, as_tuple=False).flatten().to(torch.long)
                        keep_idx = keep_idx_1d.unsqueeze(0).expand(x.shape[0], -1)
                        x_intv = x.gather(1, keep_idx)
                        intv_out = model(
                            x_intv,
                            output_attentions=False,
                            q_indices=keep_idx,
                            k_indices=keep_idx,
                            seq_len=x.shape[1],
                            update_rope=(i == 0),
                        )
                        _record_observation_intervention_target_uncertainty(
                            stats=stats,
                            logits=intv_out.logits,
                            target_positions=target_positions_intv,
                            step_idx=i,
                            mode=mode,
                            q_abs_positions=keep_idx_1d,
                        )

                if need_figure4a_attn:
                    metrics = _extract_figure4a_suffix_block_metrics(
                        attentions=model_out.attentions,
                        query_start=int(figure4a_capture_spec['query_start']),
                        query_end=int(figure4a_capture_spec['query_end']),
                        suffix_start=int(figure4a_capture_spec['suffix_start']),
                        block_size=int(block_length),
                    )
                    if metrics is not None:
                        figure4a_captured = True
                        if stats is not None:
                            stats['figure4a_layer_scores'] = metrics['layer_scores']
                            stats['figure4a_block_score'] = metrics['block_score']
                            stats['figure4a_block_entropy'] = metrics['block_entropy']
                            stats['figure4a_block_first_score'] = metrics['block_first_score']
                            stats['figure4a_block_last_score'] = metrics['block_last_score']
                            stats['figure4a_block_first_last_delta'] = metrics['block_first_last_delta']
                            stats['figure4a_num_suffix_blocks'] = int(metrics['num_suffix_blocks'])
                            stats['figure4a_suffix_len'] = int(metrics['suffix_len'])
                            stats['figure4a_mode'] = str(figure4a_capture_spec['mode'])
                            stats['figure4a_target_block_idx'] = int(figure4a_capture_spec['target_block_idx'])
                            stats['figure4a_query_start'] = int(figure4a_capture_spec['query_start'])
                            stats['figure4a_query_end'] = int(figure4a_capture_spec['query_end'])
                            stats['figure4a_suffix_start'] = int(figure4a_capture_spec['suffix_start'])
                            stats['figure4a_status'] = 'captured'

                        if bool(figure4a_exit_after_capture):
                            if stats is not None:
                                stats['figure4a_status'] = 'captured_exit_early'
                                stats['figure4a_exit_after_capture'] = True
                            if probe_collect_metrics and stats is not None:
                                _probe_a_finalize_logs(
                                    stats=stats,
                                    block_layer_metrics=probe_layer_metrics_by_block,
                                    block_step_metrics=probe_step_metrics_by_block
                                    if probe_step_metrics_by_block is not None
                                    else [{"entropy": [], "confidence": [], "top1_match": []} for _ in range(num_blocks)],
                                    cfg=probe_cfg,
                                )
                            if probe_b_collect_metrics and stats is not None:
                                _probe_b_finalize_logs(
                                    stats=stats,
                                    step_metrics=probe_b_step_metrics,
                                    cfg=probe_b_cfg,
                                )
                            # Keep decode path safe: replace remaining masks before returning.
                            x = torch.where(x == mask_id, torch.full_like(x, eos_id), x)
                            return x, nfe
                    elif stats is not None:
                        stats['figure4a_status'] = 'empty_slice'

                if collect_suffix_top1_first_block and (num_block == 0):
                    _record_suffix_top1_first_block_stats(
                        stats=stats,
                        logits=logits,
                        block_end=block_end,
                        block_length=block_length,
                        step_idx=i,
                        enabled=True,
                    )

                mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
                _record_uncertainty_stats(stats, logits, mask_index, i, steps, enabled=collect_hypothesis_stats)
                if factor is None:
                    x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
                else:
                    x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
                x[transfer_index] = x0[transfer_index]
                i += 1
                if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                    if early_termination and (x[:, block_start:block_end] == eos_id).any():
                        x[:, block_end: ] = eos_id
                        if collect_figure4a and stats is not None:
                            if 'figure4a_status' not in stats:
                                stats['figure4a_status'] = 'not_captured'
                            stats['figure4a_captured'] = bool(figure4a_captured)
                        if probe_collect_metrics and stats is not None:
                            _probe_a_finalize_logs(
                                stats=stats,
                                block_layer_metrics=probe_layer_metrics_by_block,
                                block_step_metrics=probe_step_metrics_by_block
                                if probe_step_metrics_by_block is not None
                                else [{"entropy": [], "confidence": [], "top1_match": []} for _ in range(num_blocks)],
                                cfg=probe_cfg,
                            )
                        if probe_b_collect_metrics and stats is not None:
                            _probe_b_finalize_logs(
                                stats=stats,
                                step_metrics=probe_b_step_metrics,
                                cfg=probe_b_cfg,
                            )
                        return x, nfe
                    break

        if collect_figure4a and stats is not None:
            if 'figure4a_status' not in stats:
                stats['figure4a_status'] = 'not_captured'
            stats['figure4a_captured'] = bool(figure4a_captured)
            stats['figure4a_exit_after_capture'] = bool(figure4a_exit_after_capture)
        if probe_collect_metrics and stats is not None:
            _probe_a_finalize_logs(
                stats=stats,
                block_layer_metrics=probe_layer_metrics_by_block,
                block_step_metrics=probe_step_metrics_by_block
                if probe_step_metrics_by_block is not None
                else [{"entropy": [], "confidence": [], "top1_match": []} for _ in range(num_blocks)],
                cfg=probe_cfg,
            )
        if probe_b_collect_metrics and stats is not None:
            _probe_b_finalize_logs(
                stats=stats,
                step_metrics=probe_b_step_metrics,
                cfg=probe_b_cfg,
            )
        return x, nfe
    finally:
        if trajectory_runtime is not None:
            trajectory_runtime.close()



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, eos_id=126081, threshold=None, factor=None, early_termination=False, stats=None,
             collect_hypothesis_stats=False, probe_a=None, probe_b=None):
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
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    probe_cfg = _probe_a_normalize_config(probe_a)
    probe_enabled = bool(probe_cfg.get("enable", False))
    probe_n_layers = _probe_a_get_num_layers(model) if probe_enabled else 0
    if probe_enabled and stats is not None:
        stats["probe_a_enabled"] = True
        stats["probe_a_mode"] = str(probe_cfg.get("mode", "global_shift"))
        stats["probe_a_delta"] = int(probe_cfg.get("delta", 0))

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length
        available_suffix_tokens = max(0, x.shape[1] - current_block_end)
        _record_suffix_stats(stats, available_suffix_tokens, available_suffix_tokens)
        _record_suffix_bucket_stats(stats, current_block_end, x.shape[1])

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        if probe_enabled:
            full_seq_len = int(x.shape[1])
            base_q_abs = torch.arange(0, full_seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
            base_k_abs = torch.arange(0, full_seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
            probe_q_arg, probe_k_arg, rope_seq_len = _probe_a_prepare_indices_for_call(
                base_q_abs=base_q_abs,
                base_k_abs=base_k_abs,
                full_seq_len=full_seq_len,
                suffix_start=current_block_end,
                cfg=probe_cfg,
                n_layers=probe_n_layers,
                block_idx=num_block,
                step_idx=0,
            )
            output = model(
                x,
                use_cache=True,
                q_indices=probe_q_arg,
                k_indices=probe_k_arg,
                seq_len=rope_seq_len,
                update_rope=True,
            )
        else:
            output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        _record_uncertainty_stats(stats, output.logits, mask_index, step_idx=0, total_steps=steps, enabled=collect_hypothesis_stats)
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                if early_termination and (x[:, current_block_start: current_block_end] == eos_id).any():
                    x[:, current_block_end: ] = eos_id
                    return x, nfe
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            if probe_enabled:
                full_seq_len = int(x.shape[1])
                q_len = int(x[:, current_block_start:].shape[1])
                base_q_abs = torch.arange(
                    int(current_block_start),
                    int(current_block_start) + q_len,
                    device=x.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                base_k_abs = torch.arange(0, full_seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
                probe_q_arg, probe_k_arg, rope_seq_len = _probe_a_prepare_indices_for_call(
                    base_q_abs=base_q_abs,
                    base_k_abs=base_k_abs,
                    full_seq_len=full_seq_len,
                    suffix_start=current_block_end,
                    cfg=probe_cfg,
                    n_layers=probe_n_layers,
                    block_idx=num_block,
                    step_idx=i,
                )
                logits = model(
                    x[:, current_block_start:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    q_indices=probe_q_arg,
                    k_indices=probe_k_arg,
                    seq_len=rope_seq_len,
                    update_rope=(i == 1),
                ).logits
            else:
                logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits
            _record_uncertainty_stats(stats, logits, mask_index, i, steps, enabled=collect_hypothesis_stats)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe


@ torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, eos_id=126081, threshold=None, factor=None, early_termination=False, stats=None,
            collect_hypothesis_stats=False, probe_a=None, probe_b=None):
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
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    probe_cfg = _probe_a_normalize_config(probe_a)
    probe_enabled = bool(probe_cfg.get("enable", False))
    probe_n_layers = _probe_a_get_num_layers(model) if probe_enabled else 0
    if probe_enabled and stats is not None:
        stats["probe_a_enabled"] = True
        stats["probe_a_mode"] = str(probe_cfg.get("mode", "global_shift"))
        stats["probe_a_delta"] = int(probe_cfg.get("delta", 0))

    nfe = 0  
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length
        available_suffix_tokens = max(0, x.shape[1] - current_block_end)
        _record_suffix_stats(stats, available_suffix_tokens, available_suffix_tokens)
        _record_suffix_bucket_stats(stats, current_block_end, x.shape[1])

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        if probe_enabled:
            full_seq_len = int(x.shape[1])
            base_q_abs = torch.arange(0, full_seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
            base_k_abs = torch.arange(0, full_seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
            probe_q_arg, probe_k_arg, rope_seq_len = _probe_a_prepare_indices_for_call(
                base_q_abs=base_q_abs,
                base_k_abs=base_k_abs,
                full_seq_len=full_seq_len,
                suffix_start=current_block_end,
                cfg=probe_cfg,
                n_layers=probe_n_layers,
                block_idx=num_block,
                step_idx=0,
            )
            output = model(
                x,
                use_cache=True,
                q_indices=probe_q_arg,
                k_indices=probe_k_arg,
                seq_len=rope_seq_len,
                update_rope=True,
            )
        else:
            output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        _record_uncertainty_stats(stats, output.logits, mask_index, step_idx=0, total_steps=steps, enabled=collect_hypothesis_stats)
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                if early_termination and (x[:, current_block_start: current_block_end] == eos_id).any():
                    x[:, current_block_end: ] = eos_id
                    return x, nfe
                break
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            if probe_enabled:
                full_seq_len = int(x.shape[1])
                q_len = int(current_block_end - current_block_start)
                base_q_abs = torch.arange(
                    int(current_block_start),
                    int(current_block_start) + q_len,
                    device=x.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                base_k_abs = torch.arange(0, full_seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
                probe_q_arg, probe_k_arg, rope_seq_len = _probe_a_prepare_indices_for_call(
                    base_q_abs=base_q_abs,
                    base_k_abs=base_k_abs,
                    full_seq_len=full_seq_len,
                    suffix_start=current_block_end,
                    cfg=probe_cfg,
                    n_layers=probe_n_layers,
                    block_idx=num_block,
                    step_idx=i,
                )
                logits = model(
                    x[:, current_block_start:current_block_end],
                    past_key_values=past_key_values,
                    use_cache=True,
                    replace_position=replace_position,
                    q_indices=probe_q_arg,
                    k_indices=probe_k_arg,
                    seq_len=rope_seq_len,
                    update_rope=(i == 1),
                ).logits
            else:
                logits = model(
                    x[:, current_block_start:current_block_end],
                    past_key_values=past_key_values,
                    use_cache=True,
                    replace_position=replace_position,
                ).logits
            _record_uncertainty_stats(stats, logits, mask_index, i, steps, enabled=collect_hypothesis_stats)

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], None, factor)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
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
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
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
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
