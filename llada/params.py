# Copyright 2025 Xinhua Chen, Duke CEI Center
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

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--name", type=str, default='', help="Test Name")

parser.add_argument("-t", "--task", type=str, default='gsm8k', help="Task name")
parser.add_argument("-l", "--gen_length", type=int, default=256, help="Response length")
parser.add_argument("-b", "--block_size", type=int, default=32, help="Block size")
parser.add_argument("-s", "--num_fewshot", type=int, default=0, help="Number of few-shot examples")
parser.add_argument("-th", "--threshold", type=float, default=0, help="Threshold value")
parser.add_argument("-m", "--model", type=str, default='instruct', help="Model name")

parser.add_argument(
    "-d",
    "--dropout_strategy",
    dest="d",
    type=str,
    default="null",
    choices=[
        "null",
        "gaussian",
        "ssm",
        "ssm_local_none",
        "ssm_mid_none",
        "ssm_mid_end_only",
        "ssm_mid_start_end",
        "ssm_mid_mid_only",
        "ssm_last_start_middle_end",
        "ssm_last_end_only",
        "ssm_last_start_only",
        "ssm_last_none",
    ],
    help="Suffix sampling strategy",
)
parser.add_argument("-w", "--window", type=int, default=256, help="Dropout window size")
parser.add_argument("-k", "--k_sigma", type=int, default=4, help="The end of dropout window falls at k * sigma in Gaussian Distribution")
parser.add_argument("-sc", "--scale", type=float, default=2.0, help="Scale factor for Gaussian Pruning")
parser.add_argument("-nt", "--num_tokens", dest='nt', type=int, default=0, help="Number of reserved tokens for Random Pruning")

parser.add_argument("-c", "--use_cache", dest='c', help='Use cache', action='store_true')
parser.add_argument("-dc", "--dual_cache", dest='dc', help='Dual cache', action='store_true')
parser.add_argument("-re", "--from_scratch", dest='re', help='From scratch', action='store_true')
parser.add_argument("-e", "--early_termination", dest='e', help='From scratch', action='store_true')
parser.add_argument("-p", "--main_process_port", type=int, default=None, help="Main process port for distributed training (default: None, uses accelerate default or 0 for auto)")
parser.add_argument("--accelerate_config_file", type=str, default=None, help="Optional Accelerate config override path. If omitted, run.py auto-uses scripts/accelerate_single_gpu.yaml when present")
parser.add_argument("--single_gpu_id", type=int, default=None, help="Force bind this run to one physical GPU id via CUDA_VISIBLE_DEVICES")

parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to evaluate")
parser.add_argument("-lw", "--local_window", type=int, default=128, help="Size of local continuous window for SPC Sampler")
parser.add_argument("--use_suffix_end", action="store_true", default=False, help="Keep suffix-end anchor in SSM")
parser.add_argument("--use_block_boundaries", action="store_true", default=False, help="Keep future block boundaries in SSM")
parser.add_argument("--block_boundary_mode", type=str, default="start", choices=["start", "end", "both", "fixed", "random"], help="How to choose boundary token(s) inside each future block when use_block_boundaries is enabled")
parser.add_argument("--block_boundary_offset", type=int, default=0, help="Offset for block_boundary_mode=fixed")
parser.add_argument("--use_suffix_soft_state", action="store_true", default=False, help="Enable suffix soft states (top-k soft embedding) and current-block warm start from suffix states")
parser.add_argument("-sk", "--suffix_soft_topk", type=int, default=5, help="Top-k vocabulary size for suffix soft state construction")
parser.add_argument("-sa", "--suffix_soft_alpha", type=float, default=0.5, help="Blend alpha for suffix soft state: h=(1-alpha)*e_mask + alpha*E_topk")
parser.add_argument("--suffix_soft_non_local_only", action="store_true", default=False, help="Experiment-2: apply suffix soft state only on selected suffix tokens outside local window (abs_pos >= block_end + local_window)")
