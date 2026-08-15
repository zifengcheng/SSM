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

import os
import subprocess
import shlex  
import socket
from params import parser


def _count_visible_gpus():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return None
    devices = [d.strip() for d in cvd.split(",") if d.strip() != "" and d.strip() != "-1"]
    return len(devices)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _default_accelerate_config_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "scripts", "accelerate_single_gpu.yaml")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.single_gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.single_gpu_id)

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    for model in ['instruct', '1.5', 'base']:
        if not os.path.exists(f"output/log/{model}/"):
            os.makedirs(f"output/log/{model}/")
        if not os.path.exists(f"output/debug/{model}/"):
            os.makedirs(f"output/debug/{model}/")

    if args.model == 'instruct':
        model = 'GSAI-ML/LLaDA-8B-Instruct'
    elif args.model == 'base':
        model = 'GSAI-ML/LLaDA-8B-Base'
    elif args.model == '1.5':
        model = 'GSAI-ML/LLaDA-1.5'
    else:
        raise ValueError(f"Unknown model: {args.model}")
    if args.threshold == 0:
        steps = args.gen_length
        sampling = '_t_'
        threshold = ''
    else:
        assert 0 < args.threshold < 1, "Invalid threshold"
        steps = args.gen_length // args.block_size
        sampling = '_p_'
        threshold = f'threshold={args.threshold},'

    if args.dc is True:
        args.c = True
        cache = 'dc_'
    elif args.c is True:
        cache = 'c_'
    else:
        cache = ''

    if args.e is True:
        early = 'e_'
    else:
        early = ''
 
    if args.name != '':
        args.name += '_'

    if args.d == 'gaussian':
        dropout = f"_g_sigma{args.k_sigma}_scale{args.scale}_win{args.window}"
    elif args.d == 'ssm':
        dropout = (
            f"_ssm_win{args.window}_local{args.local_window}"
            f"_se{args.use_suffix_end}_bb{args.use_block_boundaries}"
            f"_bbm{args.block_boundary_mode}_bbo{args.block_boundary_offset}"
        )
    elif args.d == 'ssm_local_none':
        dropout = (
            f"_ssm_localnone_win{args.window}_local{args.local_window}"
            f"_se{args.use_suffix_end}_bb{args.use_block_boundaries}"
            f"_bbm{args.block_boundary_mode}_bbo{args.block_boundary_offset}"
        )
    elif args.d == 'ssm_mid_none':
        dropout = (
            f"_ssm_midnone_win{args.window}_local{args.local_window}"
            f"_se{args.use_suffix_end}_bb{args.use_block_boundaries}"
            f"_bbm{args.block_boundary_mode}_bbo{args.block_boundary_offset}"
        )
    elif args.d == 'ssm_mid_end_only':
        dropout = (
            f"_ssm_midend_win{args.window}_local{args.local_window}"
            f"_se{args.use_suffix_end}_bb{args.use_block_boundaries}"
            f"_bbm{args.block_boundary_mode}_bbo{args.block_boundary_offset}"
        )
    elif args.d == 'ssm_mid_start_end':
        dropout = (
            f"_ssm_midse_win{args.window}_local{args.local_window}"
            f"_se{args.use_suffix_end}_bb{args.use_block_boundaries}"
            f"_bbm{args.block_boundary_mode}_bbo{args.block_boundary_offset}"
        )
    elif args.d == 'ssm_mid_mid_only':
        dropout = (
            f"_ssm_midonly_win{args.window}_local{args.local_window}"
            f"_se{args.use_suffix_end}_bb{args.use_block_boundaries}"
            f"_bbm{args.block_boundary_mode}_bbo{args.block_boundary_offset}"
        )
    elif args.d == 'ssm_last_start_middle_end':
        dropout = (
            f"_ssm_lastsme_win{args.window}_local{args.local_window}"
            f"_se{args.use_suffix_end}_bb{args.use_block_boundaries}"
            f"_bbm{args.block_boundary_mode}_bbo{args.block_boundary_offset}"
        )
    elif args.d == 'ssm_last_end_only':
        dropout = (
            f"_ssm_lastend_win{args.window}_local{args.local_window}"
            f"_se{args.use_suffix_end}_bb{args.use_block_boundaries}"
            f"_bbm{args.block_boundary_mode}_bbo{args.block_boundary_offset}"
        )
    elif args.d == 'ssm_last_start_only':
        dropout = (
            f"_ssm_laststart_win{args.window}_local{args.local_window}"
            f"_se{args.use_suffix_end}_bb{args.use_block_boundaries}"
            f"_bbm{args.block_boundary_mode}_bbo{args.block_boundary_offset}"
        )
    elif args.d == 'ssm_last_none':
        dropout = (
            f"_ssm_lastnone_win{args.window}_local{args.local_window}"
            f"_se{args.use_suffix_end}_bb{args.use_block_boundaries}"
            f"_bbm{args.block_boundary_mode}_bbo{args.block_boundary_offset}"
        )
    else:
        assert args.d == 'null', "Invalid dropout strategy"
        dropout = ''


    if args.use_suffix_soft_state:
        dropout += f"_sss_k{args.suffix_soft_topk}_a{args.suffix_soft_alpha}_b{args.current_warm_start_beta}"
        if args.suffix_soft_non_local_only:
            dropout += "_nlocal"

    filename = f"{args.name}{args.task}{sampling}{cache}{early}len{args.gen_length}_blk{args.block_size}{dropout}"
    log_file = f"output/log/{args.model}/{filename}.log"
    debug_file = f"output/debug/{args.model}/{filename}.log"
    save_dir = f"output/checkpoint/{args.model}/{filename}"

    selected_accelerate_config = args.accelerate_config_file
    if selected_accelerate_config is None:
        default_cfg = _default_accelerate_config_path()
        if os.path.exists(default_cfg):
            selected_accelerate_config = default_cfg

    # --- [The Elegant Way] Build and execute the command using subprocess ---

    # 1. Build the command as a list of arguments for safety and clarity.
    base_cmd = ['accelerate', 'launch']
    if selected_accelerate_config is not None:
        base_cmd.extend(['--config_file', selected_accelerate_config])

    visible_gpu_count = _count_visible_gpus()

    # If only one GPU is visible for this process, force a single worker.
    if visible_gpu_count == 1:
        base_cmd.extend(['--num_processes', '1'])

    # Hard constraint: when user explicitly pins one physical GPU, keep single process only.
    if args.single_gpu_id is not None:
        base_cmd.extend(['--num_processes', '1'])
    
    # Add port configuration if specified (must come before script name)
    if args.main_process_port is not None:
        base_cmd.extend(['--main_process_port', str(args.main_process_port)])
    else:
        # Choose a concrete free port to avoid both conflict and invalid ':0' rendezvous.
        base_cmd.extend(['--main_process_port', str(_find_free_port())])
    
    base_cmd.append('eval_llada.py')

    task_args = ['--tasks', args.task]
    if args.task == 'humaneval':
        task_args.append('--log_samples')
    else:
        task_args.extend(['--num_fewshot', str(args.num_fewshot)])
    
    if args.limit is not None:
        task_args.extend(['--limit', str(args.limit)])

    model_args_string = (
        f"model_path={model},gen_length={args.gen_length},steps={steps},block_length={args.block_size},"
        f"{threshold}"
        f"from_scratch={args.re},save_dir={save_dir},show_speed=True,"
        f"use_cache={args.c},dual_cache={args.dc},early_termination={args.e},"
        f"dropout={args.d},sigma={args.k_sigma},scale={args.scale},preserved_tokens={args.nt},window={args.window},"
        f"local_window={args.local_window},"
        f"use_suffix_soft_state={args.use_suffix_soft_state},"
        f"suffix_soft_topk={args.suffix_soft_topk},"
        f"suffix_soft_alpha={args.suffix_soft_alpha},"
        f"current_warm_start_beta={args.current_warm_start_beta},"
        f"suffix_soft_non_local_only={args.suffix_soft_non_local_only},"
    )

    # 2. Assemble the final command list.
    cmd_list = base_cmd + task_args
    cmd_list.extend(['--confirm_run_unsafe_code', '--model', 'llada_dist'])
    cmd_list.extend(['--model_args', model_args_string])

    # Add the specific output path parameter based on the task.
    if args.task == 'humaneval':
        output_path = f"output/humaneval_results/{model}/{filename}"
        cmd_list.extend(['--output_path', output_path])

    # 3. Print the command
    if selected_accelerate_config is not None:
        print(f"[accelerate] override config: {selected_accelerate_config}")
    print(f"[gpu] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(shlex.join(cmd_list))
    print("-" * 50)  # Separator

    # 4. Execute the command
    try:
        # Use a 'with' statement to open log files, ensuring they are properly closed afterward.
        with open(log_file, 'w') as log_f, open(debug_file, 'w') as debug_f:
            # Execute the command, redirecting stdout and stderr to the log files.
            print(f"   Log file: {log_file}")
            print(f"   Debug file: {debug_file}")

            run_env = os.environ.copy()
            if args.single_gpu_id is not None:
                run_env["CUDA_VISIBLE_DEVICES"] = str(args.single_gpu_id)
            if selected_accelerate_config is not None:
                if not os.path.exists(selected_accelerate_config):
                    raise FileNotFoundError(f"Accelerate config file not found: {selected_accelerate_config}")
                run_env["ACCELERATE_CONFIG_FILE"] = selected_accelerate_config

            subprocess.run(
                cmd_list,
                stdout=log_f,       # Redirect standard output
                stderr=debug_f,       # Redirect standard error
                env=run_env,
                check=True          # Raise an exception on non-zero exit codes (errors)
            )
        print(f"\n✅ Command completed successfully.")

    except FileNotFoundError:
        print(f"❌ Error")
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code: {e.returncode}.")
        print(f"   Check the debug log for details: {debug_file}")
