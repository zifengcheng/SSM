# LLaDA-8B-Instruct

# GSM8K - Diff Gen len
# DPad
# Gen len = 128
python run.py -n="Diff_Gen_len128_DPad" -t=gsm8k -m=instruct -s=4 -l=128 -b=32 -th=0.9 -e -d=gaussian -k=4 -sc=2 -w=128 -re 

# Gen len = 256
python run.py -n="Diff_Gen_len256_DPad" -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -th=0.9 -e -d=gaussian -k=4 -sc=2 -w=256 -re 

# Gen len = 512
python run.py -n="Diff_Gen_len512_DPad" -t=gsm8k -m=instruct -s=4 -l=512 -b=32 -th=0.9 -e -d=gaussian -k=4 -sc=2 -w=256 -re 

# Gen len = 1024
python run.py -n="Diff_Gen_len1024_DPad" -t=gsm8k -m=instruct -s=4 -l=1024 -b=32 -th=0.9 -e -d=gaussian -k=4 -sc=2 -w=256 -re 

# SLSP + Parallel

# Gen len = 128
python run.py -n="Diff_Gen_len128_SLSP" -t=gsm8k -m=instruct -s=4 -l=128 -b=32 -th=0.9 -e \
  -re -lw=64 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 3 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0

# Gen len = 256
python run.py -n="Diff_Gen_len256_SLSP" -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -th=0.9 -e \
  -re -lw=64 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 3 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0

# Gen len = 512
python run.py -n="Diff_Gen_len512_SLSP" -t=gsm8k -m=instruct -s=4 -l=512 -b=32 -th=0.9 -e \
  -re -lw=64 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 3 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0

# Gen len = 1024
python run.py -n="Diff_Gen_len1024_SLSP" -t=gsm8k -m=instruct -s=4 -l=1024 -b=32 -th=0.9 -e \
  -re -lw=64 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 3 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0

# HumanEval - Diff Gen len
# DPad+Par
# Gen len = 128
python run.py -n="Diff_Gen_len128_DPad" -t=humaneval -m=instruct -l=128 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=2.3 -w=128 -re 

# Gen len = 256
python run.py -n="Diff_Gen_len256_DPad" -t=humaneval -m=instruct -l=256 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=2.3 -w=256 -re

# Gen len = 512
python run.py -n="Diff_Gen_len512_DPad" -t=humaneval -m=instruct -l=512 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=2.3 -w=512 -re

# Gen len = 1024
python run.py -n="Diff_Gen_len1024_DPad" -t=humaneval -m=instruct -l=1024 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=2.3 -w=512 -re

# SLSP + Parallel
# Gen len = 128
python run.py -n="Diff_Gen_len128_SLSP" -t=humaneval -m=instruct -s=0 -l=128 -b=32 -th=0.9 -e -re -lw=40 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 7 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0

# Gen len = 256
python run.py -n="Diff_Gen_len256_SLSP" -t=humaneval -m=instruct -s=0 -l=256 -b=32 -th=0.9 -e -re -lw=80 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 7 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0

# Gen len = 512
python run.py -n="Diff_Gen_len512_SLSP" -t=humaneval -m=instruct -s=0 -l=512 -b=32 -th=0.9 -e -re -lw=160 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 7 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0

# Gen len = 1024
python run.py -n="Diff_Gen_len1024_SLSP" -t=humaneval -m=instruct -s=0 -l=1024 -b=32 -th=0.9 -e -re -lw=160 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 7 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0