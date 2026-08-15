# Vanilla
python run.py -n="Vanilla" -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -re 
python run.py -n="Vanilla" -t=minerva_math -m=instruct -s=4 -l=256 -b=32 -re 
python run.py -n="Vanilla" -t=humaneval -m=instruct -l=512 -b=32 -re 
python run.py -n="Vanilla" -t=mbpp -m=instruct -s=3 -l=512 -b=32 -re

# +SSM
python run.py -n="SSM" -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -e \
  -re -lw=72 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 5 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0

python run.py -n="SSM" -t=minerva_math -m=instruct -s=4 -l=256 -b=32 -e \
  -re -lw=72 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 5 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0  

python run.py -n="SSM" -t=humaneval -m=instruct -s=0 -l=512 -b=32 -e -re -lw=160 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 5 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0

python run.py -n="SSM" -t=mbpp -m=instruct -s=3 -l=512 -b=32 -e -re -lw=48 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 5 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0

# Vanilla + Parallel
python run.py -n="Parallel" -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -th=0.9 -re 
python run.py -n="Parallel" -t=minerva_math -m=instruct -s=4 -l=256 -b=32 -th=0.9 -re 
python run.py -n="Parallel" -t=humaneval -m=instruct -l=512 -b=32 -th=0.9 -re 
python run.py -n="Parallel" -t=mbpp -m=instruct -s=3 -l=512 -b=32 -th=0.9 -re 

# SSM + Parallel

python run.py -n="SSM+Par" -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -th=0.9 -e \
  -re -lw=64 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 3 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0

python run.py -n="SSM+Par" -t=minerva_math -m=instruct -s=4 -l=256 -b=32 -th=0.9 -e \
  -re -lw=64 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 5 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0  

python run.py -n="SSM+Par" -t=humaneval -m=instruct -s=0 -l=512 -b=32 -th=0.9 -e -re -lw=160 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 7 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0

python run.py -n="SSM+Par" -t=mbpp -m=instruct -s=3 -l=512 -b=32 -th=0.9 -e -re -lw=48 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 5 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0