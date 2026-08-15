# +Parallel+DPad
python run.py -t=gsm8k -m=1.5 -s=1 -l=1024 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=1.6 -w=256 -re 

# +Par + SSM

python run.py -n="Long_seq_SSM+Par" -t=gsm8k -m=1.5 -s=1 -l=1024 -b=32 -th=0.9 -e \
  -re -lw=72 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 3 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0


# +Parallel+PrefixCache+SSM
python run.py -n="Long_seq_SSM+Par+PC" -t=gsm8k -m=1.5 -s=1 -l=1024 -b=32 -th=0.9 -e -c \
  -re -lw=72 -d=ssm \
  --use_suffix_soft_state \
  --suffix_soft_topk 3 --suffix_soft_alpha 0.4 --current_warm_start_beta 1.0 