# Vanilla
python run.py -n="Vanilla" -t=gsm8k -m=base -s=4 -l=256 -b=32 -re 
python run.py -n="Vanilla" -t=minerva_math -m=base -s=4 -l=256 -b=32 -re 
python run.py -n="Vanilla" -t=humaneval -m=base -l=512 -b=32 -re 
python run.py -n="Vanilla" -t=mbpp -m=base -s=3 -l=512 -b=32 -re 

# SSM
python run.py -n="SSM" -t=gsm8k -m=base -s=4 -l=256 -b=32 -lw=72 -d=ssm -e -re 
python run.py -n="SSM" -t=minerva_math -m=base -s=4 -l=256 -b=32 -lw=72 -d=ssm -e -re --use_suffix_soft_state -th=0.9 \
  --suffix_soft_topk 5 --suffix_soft_alpha 0.2 --current_warm_start_beta 1.0 
python run.py -n="SSM" -t=humaneval -m=base -l=512 -b=32 -lw=72 -d=ssm -e -re --use_suffix_soft_state -th=0.9 \
  --suffix_soft_topk 5 --suffix_soft_alpha 0.2 --current_warm_start_beta 1.0  
python run.py -n="SSM" -t=mbpp -m=base -s=3 -l=512 -b=32 -lw=48 -d=ssm -e -re 


# SSM+Par
python run.py -n="SSM+Par" -t=gsm8k -m=base -s=4 -l=256 -b=32 -lw=48 -d=ssm -e -re --use_suffix_soft_state -th=0.9 \
  --suffix_soft_topk 5 --suffix_soft_alpha 0.2 --current_warm_start_beta 1.0 
python run.py -n="SSM+Par" -t=minerva_math -m=base -s=4 -l=256 -b=32 -lw=48 -d=ssm -e -re --use_suffix_soft_state -th=0.9 \
  --suffix_soft_topk 5 --suffix_soft_alpha 0.2 --current_warm_start_beta 1.0 
python run.py -n="SSM+Par" -t=humaneval -m=base -l=512 -b=32 -lw=72 -d=ssm -e -re --use_suffix_soft_state -th=0.9 \
  --suffix_soft_topk 5 --suffix_soft_alpha 0.2 --current_warm_start_beta 1.0  
python run.py -n="SSM+Par" -t=mbpp -m=base -s=3 -l=512 -b=32 -lw=48 -d=ssm -e -re --use_suffix_soft_state -th=0.9 \
  --suffix_soft_topk 5 --suffix_soft_alpha 0.2 --current_warm_start_beta 1.0  
