# Vanilla
python run.py -n="Vanilla" -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -re 
python run.py -n="Vanilla" -t=minerva_math -m=instruct -s=4 -l=256 -b=32 -re 
python run.py -n="Vanilla" -t=humaneval -m=instruct -l=512 -b=32 -re 
python run.py -n="Vanilla" -t=mbpp -m=instruct -s=3 -l=512 -b=32 -re 

# +Parallel
python run.py -n="+Par" -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -th=0.9 -re 
python run.py -n="+Par" -t=minerva_math -m=instruct -s=4 -l=256 -b=32 -th=0.9 -re 
python run.py -n="+Par" -t=humaneval -m=instruct -l=512 -b=32 -th=0.9 -re 
python run.py -n="+Par" -t=mbpp -m=instruct -s=3 -l=512 -b=32 -th=0.9 -re 

# +Parallel+DPad
python run.py -n="DPad+Par" -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -th=0.9 -e -d=gaussian -k=4 -sc=2 -w=256 -re 
python run.py -n="DPad+Par" -t=minerva_math -m=instruct -s=4 -l=256 -b=32 -th=0.9 -e -d=gaussian -k=4 -sc=2 -w=256 -re 
python run.py -n="DPad+Par" -t=humaneval -m=instruct -l=512 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=2.3 -w=512 -re 
python run.py -n="DPad+Par" -t=mbpp -m=instruct -s=3 -l=512 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=2.3 -w=128 -re 

# +Parallel+Streaming-dLLM
python run.py -n="Streaming+Par" -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -th=0.9 -e -re -lw=64 -d=streaming_dllm
python run.py -n="Streaming+Par" -t=minerva_math -m=instruct -s=4 -l=256 -b=32 -th=0.9 -e -re -lw=64 -d=streaming_dllm
python run.py -n="Streaming+Par" -t=humaneval -m=instruct -s=0 -l=512 -b=32 -th=0.9 -e -re -lw=160 -d=streaming_dllm
python run.py -n="Streaming+Par" -t=mbpp -m=instruct -s=3 -l=512 -b=32 -th=0.9 -e -re -lw=80 -d=streaming_dllm --suffix_soft_topk 7 --suffix_soft_alpha 0.3


# +Parallel+SSM
python run.py -n="SSM+Par" -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -th=0.9 -e -re -lw=64 -d=ssm --use_suffix_soft_state -sk=3 -sa=0.3
python run.py -n="SSM+Par" -t=minerva_math -m=instruct -s=4 -l=256 -b=32 -th=0.9 -e -re -lw=64 -d=ssm --use_suffix_soft_state -sk=7 -sa=0.4
python run.py -n="SSM+Par" -t=humaneval -m=instruct -s=0 -l=512 -b=32 -th=0.9 -e -re -lw=160 -d=ssm --use_suffix_soft_state -sk=7 -sa=0.4
python run.py -n="SSM+Par" -t=mbpp -m=instruct -s=3 -l=512 -b=32 -th=0.9 -e -re -lw=80 -d=ssm --use_suffix_soft_state -sk=7 -sa=0.4
