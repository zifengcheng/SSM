# Vanilla
python run.py -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -re 
python run.py -t=minerva_math -m=instruct -s=4 -l=256 -b=32 -re 
python run.py -t=humaneval -m=instruct -l=512 -b=32 -re 
python run.py -t=mbpp -m=instruct -s=3 -l=512 -b=32 -re 

# +DPad
python run.py -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -d=gaussian -k=4 -sc=2 -w=256 -e -re 
python run.py -t=minerva_math -m=instruct -s=4 -l=256 -b=32 -d=gaussian -k=4 -sc=2 -w=256 -e -re 
python run.py -t=humaneval -m=instruct -l=512 -b=32 -d=gaussian -k=3 -sc=2.3 -w=512 -e -re 
python run.py -t=mbpp -m=instruct -s=3 -l=512 -b=32 -d=gaussian -k=3 -sc=2.3 -w=128 -e -re 

# +Parallel
python run.py -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -th=0.9 -re 
python run.py -t=minerva_math -m=instruct -s=4 -l=256 -b=32 -th=0.9 -re 
python run.py -t=humaneval -m=instruct -l=512 -b=32 -th=0.9 -re 
python run.py -t=mbpp -m=instruct -s=3 -l=512 -b=32 -th=0.9 -re 

# +Parallel+DPad
python run.py -t=gsm8k -m=instruct -s=4 -l=256 -b=32 -th=0.9 -e -d=gaussian -k=4 -sc=2 -w=256 -re 
python run.py -t=minerva_math -m=instruct -s=4 -l=256 -b=32 -th=0.9 -e -d=gaussian -k=4 -sc=2 -w=256 -re 
python run.py -t=humaneval -m=instruct -l=512 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=2.3 -w=512 -re 
python run.py -t=mbpp -m=instruct -s=3 -l=512 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=2.3 -w=128 -re 
