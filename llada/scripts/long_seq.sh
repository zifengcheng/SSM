# Vanilla
python run.py -t=gsm8k -m=1.5 -s=1 -l=1024 -b=32 -re 

# Vanilla (Early Termination)
python run.py -t=gsm8k -m=1.5 -s=1 -l=1024 -b=32 -e -re 

# +DPad
python run.py -t=gsm8k -m=1.5 -s=1 -l=1024 -b=32 -d=gaussian -k=3 -sc=1.6 -w=256 -e -re 

# +Parallel
python run.py -t=gsm8k -m=1.5 -s=1 -l=1024 -b=32 -th=0.9 -re 

# +Parallel+DPad
python run.py -t=gsm8k -m=1.5 -s=1 -l=1024 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=1.6 -w=256 -re 

# +Parallel+PrefixCache
python run.py -t=gsm8k -m=1.5 -s=1 -l=1024 -b=32 -th=0.9 -c -re

# +Parallel+PrefixCache+DPad
python run.py -n="Test" -t=gsm8k -m=instruct -s=1 -l=1024 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=1.6 -w=256 -c -re 

python run.py -n="Test" -t=gsm8k -m=instruct -s=1 -l=1024 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=1.6 -w=256 -c -re 

python run.py -n="Test" -t=gsm8k -m=instruct -s=1 -l=1024 -b=32 -th=0.9 -e -d=gaussian -k=4 -sc=2 -w=256 -c -re 