import subprocess

exps = ['warmup', 'seq1', 'seq2', 'seq3', 'ewc1', 'ewc2', 'ewc3']
exps = ['warmup']
python = 'C:/Users/mhara/OneDrive/Documents/Projects/geneticfed/venv/Scripts/python.exe'
script = 'main.py'

for param in exps:
    subprocess.run([python, script, param])
