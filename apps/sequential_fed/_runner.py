import subprocess

# exps = ['warmup', 'seq1', 'seq2', 'seq3', 'ewc1', 'ewc2', 'ewc3']
exps = ['warmup1', 'warmup2', 'seq11', 'seq21', 'seq22', 'seq23', 'seq24', 'seq31', 'seq32', 'ewc11', 'ewc12',
        'ewc31', 'ewc32']
python = 'C:/Users/mhara/OneDrive/Documents/Projects/geneticfed/venv/Scripts/python.exe'
script = 'main.py'

for param in exps:
    subprocess.run([python, script, param])
