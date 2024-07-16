import subprocess

python = 'C:/Users/mhara/OneDrive/Documents/Projects/geneticfed/venv/Scripts/python.exe'
scripts = ['./seq_ga.py', './seq_none.py', './seq_rn.py']

for script in scripts:
    subprocess.run([python, script])
