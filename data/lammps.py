import os
import shutil
import subprocess

def assignVariables(file_path, variables, values):
    """
    Assigns a variable to a value.
    """
    # Read in the file
    with open('templates/hexagon.in', 'r') as f:
        filedata = f.read()

    # Replace the target string
    for i in range(len(variables)):
        filedata = filedata.replace(variables[i], str(values[i]))

    # Write the file out again
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(f'{file_path}/hexagon.in', 'w') as f:
        f.write(filedata)

    # Copy body file
    shutil.copyfile('templates/hex.conf', f'{file_path}/hex.conf')

    # Copy run file
    shutil.copyfile('templates/run.sh', f'{file_path}/run.sh')

    



# run main
if __name__ == '__main__':

    variables = ['$CUT', '$TEMP', '$R0', '$K', '$SEED', '$LOG_FREQ', '$RUNSTEPS', '$TIMESTEP' ]
    
    cut = 1.12246
    temp = 0.5
    r0 = 0
    k = 10*temp/cut/cut
    
    # takes about 50 seconds
    log_freq = 100
    runsteps = 10000000
    timestep = 0.00001


    # train
    for seed in [1, 2, 3, 4, 5]:
        values = [cut, temp, r0, k, seed, log_freq, runsteps, timestep]
        assignVariables(f'../dataset/single_T/train/{seed}', variables, values)
        
    # test
    for seed in [6, 7]:
        values = [cut, temp, r0, k, seed, log_freq, runsteps, timestep]
        assignVariables(f'../dataset/single_T/test/{seed}', variables, values)


    # validate
    for seed in [8, 9]:
        values = [cut, temp, r0, k, seed, log_freq, runsteps, timestep]
        assignVariables(f'../dataset/single_T/validation/{seed}', variables, values)
        
    

