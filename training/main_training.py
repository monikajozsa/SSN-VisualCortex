# This code runs pretraining for a general and training for a fine orientation discrimination task with a two-layer neural network model, where each layer is an SSN.

import numpy
import pandas as pd
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from util import save_code, load_parameters
from training_functions import train_ori_discr
from perturb_params import randomize_params, create_initial_parameters_df
from parameters import PretrainingPars
pretraining_pars = PretrainingPars() # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
if not pretraining_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to True in parameters.py to run training with pretraining!')

########## Initialize orientation map and gabor filters ############

# Save out initial offset and reference orientation
num_training = 20
starting_time_in_main= time.time()
initial_parameters = None

# Save scripts into scripts folder
note=f'20 test runs on longer pretrainings (min 200 SGD steps)'
results_filename, folder_path = save_code(note=note)

# Run num_training number of pretraining + training
num_FailedRuns = 0
i=0
run_indices=[]

while i < num_training and num_FailedRuns < 20:

    ##### RANDOM INITIALIZATION #####
    numpy.random.seed(i)
    
    ##### Randomize readout_pars, trained_pars, eta such that they satisfy certain conditions #####
    readout_pars_opt_dict, pretrain_pars_rand_dict, untrained_pars = randomize_params(folder_path, i)

    ##### Save initial parameters into initial_parameters variable #####
    initial_parameters = create_initial_parameters_df(folder_path, initial_parameters, readout_pars_opt_dict, pretrain_pars_rand_dict, untrained_pars.training_pars.eta, untrained_pars.filter_pars.gE_m,untrained_pars.filter_pars.gI_m, run_index = i, stage =0)

    ##### PRETRAINING: GENERAL ORIENTAION DISCRIMINATION #####
    training_output_df = train_ori_discr(
            readout_pars_opt_dict,
            pretrain_pars_rand_dict,
            untrained_pars,
            results_filename=results_filename,
            jit_on=True,
            offset_step = 0.1,
            run_index = i
        )
    
    # Handle the case when pretraining failed (possible reason can be the divergence of ssn diff equations)
    if training_output_df is None:
        print('######### Stopped run {} because of NaN values  - num failed runs = {} #########'.format(i, num_FailedRuns))
        num_FailedRuns = num_FailedRuns + 1
        continue  

    ##### Save final values into initial_parameters as initial parameters for training stage #####
    pretrained_readout_pars_dict, pretrained_pars_dict, untrained_pars = load_parameters(folder_path, run_index = i, stage =0, iloc_ind = -1)
    initial_parameters = create_initial_parameters_df(folder_path, initial_parameters, pretrained_readout_pars_dict, pretrained_pars_dict, untrained_pars.training_pars.eta, untrained_pars.filter_pars.gE_m,untrained_pars.filter_pars.gI_m, run_index = i, stage =1)
    
    run_indices.append(i)
    i = i + 1
    print('runtime of {} pretraining'.format(i), time.time()-starting_time_in_main)
    print('number of failed runs = ', num_FailedRuns)


############### FINE DISCRIMINATION ###############

for i in run_indices:
    # Load the last parameters from the pretraining
    pretrained_readout_pars_dict, trained_pars_dict, untrained_pars, offset_last, meanr_vec = load_parameters(folder_path, run_index=i, stage=0, iloc_ind = -1, for_training=True)
    
    # Change mean rate homeostatic loss
    if meanr_vec is not None:
        untrained_pars.loss_pars.lambda_r_mean = 0.25
        untrained_pars.loss_pars.Rmean_E = meanr_vec[0]
        untrained_pars.loss_pars.Rmean_I = meanr_vec[1]

    # Set the offset to the offset threshold, where a given accuracy is achieved with the parameters from the last SGD step (loaded as offset_last)
    untrained_pars.stimuli_pars.offset = min(offset_last,10)

    # Run training
    training_output_df, _ = train_ori_discr(
            pretrained_readout_pars_dict,
            trained_pars_dict,
            untrained_pars,
            results_filename=results_filename,
            jit_on=True,
            offset_step=0.1,
            run_index = i
        )
    
############### Run main analysis ###############
# Get the path to the sibling folder
sibling_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analysis')

# Get the path to the script.py file
script_path = os.path.join(sibling_folder, 'main_analysis.py')

with open(script_path) as file:
    pretraining_pars.is_on = True
    untrained_pars.pretrain_pars.is_on = True
    exec(file.read())