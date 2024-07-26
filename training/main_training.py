# This code runs pretraining for a general and training for a fine orientation discrimination task with a two-layer neural network model, where each layer is an SSN.

import numpy
import pandas as pd
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from util_gabor import init_untrained_pars
from util import save_code, load_parameters, filter_for_run
from training_functions import train_ori_discr
from perturb_params import randomize_params, create_initial_parameters_df
from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    readout_pars,
    ssn_pars,
    pretrained_pars,
    trained_pars,
    conv_pars,
    training_pars,
    loss_pars,
    pretrain_pars, # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
    randomize_pars
)

# Checking that pretrain_pars.is_on is on
if not pretrain_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to True in parameters.py to run training with pretraining!')

########## Initialize orientation map and gabor filters ############

# Save out initial offset and reference orientation
ref_ori_saved = float(stimuli_pars.ref_ori)
offset_saved = float(stimuli_pars.offset)
num_training = 3
starting_time_in_main= time.time()
initial_parameters = None

# Save scripts into scripts folder
note=f'3 test runs'
results_filename, final_folder_path = save_code(note=note)


# Run num_training number of pretraining + training
num_FailedRuns = 0
i=0
trained_pars_keys = trained_pars_dict = {attr: getattr(trained_pars, attr) for attr in dir(trained_pars)}
while i < num_training and num_FailedRuns < 20:
    numpy.random.seed(i+1)

    # Set pretraining flag to False
    pretrain_pars.is_on=True
    # Set offset and reference orientation to their initial values
    stimuli_pars.offset=offset_saved
    stimuli_pars.ref_ori=ref_ori_saved

    # Initialize untrained parameters with randomized gE_m, g_I_m (includes generating orientation map and calculating gabor filters)
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, i, folder_to_save=final_folder_path, randomize_g=randomize_pars)
    # Randomize readout_pars, trained_pars, eta such that they satisfy certain conditions
    trained_pars_stage1, pretrained_pars_stage2, untrained_pars = randomize_params(readout_pars, trained_pars, untrained_pars, randomize_pars=randomize_pars)
    # Save initial parameters into initial_parameters variable
    initial_parameters = create_initial_parameters_df(initial_parameters, trained_pars_stage1, pretrained_pars, untrained_pars.training_pars.eta, untrained_pars.filter_pars.gE_m,untrained_pars.filter_pars.gI_m)


    ##### PRETRAINING: GENERAL ORIENTAION DISCRIMINATION #####

    # Run pre-training
    training_output_df, pretraining_final_step = train_ori_discr(
            trained_pars_stage1,
            pretrained_pars_stage2,
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
    
    ##### FINE DISCRIMINATION #####
    
    # Set pretraining flag to False
    untrained_pars.pretrain_pars.is_on = False

    # Load the last parameters from the pretraining
    df = pd.read_csv(results_filename)
    df_i = filter_for_run(df, i)
    trained_pars_stage1, trained_pars_stage2, untrained_pars, offset_last, meanr_vec = load_parameters(df_i, iloc_ind = pretraining_final_step, trained_pars=trained_pars_keys, untrained_pars = untrained_pars)
    
    # Change mean rate homeostatic loss
    if meanr_vec is not None:
        untrained_pars.loss_pars.lambda_r_mean = 0.25
        untrained_pars.loss_pars.Rmean_E = meanr_vec[0]
        untrained_pars.loss_pars.Rmean_I = meanr_vec[1]

    # Set the offset to the offset threshold, where a given accuracy is achieved with the parameters from the last SGD step (loaded as offset_last)
    untrained_pars.stimuli_pars.offset = min(offset_last,10)

    # Run training
    training_output_df, _ = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            untrained_pars,
            results_filename=results_filename,
            jit_on=True,
            offset_step=0.1,
            run_index = i
        )
    
    # Save initial_parameters to csv
    initial_parameters.to_csv(os.path.join(final_folder_path, 'initial_parameters.csv'), index=False)
    
    i = i + 1
    print('runtime of {} pretraining + training run(s)'.format(i), time.time()-starting_time_in_main)
    print('number of failed runs = ', num_FailedRuns)
