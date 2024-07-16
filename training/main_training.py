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
train_only_flag = False # Setting train_only_flag to True will run an additional training without pretraining
num_training = 10
starting_time_in_main= time.time()
initial_parameters = None

# Save scripts into scripts folder and create figures and train_only folders
note=f'10 runs with new pretraining stopping criterium: stop if train task accuracy is in [3.5,6]. New randomization on initial params and no r_mean loss term.'
results_filename, final_folder_path = save_code(train_only_flag=train_only_flag, note=note)
if train_only_flag:
    results_filename_train_only = os.path.join(final_folder_path, 'train_only', "results_train_only.csv")

# Run num_training number of pretraining + training
num_FailedRuns = 0
i=0
while i < num_training and num_FailedRuns < 20:
    numpy.random.seed(i+1)

    # Set pretraining flag to False
    pretrain_pars.is_on=True
    # Set offset and reference orientation to their initial values
    stimuli_pars.offset=offset_saved
    stimuli_pars.ref_ori=ref_ori_saved

    # Initialize untrained parameters (calculate gabor filters, orientation map related variables)
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, i, folder_to_save=final_folder_path)

    ##### PRETRAINING: GENERAL ORIENTAION DISCRIMINATION #####

    # Randomize readout_pars and trained_pars and collect them into two dictionaries for the two stages of the pretraining
    # Note that orimap is regenerated if conditions do not hold!
    trained_pars_stage1, trained_pars_stage2, untrained_pars = randomize_params(readout_pars, trained_pars, untrained_pars, randomize_pars=randomize_pars)
    initial_parameters = create_initial_parameters_df(initial_parameters, trained_pars_stage2, untrained_pars.training_pars.eta)
    
    # Run pre-training
    training_output_df, pretraining_final_step = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
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
    trained_pars_stage1, trained_pars_stage2, offset_last = load_parameters(df_i, iloc_ind = pretraining_final_step, trained_pars_keys=trained_pars_stage2.keys())
    # Set the offset to the offset, where a threshold accuracy is achieved with the parameters from the last SGD step (loaded as offset_last)
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
    
    ########## TRAINING ONLY with the same initialization and orimap ##########
    if train_only_flag:
        # Load the first parameters that pretraining started with
        trained_pars_stage1, trained_pars_stage2, _ = load_parameters(df_i, iloc_ind = 0)
        # Set the offset to the original offset that pretraining started with
        untrained_pars.stimuli_pars.offset = offset_saved
        # Set the reference orientation to the original one that pretraining started with
        untrained_pars.stimuli_pars.ref_ori = ref_ori_saved
        
        # Run training
        training_output_df, _ = train_ori_discr(
                trained_pars_stage1,
                trained_pars_stage2,
                untrained_pars,
                results_filename=results_filename_train_only,
                jit_on=True,
                run_index = i
            )
        
    # Save initial_parameters to csv
    initial_parameters.to_csv(os.path.join(final_folder_path, 'initial_parameters.csv'), index=False)
    
    i = i + 1
    print('runtime of {} pretraining + training run(s)'.format(i), time.time()-starting_time_in_main)
    print('number of failed runs = ', num_FailedRuns)
