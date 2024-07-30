# THIS CODE IS A MODIFIED VERSION OF MAIN.TRAINING TO OVERWRITE PARAMETER SETTINGS AND THE PERTURBATION FUNCTION FROM MORE RECENT SETTINGS TO THOSE SUBMITTED TO CCN CONFERENCE. THIS IS TO SUPPORT CCN CONFERENCE PRESENTATION WITH ABLATION STUDY
# This code runs pretraining for a general and training for a fine orientation discrimination task with a two-layer neural network model, where each layer is an SSN.

import numpy
import jax.numpy as np
import pandas as pd
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from util_gabor import init_untrained_pars
from util import save_code, load_parameters, filter_for_run
from training_functions import train_ori_discr
from perturb_params import create_initial_parameters_df
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
    pretraining_pars, # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
)

# Checking that pretrain_pars.is_on is on
if not pretraining_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to True in parameters.py to run training with pretraining!')

######### Overwrite parameters #########
#ssn_pars.p_local_s = [1.0, 1.0] # no horizontal connections in the superficial layer
if hasattr(trained_pars, 'J_2x2_s'):
    trained_pars.J_2x2_s = (np.array([[1.82650658, -0.68194475], [2.06815311, -0.5106321]]) * np.pi * 0.774) 
else:
    ssn_pars.J_2x2_s = (np.array([[1.82650658, -0.68194475], [2.06815311, -0.5106321]]) * np.pi * 0.774)
if hasattr(trained_pars, 'J_2x2_m'):
    trained_pars.J_2x2_m = np.array([[2.5, -1.3], [4.7, -2.2]]) * 0.774 
else:
    ssn_pars.J_2x2_m = np.array([[2.5, -1.3], [4.7, -2.2]]) * 0.774
pretraining_pars.ori_dist_int = [10, 20]
pretraining_pars.offset_threshold = [0,6]
pretraining_pars.SGD_steps = 500
pretraining_pars.min_acc_check_ind = 10
training_pars.eta = 2*10e-4
training_pars.first_stage_acc_th = 0.55
loss_pars.lambda_r_mean = 0
stimuli_pars.std = 200.0

######### Define the old perturbation functions #########
import copy
import jax.numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from util import take_log, create_grating_training, sep_exponentiate, create_grating_pretraining
from util_gabor import init_untrained_pars
from SSN_classes import SSN_mid, SSN_sup
from model import evaluate_model_response
from training.perturb_params import readout_pars_from_regr

def perturb_params_supp(param_dict, percent = 0.1):
    '''Perturb all values in a dictionary by a percentage of their values. The perturbation is done by adding a uniformly sampled random value to the original value.'''
    param_randomized = copy.deepcopy(param_dict)
    for key, param_array in param_dict.items():
        if type(param_array) == float:
            random_mtx = numpy.random.uniform(low=-1, high=1)
        else:
            random_mtx = numpy.random.uniform(low=-1, high=1, size=param_array.shape)
        param_randomized[key] = param_array + percent * param_array * random_mtx
    return param_randomized

def perturb_params(readout_pars, trained_pars, untrained_pars, percent=0.1, logistic_regr=True):
    #define the parameters that get perturbed
    # List of parameters to randomize
    parameter_name_list = ['J_2x2_m', 'J_2x2_s', 'c_E', 'c_I', 'f_E', 'f_I']

    # Dictionary to store the attribute values
    params_to_perturb = {}

    # Loop through each attribute and assign values
    for attr in parameter_name_list:
        if hasattr(trained_pars, attr):
            params_to_perturb[attr] = getattr(trained_pars, attr)
        else:
            params_to_perturb[attr] = getattr(untrained_pars.ssn_pars, attr)
    
    # Perturb parameters under conditions for J_mid and convergence of the differential equations of the model
    i=0
    cond1 = False
    cond2 = False
    cond3 = False
    cond4 = False
    cond5 = False

    while not (cond1 and cond2 and cond3 and cond4 and cond5):
        params_perturbed = perturb_params_supp(params_to_perturb, percent)
        
        cond1 = numpy.abs(params_perturbed['J_2x2_m'][0,0]*params_perturbed['J_2x2_m'][1,1])*1.1 < numpy.abs(params_perturbed['J_2x2_m'][1,0]*params_perturbed['J_2x2_m'][0,1])
        cond2 = numpy.abs(params_perturbed['J_2x2_m'][0,1]*untrained_pars.filter_pars.gI_m)*1.1 < numpy.abs(params_perturbed['J_2x2_m'][1,1]*untrained_pars.filter_pars.gE_m)
        # checking the convergence of the differential equations of the model
        ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=params_perturbed['J_2x2_m'])
        ssn_sup=SSN_sup(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=params_perturbed['J_2x2_s'], p_local=untrained_pars.ssn_pars.p_local_s,s_2x2=untrained_pars.ssn_pars.s_2x2_s, sigma_oris = untrained_pars.ssn_pars.sigma_oris, ori_dist = untrained_pars.ori_dist)
        train_data = create_grating_training(untrained_pars.stimuli_pars, batch_size=1, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp)
        [r_ref, _], _, [avg_dx_mid, avg_dx_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'], untrained_pars.conv_pars, params_perturbed['c_E'], params_perturbed['c_I'], params_perturbed['f_E'], params_perturbed['f_I'], untrained_pars.gabor_filters)
        cond3 = not numpy.any(numpy.isnan(r_ref))
        cond4 = avg_dx_mid + avg_dx_sup < 50
        cond5 = min([max_E_mid, max_I_mid, max_E_sup, max_I_sup])>10 and max([max_E_mid, max_I_mid, max_E_sup, max_I_sup])<101
        if i>50:
            print("Perturbed parameters violate inequality conditions or lead to divergence in diff equation.")
            break
        else:
            i = i+1

    params_perturbed_logged = dict(
        log_J_2x2_m= take_log(params_perturbed['J_2x2_m']),
        log_J_2x2_s= take_log(params_perturbed['J_2x2_s']),
        c_E=params_perturbed['c_E'],
        c_I=params_perturbed['c_I'],
        log_f_E=np.log(params_perturbed['f_E']),
        log_f_I=np.log(params_perturbed['f_I'])
    )
    
    if logistic_regr:
        pars_stage1 = readout_pars_from_regr(params_perturbed_logged, untrained_pars)
    else:
        pars_stage1 = dict(w_sig=readout_pars.w_sig, b_sig=readout_pars.b_sig)
    pars_stage1['w_sig'] = (pars_stage1['w_sig'] / np.std(pars_stage1['w_sig']) ) * 0.25 / int(np.sqrt(len(pars_stage1['w_sig']))) # get the same std as before - see param

    # If a parameter is untrained then save the perturbed value in the untrained class
    if hasattr(untrained_pars.ssn_pars, 'c_E'):
        untrained_pars.ssn_pars.c_E = params_perturbed_logged['c_E']
        untrained_pars.ssn_pars.c_I = params_perturbed_logged['c_I']

    if hasattr(untrained_pars.ssn_pars, 'f_E'):
        untrained_pars.ssn_pars.f_E = params_perturbed['f_E']
        untrained_pars.ssn_pars.f_I = params_perturbed['f_I']

    if hasattr(untrained_pars.ssn_pars, 'J_2x2_s'):
        untrained_pars.ssn_pars.J_2x2_s = params_perturbed['J_2x2_s']

    if hasattr(untrained_pars.ssn_pars, 'J_2x2_m'):
        untrained_pars.ssn_pars.J_2x2_m = params_perturbed['J_2x2_m']

    return pars_stage1, params_perturbed_logged, untrained_pars


########## Initialize orientation map and gabor filters ############

# Save out initial offset and reference orientation
ref_ori_saved = float(stimuli_pars.ref_ori)
offset_saved = float(stimuli_pars.offset)
num_training = 50
starting_time_in_main= time.time()
initial_parameters = None

# Save scripts into scripts folder
note=f'50 runs that repeat the Apr10_v1 but with slightly different stopping criteria and flipping conditions and saving settings, etc'
results_filename, final_folder_path = save_code(note=note)


# Run num_training number of pretraining + training
num_FailedRuns = 0
i=0
trained_pars_keys = trained_pars_dict = {attr: getattr(trained_pars, attr) for attr in dir(trained_pars)}
while i < num_training and num_FailedRuns < 20:
    numpy.random.seed(i+1)

    # Set pretraining flag to False
    pretraining_pars.is_on=True
    # Set offset and reference orientation to their initial values
    stimuli_pars.offset=offset_saved
    stimuli_pars.ref_ori=ref_ori_saved

    # Initialize untrained parameters (calculate gabor filters, orientation map related variables)
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, loss_pars, training_pars, pretraining_pars, readout_pars, i, folder_to_save=final_folder_path)

    ##### PRETRAINING: GENERAL ORIENTAION DISCRIMINATION #####

    # Randomize readout_pars and trained_pars and collect them into two dictionaries for the two stages of the pretraining
    # Note that orimap is regenerated if conditions do not hold!
    trained_pars_stage1, pretrained_pars_stage2, untrained_pars = perturb_params(readout_pars, pretrained_pars, untrained_pars, percent=0.1)
    initial_parameters = create_initial_parameters_df(initial_parameters, trained_pars_stage1, pretrained_pars_stage2, untrained_pars.training_pars.eta, untrained_pars.filter_pars.gE_m,untrained_pars.filter_pars.gI_m)
    
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
    trained_pars_stage1, trained_pars_stage2, untrained_pars, offset_last, meanr_vec = load_parameters(df_i, iloc_ind = pretraining_final_step, trained_pars_keys=trained_pars_keys, untrained_pars = untrained_pars)
    
    if meanr_vec is not None:
        untrained_pars.loss_pars.lambda_r_mean = 0.25
        untrained_pars.loss_pars.Rmean_E = meanr_vec[0]
        untrained_pars.loss_pars.Rmean_I = meanr_vec[1]

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
    # Save initial_parameters to csv
    initial_parameters.to_csv(os.path.join(final_folder_path, 'initial_parameters.csv'), index=False)
    
    i = i + 1
    print('runtime of {} pretraining + training run(s)'.format(i), time.time()-starting_time_in_main)
    print('number of failed runs = ', num_FailedRuns)

############### Run main analysis as well ###############
# Get the path to the sibling folder
sibling_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analysis')

# Get the path to the script.py file
script_path = os.path.join(sibling_folder, 'main_analysis.py')

with open(script_path) as file:
    exec(file.read())

