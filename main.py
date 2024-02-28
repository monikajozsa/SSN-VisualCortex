import numpy
import copy
import time
import jax.numpy as np
import os

numpy.random.seed(0)

from util_gabor import init_untrained_pars
from util import save_code
from training import train_ori_discr
from analysis import tuning_curves
from pretraining_supp import randomize_params, load_parameters
from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    readout_pars,
    ssn_pars,
    ssn_layer_pars,
    conv_pars,
    training_pars,
    loss_pars,
    pretrain_pars # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
)

# Checking that pretrain_pars.is_on is on
if not pretrain_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to True in parameters.py to run training with pretraining!')

########## Initialize orientation map and gabor filters ############

ref_ori_saved = float(stimuli_pars.ref_ori)
offset_saved = float(stimuli_pars.offset)

# Defining the number of random initializations for pretraining + training
N_training = 5

# Save scripts
folder_to_save='C:/Users/mj555/Dropbox (Cambridge University)/Postdoc 2023-2024/results'
results_filename, final_folder_path = save_code(folder_to_save)

starting_time_in_main= time.time()
numFailedRuns = 0
i=0
loaded_orimap =  np.load(os.path.join(os.getcwd(), 'ssn_map_uniform_good.npy'))
while i < N_training and numFailedRuns < 20:
    stimuli_pars.offset=offset_saved
    stimuli_pars.ref_ori=ref_ori_saved # this changes during training because of the staircase
    pretrain_pars.is_on=True

    results_filename = f"{final_folder_path}/results_{i}.csv"
    results_filename_train_only = f"{final_folder_path}/results_train_only{i}.csv"
    tuning_curves_prepre = f"{final_folder_path}/tc_prepre_{i}.csv"
    tuning_curves_postpre = f"{final_folder_path}/tc_postpre_{i}.csv"
    tuning_curves_post = f"{final_folder_path}/tc_post_{i}.csv"
    orimap_filename = f"{final_folder_path}/orimap_{i}.npy"

    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, orimap_filename, loaded_orimap)

    ##### PRETRAINING: GENERAL ORIENTAION DISCRIMINATION #####
    # Get baseline parameters to-be-trained
    ssn_layer_pars_pretrain = copy.deepcopy(ssn_layer_pars)
    readout_pars_pretrain = copy.deepcopy(readout_pars)

    # Perturb them by percent % and collect them into two dictionaries for the two stages of the pretraining
    # Note that orimap is regenerated if conditions do not hold
    trained_pars_stage1, trained_pars_stage2, untrained_pars = randomize_params(readout_pars_pretrain, ssn_layer_pars_pretrain, untrained_pars, percent=0.1, orimap_filename=orimap_filename)

    # Pretrain parameters
    training_output_df, first_stage_final_step = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            untrained_pars,
            results_filename=results_filename,
            jit_on=True
        )
    if training_output_df is None:
        print('######### Stopped run {} because of NaN values  - num failed runs = {} #########'.format(i, numFailedRuns))
        numFailedRuns = numFailedRuns + 1
        continue

    untrained_pars.pretrain_pars.is_on=False
    untrained_pars.first_stage_final_step = first_stage_final_step
    
    trained_pars_stage1, trained_pars_stage2, _ = load_parameters(results_filename, iloc_ind = numpy.min([10,training_pars.SGD_steps[0]]))
    responses_sup_prepre, responses_mid_prepre = tuning_curves(untrained_pars, trained_pars_stage2, tuning_curves_prepre)#pretty slow

    ##### FINE DISCRIMINATION #####
    
    trained_pars_stage1, trained_pars_stage2, offset_last = load_parameters(results_filename, iloc_ind = first_stage_final_step-1)
    responses_sup_postpre, responses_mid_postpre = tuning_curves(untrained_pars, trained_pars_stage2, tuning_curves_postpre)
    untrained_pars.stimuli_pars.offset = min(offset_last,10)

    training_output_df, _ = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            untrained_pars,
            results_filename=results_filename,
            jit_on=True
        )
    
    last_row = training_output_df.iloc[-1]
    J_m_keys = ['logJ_m_EE','logJ_m_EI','logJ_m_IE','logJ_m_II'] 
    J_s_keys = ['logJ_s_EE','logJ_s_EI','logJ_s_IE','logJ_s_II']
    J_m_values = last_row[J_m_keys].values.reshape(2, 2)
    J_s_values = last_row[J_s_keys].values.reshape(2, 2)

    pars_stage2 = dict(
        log_J_2x2_m = J_m_values,
        log_J_2x2_s = J_s_values,
        c_E=last_row['c_E'],
        c_I=last_row['c_I'],
        f_E=last_row['f_E'],
        f_I=last_row['f_I'],
    )
    responses_sup_post, responses_mid_post = tuning_curves(untrained_pars, pars_stage2, tuning_curves_post)
    
    # Running training only with the same initialization and orimap
    untrained_pars.stimuli_pars.offset=offset_saved
    untrained_pars.stimuli_pars.ref_ori=ref_ori_saved # this changes during training because of the staircase
    trained_pars_stage1, trained_pars_stage2, _ = load_parameters(results_filename, iloc_ind = 0)
    training_output_df, _ = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            untrained_pars,
            results_filename=results_filename_train_only,
            jit_on=True
        )

    i = i + 1
    print('runtime of {} pretraining + training run(s)'.format(i), time.time()-starting_time_in_main)
    print('number of failed runs = ', numFailedRuns)


######### PLOT RESULTS ############

from visualization import plot_results_from_csvs, barplots_from_csvs, plot_results_from_csv#
#final_folder_path= 'results/Feb28_v10'
#N_training=1
plot_results_from_csvs(final_folder_path, N_training)

boxplot_file_name = 'boxplot_pretraining'
barplots_from_csvs(final_folder_path, boxplot_file_name)
