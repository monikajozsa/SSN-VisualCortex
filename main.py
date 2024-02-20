import os
import numpy
import copy
import time

numpy.random.seed(0)

from util_gabor import create_gabor_filters_util, BW_image_jax_supp, make_orimap
from util import save_code, cosdiff_ring
from training import train_ori_discr
from analysis import tuning_curves
from pretraining_supp import randomize_params, load_pretrained_parameters
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
# from training import binary_task_acc_test

# Checking that pretrain_pars.is_on is on
if not pretrain_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to True in parameters.py to run training with pretraining!')

########## Initialize orientation map and gabor filters ############

ssn_ori_map = numpy.load(os.path.join(os.getcwd(), "ssn_map_uniform_good.npy"))
# X = grid_pars.x_map
# Y = grid_pars.y_map
# ssn_ori_map = make_orimap(X, Y, hyper_col=None, nn=30, deterministic=False)

gabor_filters, A, A2 = create_gabor_filters_util(ssn_ori_map, ssn_pars.phases, filter_pars, grid_pars, ssn_layer_pars.gE_m, ssn_layer_pars.gI_m)
ssn_pars.A = A
ssn_pars.A2 = A2

oris = ssn_ori_map.ravel()[:, None]
ori_dist = cosdiff_ring(oris - oris.T, 180)

####################### TRAINING PARAMETERS #######################
# Collect constant parameters into single class
class ConstantPars:
    grid_pars = grid_pars
    stimuli_pars = stimuli_pars
    filter_pars = filter_pars
    ssn_ori_map = ssn_ori_map
    oris = oris
    ori_dist = ori_dist
    ssn_pars = ssn_pars
    ssn_layer_pars = ssn_layer_pars
    conv_pars = conv_pars
    loss_pars = loss_pars
    training_pars = training_pars
    gabor_filters = gabor_filters
    readout_grid_size = readout_pars.readout_grid_size
    middle_grid_ind = readout_pars.middle_grid_ind
    pretrain_pars = pretrain_pars
    BW_image_jax_inp = BW_image_jax_supp(stimuli_pars)

constant_pars = ConstantPars()

ref_ori_saved = float(stimuli_pars.ref_ori)

# Defining the number of random initializations for pretraining + training
N_training = 2

# Save scripts
results_filename, final_folder_path = save_code()

starting_time_in_main= time.time()
numFailedRuns = 0
i=0
while i < N_training and numFailedRuns < 20:
    constant_pars.stimuli_pars.ref_ori=ref_ori_saved # this changes during training because of the staircase
    constant_pars.pretrain_pars.is_on=True

    results_filename = f"{final_folder_path}/results_{i}.csv"
    tuning_curves_prepre = f"{final_folder_path}/tc_prepre_{i}.csv"
    tuning_curves_prepost = f"{final_folder_path}/tc_prepost_{i}.csv"
    tuning_curves_post = f"{final_folder_path}/tc_post_{i}.csv"

    ##### PRETRAINING: GENERAL ORIENTAION DISCRIMINATION #####
    # Get baseline parameters to-be-trained
    ssn_layer_pars_pretrain = copy.deepcopy(ssn_layer_pars)
    readout_pars_pretrain = copy.deepcopy(readout_pars)

    # Perturb them by percent % and collect them into two dictionaries for the two stages of the pretraining
    trained_pars_stage1, trained_pars_stage2 = randomize_params(readout_pars_pretrain, ssn_layer_pars_pretrain, constant_pars, percent=0.1)

    # Pretrain parameters
    training_output_df, first_stage_final_step = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            constant_pars,
            results_filename=results_filename,
            jit_on=True
        )
    if training_output_df is None:
        print('######### Stopped run {} because of NaN values  - num failed runs = {} #########'.format(i, numFailedRuns))
        numFailedRuns = numFailedRuns + 1
        continue

    constant_pars.pretrain_pars.is_on=False
    constant_pars.first_stage_final_step = first_stage_final_step
    
    trained_pars_stage1, trained_pars_stage2 = load_pretrained_parameters(results_filename, iloc_ind = numpy.min([10,training_pars.SGD_steps[0]]))
    responses_sup_prepre, responses_mid_prepre = tuning_curves(constant_pars, trained_pars_stage2, tuning_curves_prepre)#pretty slow

    ##### FINE DISCRIMINATION #####
    
    trained_pars_stage1, trained_pars_stage2 = load_pretrained_parameters(results_filename, iloc_ind = first_stage_final_step-1)
    #testing val_loss_vec, val_acc_vec = binary_task_acc_test(training_pars, trained_pars_stage2, trained_pars_stage1, constant_pars, True, 4.0)
    responses_sup_prepost, responses_mid_prepost = tuning_curves(constant_pars, trained_pars_stage2, tuning_curves_prepost)

    training_output_df, _ = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            constant_pars,
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
    responses_sup_post, responses_mid_post = tuning_curves(constant_pars, pars_stage2, tuning_curves_post)
    i = i + 1
    print('runtime of one pretraining+training', time.time()-starting_time_in_main)
    print('number of failed runs = ', numFailedRuns)

######### PLOT RESULTS ############

from visualization import plot_results_from_csvs
#final_folder_path='results/Feb20_v0'
plot_results_from_csvs(final_folder_path, N_training)

from visualization import barplots_from_csvs

boxplot_file_name = 'boxplot_pretraining'
barplots_from_csvs(final_folder_path, boxplot_file_name)
