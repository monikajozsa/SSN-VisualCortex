import os
import numpy
import copy
import time
numpy.random.seed(0)

from util_gabor import create_gabor_filters_ori_map, BW_image_jax_supp
from util import cosdiff_ring, save_code
from training import train_ori_discr
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
from analysis import tuning_curves

# Checking that pretrain_pars.is_on is off
if pretrain_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to False in parameters.py to run training without pretraining!')

#results_filename, final_folder_path = save_code()
########## Initialize orientation map and gabor filters ############

ssn_ori_map_loaded = numpy.load(os.path.join(os.getcwd(), "ssn_map_uniform_good.npy"))
gabor_filters, A, A2 = create_gabor_filters_ori_map(ssn_ori_map_loaded, ssn_pars.phases, filter_pars, grid_pars, ssn_layer_pars.gE_m, ssn_layer_pars.gI_m)
ssn_pars.A = A
ssn_pars.A2 = A2

oris = ssn_ori_map_loaded.ravel()[:, None]
ori_dist = cosdiff_ring(oris - oris.T, 180)

####################### TRAINING PARAMETERS #######################
# Collect constant parameters into single class
class ConstantPars:
    grid_pars = grid_pars
    stimuli_pars = stimuli_pars
    filter_pars = filter_pars
    ssn_ori_map = ssn_ori_map_loaded
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
'''
# Run training without pretraining

starting_time_in_main= time.time()
numFailedRuns = 0
N_training = 5
for i in range(N_training):
    if numFailedRuns<10:
        results_filename = f"{final_folder_path}/results_{i}.csv"
        
        trained_pars_stage1, trained_pars_stage2 = randomize_params(readout_pars, ssn_layer_pars, constant_pars, percent=0.1)
        training_output_df = train_ori_discr(
                    trained_pars_stage1,
                    trained_pars_stage2,
                    constant_pars,
                    results_filename=results_filename,
                    jit_on=True
                )
        if training_output_df is None:
            print('######### Stopped run {} because of NaN values #########'.format(i))
            i = i-1
            numFailedRuns = numFailedRuns+1
            continue
        
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

for i in range(N_training):
    results_filename = f"{final_folder_path}/results_{i}.csv"
    tuning_curves_pre = f"{final_folder_path}/tc_prepre_{i}.csv"
    tuning_curves_post = f"{final_folder_path}/tc_post_{i}.csv"

    trained_pars_stage1, trained_pars_stage2 = load_pretrained_parameters(results_filename, iloc_ind = 10)
    responses_sup_prepre, responses_mid_prepre = tuning_curves(constant_pars, trained_pars_stage2, tuning_curves_pre)#pretty slow
    trained_pars_stage1, trained_pars_stage2 = load_pretrained_parameters(results_filename, iloc_ind = -1)
    responses_sup_post, responses_mid_post = tuning_curves(constant_pars, trained_pars_stage2, tuning_curves_post)
        
from visualization import plot_results_from_csvs
plot_results_from_csvs(final_folder_path, N_training)

'''
from visualization import barplots_from_csvs

directory = 'results/Feb07_v6'

boxplot_file_name = 'boxplot_training'
barplots_from_csvs(directory, boxplot_file_name)
