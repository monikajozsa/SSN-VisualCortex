import os
import numpy
import pandas as pd
import copy
import time

numpy.random.seed(0)

from util_gabor import create_gabor_filters_util
from util import save_code, cosdiff_ring
from training import train_ori_discr
from analysis import tuning_curves

from parameters import pretrain_pars
# Setting pretraining to be true
pretrain_pars.is_on=True

from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    readout_pars,
    ssn_pars,
    ssn_layer_pars,
    conv_pars,
    training_pars,
    loss_pars
)
import visualization

from pretraining_supp import randomize_params, load_pretrained_parameters

########## Initialize orientation map and gabor filters ############

ssn_ori_map_loaded = numpy.load(os.path.join(os.getcwd(), "ssn_map_uniform_good.npy"))
gabor_filters, A, A2 = create_gabor_filters_util(ssn_ori_map_loaded, ssn_pars.phases, filter_pars, grid_pars, ssn_layer_pars.gE_m, ssn_layer_pars.gI_m)
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
    pretrain_pars = pretrain_pars

constant_pars = ConstantPars()

# Defining the number of random initializations for pretraining + training
N_training = 3

# Save scripts
results_filename, final_folder_path = save_code()

# Pretraining + training for N_training random initialization
for i in range(N_training):
    constant_pars.pretrain_pars.is_on=True
    results_filename = f"{final_folder_path}/results_{i}.csv"
    tuning_curves_prepre = f"{final_folder_path}/tc_prepre_{i}.csv"
    tuning_curves_prepost = f"{final_folder_path}/tc_prepost_{i}.csv"
    tuning_curves_post = f"{final_folder_path}/tc_post_{i}.csv"
    ##### PRETRAINING: GENERAL ORIENTAION DISCRIMINATION #####
    # Get baseline parameters to-be-trained
    ssn_layer_pars_pretrain = copy.copy(ssn_layer_pars)
    readout_pars_pretrain = copy.copy(readout_pars)

    # Perturb them by percent % and collect them into two dictionaries for the two stages of the pretraining
    trained_pars_stage1, trained_pars_stage2 = randomize_params(readout_pars_pretrain, ssn_layer_pars_pretrain, constant_pars, percent=0.1)

    # Pretrain parameters
    training_output_df = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            constant_pars,
            results_filename,
            jit_on=False
        )
    constant_pars.pretrain_pars.is_on=False
    
    trained_pars_stage1, trained_pars_stage2 = load_pretrained_parameters(results_filename, iloc_ind = numpy.min([10,training_pars.epochs[1]]))
    responses_sup_prepre, responses_mid_prepre = tuning_curves(constant_pars, trained_pars_stage2, tuning_curves_prepre)

    ##### FINE DISCRIMINATION #####
    
    trained_pars_stage1, trained_pars_stage2 = load_pretrained_parameters(results_filename)
    responses_sup_prepost, responses_mid_prepost = tuning_curves(constant_pars, trained_pars_stage2, tuning_curves_prepost)

    training_output_df = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            constant_pars,
            results_filename,
            jit_on=False
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

######### PLOT RESULTS ############
