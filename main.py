import os
import numpy
import pandas as pd

numpy.random.seed(0)

from util_gabor import create_gabor_filters_util
from util import take_log, save_code, cosdiff_ring, cosdiff_acc_threshold
from training import train_ori_discr
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
    pretrain_pars
)
import visualization

#from pretraining_supp import randomize_params
#randomize_params(ssn_layer_pars, stimuli_pars, percent=0.2)

########## Initialize orientation map and gabor filters ############

ssn_ori_map_loaded = numpy.load(os.path.join(os.getcwd(), "ssn_map_uniform_good.npy"))
gabor_filters = create_gabor_filters_util(ssn_ori_map_loaded, ssn_pars.phases, filter_pars, grid_pars, ssn_layer_pars.gE_m, ssn_layer_pars.gI_m)

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

# Collect trained parameters into two dictionaries for the two stages
trained_pars_stage1 = dict(w_sig=readout_pars.w_sig, b_sig=readout_pars.b_sig)

trained_pars_stage2 = dict(
    log_J_2x2_m= take_log(ssn_layer_pars.J_2x2_m),
    log_J_2x2_s= take_log(ssn_layer_pars.J_2x2_s),
    c_E=ssn_layer_pars.c_E,
    c_I=ssn_layer_pars.c_I,
    f_E=ssn_layer_pars.f_E,
    f_I=ssn_layer_pars.f_I,
    kappa_pre=ssn_layer_pars.kappa_pre,
    kappa_post=ssn_layer_pars.kappa_post,
)

####################### TRAINING #######################

results_filename, final_folder_path = save_code()
training_output_df = train_ori_discr(
        trained_pars_stage1,
        trained_pars_stage2,
        constant_pars,
        results_filename,
        jit_on=False
    )

######### PLOT RESULTS ############

df = pd.read_csv(results_filename)
fig_filename = os.path.join(final_folder_path,'results_pretraining')

visualization.plot_results_from_csv(results_filename,fig_filename)