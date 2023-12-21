import os
import numpy
import pandas as pd
import copy

numpy.random.seed(0)

from util_gabor import create_gabor_filters_util
from util import save_code, cosdiff_ring, cosdiff_acc_threshold
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

from pretraining_supp import randomize_params, load_pretrained_parameters

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
constant_pars.pretrain_pars.is_on=True
N_training=10

# Save scripts
results_filename, final_folder_path = save_code()

for i in range(N_training):
    results_filename
    ##### PRETRAINING: GENERAL ORIENTAION DISCRIMINATION #####
    # Get baseline parameters to-be-trained
    ssn_layer_pars_pretrain = copy.copy(ssn_layer_pars)
    readout_pars_pretrain = copy.copy(readout_pars)

    # Perturb them by percent % and collect them into two dictionaries for the two stages of the pretraining
    trained_pars_stage1, trained_pars_stage2 = randomize_params(readout_pars_pretrain, ssn_layer_pars_pretrain, percent=0.2)

    # Pretrain parameters
    training_output_df = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            constant_pars,
            results_filename,
            jit_on=False
        )
    constant_pars.pretrain_pars.is_on=True

    ##### FINE DISCRIMINATION #####
    readout_grid_size=5
    constant_pars.grid_pars.gridsize_Nx = readout_grid_size
    trained_pars_stage1, trained_pars_stage2 = load_pretrained_parameters(results_filename, readout_grid_size)

    training_output_df = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            constant_pars,
            results_filename,
            jit_on=False
        )
    constant_pars.pretrain_pars.is_on=True


######### PLOT RESULTS ############

df = pd.read_csv(results_filename)
fig_filename = os.path.join(final_folder_path,'results_pretraining')

visualization.plot_results_from_csv(results_filename,fig_filename)