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

from pretraining_supp import randomize_params

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
# Save scripts
results_filename, final_folder_path = save_code()

# Pretraining + training for N_training random initialization
trained_pars_stage1, trained_pars_stage2 = randomize_params(readout_pars, ssn_layer_pars, constant_pars, percent=0.0)

training_output_df = train_ori_discr(
        trained_pars_stage1,
        trained_pars_stage2,
        constant_pars,
        results_filename,
        jit_on=False
    )
