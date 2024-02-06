import os
import numpy
import copy

numpy.random.seed(0)

from util_gabor import create_gabor_filters_util, BW_image_jax_supp
from util import cosdiff_ring
from training import train_ori_discr
from pretraining_supp import randomize_params
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
from visualization import plot_results_from_csv
if pretrain_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to False in parameters.py to run training without pretraining!')

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
    middle_grid_ind = readout_pars.middle_grid_ind
    pretrain_pars = pretrain_pars
    BW_image_jax_inp = BW_image_jax_supp(stimuli_pars)

constant_pars = ConstantPars()

# Run training without pretraining
results_filename='testing_new_version'

ssn_layer_pars_pretrain = copy.copy(ssn_layer_pars)
readout_pars_pretrain = copy.copy(readout_pars)
trained_pars_stage1, trained_pars_stage2 = randomize_params(readout_pars_pretrain, ssn_layer_pars_pretrain, constant_pars, percent=0.05)
training_output_df = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            constant_pars,
            results_filename=results_filename,
            jit_on=False
        )

results_fig_filename='testing_new_version_fig'
plot_results_from_csv(results_filename,results_fig_filename)
