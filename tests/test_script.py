# note that the test does not work if the pretraining_pars.is_on is set to True in parameters.py because of the 
import os
import numpy
import pandas as pd

numpy.random.seed(0)

from util_gabor import create_gabor_filters_util
from util import take_log, save_code, cosdiff_ring
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
# Overwrite parameters that usually change to baseline - check other parameters as a first step if error is large
training_pars.batch_size=50
training_pars.validation_freq=1
training_pars.SGD_steps=5
training_pars.first_stage_acc=0.7
training_pars.sig_noise = 2.0

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
constant_pars.pretrain_pars.is_on=False

trained_pars_stage1=dict(w_sig=readout_pars.w_sig, b_sig=readout_pars.b_sig)
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

training_output_df = train_ori_discr(
        trained_pars_stage1,
        trained_pars_stage2,
        constant_pars,
        jit_on=False
    )
acc_baseline=numpy.array([0,0.42,0.58,0.46,0.62,0.64,0.58,0.64,0.54,0.62,0.56])
acc_err=numpy.mean((training_output_df['acc']-acc_baseline)/numpy.sum(acc_baseline))
readout_loss_baseline=numpy.array([0,1.375562,1.0839813,1.208399,0.85099536,0.7467426,0.82136047,0.92496383,0.90044737,0.8392517,0.8012753])
loss_err=numpy.mean((training_output_df['loss_binary']-readout_loss_baseline)/numpy.sum(readout_loss_baseline))


print("Percentage of deviation from baseline training loss and training accuracy: {}, {}".format(acc_err, loss_err))

# If there is a difference : 1) check if parameters match 2) look for random number generation such as w_sig in parameters.py or train_data, and noise generation in training.py