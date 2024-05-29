# This file illustrates Gabors to help track what locations and orientations in the orimap match the indices of gabor filters

import numpy
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from training.util_gabor import init_untrained_pars
from util import save_code, load_parameters
from training.util_gabor import init_untrained_pars
from training.training_functions import train_ori_discr
from training.perturb_params import perturb_params, create_initial_parameters_df
from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    readout_pars,
    ssn_pars,
    trained_pars,
    conv_pars,
    training_pars,
    loss_pars,
    pretrain_pars # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
)

# Checking that pretrain_pars.is_on is on
if not pretrain_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to True in parameters.py to run training with pretraining!')

########## Initialize orientation map and gabor filters ############

# Save out initial offset and reference orientation
ref_ori_saved = float(stimuli_pars.ref_ori)
offset_saved = float(stimuli_pars.offset)

# Save scripts into scripts folder and create figures and train_only folders
train_only_flag = False # Setting train_only_flag to True will run an additional training without pretraining
perturb_level=0.1
note=f'Perturbation: {perturb_level}, J baseline: {trained_pars.J_2x2_m.ravel()}, '
results_filename, final_folder_path = save_code(train_only_flag=train_only_flag, note=note)

# Run num_training number of pretraining + training
initial_parameters = None

# Set pretraining flag to False
pretrain_pars.is_on=True
# Set offset and reference orientation to their initial values
stimuli_pars.offset=offset_saved
stimuli_pars.ref_ori=ref_ori_saved

# Initialize untrained parameters (calculate gabor filters, orientation map related variables)
untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars)
import jax.numpy as np
import matplotlib.pyplot as plt
gabor_filters = untrained_pars.gabor_filters
ssn_ori_map = untrained_pars.ssn_ori_map

fig, axs = plt.subplots(2,3)
axs[0,0].imshow(np.reshape(gabor_filters[0,:], (129,129)))
# title up to 2 decimal places
axs[0,0].set_title(f'G0, ori:{ssn_ori_map[0,0]:.1f}')
axs[0,1].imshow(np.reshape(gabor_filters[1,:], (129,129)))
axs[0,1].set_title(f'G1, ori:{ssn_ori_map[0,1]:.1f}')
axs[1,0].imshow(np.reshape(gabor_filters[2,:], (129,129)))
axs[1,0].set_title(f'G2, ori{ssn_ori_map[0,2]:.1f}')
axs[1,1].imshow(np.reshape(gabor_filters[3,:], (129,129)))
axs[1,1].set_title(f'G3, ori:{ssn_ori_map[0,3]:.1f}')
axs[0,2].imshow(np.reshape(gabor_filters[9,:], (129,129)))
axs[0,2].set_title(f'G9, ori:{ssn_ori_map[1,0]:.1f}')
axs[1,2].imshow(np.reshape(gabor_filters[80,:], (129,129)))
axs[1,2].set_title(f'G80, ori:{ssn_ori_map[9,9]:.1f}')
axs[0,0].axis('off')
axs[0,1].axis('off')
axs[1,0].axis('off')
axs[1,1].axis('off')
axs[0,2].axis('off')
axs[1,2].axis('off')
plt.savefig(f'Gabor_filters.png')
            