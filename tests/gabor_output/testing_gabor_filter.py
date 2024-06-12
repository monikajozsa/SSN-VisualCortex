# visualizing Gabor output

import pandas as pd
import numpy
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
numpy.random.seed(0)

from training.util_gabor import init_untrained_pars, create_gabor_filters_ori_map
from analysis.analysis_functions import tuning_curve, SGD_step_indices
from util import load_parameters
from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    readout_pars,
    ssn_pars,
    conv_pars,
    training_pars,
    loss_pars,
    pretrain_pars # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
)

# Checking that pretrain_pars.is_on is on
if not pretrain_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to True in parameters.py to run training with pretraining!')

import jax.numpy as np
import matplotlib.pyplot as plt

from analysis.analysis_functions import gabor_tuning

tc_ori_list = numpy.arange(0,180,2)
num_training = 10
final_folder_path = os.path.join('results','Apr10_v1')
########## Calculate and save gabor outputs ############
num_phases=4
for i in range(1):
    # Load orimap
    orimap_filename = os.path.join(final_folder_path, f"orimap_{i}.npy")
    orimap_loaded = numpy.load(orimap_filename)
    
    # Calculate gaboor filters
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, orimap_loaded= orimap_loaded)
    start_time = time.time()
    gabors = create_gabor_filters_ori_map(orimap_loaded,num_phases,filter_pars,grid_pars, flatten=False)
    gabor_outputs = gabor_tuning(untrained_pars, ori_vec=tc_ori_list)
    print(f'Gabor outputs calculated for run {i}',time.time()-start_time)

####### Plotting gabor outputs ########
# first 4x81 are for E phase 0, pi/2, pi, 3pi/2 next are the same for I cells
num_phases, num_loc = 4, 81

for i in range(num_phases):
    fig, axs = plt.subplots(9, 9, figsize=(5*9, 5*9))
    phases = numpy.array([0,1,2,3])
    phase_label = ['0', 'pi/2', 'pi', '3pi/2']
    for loc1 in range(9):
        for loc2 in range(9):
            # E cells
            axs[loc1,loc2].plot(gabor_outputs[:,i,0,loc1+9*loc2])
            # I cells
            axs[loc1,loc2].plot(gabor_outputs[:,i,1,loc1+9*loc2])
            axs[loc1,loc2].set_title(f'phase {phase_label[i]}, loc:{loc1,loc2}')
    plt.savefig(f'tests/gabor_output/gabor_outputs_phase_{i}.png')
    plt.close()
print(f'gabor_outputs_phase_{i} gnerated in time:',time.time()-start_time)
# Gabor filter dimensions: (grid_size_2D, num_phases, 2, image_size)
for i in range(num_phases):
    fig, axs = plt.subplots(9, 9, figsize=(5*9, 5*9))
    phases = numpy.array([0,1,2,3])
    phase_label = ['0', 'pi/2', 'pi', '3pi/2']
    for loc1 in range(9):
        for loc2 in range(9):# when loc 2 is 4, the phase seems to be consistent, otherwise, it is not!
            # E cells            
            axs[loc1,loc2].imshow(np.reshape(gabors[i,0,loc1+9*loc2,:],(129,129)))#, cmap='seismic')#, vmin=vmin_val, vmax=vmax_val)
            axs[loc1,loc2].set_title(f'phase {phase_label[i]}, loc:{loc1,loc2}')
    plt.savefig(f'tests/gabor_output/gabors_phase_{i}.png')
    plt.close()
    print(f'gabors_phase_{i} gnerated in time:',time.time()-start_time)