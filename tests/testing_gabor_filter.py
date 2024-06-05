# visualizing Gabor output

import pandas as pd
import numpy
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
numpy.random.seed(0)

from training.util_gabor import init_untrained_pars, create_gabor_filters_ori_map, create_gabor_filters_ori_map_old
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
    #start_time = time.time()
    #gabors_old = create_gabor_filters_ori_map_old(orimap_loaded,num_phases,filter_pars,grid_pars)
    #print(time.time()-start_time)
    start_time = time.time()
    gabors = create_gabor_filters_ori_map(orimap_loaded,num_phases,filter_pars,grid_pars)
    print(time.time()-start_time)
    gabor_outputs = gabor_tuning(untrained_pars, ori_vec=tc_ori_list)

####### Plotting gabor outputs ########
# first 4x81 are for E phase 0, pi/2, pi, 3pi/2 next are the same for I cells
num_phases, num_loc = 4, 81
'''
for i in range(num_phases):
    fig, axs = plt.subplots(9, 9, figsize=(5*9, 5*9))
    phases = numpy.array([0,1,2,3])
    phase_label = ['0', 'pi/2', 'pi', '3pi/2']
    for loc1 in range(9):
        for loc2 in range(9):
            # E cells
            axs[loc1,loc2].plot(gabor_outputs[:,loc1+9*loc2,i,0])
            # I cells
            axs[loc1,loc2].plot(gabor_outputs[:,loc1+9*loc2,i,1])
            axs[loc1,loc2].set_title(f'phase {phase_label[i]}, loc:{loc1,loc2}')
    plt.savefig(f'gabor_outputs_phase_{i}.png')
    plt.close()
'''
#(grid_size_2D, num_phases, 2, image_size)
for i in range(num_phases):
    fig, axs = plt.subplots(9, 9, figsize=(5*9, 5*9))
    phases = numpy.array([0,1,2,3])
    phase_label = ['0', 'pi/2', 'pi', '3pi/2']
    #vmax_val= np.max(np.abs(gabors[0:81,0,0,:]))
    #vmin_val= -vmax_val
    for loc1 in range(9):
        for loc2 in range(9):# when loc 2 is 4, the phase seems to be consistent, otherwise, it is not!
            # E cells            
            axs[loc1,loc2].imshow(np.reshape(gabors[loc1+9*loc2,i,0,:],(129,129)))#, cmap='seismic')#, vmin=vmin_val, vmax=vmax_val)
            axs[loc1,loc2].set_title(f'phase {phase_label[i]}, loc:{loc1,loc2}')
    plt.savefig(f'gabors_phase_{i}.png')
    plt.close()

'''
fig, axs = plt.subplots(8, 6, figsize=(5*3, 5*8))
for i in range(8):
    gabors_old_0=gabors_old[0]
    gabors_0=gabors[81*i,:]
    gabors_0=np.reshape(gabors_0,(129,129))
    gabors_old_0=np.reshape(gabors_old_0[81*i,:],(129,129))
    axs[i,0].imshow(gabors_old_0)
    axs[i,0].set_title('with old code')
    axs[i,1].imshow(gabors_0)
    axs[i,1].set_title('with new code')
    axs[i,2].imshow(gabors_0-gabors_old_0)
    axs[i,2].set_title(f'tot rel diff: {np.sum(np.abs(gabors_0-gabors_old_0))/np.sum(np.abs(gabors_old_0)):.5f}')
    gabors_old_21=gabors_old[0]
    gabors_21=gabors[81*i+21,:]
    gabors_21=np.reshape(gabors_21,(129,129))
    gabors_old_21=np.reshape(gabors_old_21[81*i+21,:],(129,129))
    axs[i,3].imshow(gabors_old_21)
    axs[i,3].set_title('with old code')
    axs[i,4].imshow(gabors_21)
    axs[i,4].set_title('with new code')
    axs[i,5].imshow(gabors_21-gabors_old_21)
    axs[i,5].set_title(f'tot rel diff: {np.sum(np.abs(gabors_21-gabors_old_21))/np.sum(np.abs(gabors_old_21)):.5f}')
plt.savefig('gabors_0_21.png')
'''