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
    #start_time = time.time()
    #gabors_old = create_gabor_filters_ori_map_old(orimap_loaded,num_phases,filter_pars,grid_pars)
    #print(time.time()-start_time)
    start_time = time.time()
    gabors = create_gabor_filters_ori_map(orimap_loaded,num_phases,filter_pars,grid_pars, flatten=False)
    print(time.time()-start_time)
    gabor_outputs = gabor_tuning(untrained_pars, ori_vec=tc_ori_list)

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
            axs[loc1,loc2].plot(gabor_outputs[:,loc1+9*loc2,i,0])
            # I cells
            axs[loc1,loc2].plot(gabor_outputs[:,loc1+9*loc2,i,1])
            axs[loc1,loc2].set_title(f'phase {phase_label[i]}, loc:{loc1,loc2}')
    plt.savefig(f'tests/gabor_outputs_phase_{i}_shift_each.png')
    plt.close()
print(time.time()-start_time)
# Gabor filter dimensions: (grid_size_2D, num_phases, 2, image_size)
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
    plt.savefig(f'tests/gabors_phase_{i}_shift_each.png')
    plt.close()
print(time.time()-start_time)
'''
# The gabor_tuning function used for plotting
def gabor_tuning(untrained_pars, ori_vec=np.arange(0,180,6)):
    gabor_filters = untrained_pars.gabor_filters
    num_ori = len(ori_vec)
    # Getting the 'blank' alpha_channel and mask for a full image version stimuli with no background
    BW_image_jax_inp = BW_image_jax_supp(stimuli_pars, x0 = 0, y0=0, phase=0.0, full_grating=True) 
    alpha_channel = BW_image_jax_inp[6]
    mask = BW_image_jax_inp[7]
    if len(gabor_filters.shape)==2:
        gabor_filters = np.reshape(gabor_filters, (untrained_pars.grid_pars.gridsize_Nx **2,untrained_pars.ssn_pars.phases,2,-1)) # the third dimension 2 is for I and E cells, the last dim is the image size
    
    # testing the matching of the gabor filters and the stimuli that gives max output
    plt.close()
    plt.clf()
    last_gabor_filter = gabor_filters[-1,-1,0,:]
    fig, axs = plt.subplots(3, 4, figsize=(5*10, 5*9))
    axs_flat = axs.flatten()
    for grid_ind in range(gabor_filters.shape[0]):
        for phase_ind in range(gabor_filters.shape[1]):
            i = grid_ind % untrained_pars.grid_pars.gridsize_Nx
            j = grid_ind // untrained_pars.grid_pars.gridsize_Nx
            x0 = untrained_pars.grid_pars.x_map[i,j]
            y0 = untrained_pars.grid_pars.y_map[i,j]
            BW_image_jax_inp = BW_image_jax_supp(stimuli_pars, x0=x0, y0=y0, phase=phase_ind * np.pi/2, full_grating=True)
            x = BW_image_jax_inp[4]
            y = BW_image_jax_inp[5]
            ori_from_gabor = untrained_pars.ssn_ori_map[i,j]+90
            gabor_filter = gabor_filters[grid_ind,phase_ind,0,:]
            gabor_test = 2*np.reshape(gabor_filter/(max(gabor_filter)-min(gabor_filter))+min(gabor_filter), (129,129))
            stimuli_max_output = BW_image_jax(BW_image_jax_inp[0:4], x, y, alpha_channel, mask, ref_ori=ori_from_gabor, jitter=0)
            if grid_ind is in [0,30,60]:
                col_ind = grid_ind//30
                axs_flat[phase_ind+4*col_ind].imshow(gabor_test + np.reshape(stimuli_max_output/(max(stimuli_max_output)-min(stimuli_max_output))+min(last_gabor_filter),(129,129)))
        # give a title to all the subplots
        plt.suptitle('Max output stimuli of Gabor filters')
        plt.savefig('tests/testing_max_output_phase.png')
    
    # Initialize the gabor output array
    gabor_output = numpy.zeros((num_ori, gabor_filters.shape[0],gabor_filters.shape[1],gabor_filters.shape[2]))
    time_start = time.time()
    for grid_ind in range(gabor_filters.shape[0]):
        grid_ind_x = grid_ind//untrained_pars.grid_pars.gridsize_Nx
        grid_ind_y = grid_ind%untrained_pars.grid_pars.gridsize_Nx
        x0 = untrained_pars.grid_pars.x_map[grid_ind_x, grid_ind_y]
        y0 = untrained_pars.grid_pars.y_map[grid_ind_x, grid_ind_y]
        for phase_ind in range(gabor_filters.shape[1]):
            phase = phase_ind * np.pi/2
            BW_image_jax_inp = BW_image_jax_supp(stimuli_pars, x0=x0, y0=y0, phase=phase, full_grating=True)
            x = BW_image_jax_inp[4]
            y = BW_image_jax_inp[5]
            stimuli = BW_image_jit(untrained_pars.BW_image_jax_inp[0:4], x, y, alpha_channel, mask, ori_vec, np.zeros(num_ori))
            for ori in range(num_ori):
                gabor_output[ori,grid_ind,phase_ind,0] = gabor_filters[grid_ind,phase_ind,0,:]@(stimuli[ori,:].T) # E cells
                gabor_output[ori,grid_ind,phase_ind,1] = gabor_filters[grid_ind,phase_ind,1,:]@(stimuli[ori,:].T) # I cells
    print('Time elapsed for gabor_output calculation:', time.time()-time_start)
    
    # Testing by visualizing the last Gabor filter and a few stimuli
    plt.close()
    plt.clf()
    last_gabor_filter = gabor_filters[-1,0,0,:]
    phase = 0
    BW_image_jax_inp = BW_image_jax_supp(stimuli_pars, x0=x0, y0=y0, phase=phase, full_grating=True)
    x = BW_image_jax_inp[4]
    y = BW_image_jax_inp[5]
    stimuli = BW_image_jit(untrained_pars.BW_image_jax_inp[0:4], x, y, alpha_channel, mask, ori_vec, np.zeros(num_ori))
    fig, axs = plt.subplots(3, 4, figsize=(5*10, 5*9))
    axs_flat = axs.flatten()
    gabor_test = 2*np.reshape(last_gabor_filter/(max(last_gabor_filter)-min(last_gabor_filter))+min(last_gabor_filter), (129,129))
    for i in range(10):
        ori = i*5
        stim_ori = np.reshape(stimuli[ori,:]/(max(stimuli[ori,:])-min(stimuli[ori,:]))+min(last_gabor_filter),(129,129))
        axs_flat[i].imshow(stim_ori + gabor_test)

    x0 = untrained_pars.grid_pars.x_map[-1, -1]
    y0 = untrained_pars.grid_pars.y_map[-1, -1]
    BW_image_jax_inp = BW_image_jax_supp(stimuli_pars, x0=x0, y0=y0, phase=0, full_grating=True)
    x = BW_image_jax_inp[4]
    y = BW_image_jax_inp[5]
    ori_from_gabor = -untrained_pars.ssn_ori_map[-1,-1]
    stimuli_max_output = BW_image_jax(BW_image_jax_inp[0:4], x, y, alpha_channel, mask, ori_from_gabor, 0)
    axs_flat[10].imshow(gabor_test + np.reshape(stimuli_max_output/(max(stimuli_max_output)-min(stimuli_max_output))+min(last_gabor_filter),(129,129)))
    axs_flat[10].set_title('Max output stimuli', fontsize=50)
    axs_flat[1].set_title('Example stimuli for the output', fontsize=50)
    axs_flat[11].plot(gabor_output[:,-1,0,0])
    plt.suptitle('Output of last Gabor with 0 phase', fontsize=70)
    plt.savefig('tests/FullImages_and_Gabors.png')
    
    return gabor_output
'''
