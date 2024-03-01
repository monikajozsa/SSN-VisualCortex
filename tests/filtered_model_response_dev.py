import os
import jax.numpy as np
import pandas as pd

from pretraining_supp import load_parameters
from model import vmap_evaluate_model_response
from util import smooth_data, sep_exponentiate
from util_gabor import BW_image_jit_noisy
from SSN_classes import SSN_mid, SSN_sup

# DEVELOPMENTAL STAGE - NOT READY TO RUN

# load orimap - will be in the func from new_mahal.m 
# load parameters  [J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I] before and after training
# generate data with ref ori 55, 125, 0 (jitter = 0)
# pass it through the model and get [fp_mid, fp_sup] (both excitatory and inhibitory rates)
# add Gaussian smoothing smooth_data
# average filtered rates over trials so the final variables have sizes n_noisy_trials x n_mid_neurons and n_noisy_trials x n_sup_neurons

def filtered_model_response(file_name, untrained_pars, ori_list= np.asarray([55, 125, 0]), n_noisy_trials = 300, step_inds = np.asarray([1, -1]) ):
    #Initialise empty lists
    filtered_r_mid_all = []
    filtered_r_sup_all = []
    labels = []

    # Iterate overs SGD_step indices (default is before and after training)
    for step_ind in step_inds:
        filtered_r_mid_oris = []
        filtered_r_sup_oris = []
        
        #Load params from csv for given epoch
        trained_pars_stage1, trained_pars_stage2, _ = load_parameters(file_name, iloc_ind = step_ind)
        J_2x2_m = sep_exponentiate(trained_pars_stage2['log_J_2x2_m'])
        J_2x2_s = sep_exponentiate(trained_pars_stage2['log_J_2x2_s'])
        c_E = trained_pars_stage2['c_E']
        c_I = trained_pars_stage2['c_I']
        f_E = np.exp(trained_pars_stage2['f_E'])
        f_I = np.exp(trained_pars_stage2['f_I'])
        
        # Iterate over the orientations
        for ori in ori_list:

            #Select orientation from list
            untrained_pars.stimuli_pars.ref_ori = ori

            #Append orientation to label 
            labels.append(np.repeat(ori, n_noisy_trials))

            #Generate noisy data
            ori_vec = np.repeat(ori, n_noisy_trials)
            jitter_vec = np.repeat(0, n_noisy_trials)
            x = untrained_pars.BW_image_jax_inp[5]
            y = untrained_pars.BW_image_jax_inp[6]
            alpha_channel = untrained_pars.BW_image_jax_inp[7]
            mask = untrained_pars.BW_image_jax_inp[8]
            background = untrained_pars.BW_image_jax_inp[9]
            roi =untrained_pars.BW_image_jax_inp[10]
            
            # generate data
            test_grating = BW_image_jit_noisy(untrained_pars.BW_image_jax_inp[0:5], x, y, alpha_channel, mask, background, roi, ori_vec, jitter_vec)
            
            # Create middle and superficial SSN layers *** this is something that would be great to call from outside the SGD loop and only refresh the params that change (and what rely on them such as W)
            kappa_pre = untrained_pars.ssn_layer_pars.kappa_pre
            kappa_post = untrained_pars.ssn_layer_pars.kappa_post
            p_local_s = untrained_pars.ssn_layer_pars.p_local_s
            s_2x2 = untrained_pars.ssn_layer_pars.s_2x2_s
            sigma_oris = untrained_pars.ssn_layer_pars.sigma_oris
            ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_m)
            ssn_sup=SSN_sup(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_s, p_local=p_local_s, oris=untrained_pars.oris, s_2x2=s_2x2, sigma_oris = sigma_oris, ori_dist = untrained_pars.ori_dist, train_ori = ori, kappa_post = kappa_post, kappa_pre = kappa_pre)
    
            #Calculate fixed point for data    
            r_sup, r_mid, [r_max_mid, r_max_sup], [avg_dx_mid, avg_dx_sup], [max_E_mid, max_I_mid, max_E_sup, max_I_sup], [r_mid, r_sup] = vmap_evaluate_model_response(ssn_mid, ssn_sup, test_grating, untrained_pars.conv_pars, c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)

            #Smooth data with Gaussian filter
            filtered_r_mid_oris= smooth_data(r_mid, sigma = 1)      
            filtered_r_sup= smooth_data(r_sup, sigma = 1)  
            
            #Sum all contributions of E neurons (phases) and I neurons separately
            filtered_r_mid_oris = filtered_r_mid_oris.reshape(n_noisy_trials, 9,9, -1).sum(axis = 3)
            filtered_r_sup = filtered_r_sup.reshape(n_noisy_trials, 9,9, -1).sum(axis = 3)
            
            #Concatenate all orientation responses
            filtered_r_mid_oris.append(filtered_r_mid_oris.reshape(n_noisy_trials, -1))
            filtered_r_sup_oris.append(filtered_r_sup.reshape(n_noisy_trials, -1))
        
        #Concatenate all epoch responses
        filtered_r_mid_all.append(np.vstack(np.asarray(filtered_r_mid_oris)))
        filtered_r_sup_all.append(np.vstack(np.asarray(filtered_r_sup_oris)))

    # Save as DataFrame (helper columns: ori (3), SGD_ind (2), layer (2), type (2), grid_ind (25)) 24 x 29 csv file
    # *** this is incorrect and just sketches the idea of how to save filtered_r_mid_all 
    # I might do this in the double for loop like labels
    '''
    ori_df = []
    step_df = []
    layer_df = []
    type_df = []
    grid_ind_df = []

    for ori in ori_list:
        ori_df = ori_df.append(np.repeat(ori, len(step_inds)*4))
        for step_ind in step_inds:
            step_df.append(step_ind)
            for layer in range(2):
                layer_df.append(layer+1)
                for type in range(2):
                    type_df.append(type+1)
                    grid_ind_df.append(filtered_r_mid_all[:,:,])
    df = pd.DataFrame({
        'ori': ori_df,
        'SGD_step': step_df,
        'layer': layer_df,
        'type': type_df,
        'grid_ind': grid_ind_df,
    })
    '''

    return filtered_r_mid_all, filtered_r_sup_all
    

