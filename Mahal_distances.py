import jax.numpy as np
import numpy
from scipy.stats import zscore
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import time

from pretraining_supp import load_parameters
from model import vmap_evaluate_model_response
from util import smooth_data, sep_exponentiate
from util_gabor import BW_image_jit_noisy, init_untrained_pars
from SSN_classes import SSN_mid, SSN_sup
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

def mahal(X,Y):
    # Assuming X and Y are NumPy arrays provided earlier in your code
    rx, _ = X.shape
    ry, _ = Y.shape

    # Subtract the mean from X
    m = np.mean(X, axis=0)
    X_demean = X - np.tile(m, (rx, 1))

    # Create a matrix M by repeating m for ry rows
    M = np.tile(m, (ry, 1))
    Y_demean=(Y-M).T

    # Perform QR decomposition on C
    Q, R = np.linalg.qr(X_demean, mode='reduced')

    # Solve for ri in the equation R' * ri = (Y-M) using least squares or directly if R is square and of full rank
    ri = np.linalg.lstsq(R.T, Y_demean, rcond=None)[0]

    # Calculate d as the sum of squares of ri, scaled by (rx-1)
    d = np.sum(ri**2, axis=0) * (rx-1)

    return np.sqrt(d)


def filtered_model_response(file_name, untrained_pars, ori_list= np.asarray([55, 125, 0]), n_noisy_trials = 300, SGD_step_inds = None, sigma_filter = 1):

    if SGD_step_inds==None:
        df = pd.read_csv(file_name)
        pretrain_start_ind = df.index[df['stage'] == 0][0]
        train_start_ind = df.index[df['stage'] == 1][0]
        SGD_step_inds=[pretrain_start_ind,train_start_ind, -1]
    # Iterate overs SGD_step indices (default is before and after training)
    for step_ind in SGD_step_inds:

        # Load parameters from csv for given epoch
        trained_pars_stage1, trained_pars_stage2, _ = load_parameters(file_name, iloc_ind = step_ind)
        J_2x2_m = sep_exponentiate(trained_pars_stage2['log_J_2x2_m'])
        J_2x2_s = sep_exponentiate(trained_pars_stage2['log_J_2x2_s'])
        c_E = trained_pars_stage2['c_E']
        c_I = trained_pars_stage2['c_I']
        f_E = np.exp(trained_pars_stage2['f_E'])
        f_I = np.exp(trained_pars_stage2['f_I'])
        
        # Iterate over the orientations
        for ori in ori_list:

            # Select orientation from list
            untrained_pars.stimuli_pars.ref_ori = ori

            # Generate noisy data
            ori_vec = np.repeat(ori, n_noisy_trials)
            jitter_vec = np.repeat(0, n_noisy_trials)
            x = untrained_pars.BW_image_jax_inp[5]
            y = untrained_pars.BW_image_jax_inp[6]
            alpha_channel = untrained_pars.BW_image_jax_inp[7]
            mask = untrained_pars.BW_image_jax_inp[8]
            background = untrained_pars.BW_image_jax_inp[9]
            
            # Generate data
            test_grating = BW_image_jit_noisy(untrained_pars.BW_image_jax_inp[0:5], x, y, alpha_channel, mask, background, ori_vec, jitter_vec)
            
            # Create middle and superficial SSN layers *** this is something that would be great to call from outside the SGD loop and only refresh the params that change (and what rely on them such as W)
            kappa_pre = untrained_pars.ssn_layer_pars.kappa_pre
            kappa_post = untrained_pars.ssn_layer_pars.kappa_post
            p_local_s = untrained_pars.ssn_layer_pars.p_local_s
            s_2x2 = untrained_pars.ssn_layer_pars.s_2x2_s
            sigma_oris = untrained_pars.ssn_layer_pars.sigma_oris
            ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_m)
            ssn_sup=SSN_sup(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_s, p_local=p_local_s, oris=untrained_pars.oris, s_2x2=s_2x2, sigma_oris = sigma_oris, ori_dist = untrained_pars.ori_dist, train_ori = ori, kappa_post = kappa_post, kappa_pre = kappa_pre)
    
            # Calculate fixed point for data    
            r_sup, r_mid, [r_max_mid, r_max_sup], [avg_dx_mid, avg_dx_sup], [max_E_mid, max_I_mid, max_E_sup, max_I_sup], [r_mid, r_sup] = vmap_evaluate_model_response(ssn_mid, ssn_sup, test_grating, untrained_pars.conv_pars, c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)

            # Smooth data with Gaussian filter
            filtered_r_mid= smooth_data(r_mid, sigma = sigma_filter)  #n_noisy_trials x 648
            filtered_r_sup= smooth_data(r_sup, sigma = sigma_filter)  #n_noisy_trials x 162
            
            # Sum all contributions of E neurons (across phases) and I neurons separately
            filtered_r_mid = np.sum(np.reshape(filtered_r_mid, (n_noisy_trials, 9, 9, -1)), axis=3)
            filtered_r_sup = np.sum(np.reshape(filtered_r_sup, (n_noisy_trials, 9, 9, -1)), axis=3)
            
            # Concatenate all orientation responses
            if ori == ori_list[0] and step_ind==SGD_step_inds[0]:
                filtered_r_mid_df = np.reshape(filtered_r_mid, (n_noisy_trials, -1)) #n_noisy_trials x 81
                filtered_r_sup_df = np.reshape(filtered_r_sup, (n_noisy_trials, -1)) #n_noisy_trials x 81
                ori_df = np.repeat(ori, n_noisy_trials)
                step_df = np.repeat(step_ind, n_noisy_trials)
            else:
                filtered_r_mid_df = np.concatenate((filtered_r_mid_df, np.reshape(filtered_r_mid, (n_noisy_trials, -1))))
                filtered_r_sup_df = np.concatenate((filtered_r_sup_df, np.reshape(filtered_r_sup, (n_noisy_trials, -1))))
                ori_df = np.concatenate((ori_df, np.repeat(ori, n_noisy_trials)))
                step_df = np.concatenate((step_df, np.repeat(step_ind, n_noisy_trials)))
        
    output = dict(ori = ori_df, SGD_step = step_df, r_mid = filtered_r_mid_df, r_sup = filtered_r_sup_df )

    return output, SGD_step_inds

######### Calculate Mahalanobis distance for before pretraining, after pretraining and after training - distance between trained and control should increase more than distance between untrained and control after training #########
ori_list = numpy.asarray([55, 125, 0])
num_oris = len(ori_list)
num_PC_used=15
num_layers=2
n_noisy_trials=300
N_trainings=7
folder = 'results/Mar06_v6'
colors = ['black','blue', 'red']
labels = ['pre-pretrained', 'post-pretrained','post-trained']
start_time=time.time()
mahal_train_control_mean=numpy.zeros((num_layers,3,N_trainings))
mahal_untrain_control_mean=numpy.zeros((num_layers,3,N_trainings))
for run_ind in range(N_trainings):
    file_name = f"{folder}/results_{run_ind}.csv"
    orimap_filename = f"{folder}/orimap_{run_ind}.npy"
    loaded_orimap =  numpy.load(orimap_filename)
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                    loss_pars, training_pars, pretrain_pars, readout_pars, None, loaded_orimap)
    r_mid_sup, SGD_inds = filtered_model_response(file_name, untrained_pars, ori_list= ori_list, n_noisy_trials = n_noisy_trials)
    
    fig, axs = plt.subplots(2*num_layers, num_oris-1, figsize=(20, 30))  # Plot for Mahalanobis distances and SNR
    layer_labels = ['Sup', 'Mid']
    for layer in range(num_layers):
        for i in range(len(SGD_inds)):
            mesh_step = r_mid_sup['SGD_step'] == SGD_inds[i]
            # Specify layer (superficial/middle)!!!
            if layer==0:
                r = numpy.array(r_mid_sup['r_sup'][mesh_step])
            else:
                r = numpy.array(r_mid_sup['r_mid'][mesh_step])
            
            # Normalise data and apply PCA
            r_z = zscore(r, axis=0)
            pca = PCA(n_components=num_PC_used)
            score = pca.fit_transform(r_z)
            r_pca = score[:, :num_PC_used]

            # Separate data into orientation conditions
            mesh_train = r_mid_sup['ori'] == ori_list[0]
            mesh_train = mesh_train[mesh_step]
            mesh_untrain = r_mid_sup['ori'] == ori_list[1]
            mesh_untrain = mesh_untrain[mesh_step]
            mesh_control = r_mid_sup['ori'] == ori_list[2]
            mesh_control = mesh_control[mesh_step]

            train_data = r_pca[mesh_train,:]
            untrain_data = r_pca[mesh_untrain,:]    
            control_data = r_pca[mesh_control,:]

            # Calculate Mahalanobis distance
            mahal_train_control = mahal(control_data, train_data)    
            mahal_untrain_control = mahal(control_data, untrain_data)

            # Mean over the trials 
            mahal_train_control_mean[layer,i,run_ind] = numpy.mean(mahal_train_control)
            mahal_untrain_control_mean[layer,i,run_ind] = numpy.mean(mahal_untrain_control)

            # Calculate the standard deviation
            train_data_size = train_data.shape[0]
            mahal_within_train = numpy.zeros((n_noisy_trials-1,train_data_size))
            mahal_within_untrain = numpy.zeros((n_noisy_trials-1,train_data_size))
            
            # std of within group mahal distance
            for trial in range(train_data_size):
                # Create temporary copies excluding one sample
                mask = numpy.ones(train_data_size, dtype=bool)
                mask[trial] = False
                train_data_temp = train_data[mask]
                untrain_data_temp = untrain_data[mask]

                # Calculate distances
                train_data_trial_2d = numpy.expand_dims(train_data[trial], axis=0)
                untrain_data_trial_2d = numpy.expand_dims(untrain_data[trial], axis=0)
                mahal_within_train[:,trial] = mahal(train_data_temp, numpy.repeat(train_data_trial_2d,n_noisy_trials-1, axis=0))
                mahal_within_untrain[:,trial] = mahal(untrain_data_temp, numpy.repeat(untrain_data_trial_2d,n_noisy_trials-1, axis=0))
                
            # Calculate mean for each condition
            mahal_within_train_mean = numpy.mean(mahal_within_train)
            mahal_within_untrain_mean = numpy.mean(mahal_within_untrain)
            
            # Calculate SNR
            train_SNR = mahal_train_control / mahal_within_train_mean
            untrain_SNR = mahal_untrain_control / mahal_within_untrain_mean

            # Plotting Mahal distances for trained ori
            axs[layer,0].set_title(f'Mahalanobis dist: {layer_labels[layer]} layer, ori {ori_list[0]}')
            axs[layer,0].hist(mahal_train_control, label=labels[i], color=colors[i], alpha=0.4)  
            axs[layer,0].axvline(mahal_train_control_mean[layer,i,run_ind], color=colors[i], linestyle='dashed', linewidth=1)
            axs[layer,0].text(mahal_train_control_mean[layer,i,run_ind], axs[layer,0].get_ylim()[1]*0.95, f'{mahal_train_control_mean[layer,i,run_ind]:.2f}', color=colors[i], ha='center')
            axs[layer,0].legend(loc='lower left')
            # Plotting Mahal distances for untrained ori
            axs[layer,1].set_title(f'Mahal dist: {layer_labels[layer]} layer, ori {ori_list[1]}')
            axs[layer,1].hist(mahal_untrain_control, label=labels[i], color=colors[i], alpha=0.4)  
            axs[layer,1].axvline(mahal_untrain_control_mean[layer,i,run_ind], color=colors[i], linestyle='dashed', linewidth=1)
            axs[layer,1].text(mahal_untrain_control_mean[layer,i,run_ind], axs[layer,1].get_ylim()[1]*0.90, f'{mahal_untrain_control_mean[layer,i,run_ind]:.2f}', color=colors[i], ha='center')
            axs[layer,1].legend(loc='lower left')
            # Plotting SNR for trained ori
            train_SNR_mean=numpy.mean(train_SNR)
            axs[2+layer,0].set_title(f'SNR: layer {layer_labels[layer]}, ori {ori_list[0]}')
            axs[2+layer,0].hist(train_SNR, label=labels[i], color=colors[i], alpha=0.4)
            axs[2+layer,0].axvline(train_SNR_mean, color=colors[i], linestyle='dashed', linewidth=1)
            axs[2+layer,0].text(train_SNR_mean, axs[2+layer,0].get_ylim()[1]*0.95, f'{train_SNR_mean:.2f}', color=colors[i], ha='center')
            axs[2+layer,0].legend(loc='lower left')
            # Plotting SNR for untrained ori
            untrain_SNR_mean=numpy.mean(untrain_SNR)
            axs[2+layer,1].set_title(f'SNR: {layer_labels[layer]} layer, ori {ori_list[1]}')
            axs[2+layer,1].hist(untrain_SNR, label=labels[i], color=colors[i], alpha=0.4)
            axs[2+layer,1].axvline(untrain_SNR_mean, color=colors[i], linestyle='dashed', linewidth=1)
            axs[2+layer,1].text(untrain_SNR_mean, axs[2+layer,1].get_ylim()[1]*0.90, f'{untrain_SNR_mean:.2f}', color=colors[i], ha='center')
            axs[2+layer,1].legend(loc='lower left')
            print(time.time()-start_time)
    fig.savefig(f"{folder}/Mahal_dist_{run_ind}_control0")
fig, axs = plt.subplots(num_layers, num_oris-1, figsize=(10, 10))  # Plot for Mahalanobis distances and SNR
bp = axs[0,0].boxplot(mahal_train_control_mean[0,:,:].T, labels=labels, patch_artist=True)
axs[0,0].set_title(f'Mean Mahal dist: {layer_labels[0]} layer and {ori_list[0]} ori', fontsize=20)
for box, color in zip(bp['boxes'], colors):
    box.set_facecolor(color)
bp = axs[0,1].boxplot(mahal_untrain_control_mean[0,:,:].T, labels=labels, patch_artist=True)
axs[0,1].set_title(f'Mean Mahal dist: {layer_labels[0]} layer and {ori_list[1]} ori', fontsize=20)
for box, color in zip(bp['boxes'], colors):
    box.set_facecolor(color)
bp = axs[1,0].boxplot(mahal_train_control_mean[1,:,:].T, labels=labels, patch_artist=True)
axs[1,0].set_title(f'Mean Mahal dist: {layer_labels[1]} layer and {ori_list[0]} ori', fontsize=20)
for box, color in zip(bp['boxes'], colors):
    box.set_facecolor(color)
bp = axs[1,1].boxplot(mahal_untrain_control_mean[1,:,:].T, labels=labels, patch_artist=True)
axs[1,1].set_title(f'Mean Mahal dist: {layer_labels[1]}  layer and {ori_list[1]} ori', fontsize=20)
for box, color in zip(bp['boxes'], colors):
    box.set_facecolor(color)
fig.savefig(f"{folder}/Mahal_dist_mean_control0")
