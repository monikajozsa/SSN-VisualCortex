import jax.numpy as np
import numpy
from scipy.stats import zscore
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import time
from scipy import ndimage
from scipy.stats import ttest_1samp
import os

from model import vmap_evaluate_model_response
from util import sep_exponentiate, load_parameters
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

def smooth_data(vector, sigma = 1):
    '''
    Smooth fixed point. Data is reshaped into 9x9 grid
    '''
    
    new_data = []
    for trial_response in vector:

        trial_response = trial_response.reshape(9,9,-1)
        smoothed_data = numpy.asarray([ndimage.gaussian_filter(numpy.reshape(trial_response[:, :, i], (9,9)), sigma = sigma) for i in range(0, trial_response.shape[2])]).ravel()
        new_data.append(smoothed_data)
    
    return np.vstack(np.asarray(new_data)) 

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


def filtered_model_response(file_name, untrained_pars, ori_list= np.asarray([55, 125, 0]), n_noisy_trials = 100, SGD_step_inds = None, sigma_filter = 1):

    if SGD_step_inds==None:
        df = pd.read_csv(file_name)
        train_start_ind = df.index[df['stage'] == 1][0]
        if numpy.min(df['stage'])==0:
            pretrain_start_ind = df.index[df['stage'] == 0][0]
            SGD_step_inds=[pretrain_start_ind, train_start_ind, -1]
        else:
            SGD_step_inds=[train_start_ind, -1]
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
def Mahalanobis_dist(num_trainings, folder, folder_to_save, file_name_to_save):
    ori_list = numpy.asarray([55, 125, 0])
    num_oris = len(ori_list)
    num_PC_used=15
    num_layers=2
    num_noisy_trials=100
    colors = ['black','blue', 'red']
    labels = ['pre-pretrained', 'post-pretrained','post-trained']
    start_time=time.time()
    mahal_train_control_mean=numpy.zeros((num_layers,3,num_trainings))
    mahal_untrain_control_mean=numpy.zeros((num_layers,3,num_trainings))
    for run_ind in range(num_trainings):
        file_name = f"{folder}/results_{run_ind}.csv"
        orimap_filename = f"{folder}/orimap_{run_ind}.npy"
        if not os.path.exists(file_name):
            file_name = f"{folder}/results_train_only{run_ind}.csv"
            orimap_filename = os.path.dirname(folder)+f'/orimap_{run_ind}.npy'
        
        loaded_orimap =  numpy.load(orimap_filename)
        untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                        loss_pars, training_pars, pretrain_pars, readout_pars, None, loaded_orimap)
        r_mid_sup, SGD_inds = filtered_model_response(file_name, untrained_pars, ori_list= ori_list, n_noisy_trials = num_noisy_trials)
        
        fig, axs = plt.subplots(2*num_layers, num_oris-1, figsize=(20, 30))  # Plot for Mahalanobis distances and SNR
        layer_labels = ['Sup', 'Mid']

        train_within_all = numpy.zeros(num_layers,len(SGD_inds), num_noisy_trials)
        untrain_within_all = numpy.zeros(num_layers,len(SGD_inds), num_noisy_trials)
        train_control_all = numpy.zeros(num_layers,len(SGD_inds), num_noisy_trials)
        untrain_control_all = numpy.zeros(num_layers,len(SGD_inds), num_noisy_trials)
        train_SNR_all=numpy.zeros(num_layers,len(SGD_inds), num_noisy_trials)
        untrain_SNR_all=numpy.zeros(num_layers,len(SGD_inds), num_noisy_trials)
        LMI_across = numpy.zeros(num_layers, num_noisy_trials)
        LMI_within = numpy.zeros(num_layers, num_noisy_trials)
        LMI_ratio = numpy.zeros(num_layers, num_noisy_trials)
        LMI_ttests = numpy.zeros(num_layers,3)
        LMI_p_vals = numpy.zeros(num_layers,3)
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

                ## TO BE TESTED Get explained variance
                #variance_explained = pca.explained_variance_ratio_
                #for i, var in enumerate(variance_explained):
                #    print(f"Variance explained by PC{i+1}: {var:.2%}")

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
                train_data_size = train_data.shape[0] # is it N_cells?
                mahal_within_train = numpy.zeros((num_noisy_trials-1,train_data_size))
                mahal_within_untrain = numpy.zeros((num_noisy_trials-1,train_data_size))
                
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
                    mahal_within_train[:,trial] = mahal(train_data_temp, numpy.repeat(train_data_trial_2d,num_noisy_trials-1, axis=0))
                    mahal_within_untrain[:,trial] = mahal(untrain_data_temp, numpy.repeat(untrain_data_trial_2d,num_noisy_trials-1, axis=0))
                    
                # Save distances and ratios
                train_within_all[layer,i,:] = mahal_within_train
                untrain_within_all[layer,i,:] = mahal_within_untrain
                train_control_all[layer,i,:] =mahal_train_control
                untrain_control_all[layer,i,:] =mahal_untrain_control
                train_SNR_all[layer,i,:] = mahal_train_control / mahal_within_train
                untrain_SNR_all[layer,i,:] = mahal_untrain_control / mahal_within_untrain
                
                '''
                # Calculate SNR
                train_SNR = numpy.mean(mahal_train_control / mahal_within_train)
                untrain_SNR = numpy.mean(mahal_untrain_control / mahal_within_untrain)

                
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
                
        fig.savefig(folder_to_save + '/' + file_name_to_save + f"_{run_ind}_control_{ori_list[2]}")
        '''
        # t-test on learning modulation index (LMI = [post-test ratio for trained orientation - pre-test ratio for trained orientation - post-test ratio for untrained orientation + pre-test ratio for untrained orientation]), 
        LMI_across[layer,:] = (train_control_all[layer,len(SGD_inds),:] - train_control_all[layer,0,:]) - (untrain_control_all[layer,len(SGD_inds),:] - untrain_control_all[layer,0,:])
        LMI_within[layer,:] = (train_within_all[layer,len(SGD_inds),:] - train_within_all[layer,0,:]) - (untrain_within_all[layer,len(SGD_inds),:] - untrain_within_all[layer,0,:])
        LMI_ratio[layer,:] = (train_SNR_all[layer,len(SGD_inds),:] - train_SNR_all[layer,0,:]) - (untrain_SNR_all[layer,len(SGD_inds),:] - untrain_SNR_all[layer,0,:])
        LMI_ttests[layer,0], LMI_p_vals[layer,0] = ttest_1samp(LMI_across[layer,:],0)
        LMI_ttests[layer,1], LMI_p_vals[layer,1] = ttest_1samp(LMI_within[layer,:],0)
        LMI_ttests[layer,2], LMI_p_vals[layer,2] = ttest_1samp(LMI_ratio[layer,:],0)
        
        ## TO BE TESTED For the ANOVA, we need to arrange the orientations (train_SNR_all and untrain_SNR_all) into a 2d numpy array, called data and then 
        # factor1_levels = np.repeat(['pre', 'post'], repeats=num_noisy_trials)
        # factor2_levels = np.tile(np.arange(1, 2), reps=num_noisy_trials)
        # indices = np.column_stack((factor1_levels, factor2_levels))
        # df = pd.DataFrame(data, index=indices)
        # df.index.names = ['pre_or_post', 'orientations']
        # df.reset_index(inplace=True)

        ## Perform repeated measures ANOVA
        ## Assuming 'subject' is the column containing the subject IDs
        ## 'dv' is the dependent variable
        ## 'within' specifies the repeated measures factors
        ## 'subject' specifies the subject ID column
        ## 'correction' specifies the correction method for violation of sphericity assumption
        ## 'effsize' specifies the effect size calculation method
        ## 'detailed' provides detailed ANOVA results
        # rm_anova = pg.rm_anova(dv='dependent_variable_column', within=['Factor1', 'Factor2'], subject='subject', data=df, correction=True, effsize="np2", detailed=True)

        ## Extract F-statistic and p-value
        #F_statistic = rm_anova['F'][0]  # Extract F-statistic (index 0 because it's the first factor)
        #p_value = rm_anova['p-unc'][0]  # Extract p-value (index 0 because it's the first factor)
    '''
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
    fig.savefig(f"{folder_to_save}/Mahal_dist_mean_control_{ori_list[2]}")
    '''
    return LMI_ttests 

    
