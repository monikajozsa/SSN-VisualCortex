import jax.numpy as np
from jax import vmap
import numpy
from scipy.stats import zscore
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
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
    '''
    D2 = MAHAL(Y,X) returns the Mahalanobis distance (in squared units) of
    each observation (point) in Y from the sample data in X, i.e.,
    D2(I) = (Y(I,:)-MU) * SIGMA^(-1) * (Y(I,:)-MU)',
    where MU and SIGMA are the sample mean and covariance of the data in X.
    Rows of Y and X correspond to observations, and columns to variables.
    '''
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

#vmap_mahal = vmap(mahal, in_axes=[0,0])

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
        f_E = np.exp(trained_pars_stage2['log_f_E'])
        f_I = np.exp(trained_pars_stage2['log_f_I'])
        
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
def Mahalanobis_dist(num_trainings, folder, folder_to_save, file_to_save, SGD_inds=[0,-1]):
    ori_list = numpy.asarray([55, 125, 0])
    num_PC_used=15
    num_layers=2
    num_noisy_trials=100
    len_SGD_inds=len(SGD_inds)

    LMI_across = numpy.zeros((num_trainings,num_layers,len_SGD_inds-1))
    LMI_within = numpy.zeros((num_trainings,num_layers,len_SGD_inds-1))
    LMI_ratio = numpy.zeros((num_trainings,num_layers,len_SGD_inds-1))
    LMI_ttests = numpy.zeros((num_layers,3,len_SGD_inds-1))
    LMI_p_vals = numpy.zeros((num_layers,3,len_SGD_inds-1))

    mahal_within_train_all = numpy.zeros((num_trainings,num_layers,len_SGD_inds, num_noisy_trials))
    mahal_within_untrain_all = numpy.zeros((num_trainings,num_layers,len_SGD_inds, num_noisy_trials))
    mahal_train_control_all = numpy.zeros((num_trainings,num_layers,len_SGD_inds, num_noisy_trials))
    mahal_untrain_control_all = numpy.zeros((num_trainings,num_layers,len_SGD_inds, num_noisy_trials))
    train_SNR_all=numpy.zeros((num_trainings,num_layers,len_SGD_inds, num_noisy_trials))
    untrain_SNR_all=numpy.zeros((num_trainings,num_layers,len_SGD_inds, num_noisy_trials))
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
                
        for layer in range(num_layers):
            for SGD_ind in range(len_SGD_inds):
                mesh_step = r_mid_sup['SGD_step'] == SGD_inds[SGD_ind]
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

                ## Explained variance is above 99% typically
                #variance_explained = pca.explained_variance_ratio_
                #for i, var in enumerate(variance_explained):
                #    print(f"Variance explained by {i+1} PCs: {numpy.sum(variance_explained[0:i+1]):.2%}")

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

                # Calculate Mahalanobis distance - mean and std of control data is calculated (along axis 0) and compared to the train and untrain data
                mahal_train_control = mahal(control_data, train_data)    
                mahal_untrain_control = mahal(control_data, untrain_data)

                # Calculate the within group Mahal distances
                train_data_size = train_data.shape[0] # is it N_noisy_trials x 
                mahal_within_train = numpy.zeros(train_data_size)
                mahal_within_untrain = numpy.zeros(train_data_size)
                
                for trial in range(train_data_size):
                    # Create temporary copies excluding one sample
                    mask = numpy.ones(train_data_size, dtype=bool)
                    mask[trial] = False
                    train_data_temp = train_data[mask]
                    untrain_data_temp = untrain_data[mask]

                    # Calculate distances
                    train_data_trial_2d = numpy.expand_dims(train_data[trial], axis=0)
                    untrain_data_trial_2d = numpy.expand_dims(untrain_data[trial], axis=0)
                    mahal_within_train[trial] = numpy.mean(mahal(train_data_temp, numpy.repeat(train_data_trial_2d,num_noisy_trials-1, axis=0))) # averaging over the distances between one sample and the other samples ***
                    mahal_within_untrain[trial] = numpy.mean(mahal(untrain_data_temp, numpy.repeat(untrain_data_trial_2d,num_noisy_trials-1, axis=0))) # averaging over the distances between one sample and the other samples

                # Save distances and ratios
                mahal_train_control_all[run_ind,layer,SGD_ind,:] = mahal_train_control
                mahal_untrain_control_all[run_ind,layer,SGD_ind,:] = mahal_untrain_control
                mahal_within_train_all[run_ind,layer,SGD_ind,:] = mahal_within_train
                mahal_within_untrain_all[run_ind,layer,SGD_ind,:] = mahal_within_untrain
                train_SNR_all[run_ind,layer,SGD_ind,:] = mahal_train_control / mahal_within_train
                untrain_SNR_all[run_ind,layer,SGD_ind,:] = mahal_untrain_control / mahal_within_untrain
        
            # learning modulation indices (LMI)
            for SGD_ind in range(len_SGD_inds-1):
                mahal_pre_trained_ori = numpy.mean(mahal_train_control_all[run_ind,layer,SGD_ind,:])
                mahal_post_trained_ori = numpy.mean(mahal_train_control_all[run_ind,layer,SGD_ind+1,:])
                mahal_pre_untrained_ori = numpy.mean(mahal_untrain_control_all[run_ind,layer,SGD_ind,:])
                mahal_post_untrained_ori = numpy.mean(mahal_untrain_control_all[run_ind,layer,SGD_ind+1,:])
                LMI_across[run_ind,layer] = (mahal_post_trained_ori - mahal_pre_trained_ori) - (mahal_post_untrained_ori - mahal_pre_untrained_ori)

                mahal_within_pre_trained_ori = numpy.mean(mahal_within_train_all[run_ind,layer,SGD_ind,:])
                mahal_within_post_trained_ori = numpy.mean(mahal_within_train_all[run_ind,layer,SGD_ind+1,:])
                mahal_within_pre_untrained_ori = numpy.mean(mahal_within_untrain_all[run_ind,layer,SGD_ind,:])
                mahal_within_post_untrained_ori = numpy.mean(mahal_within_untrain_all[run_ind,layer,SGD_ind+1,:])
                LMI_within[run_ind,layer] = (mahal_within_post_trained_ori - mahal_within_pre_trained_ori) - (mahal_within_post_untrained_ori - mahal_within_pre_untrained_ori)

                mahal_SNR_pre_trained_ori = numpy.mean(train_SNR_all[run_ind,layer,SGD_ind,:])
                mahal_SNR_post_trained_ori = numpy.mean(train_SNR_all[run_ind,layer,SGD_ind+1,:])
                mahal_SNR_pre_untrained_ori = numpy.mean(untrain_SNR_all[run_ind,layer,SGD_ind,:])
                mahal_SNR_post_untrained_ori = numpy.mean(untrain_SNR_all[run_ind,layer,SGD_ind+1,:])
                LMI_ratio[run_ind,layer] = (mahal_SNR_post_trained_ori - mahal_SNR_pre_trained_ori) - (mahal_SNR_post_untrained_ori - mahal_SNR_pre_untrained_ori)

    # Apply t-test per layer (samples are the different runs)
    for layer in range(num_layers):
        LMI_ttests[layer,0], LMI_p_vals[layer,0] = ttest_1samp(LMI_across[:,layer],0) # compare it to mean 0
        LMI_ttests[layer,1], LMI_p_vals[layer,1] = ttest_1samp(LMI_within[:,layer],0)
        LMI_ttests[layer,2], LMI_p_vals[layer,2] = ttest_1samp(LMI_ratio[:,layer],0)
    
    ## Example usage of ANOVA
    ## I need to reorganize the data such that Factor1 (SGD_inds) and Factor2 (control, untrained, trained) are the first dim in the data (6) and samples are the second
    #data = np.random.randn(6, 100)  # Example 6x100 data matrix, where factor 1 has 2 values and factor 2 has 3 values
    ## Define factor levels
    #factor1_levels = np.repeat(['pre', 'post'], repeats=3)
    #factor2_levels = np.tile(np.arange(1, 4), reps=2)
    ## Add indices
    #indices = np.column_stack((factor1_levels, factor2_levels))                
    ## Create DataFrame with data and indices
    #df = pd.DataFrame(data, index=indices)
    # Name the index levels
    #df.index.names = ['Session', 'Ori']
    #rm_anova = pg.rm_anova(dv='dependent_variable_column', within=['Session', 'Ori'], subject='subject', data=df, correction=True, effsize="np2", detailed=True)
    ## ANOVA args:
    ## 'dv' is the dependent variable
    ## 'within' specifies the repeated measures factors
    ## 'subject' specifies the subject ID column
    ## 'correction' specifies the correction method for violation of sphericity assumption
    ## 'effsize' specifies the effect size calculation method
    ## 'detailed' provides detailed ANOVA results

    ## Extract F-statistic and p-value
    #F_statistic = rm_anova['F'][0]  # Extract F-statistic (index 0 because it's the first factor)
    #p_value = rm_anova['p-unc'][0]  # Extract p-value (index 0 because it's the first factor)
   
    return LMI_ttests, mahal_train_control_all, mahal_untrain_control_all, mahal_within_train_all, mahal_within_untrain_all, SGD_inds
    
def plot_Mahalanobis_dist(num_trainings, SGD_inds, mahal_train_control, mahal_untrain_control, mahal_within_train, mahal_within_untrain, folder_to_save, file_to_save):
    ori_list = numpy.asarray([55, 125, 0])
    num_oris = len(ori_list)
    num_layers=2
    colors = ['black','blue', 'red']
    labels = ['pre-pretrained', 'post-pretrained','post-trained']

    # Histogram plots (samples are per trial)
    mahal_SNR_train = mahal_train_control / mahal_within_train
    mahal_SNR_untrain = mahal_untrain_control / mahal_within_untrain
    for run_ind in range(num_trainings):
        fig, axs = plt.subplots(2*num_layers, num_oris-1, figsize=(20, 30))  # Plot for Mahalanobis distances and SNR
        layer_labels = ['Sup', 'Mid']
        for layer in range(num_layers):
            for SGD_ind in range(len(SGD_inds)):
                # Plotting Mahal distances for trained ori
                axs[layer,0].set_title(f'Mahalanobis dist: {layer_labels[layer]} layer, ori {ori_list[0]}')
                axs[layer,0].hist(mahal_train_control[run_ind,layer,SGD_ind,:], label=labels[SGD_ind], color=colors[SGD_ind], alpha=0.4) 
                mean_val=numpy.mean(mahal_train_control[run_ind,layer,SGD_ind,:])
                axs[layer,0].axvline(mean_val, color=colors[SGD_ind], linestyle='dashed', linewidth=1)
                axs[layer,0].text(mean_val, axs[layer,0].get_ylim()[1]*0.95, f'{mean_val:.2f}', color=colors[SGD_ind], ha='center')
                axs[layer,0].legend(loc='lower left')
                # Plotting Mahal distances for untrained ori
                axs[layer,1].set_title(f'Mahal dist: {layer_labels[layer]} layer, ori {ori_list[1]}')
                axs[layer,1].hist(mahal_untrain_control[run_ind,layer,SGD_ind,:], label=labels[SGD_ind], color=colors[SGD_ind], alpha=0.4)
                mean_val=numpy.mean(mahal_untrain_control[run_ind,layer,SGD_ind,:]) 
                axs[layer,1].axvline(mean_val, color=colors[SGD_ind], linestyle='dashed', linewidth=1)
                axs[layer,1].text(mean_val, axs[layer,0].get_ylim()[1]*0.95, f'{mean_val:.2f}', color=colors[SGD_ind], ha='center')
                axs[layer,1].legend(loc='lower left')
                # Plotting SNR for trained ori
                axs[2+layer,0].set_title(f'SNR: layer {layer_labels[layer]}, ori {ori_list[0]}')
                axs[2+layer,0].hist(mahal_SNR_train[run_ind,layer,SGD_ind,:], label=labels[SGD_ind], color=colors[SGD_ind], alpha=0.4)
                mean_val = numpy.mean(mahal_SNR_train[run_ind,layer,SGD_ind,:])
                axs[2+layer,0].axvline(mean_val, color=colors[SGD_ind], linestyle='dashed', linewidth=1)
                axs[2+layer,0].text(mean_val, axs[2+layer,0].get_ylim()[1]*0.95, f'{mean_val:.2f}', color=colors[SGD_ind], ha='center')
                axs[2+layer,0].legend(loc='lower left')
                # Plotting SNR for untrained ori
                axs[2+layer,1].set_title(f'SNR: {layer_labels[layer]} layer, ori {ori_list[1]}')
                axs[2+layer,1].hist(mahal_SNR_untrain[run_ind,layer,SGD_ind,:], label=labels[SGD_ind], color=colors[SGD_ind], alpha=0.4)
                mean_val = numpy.mean(mahal_SNR_untrain[run_ind,layer,SGD_ind,:])
                axs[2+layer,1].axvline(mean_val, color=colors[SGD_ind], linestyle='dashed', linewidth=1)
                axs[2+layer,1].text(mean_val, axs[2+layer,1].get_ylim()[1]*0.90, f'{mean_val:.2f}', color=colors[SGD_ind], ha='center')
                axs[2+layer,1].legend(loc='lower left')
                
        fig.savefig(folder_to_save + '/' + file_to_save + f"_{run_ind}_control_{ori_list[2]}")
    
    # Middle layer mahal boxplots - boxes correspond to different runs
    fig, axs = plt.subplots(num_layers, num_oris-1, figsize=(10, 10)) 
    bp = axs[0,0].boxplot(mahal_train_control[0,1,:,:]-mahal_train_control[0,0,:,:], labels=labels, patch_artist=True)
    axs[0,0].set_title(f'Mahal dist difference post-pre: {layer_labels[0]} layer and {ori_list[0]} ori', fontsize=20)
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
    bp = axs[0,1].boxplot(mahal_untrain_control[0,1,:,:]-mahal_untrain_control[0,0,:,:], labels=labels, patch_artist=True)
    axs[0,1].set_title(f'Mahal dist difference post-pre: {layer_labels[0]} layer and {ori_list[1]} ori', fontsize=20)
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
    # Superficial layer mahal boxplots - boxes correspond to different runs
    bp = axs[1,0].boxplot(mahal_train_control[1,1,:,:]-mahal_train_control[1,0,:,:], labels=labels, patch_artist=True)
    axs[1,0].set_title(f'Mahal dist difference post-pre: {layer_labels[1]} layer and {ori_list[0]} ori', fontsize=20)
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
    bp = axs[1,1].boxplot(mahal_untrain_control[1,1,:,:]-mahal_untrain_control[1,0,:,:], labels=labels, patch_artist=True)
    axs[1,1].set_title(f'Mahal dist difference post-pre: {layer_labels[1]}  layer and {ori_list[1]} ori', fontsize=20)
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
    fig.savefig(f"{folder_to_save}/Mahal_dist_mean_control_{ori_list[2]}")
    
    return 

num_training=3
final_folder_path = 'results/Mar22_v0'
folder_to_save = final_folder_path + '/figures'

mahal_train_control, mahal_untrain_control, mahal_within_train, mahal_within_untrain, SGD_inds = Mahalanobis_dist(num_training, final_folder_path,final_folder_path, 'mahal_dist')
plot_Mahalanobis_dist(num_training, SGD_inds, mahal_train_control, mahal_untrain_control, mahal_within_train, mahal_within_untrain, folder_to_save, 'mahal_dist')
