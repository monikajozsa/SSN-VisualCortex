import time
import jax.numpy as np
import numpy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
#import pingouin as pg

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from analysis_functions import filtered_model_response, mahal, LMI_Mahal_df


######### Calculate MVPA and Mahalanobis distance for before pretraining, after pretraining and after training #########
def MVPA_Mahal_analysis(folder,num_training, num_SGD_inds=2, r_noise = True, sigma_filter=1, plot_flag=False):
    # Shared parameters
    ori_list = numpy.asarray([55, 125, 0])
    num_layers=2 # number of layers
    num_noisy_trials=100

    ####### Setup for the MVPA analysis #######
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)) # SVM classifier

    # Initialize the MVPA scores matrix
    MVPA_scores = numpy.zeros((num_training, num_layers, num_SGD_inds, len(ori_list)-1))

    ####### Setup for the Mahalanobis distance analysis #######
    num_PC_used=15 # number of principal components used for the analysis
    
    # Initialize arrays to store Mahalanobis distances and related metrics
    LMI_across = numpy.zeros((num_training,num_layers,num_SGD_inds-1))
    LMI_within = numpy.zeros((num_training,num_layers,num_SGD_inds-1))
    LMI_ratio = numpy.zeros((num_training,num_layers,num_SGD_inds-1))
    
    mahal_within_train_all = numpy.zeros((num_training,num_layers,num_SGD_inds, num_noisy_trials))
    mahal_within_untrain_all = numpy.zeros((num_training,num_layers,num_SGD_inds, num_noisy_trials))
    mahal_train_control_all = numpy.zeros((num_training,num_layers,num_SGD_inds, num_noisy_trials))
    mahal_untrain_control_all = numpy.zeros((num_training,num_layers,num_SGD_inds, num_noisy_trials))
    train_SNR_all=numpy.zeros((num_training,num_layers,num_SGD_inds, num_noisy_trials))
    untrain_SNR_all=numpy.zeros((num_training,num_layers,num_SGD_inds, num_noisy_trials))

    mahal_train_control_mean = numpy.zeros((num_training,num_layers,num_SGD_inds))
    mahal_untrain_control_mean = numpy.zeros((num_training,num_layers,num_SGD_inds))
    mahal_within_train_mean = numpy.zeros((num_training,num_layers,num_SGD_inds))
    mahal_within_untrain_mean = numpy.zeros((num_training,num_layers,num_SGD_inds))
    train_SNR_mean = numpy.zeros((num_training,num_layers,num_SGD_inds))
    untrain_SNR_mean = numpy.zeros((num_training,num_layers,num_SGD_inds))
    
    # Define pca model
    pca = PCA(n_components=num_PC_used)
      
    # Iterate over the different parameter initializations (runs)
    for run_ind in range(num_training):
        start_time=time.time()
                
        # Calculate num_noisy_trials filtered model response for each oris in ori list and for each parameter set (that come from file_name at num_SGD_inds rows)
        r_mid_sup, SGD_step_inds = filtered_model_response(folder, run_ind, ori_list= ori_list, num_noisy_trials = num_noisy_trials, num_SGD_inds=num_SGD_inds,r_noise=r_noise, sigma_filter = sigma_filter, plot_flag=plot_flag)
        # Note: r_mid_sup is a dictionary with the oris and SGD_step_inds saved in them
        r_ori = r_mid_sup['ori']
        mesh_train_ = r_ori == ori_list[0] 
        mesh_untrain_ = r_ori == ori_list[1]
        mesh_control_ = r_ori == ori_list[2]
        # Iterate over the layers and SGD steps
        num_PCA_plots= 6
        if plot_flag and run_ind<num_PCA_plots:
            # make grid of plots for each layer and SGD step
            fig, axs = plt.subplots(num_layers, num_SGD_inds+1, figsize=(5*(num_SGD_inds+2), 5*num_layers))
        for layer in range(num_layers):
            if layer == 0:
                r_l = r_mid_sup['r_sup']
            else:
                r_l = r_mid_sup['r_mid']
            r_l = r_l.reshape((r_l.shape[0], -1))
            score = pca.fit_transform(r_l)
            variance_explained = pca.explained_variance_ratio_
            # Define the number of PCs to use for the current run (set it to min of 2 and otherwise, where the variance explained is above 80%)
            variance_explained_cumsum = numpy.cumsum(variance_explained)
            variance_explained_cumsum[-1]=1
            num_PC_used_run = numpy.argmax(variance_explained_cumsum > 0.7) + 1
            num_PC_used_run = max(num_PC_used_run, 2)         
            r_pca = score[:, :num_PC_used_run]
            print(f"Variance explained by {num_PC_used_run+1} PCs: {numpy.sum(variance_explained[0:num_PC_used_run+1]):.2%}")

            for SGD_ind in range(num_SGD_inds):                        
                # Define filter to select the responses corresponding to SGD_ind
                SGD_mask = r_mid_sup['SGD_step'] == SGD_step_inds[SGD_ind]
                 
                # Separate data into orientation conditions
                train_data = r_pca[mesh_train_ & SGD_mask,:]
                untrain_data = r_pca[mesh_untrain_ & SGD_mask,:]
                control_data = r_pca[mesh_control_ & SGD_mask,:]
                ############################# MVPA analysis #############################
                
                # MVPA for distinguishing trained orientation and control orientation
                # Combine the responses for the 0 and 2 oris (along axis-0)
                train_control_data = numpy.concatenate((train_data, control_data))
                train_control_data = train_control_data.reshape(train_control_data.shape[0],-1)
                train_control_label = numpy.concatenate((numpy.zeros(num_noisy_trials), numpy.ones(num_noisy_trials)))
                # make test-train split
                X_train, X_test, y_train, y_test = train_test_split(train_control_data, train_control_label, test_size=0.5, random_state=42)
                MVPA_scores[run_ind,layer,SGD_ind, 0] = clf.fit(X_train, y_train).score(X_test, y_test)
                
                # MVPA for distinguishing untrained orientation and control orientation
                # Combine the responses for the 1 and 2 oris
                untrain_control_data = numpy.concatenate((untrain_data, control_data))
                untrain_control_data = untrain_control_data.reshape(untrain_control_data.shape[0],-1)
                untrain_control_label = numpy.concatenate((numpy.zeros(num_noisy_trials), numpy.ones(num_noisy_trials)))
                              
                # fit the classifier for 10 randomly selected trial and test data and average the scores
                scores = []
                for i in range(10):
                    X_train, X_test, y_train, y_test = train_test_split(untrain_control_data, untrain_control_label, test_size=0.5, random_state=i)
                    score_i = clf.fit(X_train, y_train).score(X_test, y_test)
                    scores.append(score_i)
                MVPA_scores[run_ind,layer,SGD_ind, 1] = np.mean(np.array(scores))#clf.fit(X_train, y_train).score(X_test, y_test)

                ############################# Mahalanobis distance analysis #############################
                # Calculate Mahalanobis distance - mean and std of control data is calculated (along axis 0) and compared to the train and untrain data
                mahal_train_control = mahal(control_data, train_data)
                mahal_untrain_control = mahal(control_data, untrain_data)

                # Calculate the within group Mahal distances
                num_noisy_trials = train_data.shape[0] 
                mahal_within_train = numpy.zeros(num_noisy_trials)
                mahal_within_untrain = numpy.zeros(num_noisy_trials)
                                
                # Iterate over the trials to calculate the Mahal distances
                for trial in range(num_noisy_trials):
                    # Create temporary copies excluding one sample
                    mask = numpy.ones(num_noisy_trials, dtype=bool)
                    mask[trial] = False
                    train_data_temp = train_data[mask]
                    untrain_data_temp = untrain_data[mask]

                    # Calculate distances
                    train_data_trial_2d = numpy.expand_dims(train_data[trial], axis=0)
                    untrain_data_trial_2d = numpy.expand_dims(untrain_data[trial], axis=0)
                    mahal_within_train[trial] = mahal(train_data_temp, train_data_trial_2d)[0]
                    mahal_within_untrain[trial] = mahal(untrain_data_temp, untrain_data_trial_2d)[0]

                # PCA scatter plot the three conditions with different colors
                symbols = ['o', 's', '^']
                SGD_labels = ['prepre', 'pre', 'post']
                if plot_flag and run_ind < num_PCA_plots:
                    axs[layer,SGD_ind].scatter(control_data[:,0], control_data[:,1], label='control '+SGD_labels[SGD_ind], color='tab:green', s=5, marker=symbols[SGD_ind])
                    axs[layer,SGD_ind].scatter(train_data[:,0], train_data[:,1], label='trained '+SGD_labels[SGD_ind], color='blue', s=5, marker=symbols[SGD_ind])
                    axs[layer,SGD_ind].scatter(untrain_data[:,0], untrain_data[:,1], label='untrained '+SGD_labels[SGD_ind], color='red', s=5, marker=symbols[SGD_ind])
                    axs[layer,SGD_ind].set_title(f'Layer {layer}, run {run_ind}')
                    # Add lines between the mean of the conditions and write the Euclidean distance between them
                    mean_control = numpy.mean(control_data, axis=0)
                    mean_train = numpy.mean(train_data, axis=0)
                    mean_untrain = numpy.mean(untrain_data, axis=0)
                    axs[layer,SGD_ind].plot([mean_control[0], mean_train[0]], [mean_control[1], mean_train[1]], color='gray')
                    axs[layer,SGD_ind].plot([mean_control[0], mean_untrain[0]], [mean_control[1], mean_untrain[1]], color='gray')
                    axs[layer,SGD_ind].plot([mean_train[0], mean_untrain[0]], [mean_train[1], mean_untrain[1]], color='gray')
                    # add two lines of title, one with Eucledean distances and one with Mahalanobis distances
                    axs[layer,SGD_ind].set_title(f'train:{numpy.linalg.norm(mean_control-mean_train):.2f},untrain:{numpy.linalg.norm(mean_control-mean_untrain):.2f} \n train:{np.mean(mahal_train_control):.2f},untrain:{np.mean(mahal_untrain_control):.2f} within: {np.mean(mahal_within_train):.2f}, {np.mean(mahal_within_untrain):.2f}')
                    axs[layer,SGD_ind].legend()
                    
                # Save Mahal distances and ratios
                mahal_train_control_all[run_ind,layer,SGD_ind,:] = mahal_train_control
                mahal_untrain_control_all[run_ind,layer,SGD_ind,:] = mahal_untrain_control
                mahal_within_train_all[run_ind,layer,SGD_ind,:] = mahal_within_train
                mahal_within_untrain_all[run_ind,layer,SGD_ind,:] = mahal_within_untrain
                train_SNR_all[run_ind,layer,SGD_ind,:] = mahal_train_control / mahal_within_train
                untrain_SNR_all[run_ind,layer,SGD_ind,:] = mahal_untrain_control / mahal_within_untrain

                # Average over trials
                mahal_train_control_mean[run_ind,layer,SGD_ind] = numpy.mean(mahal_train_control)
                mahal_untrain_control_mean[run_ind,layer,SGD_ind] = numpy.mean(mahal_untrain_control)
                mahal_within_train_mean[run_ind,layer,SGD_ind] = numpy.mean(mahal_within_train)
                mahal_within_untrain_mean[run_ind,layer,SGD_ind] = numpy.mean(mahal_within_untrain)
                train_SNR_mean[run_ind,layer,SGD_ind] = numpy.mean(train_SNR_all[run_ind,layer,SGD_ind,:])
                untrain_SNR_mean[run_ind,layer,SGD_ind] = numpy.mean(untrain_SNR_all[run_ind,layer,SGD_ind,:])

            # Add Mahal distances as the last column of the plot as bar plots
            if plot_flag and run_ind < num_PCA_plots:
                axs[layer,num_SGD_inds].bar([1,2,3],mahal_train_control_mean[run_ind,layer,:], color='blue', alpha=0.5)
                axs[layer,num_SGD_inds].bar([5,6,7],mahal_untrain_control_mean[run_ind,layer,:], color='red', alpha=0.5)
                axs[layer,num_SGD_inds].set_xticks([1,2,3,5,6,7])
                axs[layer,num_SGD_inds].set_xticklabels(['tr0', 'tr1', 'tr2', 'ut0', 'ut1', 'ut2'])
                fig.savefig(folder + f"/figures/PCA_{run_ind}")
                plt.close()

            # Calculate learning modulation indices (LMI)
            for SGD_ind_ in range(num_SGD_inds-1):
                LMI_across[run_ind,layer,SGD_ind_] = (mahal_train_control_mean[run_ind,layer,SGD_ind_+1] - mahal_train_control_mean[run_ind,layer,SGD_ind_]) - (mahal_untrain_control_mean[run_ind,layer,SGD_ind_+1] - mahal_untrain_control_mean[run_ind,layer,SGD_ind_] )
                LMI_within[run_ind,layer,SGD_ind_] = (mahal_within_train_mean[run_ind,layer,SGD_ind_+1] - mahal_within_train_mean[run_ind,layer,SGD_ind_]) - (mahal_within_untrain_mean[run_ind,layer,SGD_ind_+1] - mahal_within_untrain_mean[run_ind,layer,SGD_ind_] )
                LMI_ratio[run_ind,layer,SGD_ind_] = (train_SNR_mean[run_ind,layer,SGD_ind_+1] - train_SNR_mean[run_ind,layer,SGD_ind_]) - (untrain_SNR_mean[run_ind,layer,SGD_ind_+1] - untrain_SNR_mean[run_ind,layer,SGD_ind_] )
        
        # Print the results for the current run
        print(MVPA_scores[run_ind,:,:,0], 'trained vs control')
        print(MVPA_scores[run_ind,:,:,1], 'untrained vs control')
        print([np.mean(mahal_train_control_all[run_ind,0,0,:]),np.mean(mahal_train_control_all[run_ind,0,1,:]),np.mean(mahal_train_control_all[run_ind,0,2,:])] ,'train')
        print([np.mean(mahal_untrain_control_all[run_ind,0,0,:]),np.mean(mahal_untrain_control_all[run_ind,0,1,:]),np.mean(mahal_untrain_control_all[run_ind,0,2,:])],'untrain')

        print(f'runtime of run {run_ind}:',time.time()-start_time)
        ################# Create dataframes #################
    # Create dataframes for the Mahalanobis distances and LMI
    df_mahal, df_LMI = LMI_Mahal_df(num_training, num_layers, num_SGD_inds, mahal_train_control_mean, mahal_untrain_control_mean, mahal_within_train_mean, mahal_within_untrain_mean, train_SNR_mean, untrain_SNR_mean, LMI_across, LMI_within, LMI_ratio)

    return MVPA_scores, df_mahal, df_LMI, mahal_train_control_all, mahal_untrain_control_all, mahal_within_train_all, mahal_within_untrain_all
    
def plot_Mahalanobis_dist(num_trainings, num_SGD_inds, mahal_train_control, mahal_untrain_control, mahal_within_train, mahal_within_untrain, folder_to_save, file_to_save):
    ori_list = numpy.asarray([55, 125, 0])
    num_oris = len(ori_list)
    num_layers=2
    colors = ['black','blue', 'red']
    labels = ['pre-pretrained', 'post-pretrained','post-trained']

    # Histogram plots (samples are per trial)
    mahal_SNR_train = mahal_train_control / mahal_within_train # dimensions are: run x layer x SGDstep x trial
    mahal_SNR_untrain = mahal_untrain_control / mahal_within_untrain

    for run_ind in range(num_trainings):
        fig, axs = plt.subplots(2*num_layers, num_oris-1, figsize=(20, 30))  # Plot for Mahalanobis distances and SNR
        layer_labels = ['Sup', 'Mid']
        for layer in range(num_layers):
            for SGD_ind in range(num_SGD_inds):
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
    # Boxplots - boxes within a subplot correspond to different orientations over different runs
    colors = ['black','tab:green']
    labels = [f'ori:{ori_list[0]}', f'ori:{ori_list[1]}']
    training_type_text = ['training']
    mahal_diff_train_l0_0=np.mean(mahal_train_control[:,0,1,:]-mahal_train_control[:,0,0,:], axis=1)/np.mean(mahal_train_control[:,0,1,:], axis=1)
    mahal_diff_untrain_l0_0=np.mean(mahal_untrain_control[:,0,1,:]-mahal_untrain_control[:,0,0,:], axis=1)/np.mean(mahal_untrain_control[:,0,1,:], axis=1)
    mahal_diff_l0=np.stack((mahal_diff_train_l0_0,mahal_diff_untrain_l0_0), axis=1)
    
    mahal_diff_train_l1_0=np.mean(mahal_train_control[:,1,1,:]-mahal_train_control[:,1,0,:], axis=1)/np.mean(mahal_train_control[:,1,1,:], axis=1)
    mahal_diff_untrain_l1_0=np.mean(mahal_untrain_control[:,1,1,:]-mahal_untrain_control[:,1,0,:], axis=1)/np.mean(mahal_untrain_control[:,1,1,:], axis=1)
    mahal_diff_l1=np.stack((mahal_diff_train_l1_0,mahal_diff_untrain_l1_0), axis=1)
    # define the matrices to plot for trainig when pretraining was present
    num_training_type=mahal_train_control.shape[2]-1
    if num_training_type==2:
        training_type_text = ['pretraining','training']
        mahal_diff_train_l0_1=np.mean(mahal_train_control[:,0,2,:]-mahal_train_control[:,0,1,:], axis=1)/np.mean(mahal_train_control[:,0,2,:], axis=1)
        mahal_diff_untrain_l0_1=np.mean(mahal_untrain_control[:,0,2,:]-mahal_untrain_control[:,0,1,:], axis=1)/np.mean(mahal_untrain_control[:,0,2,:], axis=1)
        mahal_diff_l0_traintype2=np.stack((mahal_diff_train_l0_1,mahal_diff_untrain_l0_1), axis=1)
        mahal_diff_l0=np.concatenate((mahal_diff_l0,mahal_diff_l0_traintype2),axis=1) 
        mahal_diff_train_l1_1=np.mean(mahal_train_control[:,1,2,:]-mahal_train_control[:,1,1,:], axis=1)/np.mean(mahal_train_control[:,1,2,:], axis=1)
        mahal_diff_untrain_l1_1=np.mean(mahal_untrain_control[:,1,2,:]-mahal_untrain_control[:,1,1,:], axis=1)/np.mean(mahal_untrain_control[:,1,2,:], axis=1)
        mahal_diff_l1_traintype2=np.stack((mahal_diff_train_l1_1,mahal_diff_untrain_l1_1), axis=1) 
        mahal_diff_l1=np.concatenate((mahal_diff_l1,mahal_diff_l1_traintype2),axis=1) # num_trainings x (55ori-pretrain, 125ori-pretrain, 55ori-train, 125ori-train)
    fig, axs = plt.subplots(num_training_type, num_layers, figsize=(num_training_type*5, num_layers*5))
    # within a subplot, the two boxplots are for the trained and untrained orientation
    for training_type in range(num_training_type):
        try:
            bp = axs[training_type,0].boxplot(mahal_diff_l0[:,training_type*2:training_type*2+2], labels=labels ,patch_artist=True)
        except:
            bp = axs[training_type,0].boxplot(mahal_diff_l0[:,training_type*2:training_type*2+2].T, labels=labels ,patch_artist=True)
        axs[training_type,0].set_title(f'{layer_labels[0]} layer, {training_type_text[training_type]}', fontsize=20)
        for box, color in zip(bp['boxes'], colors):
            box.set_facecolor(color)
        try:
            bp = axs[training_type,1].boxplot(mahal_diff_l1[:,training_type*2:training_type*2+2], labels=labels, patch_artist=True)
        except:
            bp = axs[training_type,1].boxplot(mahal_diff_l1[:,training_type*2:training_type*2+2].T, labels=labels, patch_artist=True)
        axs[training_type,1].set_title(f'{layer_labels[1]} layer', fontsize=20)
        for box, color in zip(bp['boxes'], colors):
            box.set_facecolor(color)
        
    fig.savefig(f"{folder_to_save}/Mahal_dist_diff")
    return 


def plot_Mahal_LMI_hists(df_LMI, df_mahal, folder, num_SGD_inds):
    num_layers=2
    LMI_across=numpy.array(df_LMI['LMI_across'].values.reshape(-1,num_layers,num_SGD_inds-1))
    LMI_within=numpy.array(df_LMI['LMI_within'].values.reshape(-1,num_layers,num_SGD_inds-1))
    LMI_ratio=numpy.array(df_LMI['LMI_ratio'].values.reshape(-1,num_layers,num_SGD_inds-1))
    mahal_train_control_mean = numpy.array(df_mahal['ori55_across'].values.reshape(-1,num_layers,num_SGD_inds))
    mahal_untrain_control_mean = numpy.array(df_mahal['ori125_across'].values.reshape(-1,num_layers,num_SGD_inds))
    mahal_within_train_mean = numpy.array(df_mahal['ori55_within'].values.reshape(-1,num_layers,num_SGD_inds))
    mahal_within_untrain_mean = numpy.array(df_mahal['ori125_within'].values.reshape(-1,num_layers,num_SGD_inds))
    
    fig, axs = plt.subplots(num_layers, 3, figsize=(30, 20))
    for layer in range(num_layers):
        for SGD_ind in range(num_SGD_inds-1):
            axs[layer,0].hist(LMI_across[:,layer,SGD_ind], label='across', color='blue', alpha=0.5*(SGD_ind+1)) #  fainter for the pretraining
            axs[layer,1].hist(LMI_within[:,layer,SGD_ind], label='within', color='red', alpha=0.5*(SGD_ind+1))
            axs[layer,2].hist(LMI_ratio[:,layer,SGD_ind], label='ratio', color='green', alpha=0.5*(SGD_ind+1))
            # add mean values as vertical lines
            mean_LMI_across = numpy.mean(LMI_across[:,layer,SGD_ind])
            mean_LMI_within = numpy.mean(LMI_within[:,layer,SGD_ind])
            mean_LMI_ratio = numpy.mean(LMI_ratio[:,layer,SGD_ind])
            axs[layer,0].axvline(mean_LMI_across, color='blue', linestyle='dashed', linewidth=0.6*(SGD_ind+1))
            axs[layer,1].axvline(mean_LMI_within, color='red', linestyle='dashed', linewidth=0.6*(SGD_ind+1))
            axs[layer,2].axvline(mean_LMI_ratio, color='green', linestyle='dashed', linewidth=0.6*(SGD_ind+1))
        axs[layer,0].set_title(f'LMI for layer {layer}, SGD step {SGD_ind}')
        axs[layer,0].legend()
        axs[layer,1].legend()
        axs[layer,2].legend()
        # Add black vertical lines at 0
        axs[layer,0].axvline(0, color='black', linestyle='dashed', linewidth=1)
        axs[layer,1].axvline(0, color='black', linestyle='dashed', linewidth=1)
        axs[layer,2].axvline(0, color='black', linestyle='dashed', linewidth=1)
    fig.savefig(folder + f"/figures/LMI_histograms")
    plt.close()

    # Plot histograms of the mahal_train_control_mean, mahal_untrain_control_mean, mahal_within_train_mean, mahal_within_untrain_mean across the runs
    fig, axs = plt.subplots(num_layers, 4, figsize=(40, 20))
    for layer in range(num_layers):
        for SGD_ind in range(num_SGD_inds-1):
            # plot histograms with contoured colors
            axs[layer,0].hist(mahal_train_control_mean[:,layer,SGD_ind], label='train-control', color='blue', alpha=0.33*(SGD_ind+1))
            axs[layer,1].hist(mahal_untrain_control_mean[:,layer,SGD_ind], label='untrain-control', color='red', alpha=0.33*(SGD_ind+1))
            axs[layer,2].hist(mahal_within_train_mean[:,layer,SGD_ind], label='within-train', color='green', alpha=0.33*(SGD_ind+1))
            axs[layer,3].hist(mahal_within_untrain_mean[:,layer,SGD_ind], label='within-untrain', color='purple', alpha=0.33*(SGD_ind+1))
            # add mean values as vertical lines
            mean_mahal_train_control = numpy.mean(mahal_train_control_mean[:,layer,SGD_ind])
            mean_mahal_untrain_control = numpy.mean(mahal_untrain_control_mean[:,layer,SGD_ind])
            mean_mahal_within_train = numpy.mean(mahal_within_train_mean[:,layer,SGD_ind])
            mean_mahal_within_untrain = numpy.mean(mahal_within_untrain_mean[:,layer,SGD_ind])
            axs[layer,0].axvline(mean_mahal_train_control, color='blue', linestyle='dashed', linewidth=0.6*(SGD_ind+1))
            axs[layer,1].axvline(mean_mahal_untrain_control, color='red', linestyle='dashed', linewidth=0.6*(SGD_ind+1))
            axs[layer,2].axvline(mean_mahal_within_train, color='green', linestyle='dashed', linewidth=0.6*(SGD_ind+1))
            axs[layer,3].axvline(mean_mahal_within_untrain, color='purple', linestyle='dashed', linewidth=0.6*(SGD_ind+1))
            
            axs[layer,0].set_title(f'train-control Mahal dist, layer {layer}')
            axs[layer,1].set_title(f'untrain-control Mahal dist, layer {layer}')
            axs[layer,2].set_title(f'within-train Mahal dist, layer {layer}')
            axs[layer,3].set_title(f'within-untrain Mahal dist, layer {layer}')
            axs[layer,0].legend()
    fig.savefig(folder + f"/figures/Mahal_histograms")
    plt.close()


def MVPA_Mahal_from_csv(folder, num_training, num_SGD_inds=2, sigma_filter=5, r_noise=True, plot_flag=False):
    ''' Calculate MVPA scores for before pretraining, after pretraining and after training - score should increase for trained ori more than for other two oris especially in superficial layer'''
    MVPA_scores, df_mahal, df_LMI, mahal_train_control, mahal_untrain_control, mahal_within_train, mahal_within_untrain = MVPA_Mahal_analysis(folder,num_training, num_SGD_inds, r_noise = r_noise, sigma_filter=sigma_filter, plot_flag=plot_flag)
    
    # save the output into folder_to_save as npy files
    folder_to_save = folder + f'/sigmafilt_{sigma_filter}'
    # create the folder if it does not exist
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
    numpy.save(folder_to_save +'/MVPA_scores.npy', MVPA_scores)    
    df_mahal.to_csv(folder_to_save + '/df_mahal.csv', index=False)
    df_LMI.to_csv(folder_to_save + '/df_LMI.csv', index=False)
    
    print('Pre-pre, pre and post training for 55~0, sup layer:',[np.mean(MVPA_scores[:,0,0,0]),np.mean(MVPA_scores[:,0,1,0]),np.mean(MVPA_scores[:,0,-1,0])])
    print('Pre-pre, pre and post training for 55~0, mid layer:',[np.mean(MVPA_scores[:,1,0,0]),np.mean(MVPA_scores[:,1,1,0]),np.mean(MVPA_scores[:,1,-1,0])])
    print('Pre-pre, pre and post training for 125~0, sup layer:',[np.mean(MVPA_scores[:,0,0,1]),np.mean(MVPA_scores[:,0,1,1]),np.mean(MVPA_scores[:,0,-1,1])])
    print('Pre-pre, pre and post training for 125~0, mid layer:',[np.mean(MVPA_scores[:,1,0,1]),np.mean(MVPA_scores[:,1,1,1]),np.mean(MVPA_scores[:,1,-1,1])])

    # Plot histograms of the LMI acorss the runs
    if plot_flag:
        plot_Mahal_LMI_hists(df_LMI, df_mahal, folder, num_SGD_inds)
        file_name = 'Mahal_dist'
        plot_Mahalanobis_dist(num_training, num_SGD_inds, mahal_train_control, mahal_untrain_control, mahal_within_train, mahal_within_untrain, folder_to_save, file_name)
    
    return MVPA_scores, df_mahal, df_LMI