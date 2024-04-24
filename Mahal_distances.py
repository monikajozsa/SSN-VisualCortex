import time
import jax.numpy as np
import numpy
from scipy.stats import zscore
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
#import pingouin as pg
from scipy import ndimage
from scipy.stats import ttest_1samp
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

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

def select_type_mid(r_mid, cell_type='E'):
    '''Selects the excitatory or inhibitory cell responses. This function assumes that r_mid is 3D (trials x grid points x celltype and phase)'''
    if cell_type=='E':
        map_numbers = np.arange(1, 2 * ssn_pars.phases, 2)-1 # 0,2,4,6
    else:
        map_numbers = np.arange(2, 2 * ssn_pars.phases + 1, 2)-1 # 1,3,5,7
    
    out = numpy.zeros((r_mid.shape[0],r_mid.shape[1], int(r_mid.shape[2]/2)))
    for i in range(len(map_numbers)):
        out[:,:,i] = r_mid[:,:,map_numbers[i]]

    return np.array(out)

def smooth_data(X, gridsize_Nx =9, sigma = 1):
    '''
    Smooth data matrix (trials x cells) over grid points. Data is reshaped into 9x9 grid before smoothing and then flattened again.
    '''
    N_grid_points=gridsize_Nx*gridsize_Nx
    N_phases = int(X.shape[1]/N_grid_points)
    N_trials = X.shape[0]
    smoothed_data= numpy.zeros((N_trials,N_grid_points,N_phases))
    for trial in range(N_trials):
        for phase in range(N_phases):
            trial_response = X[trial,phase*N_grid_points:(phase+1)*N_grid_points]
            trial_response = trial_response.reshape(gridsize_Nx,gridsize_Nx,-1)
            smoothed_data[trial,:, phase] =numpy.asarray(ndimage.gaussian_filter(trial_response, sigma = sigma)).ravel()

    return smoothed_data

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

    # Perform QR decomposition on C
    Q, R = np.linalg.qr(X_demean, mode='reduced')
    
    #if ry==1:
    #    cov_X_approx = np.dot(R.T, R) / (X.shape[0] - 1)
    #    d = (Y-m) @ numpy.linalg.inv(cov_X_approx) @ np.transpose(Y-m)
    #else:
    
    # Solve for ri in the equation R' * ri = (Y-M) using least squares or directly if R is square and of full rank
    Y_demean=(Y-M).T
    ri = np.linalg.lstsq(R.T, Y_demean, rcond=None)[0]

    # Calculate d as the sum of squares of ri, scaled by (rx-1)
    d = np.sum(ri**2, axis=0) * (rx-1)
    
    return np.sqrt(d)


def filtered_model_response(file_name, untrained_pars, ori_list= np.asarray([55, 125, 0]), n_noisy_trials = 100, num_SGD_inds = 2, sigma_filter = 1):
    df = pd.read_csv(file_name)
    train_start_ind = df.index[df['stage'] == 1][0]
    if num_SGD_inds==3:        
        if numpy.min(df['stage'])==0:
            pretrain_start_ind = df.index[df['stage'] == 0][0]
            SGD_step_inds=[pretrain_start_ind, train_start_ind, -1]
        else:
            print('Warning: There is no 0 stage but Mahal dist was asked to be calculated for pretraining!')
            SGD_step_inds=[train_start_ind, -1]
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
            _, _, [_, _], [_, _], [_, _, _, _], [r_mid, r_sup] = vmap_evaluate_model_response(ssn_mid, ssn_sup, test_grating, untrained_pars.conv_pars, c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)

            # Smooth data for each celltype separately with Gaussian filter
            filtered_r_mid_EI= smooth_data(r_mid)  #n_noisy_trials x 648
            filtered_r_mid_E=select_type_mid(filtered_r_mid_EI,'E')
            filtered_r_mid_I=select_type_mid(filtered_r_mid_EI,'I')
            filtered_r_mid=np.sum(0.8*filtered_r_mid_E + 0.2 *filtered_r_mid_I, axis=2)# order of summing up phases and mixing I-E matters if we change to sum of squares!

            filtered_r_sup_EI= smooth_data(r_sup, sigma = sigma_filter) 
            filtered_r_sup_E=filtered_r_sup_EI[:,:,0]
            filtered_r_sup_I=filtered_r_sup_EI[:,:,1]
            filtered_r_sup=0.8*filtered_r_sup_E + 0.2 *filtered_r_sup_I
            
            # Concatenate all orientation responses
            if ori == ori_list[0] and step_ind==SGD_step_inds[0]:
                filtered_r_mid_df = filtered_r_mid
                filtered_r_sup_df = filtered_r_sup
                ori_df = np.repeat(ori, n_noisy_trials)
                step_df = np.repeat(step_ind, n_noisy_trials)
            else:
                filtered_r_mid_df = np.concatenate((filtered_r_mid_df, filtered_r_mid))
                filtered_r_sup_df = np.concatenate((filtered_r_sup_df, filtered_r_sup))
                ori_df = np.concatenate((ori_df, np.repeat(ori, n_noisy_trials)))
                step_df = np.concatenate((step_df, np.repeat(step_ind, n_noisy_trials)))

    output = dict(ori = ori_df, SGD_step = step_df, r_mid = filtered_r_mid_df, r_sup = filtered_r_sup_df )

    return output, SGD_step_inds, r_mid, r_sup

######### Calculate Mahalanobis distance for before pretraining, after pretraining and after training - distance between trained and control should increase more than distance between untrained and control after training #########
def Mahalanobis_dist(num_training, folder, num_SGD_inds=2):
    ori_list = numpy.asarray([55, 125, 0])
    num_PC_used=15 # number of principal components used for the analysis
    num_layers=2 # number of layers
    num_noisy_trials=100 # Note: do not run this with small trial number because the estimation error of covariance matrix of the response for the control orientation stimuli will introduce a bias

    # Initialize arrays to store Mahalanobis distances and related metrics
    LMI_across = numpy.zeros((num_training,num_layers,num_SGD_inds-1))
    LMI_within = numpy.zeros((num_training,num_layers,num_SGD_inds-1))
    LMI_ratio = numpy.zeros((num_training,num_layers,num_SGD_inds-1))
    LMI_ttests = numpy.zeros((num_layers,3,num_SGD_inds-1))# 3 is for across, within and ratio
    LMI_ttest_p = numpy.zeros((num_layers,3,num_SGD_inds-1))

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
        file_name = f"{folder}/results_{run_ind}.csv"
        orimap_filename = f"{folder}/orimap_{run_ind}.npy"
        if not os.path.exists(file_name):
            file_name = f"{folder}/results_train_only{run_ind}.csv"
            orimap_filename = os.path.dirname(folder)+f'/orimap_{run_ind}.npy'
        
        loaded_orimap =  numpy.load(orimap_filename)
        untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                        loss_pars, training_pars, pretrain_pars, readout_pars, None, orimap_loaded=loaded_orimap)
        
        # Calculate num_noisy_trials filtered model response for each oris in ori list and for each parameter set (that come from file_name at num_SGD_inds rows)
        r_mid_sup, SGD_steps, _, _ = filtered_model_response(file_name, untrained_pars, ori_list= ori_list, n_noisy_trials = num_noisy_trials, num_SGD_inds=num_SGD_inds)
        # Note: r_mid_sup is a dictionary with the oris and SGD_steps saved in them
        
        # Separate the responses into orientation conditions
        mesh_train = r_mid_sup['ori'] == ori_list[0]
        mesh_untrain = r_mid_sup['ori'] == ori_list[1]
        mesh_control = r_mid_sup['ori'] == ori_list[2]
        
        # Iterate over the layers and SGD steps
        for layer in range(num_layers):
            for SGD_ind in range(num_SGD_inds):
                # Define filter to select the responses corresponding to SGD_ind
                mesh_step = r_mid_sup['SGD_step'] == SGD_steps[SGD_ind]
                mesh_train_ = mesh_train[mesh_step]
                mesh_untrain_ = mesh_untrain[mesh_step]
                mesh_control_ = mesh_control[mesh_step]
                # Select the layer (superficial=0 or middle=1) and the SGD_step
                if layer==0:
                    r = numpy.array(r_mid_sup['r_sup'][mesh_step])
                else:
                    r = numpy.array(r_mid_sup['r_mid'][mesh_step])
                
                # Attempt to stabilize the Mahalanobis distances (they chage too much from seed to seed) 
                # Save/Load the PCA object to a file
                # with open(f'pca_model_l{layer}_SGD{SGD_ind}.pkl', 'rb') as f:
                #    pca = pickle.load(f)                
                # with open(f'pca_model_l{layer}_SGD{SGD_ind}.pkl', 'wb') as f:
                #    pickle.dump(pca, f)
                # r matches up tp 3% but r_z has an average of 100% relative error because the small responsees are very unstable
                
                # Normalize data across trials
                #r_z = zscore(r, axis=0) # trials*n_oris x grid points (only E cells considered and averaged over phases for middle layer)
                score = pca.fit_transform(r)
                r_pca = score[:, :num_PC_used]
                
                ## Explained variance is above 99% for 15 components (70% for the experiments)
                #variance_explained = pca.explained_variance_ratio_
                #for i, var in enumerate(variance_explained):
                #    print(f"Variance explained by {i+1} PCs: {numpy.sum(variance_explained[0:i+1]):.2%}")

                # Separate data into orientation conditions, mesh for orientation and SGD step
                train_data = r_pca[mesh_train_,:]
                untrain_data = r_pca[mesh_untrain_,:]    
                control_data = r_pca[mesh_control_,:]
                
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
                    mahal_within_train[trial] = mahal(train_data_temp,train_data_trial_2d) 
                    mahal_within_untrain[trial] = mahal(untrain_data_temp, untrain_data_trial_2d)

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
                
            # Calculate learning modulation indices (LMI)
            for SGD_ind in range(num_SGD_inds-1):
                LMI_across[run_ind,layer,SGD_ind] = (mahal_train_control_mean[run_ind,layer,SGD_ind+1] - mahal_train_control_mean[run_ind,layer,SGD_ind]) - (mahal_untrain_control_mean[run_ind,layer,SGD_ind+1] - mahal_untrain_control_mean[run_ind,layer,SGD_ind] )
                LMI_within[run_ind,layer,SGD_ind] = (mahal_within_train_mean[run_ind,layer,SGD_ind+1] - mahal_within_train_mean[run_ind,layer,SGD_ind]) - (mahal_within_untrain_mean[run_ind,layer,SGD_ind+1] - mahal_within_untrain_mean[run_ind,layer,SGD_ind] )
                LMI_ratio[run_ind,layer,SGD_ind] = (train_SNR_mean[run_ind,layer,SGD_ind+1] - train_SNR_mean[run_ind,layer,SGD_ind]) - (untrain_SNR_mean[run_ind,layer,SGD_ind+1] - untrain_SNR_mean[run_ind,layer,SGD_ind] )
        print(run_ind)
        print(time.time()-start_time)

    # Apply t-test per layer and SGD_ind difference (samples are the different runs)
    for layer in range(num_layers):
        for SGD_ind in range(num_SGD_inds-1):
            LMI_ttests[layer,0,SGD_ind], LMI_ttest_p[layer,0,SGD_ind] = ttest_1samp(LMI_across[:,layer,SGD_ind],0) # compare it to mean 0
            LMI_ttests[layer,1,SGD_ind], LMI_ttest_p[layer,1,SGD_ind] = ttest_1samp(LMI_within[:,layer,SGD_ind],0)
            LMI_ttests[layer,2,SGD_ind], LMI_ttest_p[layer,2,SGD_ind] = ttest_1samp(LMI_ratio[:,layer,SGD_ind],0)
    
    # Create dataframes for the Mahalanobis distances and LMI
    run_layer_df=np.repeat(np.arange(num_training),num_layers)
    SGD_ind_df = numpy.zeros(num_training*num_layers)
    for i in range(1, num_SGD_inds):
        SGD_ind_df=numpy.hstack((SGD_ind_df,i * numpy.ones(num_training * num_layers)))
    # switch dimensions to : layer, run, SGD_ind
    mahal_train_control_mean = np.transpose(mahal_train_control_mean, (1, 0, 2))
    mahal_untrain_control_mean = np.transpose(mahal_untrain_control_mean, (1, 0, 2))
    mahal_within_train_mean = np.transpose(mahal_within_train_mean, (1, 0, 2))
    mahal_within_untrain_mean = np.transpose(mahal_within_untrain_mean, (1, 0, 2))
    train_SNR_mean = np.transpose(train_SNR_mean, (1, 0, 2))
    untrain_SNR_mean = np.transpose(untrain_SNR_mean, (1, 0, 2))
    df_mahal = pd.DataFrame({
        'run': np.tile(run_layer_df, num_SGD_inds), # 1,1,2,2,3,3,4,4,... , 1,1,2,2,3,3,4,4,... 
        'layer': np.tile(np.arange(num_layers),num_training*num_SGD_inds),# 1,2,1,2,1,2,...
        'SGD_ind': SGD_ind_df, # 0,0,0,... 1,1,1,...
        'ori55_across': mahal_train_control_mean.ravel(),
        'ori125_across': mahal_untrain_control_mean.ravel(),
        'ori55_within': mahal_within_train_mean.ravel(),
        'ori125_within': mahal_within_untrain_mean.ravel(),
        'ori55_SNR': train_SNR_mean.ravel(),
        'ori125_SNR': untrain_SNR_mean.ravel()
    })

    SGD_ind_df = numpy.zeros(num_training*num_layers)
    for i in range(1, num_SGD_inds-1):
        SGD_ind_df=numpy.hstack((SGD_ind_df,i * numpy.ones(num_training * num_layers)))
    LMI_across = np.transpose(LMI_across,(1,0,2))
    LMI_within = np.transpose(LMI_within,(1,0,2))
    LMI_ratio = np.transpose(LMI_ratio,(1,0,2))
    df_LMI = pd.DataFrame({
        'run': np.tile(run_layer_df, num_SGD_inds-1), # 1,1,2,2,3,3,4,4,... , 1,1,2,2,3,3,4,4,... 
        'layer': np.tile(np.arange(num_layers),num_training*(num_SGD_inds-1)),# 1,2,1,2,1,2,...
        'SGD_ind': SGD_ind_df,
        'LMI_across': LMI_across.ravel(),# layer, SGD
        'LMI_within': LMI_within.ravel(),
        'LMI_ratio': LMI_ratio.ravel()
    })

    SGD_ind_df = numpy.zeros(3*num_layers)
    for i in range(1, num_SGD_inds-1):
        SGD_ind_df=numpy.hstack((SGD_ind_df,i * numpy.ones(3 * num_layers)))

    LMI_type=['accross','accross', 'within', 'within', 'ratio', 'ratio']
    df_stats = pd.DataFrame({
        'layer': np.tile(np.arange(num_layers),3*(num_SGD_inds-1)),# 1,2,1,2,1,2,...
        'LMI_type': LMI_type *(num_SGD_inds-1),
        'SGD_ind': SGD_ind_df,
        'LMI_ttests': LMI_ttests.ravel(),
        'LMI_ttest_p': LMI_ttest_p.ravel()
    })

    # t-test and ANOVA on mahal distances directly
    # ANOVA on LMI_across, within and ratio
    
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
    
    return df_mahal, df_LMI, df_stats, mahal_train_control_all, mahal_untrain_control_all, mahal_within_train_all, mahal_within_untrain_all
    
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
        bp = axs[training_type,0].boxplot(np.transpose(mahal_diff_l0[:,training_type*2:training_type*2+2]), labels=labels ,patch_artist=True)
        axs[training_type,0].set_title(f'{layer_labels[0]} layer, {training_type_text[training_type]}', fontsize=20)
        for box, color in zip(bp['boxes'], colors):
            box.set_facecolor(color)
        bp = axs[training_type,1].boxplot(np.transpose(mahal_diff_l1[:,training_type*2:training_type*2+2]), labels=labels, patch_artist=True)
        axs[training_type,1].set_title(f'{layer_labels[1]} layer', fontsize=20)
        for box, color in zip(bp['boxes'], colors):
            box.set_facecolor(color)
        
    fig.savefig(f"{folder_to_save}/Mahal_dist_diff")
    return 

#num_training=3
#final_folder_path = 'results/Mar22_v0'
#folder_to_save = final_folder_path + '/figures'

def Mahal_dist_from_csv(num_training, final_folder_path, folder_to_save, file_name='mahal_dist', num_SGD_inds=2):
    plt.close()
    df_mahal, df_LMI, df_stats, mahal_train_control, mahal_untrain_control, mahal_within_train, mahal_within_untrain = Mahalanobis_dist(num_training, final_folder_path, num_SGD_inds)
    df_mahal.to_csv(final_folder_path+'/df_mahal.csv', index=False)
    df_LMI.to_csv(final_folder_path+'/df_LMI.csv', index=False)
    df_stats.to_csv(final_folder_path+'/df_stats.csv', index=False)
    plot_Mahalanobis_dist(num_training, num_SGD_inds, mahal_train_control, mahal_untrain_control, mahal_within_train, mahal_within_untrain, folder_to_save, file_name)
