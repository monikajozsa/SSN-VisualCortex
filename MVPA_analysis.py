import time
import jax
from jax import vmap
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
from util import sep_exponentiate, load_parameters, create_grating_training
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
from training import generate_noise

def gaussian_kernel(size: int, sigma: float):
    """Generates a 2D Gaussian kernel."""
    x = np.arange(-size // 2 + 1., size // 2 + 1.)
    y = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def gaussian_filter_jax(image, sigma: float):
    """Applies Gaussian filter to a 2D JAX array (image)."""
    size = int(np.ceil(3 * sigma) * 2 + 1)  # Kernel size
    kernel = gaussian_kernel(size, sigma)
    smoothed_image = jax.scipy.signal.convolve2d(image, kernel, mode='same')
    return smoothed_image

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
    Smooth data for a single trial over grid points. Data is reshaped into 9x9 grid before smoothing and then flattened again.
    '''
    N_grid_points=gridsize_Nx*gridsize_Nx
    N_phases = X.shape[0]//N_grid_points
    smoothed_data= numpy.zeros((N_grid_points,N_phases))
    for phase in range(N_phases):
        trial_response = X[phase*N_grid_points:(phase+1)*N_grid_points]
        trial_response = trial_response.reshape(gridsize_Nx,gridsize_Nx)
        smoothed_data_temp = gaussian_filter_jax(trial_response, sigma = sigma).ravel()
        smoothed_data.at[:, phase].set(smoothed_data_temp) # Error to be fixed ***

    return smoothed_data

# Vectorize the function over the first axis (trials)
vmap_smooth_data = vmap(smooth_data, in_axes=(0, None, None))

def filtered_model_response_task(file_name, untrained_pars, ori_list= np.asarray([55, 125, 0]), n_noisy_trials = 100, num_SGD_inds = 2, r_noise=None, sigma_filter = 1,gridsize_Nx=9):
    '''Calculate filtered model response for each orientation in ori_list and for each parameter set (that come from file_name at num_SGD_inds rows)'''
    start_time = time.time()
    df = pd.read_csv(file_name)
    train_start_ind = df.index[df['stage'] == 1][0]
    if num_SGD_inds==3:        
        if numpy.min(df['stage'])==0:
            pretrain_start_ind = df.index[df['stage'] == 0][0]
            SGD_step_inds=[pretrain_start_ind, train_start_ind, -1]
        else:
            print('Warning: There is no 0 stage but MVPA score was asked to be calculated for pretraining!')
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

            # Generate data
            test_gratings = create_grating_training(untrained_pars.stimuli_pars, n_noisy_trials,untrained_pars.BW_image_jax_inp)
            label = test_gratings['label']
            # Create middle and superficial SSN layers *** this is something that would be great to call from outside the SGD loop and only refresh the params that change (and what rely on them such as W)
            kappa_pre = untrained_pars.ssn_layer_pars.kappa_pre
            kappa_post = untrained_pars.ssn_layer_pars.kappa_post
            p_local_s = untrained_pars.ssn_layer_pars.p_local_s
            s_2x2 = untrained_pars.ssn_layer_pars.s_2x2_s
            sigma_oris = untrained_pars.ssn_layer_pars.sigma_oris
            ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_m)
            ssn_sup=SSN_sup(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_s, p_local=p_local_s, oris=untrained_pars.oris, s_2x2=s_2x2, sigma_oris = sigma_oris, ori_dist = untrained_pars.ori_dist, train_ori = ori, kappa_post = kappa_post, kappa_pre = kappa_pre)
    
            # Calculate fixed point for data    
            _, _, [_, _], [_, _], [_, _, _, _], [r_mid_ref, r_sup_ref] = vmap_evaluate_model_response(ssn_mid, ssn_sup, test_gratings['ref'], untrained_pars.conv_pars, c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)
            _, _, [_, _], [_, _], [_, _, _, _], [r_mid_target, r_sup_target] = vmap_evaluate_model_response(ssn_mid, ssn_sup, test_gratings['target'], untrained_pars.conv_pars, c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)

            # Add noise to the responses
            if r_noise is not None:
                noise_mid_ref = generate_noise(n_noisy_trials, length = r_mid_ref.shape[1], N_readout = untrained_pars.N_readout_noise)
                noise_mid_target = generate_noise(n_noisy_trials, length = r_mid_target.shape[1], N_readout = untrained_pars.N_readout_noise)
                r_mid_ref_noisy = r_mid_ref + noise_mid_ref*np.sqrt(jax.nn.softplus(r_mid_ref))
                r_mid_target_noisy = r_mid_target + noise_mid_target*np.sqrt(jax.nn.softplus(r_mid_target))
                noise_sup_ref = generate_noise(n_noisy_trials, length = r_sup_ref.shape[1], N_readout = untrained_pars.N_readout_noise)
                noise_sup_target = generate_noise(n_noisy_trials, length = r_sup_target.shape[1], N_readout = untrained_pars.N_readout_noise)
                r_sup_ref_noisy = r_sup_ref + noise_sup_ref*np.sqrt(jax.nn.softplus(r_sup_ref))
                r_sup_target_noisy = r_sup_target + noise_sup_target*np.sqrt(jax.nn.softplus(r_sup_target))
            else:
                r_mid_ref_noisy = r_mid_ref 
                r_mid_target_noisy = r_mid_target
                r_sup_ref_noisy = r_sup_ref 
                r_sup_target_noisy = r_sup_target

            # Smooth data for each celltype, layer and stimulus type (ref or target) separately with Gaussian filter
            filtered_r_mid_ref_EI= vmap_smooth_data(r_mid_ref_noisy,gridsize_Nx,sigma_filter)  #n_noisy_trials x 648
            filtered_r_mid_ref_E = select_type_mid(filtered_r_mid_ref_EI,'E')
            filtered_r_mid_ref_I = select_type_mid(filtered_r_mid_ref_EI,'I')
            filtered_r_mid_ref = np.sum(0.8*filtered_r_mid_ref_E + 0.2 *filtered_r_mid_ref_I, axis=2) # sum up along phases - should it be sum of squares?
            filtered_r_mid_target_EI= vmap_smooth_data(r_mid_target_noisy,sigma_filter)  #n_noisy_trials x 648
            filtered_r_mid_target_E = select_type_mid(filtered_r_mid_target_EI,'E')
            filtered_r_mid_target_I = select_type_mid(filtered_r_mid_target_EI,'I')
            filtered_r_mid_target = np.sum(0.8*filtered_r_mid_target_E + 0.2 *filtered_r_mid_target_I, axis=2) # order of summing up phases and mixing I-E matters if we change to sum of squares!

            filtered_r_sup_ref_EI = vmap_smooth_data(r_sup_ref_noisy,sigma_filter)
            filtered_r_sup_ref_E = filtered_r_sup_ref_EI[:,:,0]
            filtered_r_sup_ref_I = filtered_r_sup_ref_EI[:,:,1]
            filtered_r_sup_ref = 0.8*filtered_r_sup_ref_E + 0.2 *filtered_r_sup_ref_I
            filtered_r_sup_target_EI = vmap_smooth_data(r_sup_target_noisy,sigma_filter)
            filtered_r_sup_target_E = filtered_r_sup_target_EI[:,:,0]
            filtered_r_sup_target_I = filtered_r_sup_target_EI[:,:,1]
            filtered_r_sup_target = 0.8*filtered_r_sup_target_E + 0.2 *filtered_r_sup_target_I

            # Concatenate along orientation responses
            if ori == ori_list[0] and step_ind==SGD_step_inds[0]:
                filtered_r_mid_ref_df = filtered_r_mid_ref
                filtered_r_mid_target_df = filtered_r_mid_target
                filtered_r_sup_ref_df = filtered_r_sup_ref
                filtered_r_sup_target_df = filtered_r_sup_target

                ori_df = np.repeat(ori, n_noisy_trials)
                step_df = np.repeat(step_ind, n_noisy_trials)
                labels = label
            else:
                filtered_r_mid_ref_df = np.concatenate((filtered_r_mid_ref_df, filtered_r_mid_ref))
                filtered_r_mid_target_df = np.concatenate((filtered_r_mid_target_df, filtered_r_mid_target))
                filtered_r_sup_ref_df = np.concatenate((filtered_r_sup_ref_df, filtered_r_sup_ref))
                filtered_r_sup_target_df = np.concatenate((filtered_r_sup_target_df, filtered_r_sup_target))
                ori_df = np.concatenate((ori_df, np.repeat(ori, n_noisy_trials)))
                step_df = np.concatenate((step_df, np.repeat(step_ind, n_noisy_trials)))
                labels = np.concatenate((labels, label))
            print('filtered model response task done for step ind and ori: ',[step_ind, ori])
            print(time.time()-start_time)

    # Define output dictionary with keys: ori, SGD_step, r_mid_ref, r_mid_target, r_sup_ref, r_sup_target, labels
    output = dict(ori = ori_df, SGD_step = step_df, r_mid_ref = filtered_r_mid_ref_df, r_mid_target = filtered_r_mid_target_df, r_sup_ref = filtered_r_sup_ref_df , r_sup_target = filtered_r_sup_target_df , labels = labels)

    return output, SGD_step_inds

######### Calculate MVPA scores for before pretraining, after pretraining and after training - score should increase for trained ori more than for other two oris especially in superficial layer #########
def MVPA_score(num_training, folder, num_SGD_inds=2):
    ori_list = numpy.asarray([55, 125, 0])
    num_layers=2 # number of layers
    num_noisy_trials=200 # Note: do not run this with small trial number because the estimation error of covariance matrix of the response for the control orientation stimuli will introduce a bias
    
    # parameters of the SVM classifier (MVPA analysis)
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    MVPA_scores = numpy.zeros((num_training,num_layers,num_SGD_inds, len(ori_list)))
                
    # Iterate over the different parameter initializations (runs)
    for run_ind in range(num_training):
        file_name = f"{folder}/results_{run_ind}.csv"
        orimap_filename = f"{folder}/orimap_{run_ind}.npy"
        if not os.path.exists(file_name):
            file_name = f"{folder}/results_train_only{run_ind}.csv"
            orimap_filename = os.path.dirname(folder)+f'/orimap_{run_ind}.npy'
        
        loaded_orimap =  numpy.load(orimap_filename)
        untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                        loss_pars, training_pars, pretrain_pars, readout_pars, None, orimap_loaded=loaded_orimap)
        
        # Calculate num_noisy_trials filtered model response for each oris in ori list and for each parameter set (that come from file_name at num_SGD_inds rows)
        r_mid_sup, SGD_step_inds = filtered_model_response_task(file_name, untrained_pars, ori_list= ori_list, n_noisy_trials = num_noisy_trials, num_SGD_inds=num_SGD_inds, r_noise=True)
        
        # Separate the responses into orientation conditions
        mesh_train = r_mid_sup['ori'] == ori_list[0]
        mesh_untrain = r_mid_sup['ori'] == ori_list[1]
        mesh_control = r_mid_sup['ori'] == ori_list[2]
        
        # Iterate over the layers and SGD steps
        for layer in range(num_layers):
            for SGD_ind in range(num_SGD_inds):
                # Define filter to select the responses corresponding to SGD_ind
                mesh_step = r_mid_sup['SGD_step'] == SGD_step_inds[SGD_ind]
                # Select the layer (superficial=0 or middle=1) and the SGD_step and subtract the target from the reference response
                label = numpy.array(r_mid_sup['labels'][mesh_step])
                if layer==0:
                    r = numpy.array(r_mid_sup['r_sup_ref'][mesh_step]) - numpy.array(r_mid_sup['r_sup_target'][mesh_step])
                else:
                    r = numpy.array(r_mid_sup['r_mid_ref'][mesh_step])- numpy.array(r_mid_sup['r_mid_target'][mesh_step])
                # Define filter to select the responses corresponding to the orientation condition
                for ori_ind in range(len(ori_list)):
                    if ori_ind==0:
                        mesh_ori = mesh_train[mesh_step]
                    if ori_ind==1:
                        mesh_ori = mesh_untrain[mesh_step]
                    if ori_ind==2:
                        mesh_ori = mesh_control[mesh_step]
                    r_ori = r[mesh_ori]
                    label_ori = label[mesh_ori]

                    # MVPA analysis
                    X_train, X_test, y_train, y_test = train_test_split(r_ori, label_ori, test_size=0.2, random_state=42)
                    MVPA_scores[run_ind,layer,SGD_ind, ori_ind]=clf.fit(X_train, y_train).score(X_test, y_test)
                
    return MVPA_scores
    

def MVPA_score_from_csv(num_training, final_folder_path, folder_to_save, file_name='MVPA_scores', num_SGD_inds=2):
    ''' Calculate MVPA scores for before pretraining, after pretraining and after training - score should increase for trained ori more than for other two oris especially in superficial layer'''
    plt.close()
    MVPA_scores = MVPA_score(num_training, final_folder_path, num_SGD_inds)
