import pandas as pd
import os
import jax.numpy as np
import numpy
import time
import scipy
from scipy import ndimage
import jax
from jax import vmap
import matplotlib.pyplot as plt
import seaborn as sns

from model import evaluate_model_response, vmap_evaluate_model_response
from util import sep_exponentiate
from SSN_classes import SSN_mid, SSN_sup
from util_gabor import BW_image_jit, BW_image_jit_noisy
from training import mean_training_task_acc_test, offset_at_baseline_acc, generate_noise
from util import load_parameters, create_grating_training
from util_gabor import init_untrained_pars
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
    pretrain_pars # Setting pretraining (pretrain_pars.is_on) should happen in parameters.py because w_sig depends on it
)

############## Analysis functions ##########

def rel_changes_from_csvs(folder, num_trainings=None, num_indices = 3, offset_calc=True):
    '''read CSV files and calculate the correlation between the changes of accuracy and J (J_II and J_EI are summed up and J_EE and J_IE are summed up) for each file'''
    
    # Check if num_trainings is None and calculate the number of result files in the folder
    if num_trainings is None:
        num_trainings = 0
        for filename in os.listdir(folder):
            if filename.endswith('.csv') and filename.startswith('result'):
                num_trainings += 1

    file_pattern = folder + '/results_{}'

    # Initialize the arrays to store the results in
    J_m_diff = numpy.zeros((num_trainings*(num_indices-2),7))
    J_s_diff = numpy.zeros((num_trainings*(num_indices-2),7))
    f_diff = numpy.zeros((num_trainings*(num_indices-2),2))
    c_diff = numpy.zeros((num_trainings*(num_indices-2),2))
    offset_th = numpy.zeros((num_trainings*(num_indices-2),2))
    offset_th_125 = numpy.zeros((num_trainings*(num_indices-2),2))
    offset_th_diff = numpy.zeros(num_trainings*(num_indices-2))
    offset_th_diff_125 = numpy.zeros(num_trainings*(num_indices-2))

    # Initialize the test offset vector for the threshold calculation
    test_offset_vec = numpy.array([1, 2, 3, 4, 6]) 

    start_time = time.time()
    sample_ind = 0

    # Check if offset_th.csv is already present and if it is the same size as the number of trainings * (num_indices-2)
    if offset_calc:
        if 'offset_th.csv' in os.listdir(folder):
            offset_th = numpy.loadtxt(folder + '/offset_th.csv', delimiter=',')
            offset_th_125 = numpy.loadtxt(folder + '/offset_th_125.csv', delimiter=',')
            if offset_th.shape[0]==num_trainings*(num_indices-2):
                offset_calc = False
    
    ref_ori_saved = float(stimuli_pars.ref_ori)
    for i in range(num_trainings):
        # Construct the file name
        file_name = file_pattern.format(i) + '.csv'
        if offset_calc:
            # Load the orimap file and define the untrained parameters
            orimap_filename = folder + '/orimap_{}.npy'.format(i)
            loaded_orimap =  numpy.load(orimap_filename)
            untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                            loss_pars, training_pars, pretrain_pars, readout_pars, None, orimap_loaded=loaded_orimap)
            untrained_pars.pretrain_pars.is_on = False        
        
        # Read the file
        df = pd.read_csv(file_name)

        # Calculate the J differences (J_m_EE	J_m_EI	J_m_IE	J_m_II	J_s_EE	J_s_EI	J_s_IE	J_s_II) at start and end of training
        relative_changes, time_inds = rel_changes(df, num_indices)
        training_start = time_inds[1]
        J_m_EE = df['J_m_EE']
        J_m_IE = df['J_m_IE']
        J_s_EE = df['J_s_EE']
        J_s_IE = df['J_s_IE']
        J_m_EI = [numpy.abs(df['J_m_EI'][i]) for i in range(len(df['J_m_EI']))]
        J_m_II = [numpy.abs(df['J_m_II'][i]) for i in range(len(df['J_m_II']))]
        J_s_EI = [numpy.abs(df['J_s_EI'][i]) for i in range(len(df['J_s_EI']))]
        J_s_II = [numpy.abs(df['J_s_II'][i]) for i in range(len(df['J_s_II']))]

        if offset_calc:
            # Calculate the offset threshold
            trained_pars_stage1, trained_pars_stage2, _ = load_parameters(file_name, iloc_ind = training_start)
            acc_mean, _, _ = mean_training_task_acc_test(trained_pars_stage2, trained_pars_stage1, untrained_pars, jit_on=True, offset_vec=test_offset_vec, sample_size = 1 )
            offset_temp = numpy.atleast_1d(offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= 0.85))[0]
            offset_th[sample_ind : sample_ind + num_indices-2,0] = numpy.repeat(offset_temp, num_indices-2)
            untrained_pars.stimuli_pars.ref_ori = 125
            acc_mean, _, _ = mean_training_task_acc_test(trained_pars_stage2, trained_pars_stage1, untrained_pars, jit_on=True, offset_vec=test_offset_vec, sample_size = 1 )
            offset_temp = numpy.atleast_1d(offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= 0.85))[0]
            offset_th_125[sample_ind : sample_ind + num_indices-2,0] = numpy.repeat(offset_temp, num_indices-2)
            
        for j in range(2,num_indices):
            training_end = time_inds[j]
            # Calculate the relative changes in the combined J_m_E J_m_I, J_s_E and J_s_I parameters
            J_m_E_0 = (J_m_EE[training_start]+J_m_IE[training_start])
            J_m_I_0 = (J_m_II[training_start]+J_m_EI[training_start])
            J_s_E_0 = (J_s_EE[training_start]+J_s_IE[training_start])
            J_s_I_0 = (J_s_II[training_start]+J_s_EI[training_start])
            J_m_E_1 = (J_m_EE[training_end]+J_m_IE[training_end])
            J_m_I_1 = (J_m_II[training_end]+J_m_EI[training_end])
            J_s_E_1 = (J_s_EE[training_end]+J_s_IE[training_end])
            J_s_I_1 = (J_s_II[training_end]+J_s_EI[training_end])

            J_m_diff[sample_ind,4] = (J_m_E_1 - J_m_E_0) / J_m_E_0
            J_m_diff[sample_ind,5] = (J_m_I_1 - J_m_I_0) / J_m_I_0
            J_m_diff[sample_ind,6] = (J_m_I_1/J_m_E_1 - J_m_I_0/J_m_E_0) / (J_m_I_0/J_m_E_0)
            J_s_diff[sample_ind,4] = (J_s_E_1 - J_s_E_0) / J_s_E_0       
            J_s_diff[sample_ind,5] = (J_s_I_1 - J_s_I_0) / J_s_I_0
            J_s_diff[sample_ind,6] = (J_s_I_1/J_s_E_1 - J_s_I_0/J_s_E_0) / (J_s_I_0/J_s_E_0)
            # Calculate the relative changes in the parameters
            J_m_diff[sample_ind,0:4]= relative_changes[0:4,1] *100
            J_s_diff[sample_ind,0:4]= relative_changes[4:8,1] *100
            c_diff[sample_ind,:] = relative_changes[8:10,1] *100
            f_diff[sample_ind,:] = relative_changes[10:12,1] *100
                    
            if offset_calc:
                # Calculate the offset threshold
                trained_pars_stage1, trained_pars_stage2, _ = load_parameters(file_name, iloc_ind = training_end)
                acc_mean, _, _ = mean_training_task_acc_test(trained_pars_stage2, trained_pars_stage1, untrained_pars, jit_on=True, offset_vec=test_offset_vec )
                offset_temp = numpy.atleast_1d(offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= 0.85))[0]
                offset_th[sample_ind,1] = offset_temp
                
                untrained_pars.stimuli_pars.ref_ori = 125
                acc_mean, _, _ = mean_training_task_acc_test(trained_pars_stage2, trained_pars_stage1, untrained_pars, jit_on=True, offset_vec=test_offset_vec )
                offset_temp = numpy.atleast_1d(offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= 0.85))[0]
                offset_th_125[sample_ind,1] = offset_temp
                untrained_pars.stimuli_pars.ref_ori = ref_ori_saved
                
            offset_th_diff[sample_ind] = -(offset_th[sample_ind,1] - offset_th[sample_ind,0]) / offset_th[sample_ind,0] *100
            offset_th_diff_125[sample_ind] = -(offset_th_125[sample_ind,1] - offset_th_125[sample_ind,0]) / offset_th_125[sample_ind,0] *100
        
            # increment the sample index
            sample_ind += 1

        print('Finished reading file', i, 'time elapsed:', time.time() - start_time)

    # save out offset_th in a csv if it is not already present
    if offset_calc:
        numpy.savetxt(folder + '/offset_th.csv', offset_th, delimiter=',')

    # Check if the offset is valid (180 is a default value for when offset_th is not found within range)    
    mesh_offset_th=numpy.sum(offset_th, axis=1)<180
    # Filter the data based on the valid offset values
    #offset_th = offset_th[mesh_offset_th,:]
    offset_th_diff = offset_th_diff[mesh_offset_th]
    #offset_th_125 = offset_th_125[mesh_offset_th,:]
    offset_th_diff_125 = offset_th_diff_125[mesh_offset_th]
    J_m_diff = J_m_diff[mesh_offset_th,:]
    J_s_diff = J_s_diff[mesh_offset_th,:]
    f_diff = f_diff[mesh_offset_th,:]
    c_diff = c_diff[mesh_offset_th,:]
    
    print('Number of samples:', sum(mesh_offset_th))
    
    return  offset_th_diff, offset_th_diff_125, J_m_diff, J_s_diff, f_diff, c_diff


def rel_changes(df, num_indices=3):
    # Calculate relative changes in Jm and Js
    J_m_EE = df['J_m_EE']
    J_m_IE = df['J_m_IE']
    J_m_EI = [np.abs(df['J_m_EI'][i]) for i in range(len(df['J_m_EI']))]
    J_m_II = [np.abs(df['J_m_II'][i]) for i in range(len(df['J_m_II']))]
    J_s_EE = df['J_s_EE']
    J_s_IE = df['J_s_IE']
    J_s_EI = [np.abs(df['J_s_EI'][i]) for i in range(len(df['J_s_EI']))]
    J_s_II = [np.abs(df['J_s_II'][i]) for i in range(len(df['J_s_II']))]
    c_E = df['c_E']
    c_I = df['c_I']
    f_E = df['f_E']
    f_I = df['f_I']
    acc = df['acc']
    offset = df['offset']
    maxr_E_mid = df['maxr_E_mid']
    maxr_I_mid = df['maxr_I_mid']
    maxr_E_sup = df['maxr_E_sup']
    maxr_I_sup = df['maxr_I_sup']
    relative_changes = numpy.zeros((18,num_indices-1))

    ############### Calculate relative changes in parameters and other metrics before and after training ###############
    # Define time indices for pretraining and training
    time_inds = SGD_step_indices(df, num_indices)

    # Calculate relative changes for pretraining and training (additional time points may be included)
    for j in range(2):
        if j==0:
            # changes during pretraining
            start_ind = time_inds[0]
            relative_changes[0,0] =(J_m_EE[time_inds[1]] - J_m_EE[start_ind]) / J_m_EE[start_ind] # J_m_EE
            relative_changes[1,0] =(J_m_IE[time_inds[1]] - J_m_IE[start_ind]) / J_m_IE[start_ind] # J_m_IE
            relative_changes[2,0] =(J_m_EI[time_inds[1]] - J_m_EI[start_ind]) / J_m_EI[start_ind] # J_m_EI
            relative_changes[3,0] =(J_m_II[time_inds[1]] - J_m_II[start_ind]) / J_m_II[start_ind] # J_m_II
            relative_changes[4,0] =(J_s_EE[time_inds[1]] - J_s_EE[start_ind]) / J_s_EE[start_ind] # J_s_EE
            relative_changes[5,0] =(J_s_IE[time_inds[1]] - J_s_IE[start_ind]) / J_s_IE[start_ind] # J_s_IE
            relative_changes[6,0] =(J_s_EI[time_inds[1]] - J_s_EI[start_ind]) / J_s_EI[start_ind] # J_s_EI
            relative_changes[7,0] =(J_s_II[time_inds[1]] - J_s_II[start_ind]) / J_s_II[start_ind] # J_s_II
            relative_changes[8,0] = (c_E[time_inds[1]] - c_E[start_ind]) / c_E[start_ind] # c_E
            relative_changes[9,0] = (c_I[time_inds[1]] - c_I[start_ind]) / c_I[start_ind] # c_I
            relative_changes[10,0] = (f_E[time_inds[1]] - f_E[start_ind]) / f_E[start_ind] # f_E
            relative_changes[11,0] = (f_I[time_inds[1]] - f_I[start_ind]) / f_I[start_ind] # f_I
            relative_changes[12,0] = (acc[time_inds[1]] - acc[start_ind]) / acc[start_ind] # accuracy
            relative_changes[13,1] = (offset[time_inds[1]] - offset[start_ind]) / offset[start_ind] # offset and offset threshold
            relative_changes[14,0] = (maxr_E_mid[time_inds[1]] - maxr_E_mid[start_ind]) /maxr_E_mid[start_ind] # r_E_mid
            relative_changes[15,0] = (maxr_I_mid[time_inds[1]] -maxr_I_mid[start_ind]) / maxr_I_mid[start_ind] # r_I_mid
            relative_changes[16,0] = (maxr_E_sup[time_inds[1]] - maxr_E_sup[start_ind]) / maxr_E_sup[start_ind] # r_E_sup
            relative_changes[17,0] = (maxr_I_sup[time_inds[1]] - maxr_I_sup[start_ind]) / maxr_I_sup[start_ind] # r_I_sup
        else: 
            # changes during training
            start_ind = time_inds[1]
            for i in range(num_indices-2):
                relative_changes[0,i+j] =(J_m_EE[time_inds[i+2]] - J_m_EE[start_ind]) / J_m_EE[start_ind] # J_m_EE
                relative_changes[1,i+j] =(J_m_IE[time_inds[i+2]] - J_m_IE[start_ind]) / J_m_IE[start_ind] # J_m_IE
                relative_changes[2,i+j] =(J_m_EI[time_inds[i+2]] - J_m_EI[start_ind]) / J_m_EI[start_ind] # J_m_EI
                relative_changes[3,i+j] =(J_m_II[time_inds[i+2]] - J_m_II[start_ind]) / J_m_II[start_ind] # J_m_II
                relative_changes[4,i+j] =(J_s_EE[time_inds[i+2]] - J_s_EE[start_ind]) / J_s_EE[start_ind] # J_s_EE
                relative_changes[5,i+j] =(J_s_IE[time_inds[i+2]] - J_s_IE[start_ind]) / J_s_IE[start_ind] # J_s_IE
                relative_changes[6,i+j] =(J_s_EI[time_inds[i+2]] - J_s_EI[start_ind]) / J_s_EI[start_ind] # J_s_EI
                relative_changes[7,i+j] =(J_s_II[time_inds[i+2]] - J_s_II[start_ind]) / J_s_II[start_ind] # J_s_II
                relative_changes[8,i+j] = (c_E[time_inds[i+2]] - c_E[start_ind]) / c_E[start_ind] # c_E
                relative_changes[9,i+j] = (c_I[time_inds[i+2]] - c_I[start_ind]) / c_I[start_ind] # c_I
                relative_changes[10,i+j] = (f_E[time_inds[i+2]] - f_E[start_ind]) / f_E[start_ind] # f_E
                relative_changes[11,i+j] = (f_I[time_inds[i+2]] - f_I[start_ind]) / f_I[start_ind] # f_I
                relative_changes[12,i+j] = (acc[time_inds[i+2]] - acc[start_ind]) / acc[start_ind] # accuracy
                relative_changes[13,i+j] = (offset[time_inds[i+2]] - offset[start_ind]) / offset[start_ind]
                relative_changes[14,i+j] = (maxr_E_mid[time_inds[i+2]] - maxr_E_mid[start_ind]) / maxr_E_mid[start_ind] # r_E_mid
                relative_changes[15,i+j] = (maxr_I_mid[time_inds[i+2]] - maxr_I_mid[start_ind]) / maxr_I_mid[start_ind] # r_I_mid
                relative_changes[16,i+j] = (maxr_E_sup[time_inds[i+2]] - maxr_E_sup[start_ind]) /maxr_E_sup[start_ind] # r_E_sup
                relative_changes[17,i+j] = (maxr_I_sup[time_inds[i+2]] - maxr_I_sup[start_ind]) /maxr_I_sup[start_ind] # r_I_sup

    return relative_changes, time_inds


def tuning_curve(untrained_pars, trained_pars, tuning_curves_filename=None, ori_vec=np.arange(0,180,6)):
    '''
    Calculate responses of middle and superficial layers to different orientations.
    '''
    ref_ori_saved = float(untrained_pars.stimuli_pars.ref_ori)
    for key in list(trained_pars.keys()):  # Use list to make a copy of keys to avoid RuntimeError
        # Check if key starts with 'log'
        if key.startswith('log'):
            # Create a new key by removing 'log' prefix
            new_key = key[4:]
            # Exponentiate the values and assign to the new key
            if np.isscalar(trained_pars[key]) or numpy.isscalar(trained_pars[key]):
                if key.startswith('log_J') and key.endswith('I'):
                    trained_pars[new_key] = -numpy.exp(trained_pars[key])
                else:
                    trained_pars[new_key] = numpy.exp(trained_pars[key])
            else:
                trained_pars[new_key] = sep_exponentiate(trained_pars[key])
    
    ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=trained_pars['J_2x2_m'])
    
    num_ori = len(ori_vec)
    new_rows = []
    x = untrained_pars.BW_image_jax_inp[5]
    y = untrained_pars.BW_image_jax_inp[6]
    alpha_channel = untrained_pars.BW_image_jax_inp[7]
    mask = untrained_pars.BW_image_jax_inp[8]
    background = untrained_pars.BW_image_jax_inp[9]
    
    train_data = BW_image_jit(untrained_pars.BW_image_jax_inp[0:5], x, y, alpha_channel, mask, background, ori_vec, np.zeros(num_ori))
    for i in range(num_ori):
        ssn_sup=SSN_sup(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=trained_pars['J_2x2_s'], p_local=untrained_pars.ssn_layer_pars.p_local_s, oris=untrained_pars.oris, s_2x2=untrained_pars.ssn_layer_pars.s_2x2_s, sigma_oris = untrained_pars.ssn_layer_pars.sigma_oris, ori_dist = untrained_pars.ori_dist, train_ori = untrained_pars.stimuli_pars.ref_ori)
        _, _, [_,_], [_,_], [_,_,_,_], [r_mid_i, r_sup_i] = evaluate_model_response(ssn_mid, ssn_sup, train_data[i,:], untrained_pars.conv_pars, trained_pars['c_E'], trained_pars['c_I'], trained_pars['f_E'], trained_pars['f_I'], untrained_pars.gabor_filters)
        # testing tuning curve differece ***
        #constant_vector_mid = constant_to_vec(c_E = trained_pars['c_E'], c_I = trained_pars['c_I'], ssn= ssn_mid)
        #constant_vector_sup = constant_to_vec(c_E = trained_pars['c_E'], c_I = trained_pars['c_I'], ssn = ssn_sup, sup=True)
        #_, _, _, _, [fp_mid, fp_sup] = two_layer_model(ssn_mid, ssn_sup, train_data[i,:], untrained_pars.conv_pars, constant_vector_mid, constant_vector_sup,trained_pars['f_E'], trained_pars['f_I'])
        if i==0:
            responses_mid = numpy.zeros((num_ori,len(r_mid_i)))
            responses_sup = numpy.zeros((num_ori,len(r_sup_i)))
        responses_mid[i,:] = r_mid_i
        responses_sup[i,:] = r_sup_i
    
        # Save responses into csv file
        if tuning_curves_filename is not None:
 
            # Concatenate the new data as additional rows
            new_row = numpy.concatenate((r_mid_i, r_sup_i), axis=0)
            new_rows.append(new_row)

    if tuning_curves_filename is not None:
        new_rows_df = pd.DataFrame(new_rows)
        if os.path.exists(tuning_curves_filename):
            # Read existing data and concatenate new data
            existing_df = pd.read_csv(tuning_curves_filename)
            df = pd.concat([existing_df, new_rows_df], axis=0)
        else:
            # If CSV does not exist, use new data as the DataFrame
            df = new_rows_df

        # Write the DataFrame to CSV file
        df.to_csv(tuning_curves_filename, index=False)

    untrained_pars.stimuli_pars.ref_ori = ref_ori_saved

    return responses_sup, responses_mid


def tc_slope(tuning_curve, x_axis, x1, x2, normalised=False):
    """
    Calculates slope of normalized tuning_curve between points x1 and x2. tuning_curve is given at x_axis points.
    """
    #Remove baseline if normalising
    if normalised == True:
        tuning_curve = (tuning_curve - tuning_curve.min()) / tuning_curve.max()
    
    #Find indices corresponding to desired x values
    idx_1 = (np.abs(x_axis - x1)).argmin()
    idx_2 = (np.abs(x_axis - x2)).argmin()
    x1, x2 = x_axis[[idx_1, idx_2]]
     
    grad =(np.abs(tuning_curve[idx_2] - tuning_curve[idx_1]))/(x2-x1)
    
    return grad


def full_width_half_max(vector, d_theta):
    
    #Remove baseline
    vector = vector-vector.min()
    half_height = vector.max()/2
    points_above = len(vector[vector>half_height])

    distance = d_theta * points_above
    
    return distance


def tc_features(file_name, ori_list=numpy.arange(0,180,6), expand_dims=False):
    
    # Tuning curve of given cell indices
    tuning_curve = numpy.array(pd.read_csv(file_name))
    num_cells = tuning_curve.shape[1]
    
    # Find preferred orientation and center it at 55
    pref_ori = ori_list[np.argmax(tuning_curve, axis = 0)]
    norm_pref_ori = pref_ori -55

    # Full width half height
    full_width_half_max_vec = numpy.zeros(num_cells) 
    d_theta = ori_list[1]-ori_list[0]
    for i in range(0, num_cells):
        full_width_half_max_vec[i] = full_width_half_max(tuning_curve[:,i], d_theta = d_theta)

    # Norm slope
    avg_slope_vec =numpy.zeros(num_cells) 
    for i in range(num_cells):
        avg_slope_vec[i] = tc_slope(tuning_curve[:, i], x_axis = ori_list, x1 = 52, x2 = 58, normalised =False)
    if expand_dims:
        avg_slope_vec = numpy.expand_dims(avg_slope_vec, axis=0)
        full_width_half_max_vec = numpy.expand_dims(full_width_half_max_vec, axis=0)
        norm_pref_ori = numpy.expand_dims(norm_pref_ori, axis=0)

    return avg_slope_vec, full_width_half_max_vec, norm_pref_ori


def MVPA_param_offset_correlations(folder, num_trainings, num_time_inds=3, x_labels=None, plot_flag=False):
    offset_th_diff, offset_th_diff_125, J_m_diff, J_s_diff, f_diff, c_diff = rel_changes_from_csvs(folder, num_trainings, num_time_inds)
    
    # Convert relative parameter differences to pandas DataFrame
    data = pd.DataFrame({'offset_th_diff': offset_th_diff, 'offset_th_diff_125': offset_th_diff_125, 'J_m_EE_diff': J_m_diff[:, 0], 'J_m_IE_diff': J_m_diff[:, 1], 'J_m_EI_diff': J_m_diff[:, 2], 'J_m_II_diff': J_m_diff[:, 3], 'J_s_EE_diff': J_s_diff[:, 0], 'J_s_IE_diff': J_s_diff[:, 1], 'J_s_EI_diff': J_s_diff[:, 2], 'J_s_II_diff': J_s_diff[:, 3], 'f_E_diff': f_diff[:, 0], 'f_I_diff': f_diff[:, 1], 'c_E_diff': c_diff[:, 0], 'c_I_diff': c_diff[:, 1]})

    ##################### Correlate offset_th_diff with the combintation of the J_m_EE and J_m_IE, J_m_EI and J_m_II, etc. #####################

    # combine the J_m_EE and J_m_IE, J_m_EI and J_m_II, J_s_EE and J_s_IE, J_s_EI and J_s_II and add them to the data
    data['J_m_E_diff'] = J_m_diff[:, 4]
    data['J_m_I_diff'] = J_m_diff[:, 5]
    data['J_s_E_diff'] = J_s_diff[:, 4]
    data['J_s_I_diff'] = J_s_diff[:, 5]
    data['J_m_ratio_diff'] = J_m_diff[:, 6]
    data['J_s_ratio_diff'] = J_s_diff[:, 6]
    offset_pars_corr = []
    if x_labels is None:
        x_labels = ['J_m_E_diff', 'J_m_I_diff', 'J_s_E_diff', 'J_s_I_diff', 'f_E_diff','f_I_diff', 'c_E_diff', 'c_I_diff']
    for i in range(len(x_labels)):
        # Calculate the Pearson correlation coefficient and the p-value
        corr, p_value = scipy.stats.pearsonr(data['offset_th_diff'], data[x_labels[i]])
        offset_pars_corr.append({'corr': corr, 'p_value': p_value})
    
    # Load MVPA_scores and correlate them with the offset threshold and the parameter differences (samples are the different trainings)
    MVPA_scores = numpy.load(folder + '/MVPA_scores.npy') # num_trainings x layer x SGD_ind x ori_ind
    MVPA_scores_diff = MVPA_scores[:,:,1,:] - MVPA_scores[:,:,-1,:] # num_trainings x layer x ori_ind
    MVPA_offset_corr = []
    for i in range(MVPA_scores_diff.shape[1]):
        for j in range(MVPA_scores_diff.shape[2]):
            corr, p_value = scipy.stats.pearsonr(data['offset_th_diff'], MVPA_scores_diff[:,i,j])
            MVPA_offset_corr.append({'corr': corr, 'p_value': p_value})
    MVPA_pars_corr = [] # (J_m_I,J_m_E,J_s_I,J_s_E,f_E,f_I,c_E,c_I) x ori_ind
    for j in range(MVPA_scores_diff.shape[2]):
        for i in range(MVPA_scores_diff.shape[1]):        
            if i==0:
                corr_m_J_I, p_val_m_J_I = scipy.stats.pearsonr(data['J_m_I_diff'], MVPA_scores_diff[:,i,j])
                corr_m_J_E, p_val_m_J_E = scipy.stats.pearsonr(data['J_m_E_diff'], MVPA_scores_diff[:,i,j])
                corr_m_f_E, p_val_m_f_E = scipy.stats.pearsonr(data['f_E_diff'], MVPA_scores_diff[:,i,j])
                corr_m_f_I, p_val_m_f_I = scipy.stats.pearsonr(data['f_I_diff'], MVPA_scores_diff[:,i,j])
                corr_m_c_E, p_val_m_c_E = scipy.stats.pearsonr(data['c_E_diff'], MVPA_scores_diff[:,i,j])
                corr_m_c_I, p_val_m_c_I = scipy.stats.pearsonr(data['c_I_diff'], MVPA_scores_diff[:,i,j])
            if i==1:
                corr_s_J_I, p_val_s_J_I = scipy.stats.pearsonr(data['J_s_I_diff'], MVPA_scores_diff[:,i,j])
                corr_s_J_E, p_val_s_J_E = scipy.stats.pearsonr(data['J_s_E_diff'], MVPA_scores_diff[:,i,j])                
                corr_s_f_E, p_val_s_f_E = scipy.stats.pearsonr(data['f_E_diff'], MVPA_scores_diff[:,i,j])
                corr_s_f_I, p_val_s_f_I = scipy.stats.pearsonr(data['f_I_diff'], MVPA_scores_diff[:,i,j])
                corr_s_c_E, p_val_s_c_E = scipy.stats.pearsonr(data['c_E_diff'], MVPA_scores_diff[:,i,j])
                corr_s_c_I, p_val_s_c_I = scipy.stats.pearsonr(data['c_I_diff'], MVPA_scores_diff[:,i,j])
            
        corr = [corr_m_J_E, corr_m_J_I, corr_s_J_E, corr_s_J_I, corr_m_f_E, corr_m_f_I, corr_m_c_E, corr_m_c_I, corr_s_f_E, corr_s_f_I, corr_s_c_E, corr_s_c_I]
        p_value = [p_val_m_J_E, p_val_m_J_I, p_val_s_J_E, p_val_s_J_I, p_val_m_f_E, p_val_m_f_I, p_val_m_c_E, p_val_m_c_I, p_val_s_f_E, p_val_s_f_I, p_val_s_c_E, p_val_s_c_I]
        MVPA_pars_corr.append({'corr': corr, 'p_value': p_value})
    # combine MVPA_offset_corr and MVPA_pars_corr into a single list
    MVPA_corrs = MVPA_offset_corr + MVPA_pars_corr

    if plot_flag:
        # Create a 6x6 grid of subplots
        fig, axes = plt.subplots(6, 6, figsize=(20, 20))
        axes_flat = axes.flatten()
        # x-axis labels
        x_labels = ['J_m_ratio_diff', 'J_s_ratio_diff', 'J_m_E_diff', 'J_m_I_diff', 'J_s_E_diff', 'J_s_I_diff']
        # y-axis labels
        y_axes = ['offset_th_diff', 'offset_th_diff_125', 'MVPA_m_55', 'MVPA_s_55', 'offset_125', 'MVPA_m_125', 'MVPA_s_125']

        for i in range(6):
            for j in range(6):
                # Create lmplot for each pair of variables
                sns.regplot(x=x_labels[i], y='offset_th_diff', data=data, ax=axes_flat[i], ci=95, color='red', 
                        line_kws={'color':'darkred'}, scatter_kws={'alpha':0.3, 'color':'red'})
                
                # Close the lmplot's figure to prevent overlapping
                axes[i,j].set_title( f'Corr: {corr:.2f}, p-val: {p_value:.2f}')

        plt.tight_layout()
        plt.savefig(folder + "/figures/Offset_MVPA_corr_Jall.png")
        plt.close()

        # Create a 4x6 grid of subplots
        fig, axes = plt.subplots(4, 6, figsize=(20, 15))
        axes_flat = axes.flatten()
        x_axes_labels = ['f_E', 'f_I', 'c_E', 'c_I']
        x_labels=['f_E_diff','f_I_diff', 'c_E_diff', 'c_I_diff']

    return offset_pars_corr, MVPA_corrs, data  # Returns a list of dictionaries for each training run

############################## helper functions for MVPA and Mahal distance analysis ##############################

def select_type_mid(r_mid, cell_type='E'):
    '''Selects the excitatory or inhibitory cell responses. This function assumes that r_mid is 3D (trials x grid points x celltype and phase)'''
    if cell_type=='E':
        map_numbers = np.arange(1, 2 * ssn_pars.phases, 2)-1 # 0,2,4,6
    else:
        map_numbers = np.arange(2, 2 * ssn_pars.phases + 1, 2)-1 # 1,3,5,7
    
    out = np.zeros((r_mid.shape[0],r_mid.shape[1], int(r_mid.shape[2]/2)))
    for i in range(len(map_numbers)):
        out = out.at[:,:,i].set(r_mid[:,:,map_numbers[i]])

    return np.array(out)

vmap_select_type_mid = jax.vmap(select_type_mid, in_axes=(0, None))


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


def smooth_trial(X_trial, num_phases, gridsize_Nx, sigma, num_grid_points):
    smoothed_data_trial = np.zeros((gridsize_Nx,gridsize_Nx,num_phases))
    for phase in range(num_phases):
        trial_response = X_trial[phase*num_grid_points:(phase+1)*num_grid_points]
        trial_response = trial_response.reshape(gridsize_Nx,gridsize_Nx)
        smoothed_data_trial_at_phase =  gaussian_filter_jax(trial_response, sigma = sigma)
        smoothed_data_trial=smoothed_data_trial.at[:,:, phase].set(smoothed_data_trial_at_phase)
    return smoothed_data_trial

vmap_smooth_trial = jax.vmap(smooth_trial, in_axes=(0, None, None, None, None))


def smooth_data(X, gridsize_Nx =9, sigma = 1):
    '''
    Smooth data matrix (trials x cells or single trial) over grid points. Data is reshaped into 9x9 grid before smoothing and then flattened again.
    '''
    # if it is a single trial, add a batch dimension for vectorization
    original_dim = X.ndim
    if original_dim == 1:
        X = X.reshape(1, -1) 
    num_grid_points=gridsize_Nx*gridsize_Nx
    num_phases = int(X.shape[1]/num_grid_points)

    smoothed_data = vmap_smooth_trial(X, num_phases, gridsize_Nx, sigma, num_grid_points)

    # if it was a single trial, remove the batch dimension before returning
    if original_dim == 1:
        smoothed_data = smoothed_data.reshape(-1, num_grid_points)
    
    return smoothed_data


def load_orientation_map(folder, run_ind):
    '''Loads the orientation map from the folder for the training indexed by run_ind.'''
    orimap_filename = os.path.join(folder, f"orimap_{run_ind}.npy")
    orimap = np.load(orimap_filename)
    return orimap


def vmap_model_response(untrained_pars, ori, n_noisy_trials = 100, J_2x2_m = None, J_2x2_s = None, c_E = None, c_I = None, f_E = None, f_I = None):
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

    return r_mid, r_sup


def SGD_step_indices(df, num_indices=2):
    # get the number of rows in the dataframe
    num_SGD_steps = len(df)
    SGD_step_inds = numpy.zeros(num_indices, dtype=int)
    if num_indices>2:
        SGD_step_inds[0]=df.index[df['stage'] == 0][0] #index of when pretraining starts
        SGD_step_inds[1]= df.index[df['stage'] == 0][-1]+1 #index of when training starts
        SGD_step_inds[-1]=num_SGD_steps-1 #index of when training ends
        # Fill in the rest of the indices with equidistant points between end of pretraining and end of training
        for i in range(2,num_indices-1):
            SGD_step_inds[i] = int(SGD_step_inds[1] + (SGD_step_inds[-1]-SGD_step_inds[1])*(i-1)/(num_indices-2))
    else:
        SGD_step_inds[0]= df.index[df['stage'] == 2][0] # index of when training starts (second stage of training)
        SGD_step_inds[-1]=num_SGD_steps-1 #index of when training ends    
    return SGD_step_inds


def select_response(responses, sgd_step, layer, ori):
    '''
    Selects the response for a given sgd_step, layer and ori from the responses dictionary. If the dictionary has ref and target responses, it returns the difference between them.
    The response is the output from filtered_model_response or filtered_model_response_task functions.    
    '''
    step_mask = responses['SGD_step'] == sgd_step
    if ori is None:
        ori_mask = responses['ori'] >-1
        ori_out = responses['ori'][step_mask]
    else:
        ori_mask = responses['ori'] == ori
    combined_mask = step_mask & ori_mask
    # fine discrimination task
    if len(responses)>4:
        if layer == 0:
            response = responses['r_sup_ref'][combined_mask] - responses['r_sup_target'][combined_mask]
        else:
            response = responses['r_mid_ref'][combined_mask] - responses['r_mid_target'][combined_mask]
        labels = responses['labels'][combined_mask]
    # crude discrimination task
    else:
        if layer == 0:
            response_sup = responses['r_sup']
            response = response_sup[combined_mask]
        else:
            response_mid = responses['r_mid']
            response = response_mid[combined_mask]
        labels = None
    if ori is None:
        return response, labels, ori_out
    else:
        return response, labels


def filtered_model_response(folder,run_ind, ori_list= np.asarray([55, 125, 0]), num_noisy_trials = 100, num_SGD_inds = 2, r_noise=False, sigma_filter = 1):
    file_name = f"{folder}/results_{run_ind}.csv"
    loaded_orimap = load_orientation_map(folder, run_ind)

    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                    loss_pars, training_pars, pretrain_pars, readout_pars, None, orimap_loaded=loaded_orimap)
    df = pd.read_csv(file_name)
    SGD_step_inds = SGD_step_indices(df, num_SGD_inds)

    # Iterate overs SGD_step indices (default is before and after training)
    for step_ind in SGD_step_inds:
        # Load parameters from csv for given epoch
        _, trained_pars_stage2, _ = load_parameters(file_name, iloc_ind = step_ind)
        J_2x2_m = sep_exponentiate(trained_pars_stage2['log_J_2x2_m'])
        J_2x2_s = sep_exponentiate(trained_pars_stage2['log_J_2x2_s'])
        c_E = trained_pars_stage2['c_E']
        c_I = trained_pars_stage2['c_I']
        f_E = np.exp(trained_pars_stage2['log_f_E'])
        f_I = np.exp(trained_pars_stage2['log_f_I'])
        
        # Iterate over the orientations
        for ori in ori_list:
            # Calculate model response for each orientation
            r_mid, r_sup = vmap_model_response(untrained_pars, ori, num_noisy_trials, J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I)
            if r_noise:
                # Add noise to the responses
                noise_mid = generate_noise(num_noisy_trials, length = r_mid.shape[1], num_readout_noise = untrained_pars.num_readout_noise)
                r_mid = r_mid + noise_mid*np.sqrt(jax.nn.softplus(r_mid))
                noise_sup = generate_noise(num_noisy_trials, length = r_sup.shape[1], num_readout_noise = untrained_pars.num_readout_noise)
                r_sup = r_sup + noise_sup*np.sqrt(jax.nn.softplus(r_sup))

            # Smooth data for each celltype separately with Gaussian filter
            filtered_r_mid_EI= smooth_data(r_mid)  #num_noisy_trials x 648
            filtered_r_mid_E=vmap_select_type_mid(filtered_r_mid_EI,'E')
            filtered_r_mid_I=vmap_select_type_mid(filtered_r_mid_EI,'I')
            filtered_r_mid=np.sum(0.8*filtered_r_mid_E + 0.2 *filtered_r_mid_I, axis=-1)# order of summing up phases and mixing I-E matters if we change to sum of squares!

            filtered_r_sup_EI= smooth_data(r_sup, sigma = sigma_filter)
            if filtered_r_sup_EI.ndim == 3:
                filtered_r_sup_E=filtered_r_sup_EI[:,:,0]
                filtered_r_sup_I=filtered_r_sup_EI[:,:,1]
            if filtered_r_sup_EI.ndim == 4:
                filtered_r_sup_E=filtered_r_sup_EI[:,:,:,0]
                filtered_r_sup_I=filtered_r_sup_EI[:,:,:,1]
            filtered_r_sup=0.8*filtered_r_sup_E + 0.2 *filtered_r_sup_I
            
            # Concatenate all orientation responses
            if ori == ori_list[0] and step_ind==SGD_step_inds[0]:
                filtered_r_mid_df = filtered_r_mid
                filtered_r_sup_df = filtered_r_sup
                ori_df = np.repeat(ori, num_noisy_trials)
                step_df = np.repeat(step_ind, num_noisy_trials)
            else:
                filtered_r_mid_df = np.concatenate((filtered_r_mid_df, filtered_r_mid))
                filtered_r_sup_df = np.concatenate((filtered_r_sup_df, filtered_r_sup))
                ori_df = np.concatenate((ori_df, np.repeat(ori, num_noisy_trials)))
                step_df = np.concatenate((step_df, np.repeat(step_ind, num_noisy_trials)))

    output = dict(ori = ori_df, SGD_step = step_df, r_mid = filtered_r_mid_df, r_sup = filtered_r_sup_df )

    return output, SGD_step_inds


def filtered_model_response_task(folder, run_ind, ori_list= np.asarray([55, 125, 0]), num_noisy_trials = 100, num_SGD_inds = 2, r_noise=False, sigma_filter = 1, gridsize_Nx=9):
    '''Calculate filtered model response for each orientation in ori_list and for each parameter set (that come from file_name at num_SGD_inds rows)'''
    file_name = f"{folder}/results_{run_ind}.csv"
    loaded_orimap = load_orientation_map(folder, run_ind)

    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                    loss_pars, training_pars, pretrain_pars, readout_pars, None, orimap_loaded=loaded_orimap)
    df = pd.read_csv(file_name)
    SGD_step_inds = SGD_step_indices(df, num_SGD_inds)

    # Iterate overs SGD_step indices (default is before and after training)
    for step_ind in SGD_step_inds:
        # Load parameters from csv for given epoch
        _, trained_pars_stage2, _ = load_parameters(file_name, iloc_ind = step_ind)
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
            test_gratings = create_grating_training(untrained_pars.stimuli_pars, num_noisy_trials,untrained_pars.BW_image_jax_inp)
           
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
            if r_noise:
                noise_mid_ref = generate_noise(num_noisy_trials, length = r_mid_ref.shape[1], num_readout_noise = untrained_pars.num_readout_noise)
                noise_mid_target = generate_noise(num_noisy_trials, length = r_mid_target.shape[1], num_readout_noise = untrained_pars.num_readout_noise)
                r_mid_ref_noisy = r_mid_ref + noise_mid_ref*np.sqrt(jax.nn.softplus(r_mid_ref))
                r_mid_target_noisy = r_mid_target + noise_mid_target*np.sqrt(jax.nn.softplus(r_mid_target))
                noise_sup_ref = generate_noise(num_noisy_trials, length = r_sup_ref.shape[1], num_readout_noise = untrained_pars.num_readout_noise)
                noise_sup_target = generate_noise(num_noisy_trials, length = r_sup_target.shape[1], num_readout_noise = untrained_pars.num_readout_noise)
                r_sup_ref_noisy = r_sup_ref + noise_sup_ref*np.sqrt(jax.nn.softplus(r_sup_ref))
                r_sup_target_noisy = r_sup_target + noise_sup_target*np.sqrt(jax.nn.softplus(r_sup_target))
            else:
                r_mid_ref_noisy = r_mid_ref 
                r_mid_target_noisy = r_mid_target
                r_sup_ref_noisy = r_sup_ref 
                r_sup_target_noisy = r_sup_target

            # Smooth responses for each celltype, layer and stimulus type (ref or target) separately with Gaussian filter
            filtered_r_mid_ref_EI= smooth_data(r_mid_ref_noisy,gridsize_Nx,sigma_filter)  #n_noisy_trials x 648
            filtered_r_mid_ref_E = vmap_select_type_mid(filtered_r_mid_ref_EI,'E')
            filtered_r_mid_ref_I = vmap_select_type_mid(filtered_r_mid_ref_EI,'I')
            filtered_r_mid_ref = np.sum(0.8*filtered_r_mid_ref_E + 0.2 *filtered_r_mid_ref_I, axis=-1) # sum up along phases - should it be sum of squares?
            filtered_r_mid_target_EI= smooth_data(r_mid_target_noisy,gridsize_Nx,sigma_filter)  #n_noisy_trials x 648
            filtered_r_mid_target_E = vmap_select_type_mid(filtered_r_mid_target_EI,'E')
            filtered_r_mid_target_I = vmap_select_type_mid(filtered_r_mid_target_EI,'I')
            filtered_r_mid_target = np.sum(0.8*filtered_r_mid_target_E + 0.2 *filtered_r_mid_target_I, axis=-1) # order of summing up phases and mixing I-E matters if we change to sum of squares!

            filtered_r_sup_ref_EI = smooth_data(r_sup_ref_noisy,gridsize_Nx,sigma_filter)
            if filtered_r_sup_ref_EI.ndim == 3:
                filtered_r_sup_ref_E = filtered_r_sup_ref_EI[:,:,0]
                filtered_r_sup_ref_I = filtered_r_sup_ref_EI[:,:,1]
            if filtered_r_sup_ref_EI.ndim == 4:
                filtered_r_sup_ref_E = filtered_r_sup_ref_EI[:,:,:,0]
                filtered_r_sup_ref_I = filtered_r_sup_ref_EI[:,:,:,1]
            filtered_r_sup_ref = 0.8*filtered_r_sup_ref_E + 0.2 *filtered_r_sup_ref_I
            filtered_r_sup_target_EI = smooth_data(r_sup_target_noisy,gridsize_Nx,sigma_filter)
            if filtered_r_sup_target_EI.ndim == 3:
                filtered_r_sup_target_E = filtered_r_sup_target_EI[:,:,0]
                filtered_r_sup_target_I = filtered_r_sup_target_EI[:,:,1]
            if filtered_r_sup_target_EI.ndim == 4:
                filtered_r_sup_target_E = filtered_r_sup_target_EI[:,:,:,0]
                filtered_r_sup_target_I = filtered_r_sup_target_EI[:,:,:,1]
            filtered_r_sup_target = 0.8*filtered_r_sup_target_E + 0.2 *filtered_r_sup_target_I

            # Concatenate along orientation responses
            if ori == ori_list[0] and step_ind==SGD_step_inds[0]:
                filtered_r_mid_ref_df = filtered_r_mid_ref
                filtered_r_mid_target_df = filtered_r_mid_target
                filtered_r_sup_ref_df = filtered_r_sup_ref
                filtered_r_sup_target_df = filtered_r_sup_target

                ori_df = np.repeat(ori, num_noisy_trials)
                step_df = np.repeat(step_ind, num_noisy_trials)
                labels = label
            else:
                filtered_r_mid_ref_df = np.concatenate((filtered_r_mid_ref_df, filtered_r_mid_ref))
                filtered_r_mid_target_df = np.concatenate((filtered_r_mid_target_df, filtered_r_mid_target))
                filtered_r_sup_ref_df = np.concatenate((filtered_r_sup_ref_df, filtered_r_sup_ref))
                filtered_r_sup_target_df = np.concatenate((filtered_r_sup_target_df, filtered_r_sup_target))
                ori_df = np.concatenate((ori_df, np.repeat(ori, num_noisy_trials)))
                step_df = np.concatenate((step_df, np.repeat(step_ind, num_noisy_trials)))
                labels = np.concatenate((labels, label))

    # Define output dictionary with keys: ori, SGD_step, r_mid_ref, r_mid_target, r_sup_ref, r_sup_target, labels
    output = dict(ori = ori_df, SGD_step = step_df, r_mid_ref = filtered_r_mid_ref_df, r_mid_target = filtered_r_mid_target_df, r_sup_ref = filtered_r_sup_ref_df , r_sup_target = filtered_r_sup_target_df , labels = labels)

    return output, SGD_step_inds