import jax
import jax.numpy as np
from jax import vmap
import numpy
import pandas as pd
import scipy.stats
from scipy.interpolate import interp1d
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from training.model import vmap_evaluate_model_response, vmap_evaluate_model_response_mid
from training.SSN_classes import SSN_mid, SSN_sup
from training.training_functions import generate_noise
from util import load_parameters, filter_for_run_and_stage, unpack_ssn_parameters, check_header
from training.util_gabor import BW_image_jit_noisy, BW_image_jax_supp, BW_image_jit

############## Analysis functions ##########

def exclude_runs(folder_path, input_vector):
    """Exclude runs from the analysis by removing them from the CSV files."""
    # Read the original CSV file
    folder_path_from_analysis = os.path.join(os.path.dirname(os.path.dirname(__file__)),folder_path)
    file_path = os.path.join(folder_path_from_analysis,'pretraining_results.csv')
    df_pretraining_results = pd.read_csv(file_path)
    file_path = os.path.join(folder_path_from_analysis,'training_results.csv')
    df_training_results = pd.read_csv(file_path)
    file_path = os.path.join(folder_path_from_analysis,'orimap.csv')
    df_orimap = pd.read_csv(file_path)
    file_path = os.path.join(folder_path_from_analysis,'initial_parameters.csv')
    df_init_params = pd.read_csv(file_path)
    
    # Save the original dataframe as results_complete.csv
    df_pretraining_results.to_csv(os.path.join(folder_path_from_analysis,'pretraining_results_complete.csv'), index=False)
    df_training_results.to_csv(os.path.join(folder_path_from_analysis,'training_results_complete.csv'), index=False)
    df_orimap.to_csv(os.path.join(folder_path_from_analysis,'orimap_complete.csv'), index=False)
    df_init_params.to_csv(os.path.join(folder_path_from_analysis,'initial_parameters_complete.csv'), index=False)
    
    # Exclude rows where 'runs' column is in the input_vector
    df_pretraining_results_filtered = df_pretraining_results[~df_pretraining_results['run_index'].isin(input_vector)]
    df_training_results_filtered = df_training_results[~df_training_results['run_index'].isin(input_vector)]
    df_orimap_filtered = df_orimap[~df_orimap['run_index'].isin(input_vector)]
    df_init_params_filtered = df_init_params[~df_init_params['run_index'].isin(input_vector)]

    # Adjust the 'run_index' column
    df_orimap_filtered['run_index'] = range(len(df_orimap_filtered))
    df_init_params_filtered['run_index'][df_init_params_filtered['stage']==0] = range(len(df_orimap_filtered))
    df_init_params_filtered['run_index'][df_init_params_filtered['stage']==1] = range(len(df_orimap_filtered))
    for i in range(df_pretraining_results_filtered['run_index'].max() + 1):
        if i not in input_vector:
            shift_val = sum(x < i for x in input_vector)
            df_pretraining_results_filtered.loc[df_pretraining_results_filtered['run_index'] == i, 'run_index'] = i - shift_val    
            df_training_results_filtered.loc[df_training_results_filtered['run_index'] == i, 'run_index'] = i - shift_val            
    
    # Save the filtered dataframes as csv files
    df_pretraining_results_filtered.to_csv(os.path.join(folder_path_from_analysis,'pretraining_results.csv'), index=False)
    df_training_results_filtered.to_csv(os.path.join(folder_path_from_analysis,'training_results.csv'), index=False)
    df_orimap_filtered.to_csv(os.path.join(folder_path_from_analysis,'orimap.csv'), index=False)
    df_init_params_filtered.to_csv(os.path.join(folder_path_from_analysis,'initial_parameters.csv'), index=False)


def data_from_run(folder, run_index=0, num_indices=3):
    """Read CSV files, filter them for run and return the combined dataframe together with the time indices where stages change."""
    
    pretrain_filepath = os.path.join(folder, 'pretraining_results.csv')
    train_filepath = os.path.join(folder, 'training_results.csv')
    df_pretrain = pd.read_csv(pretrain_filepath)
    df_train = pd.read_csv(train_filepath)
    df_pretrain_i = filter_for_run_and_stage(df_pretrain, run_index)
    df_train_i = filter_for_run_and_stage(df_train, run_index)
    if df_train_i.empty:
        Warning('No data for run_index = {}'.format(run_index))
    df_i = pd.concat((df_pretrain_i,df_train_i))
    df_i.reset_index(inplace=True)
    stage_time_inds = SGD_indices_at_stages(df_i, num_indices)

    return df_i, stage_time_inds


def calc_rel_change_supp(variable, time_start, time_end):
    """Calculate the relative change in a variable between two time points."""
    # Find the first non-None value after training_start and the last non-None value before training_end
    start_value = None
    for i in range(time_start, len(variable)):
        if not pd.isna(variable[i]):
            start_value = variable[i]
            break
    end_value = None
    for j in range(time_end, i, -1):
        if not pd.isna(variable[j]):
            end_value = variable[j]
            break
    
    # Calculate the relative change
    if start_value is None or end_value is None:
        return None
    else:
        if start_value == 0:
            return (end_value - start_value)
        else:
            return 100*(end_value - start_value) / start_value
    

def rel_change_for_run(folder, training_ind=0, num_indices=3):
    """Calculate the relative changes in the parameters for a single run."""
    data, time_inds = data_from_run(folder, training_ind, num_indices)
    training_end = time_inds[-1]
    columns_to_drop = ['stage', 'SGD_steps']  # Replace with the actual column names you want to drop
    data = data.drop(columns=columns_to_drop)
    data['J_I_m'] = numpy.abs(data['J_II_m'] + data['J_EI_m'])
    data['J_I_s'] = numpy.abs(data['J_II_s'] + data['J_EI_s'])
    data['J_E_m'] = numpy.abs(data['J_IE_m'] + data['J_EE_m'])
    data['J_E_s'] = numpy.abs(data['J_IE_s'] + data['J_EE_s'])

    data['EI_ratio_J_m'] = data['J_I_m'] / data['J_E_m']
    data['EI_ratio_J_s'] = data['J_I_s'] / data['J_E_s']
    data['EI_ratio_J_ms'] = (data['J_I_m'] + data['J_I_s']) / (data['J_E_m'] + data['J_E_s'])
    if num_indices == 3:
        pretraining_start = time_inds[0]
        training_start = time_inds[1]-1
        rel_change_pretrain = {}
        for key, value in data.items():
            item_rel_change = calc_rel_change_supp(value, pretraining_start, training_start)
            if item_rel_change is not None:
                rel_change_pretrain[key] = item_rel_change
    else:
        training_start = time_inds[0]
        rel_change_pretrain = None
    rel_change_train = {}
    for key, value in data.items():
        item_rel_change = calc_rel_change_supp(value, training_start, training_end)
        if item_rel_change is not None:
            rel_change_train[key] = item_rel_change 
    return rel_change_train, rel_change_pretrain, time_inds


def rel_change_for_runs(folder, num_indices=3, num_runs=None):
    """Calculate the relative changes in the parameters for all runs."""
    # if os.path.join(folder, 'rel_changes_train.csv') exists, load it and return the data
    if os.path.exists(os.path.join(folder, 'rel_changes_train.csv')):
        rel_changes_train_df = pd.read_csv(os.path.join(folder, 'rel_changes_train.csv'))
        rel_changes_train = {key: rel_changes_train_df[key].to_numpy() for key in rel_changes_train_df.columns}
    else:
        # Initialize the arrays to store the results in
        if num_runs is None:
            filepath = os.path.join(folder, 'pretraining_results.csv')
            df = pd.read_csv(filepath)
            num_runs = df['run_index'].iloc[-1]+1

        # Calculate the relative changes for all runs
        for i in range(num_runs):
            rel_change_train, rel_change_pretrain, _ = rel_change_for_run(folder, i, num_indices)
            if i == 0:
                # Initialize the arrays to store the results in
                rel_changes_train = {key: numpy.zeros(num_runs) for key in rel_change_train.keys()}
                if rel_change_pretrain is not None:
                    rel_changes_pretrain = {key: numpy.zeros(num_runs) for key in rel_change_pretrain.keys()}
                else:
                    rel_changes_pretrain = None
            for key, value in rel_change_train.items():
                rel_changes_train[key][i] = value
                if rel_change_pretrain is not None:
                    rel_changes_pretrain[key][i] = rel_change_pretrain[key]
        
        # Save the results into a csv file
        rel_changes_train_df = pd.DataFrame(rel_changes_train)
        rel_changes_train_df.to_csv(os.path.join(folder, 'rel_changes_train.csv'), index=False)
    
    save_pretrain = False
    
    if num_indices==3 and os.path.exists(os.path.join(os.path.dirname(folder), 'rel_changes_pretrain.csv')):
        rel_changes_pretrain_df = pd.read_csv(os.path.join(os.path.dirname(folder), 'rel_changes_pretrain.csv'))
        rel_changes_pretrain = {key: rel_changes_pretrain_df[key].to_numpy() for key in rel_changes_pretrain_df.columns}
    elif num_indices==3:
        if num_runs is None:
            df = pd.read_csv(os.path.join(os.path.dirname(folder), 'pretraining_results.csv'))
            num_runs = df['run_index'].iloc[-1]+1
        for i in range(num_runs):
            _, rel_change_pretrain, _ = rel_change_for_run(folder, i, num_indices)
            if i == 0:
                rel_changes_pretrain = {key: numpy.zeros(num_runs) for key in rel_change_pretrain.keys()}
            else:
                for key, value in rel_changes_pretrain.items():
                    rel_changes_pretrain[key][i] = rel_change_pretrain[key]
        save_pretrain = True
    else:
        rel_changes_pretrain = None
    
    if save_pretrain:
        rel_changes_pretrain_df = pd.DataFrame(rel_changes_pretrain)
        rel_changes_pretrain_df.to_csv(os.path.join(os.path.dirname(folder), 'rel_changes_pretrain.csv'), index=False)
        
    return rel_changes_train, rel_changes_pretrain
    

def gabor_tuning(untrained_pars, ori_vec=np.arange(0,180,6)):
    """Calculate the responses of the gabor filters to stimuli with different orientations."""
    gabor_filters = untrained_pars.gabor_filters
    num_ori = len(ori_vec)
    # Getting the 'blank' alpha_channel and mask for a full image version stimuli with no background
    BW_image_jax_inp = BW_image_jax_supp(untrained_pars.stimuli_pars, x0 = 0, y0=0, phase=0.0, full_grating=True) 
    alpha_channel = BW_image_jax_inp[6]
    mask = BW_image_jax_inp[7]
    if len(gabor_filters.shape)==2:
        gabor_filters = np.reshape(gabor_filters, (untrained_pars.ssn_pars.phases,2,untrained_pars.grid_pars.gridsize_Nx **2,-1)) # the second dimension 2 is for I and E cells, the last dim is the image size
    
    # Initialize the gabor output array
    gabor_output = numpy.zeros((num_ori, untrained_pars.ssn_pars.phases,2,untrained_pars.grid_pars.gridsize_Nx **2))
    time_start = time.time()
    for grid_ind in range(gabor_filters.shape[2]):
        grid_ind_x = grid_ind//untrained_pars.grid_pars.gridsize_Nx # For the right order, it is important to match how gabors_demean is filled up in create_gabor_filters_ori_map, currently, it is [grid_size_1D*i+j,phases_ind,:], where x0 = x_map[i, j]
        grid_ind_y = grid_ind%untrained_pars.grid_pars.gridsize_Nx
        x0 = untrained_pars.grid_pars.x_map[grid_ind_x, grid_ind_y]
        y0 = untrained_pars.grid_pars.y_map[grid_ind_x, grid_ind_y]
        for phase_ind in range(gabor_filters.shape[0]):
            phase = phase_ind * np.pi/2
            BW_image_jax_inp = BW_image_jax_supp(untrained_pars.stimuli_pars, x0=x0, y0=y0, phase=phase, full_grating=True)
            x = BW_image_jax_inp[4]
            y = BW_image_jax_inp[5]
            stimuli = BW_image_jit(BW_image_jax_inp[0:4], x, y, alpha_channel, mask, ori_vec, np.zeros(num_ori)) 
            for ori in range(num_ori):
                gabor_output[ori,phase_ind,0,grid_ind] = gabor_filters[phase_ind,0,grid_ind,:]@(stimuli[ori,:].T) # E cells
                gabor_output[ori,phase_ind,1,grid_ind] = gabor_filters[phase_ind,1,grid_ind,:]@(stimuli[ori,:].T) # I cells
    print('Time elapsed for gabor_output calculation:', time.time()-time_start)
     
    return gabor_output


def tc_grid_point(inds_maps_flat, ssn_mid, ssn_sup, num_phases, untrained_pars, ori_vec, num_ori, grid_size, cE_m, cI_m, cE_s, cI_s, f_E, f_I):
    """Calculate the responses of the middle and superficial layers to gratings at a single grid point - used for phase matched tuning curve calculation."""
    
    x_ind = inds_maps_flat[0]
    y_ind = inds_maps_flat[1]
    x_grid = inds_maps_flat[2]
    y_grid = inds_maps_flat[3]
    grid_size_1D = np.sqrt(grid_size).astype(int)
    
    # Initialize the arrays to store the responses
    responses_mid_phase_match = np.zeros((num_ori, 2, num_phases))
    responses_sup_phase_match = np.zeros((num_ori, 2))

    # Loop over the different phases
    for phase_ind in range(num_phases):
        phase = phase_ind * np.pi / 2
        
        # Generate stimulus
        BW_image_jax_inp = BW_image_jax_supp(untrained_pars.stimuli_pars, x0=x_grid, y0=y_grid, phase=phase, full_grating=True)
        x = BW_image_jax_inp[4]
        y = BW_image_jax_inp[5]
        alpha_channel = BW_image_jax_inp[6]
        mask = BW_image_jax_inp[7]
        
        stimuli = BW_image_jit(BW_image_jax_inp[0:4], x, y, alpha_channel, mask, ori_vec, np.zeros(num_ori))
        
        # Calculate model response for superficial layer cells (phase-invariant)
        if phase_ind == 0:
            _, [responses_mid, responses_sup], _, _, _, = vmap_evaluate_model_response(ssn_mid, ssn_sup, stimuli, untrained_pars.conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters)
            # Fill in the responses_sup_phase_match array at the indices corresponding to the grid point
            sup_cell_ind = np.array(x_ind*grid_size_1D+y_ind).astype(int)
            responses_sup_phase_match = responses_sup_phase_match.at[:,0].set(responses_sup[:, sup_cell_ind]) # E cell
            responses_sup_phase_match = responses_sup_phase_match.at[:,1].set(responses_sup[:, grid_size+sup_cell_ind]) # I cell
        else:
            # Calculate model response for middle layer cells
            _, responses_mid, _, _, _, _, _ = vmap_evaluate_model_response_mid(ssn_mid, stimuli, untrained_pars.conv_pars, cE_m, cI_m, untrained_pars.gabor_filters)
        # Fill in the responses_mid_phase_match array at the indices corresponding to the grid point
        mid_cell_ind_E = np.array(phase_ind*2*grid_size + x_ind*grid_size_1D+y_ind).astype(int)
        mid_cell_ind_I = np.array(phase_ind*2*grid_size + x_ind*grid_size_1D+y_ind+grid_size).astype(int)
        responses_mid_phase_match = responses_mid_phase_match.at[:, 0, phase_ind].set(responses_mid[:, mid_cell_ind_E])
        responses_mid_phase_match = responses_mid_phase_match.at[:, 1, phase_ind].set(responses_mid[:, mid_cell_ind_I])
    
    return responses_mid_phase_match, responses_sup_phase_match

# Vectorizing over the flattened indices of the grid
vmapped_tc_grid_point = vmap(tc_grid_point, in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None))


def tuning_curve(untrained_pars, trained_pars, file_path=None, ori_vec=np.arange(0,180,6), training_stage=1, run_index=0, header = False):
    """ Calculate responses of middle and superficial layers to gratings (of full images without added noise) with different orientations."""
    # Get the parameters from the trained_pars dictionary and untreatned_pars class
    ref_ori_saved = float(untrained_pars.stimuli_pars.ref_ori)
    J_2x2_m, J_2x2_s, cE_m, cI_m, cE_s, cI_s, f_E, f_I, kappa = unpack_ssn_parameters(trained_pars, untrained_pars.ssn_pars)
    ssn_pars = untrained_pars.ssn_pars
    num_ori = len(ori_vec)
    x_map = untrained_pars.grid_pars.x_map
    y_map = untrained_pars.grid_pars.y_map
    grid_size = x_map.shape[0]*x_map.shape[1]
    
    # Define the SSN layers
    ssn_mid = SSN_mid(ssn_pars, untrained_pars.grid_pars, J_2x2_m)
    ssn_sup = SSN_sup(ssn_pars, untrained_pars.grid_pars, J_2x2_s, untrained_pars.dist_from_single_ori, untrained_pars.ori_dist, kappa)
    
    # Flatten the grid indices and the x, y coordinates for vmap    
    num_rows, num_cols = x_map.shape
    i_indices, j_indices = numpy.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
    inds_maps_flat_0 = i_indices.flatten()  # i index (row index)
    inds_maps_flat_1 = j_indices.flatten()  # j index (column index)
    inds_maps_flat_2 = x_map.flatten()
    inds_maps_flat_3 = y_map.flatten()
    inds_maps_flat = np.stack((inds_maps_flat_0, inds_maps_flat_1, inds_maps_flat_2, inds_maps_flat_3), axis=1)

    responses_mid_phase_match, responses_sup_phase_match = vmapped_tc_grid_point(inds_maps_flat, ssn_mid, ssn_sup, ssn_pars.phases, untrained_pars, ori_vec, num_ori, grid_size, cE_m, cI_m, cE_s, cI_s, f_E, f_I)
    
    # rearrange responses to 2D, where first dim is oris and second dim is cells
    responses_sup_phase_match_2D = numpy.zeros((num_ori,grid_size*2))
    responses_mid_phase_match_2D = numpy.zeros((num_ori,grid_size*ssn_pars.phases*2))
    for i in range(num_ori):
        responses_sup_phase_match_2D[i,:] = responses_sup_phase_match[:, i, :].flatten(order='F')
        responses_mid_phase_match_2D[i,:] = responses_mid_phase_match[:, i, :, :].flatten(order='F')

    # Save responses into csv file - overwrite the file if it already exists
    if file_path is not None:
        if os.path.exists(file_path) and header is not False:
            Warning('Tuning curve csv file will get multiple headers and will possibly have repeated rows!')
        # repeat training_stage run_index and expand dimension to add as the first two columns of the new_rows
        run_index_vec = numpy.repeat(run_index, num_ori)
        training_stage_vec = numpy.repeat(training_stage, num_ori)
        run_index_vec = numpy.expand_dims(run_index_vec, axis=1)
        training_stage_vec = numpy.expand_dims(training_stage_vec, axis=1)
        
        responses_combined=np.concatenate((responses_mid_phase_match_2D, responses_sup_phase_match_2D), axis=1)
        new_rows = numpy.concatenate((run_index_vec, training_stage_vec, responses_combined), axis=1)
        new_rows_df = pd.DataFrame(new_rows)
        new_rows_df.to_csv(file_path, mode='a', header=header, index=False, float_format='%.4f')

    untrained_pars.stimuli_pars.ref_ori = ref_ori_saved

    return responses_sup_phase_match_2D, responses_mid_phase_match_2D


def full_width_half_max(vector, d_theta):
    """ Calculate width of tuning curve at half-maximum. This method should not be applied when tuning curve has multiple bumps. """
    # Remove baseline, calculate half-max
    vector = vector-vector.min()
    half_height = vector.max()/2

    # Get the number of points above half-max and multiply it by the delta angle to get width in angle
    points_above = len(vector[vector>half_height])
    distance = d_theta * points_above
    
    return distance


def tc_cubic(x,y, d_theta=0.5):
    """ Cubic interpolation of tuning curve data. """

    # add first value as last value to make the interpolation periodic
    x = numpy.append(x, 180)
    y = numpy.append(y, y[0])

    # Create cubic interpolation object
    cubic_interpolator = interp1d(x, y, kind='cubic')

    # Create new x values and get interpolated values
    x_new = numpy.arange(0, 180, d_theta)
    y_new = cubic_interpolator(x_new)

    return x_new, y_new


def save_tc_features(training_tc_file, num_runs=1, ori_list=np.arange(0,180,6), ori_to_center_slope=[55, 125], stages=[1, 2], filename='tuning_curve_features.csv'):
    """
    Calls compute_features for each stage and run index and saves the results into a CSV file.
    """

    # Load training tuning curve data (no headers)
    header_flag = check_header(training_tc_file)
    training_tc_all_run_df = pd.read_csv(training_tc_file, header=header_flag)
    training_tc_all_run = training_tc_all_run_df.to_numpy()

    if header_flag is None:
        # get header from the pretraining tuning curve file
        pretraining_tc_file = os.path.join(os.path.dirname(os.path.dirname(training_tc_file)), 'pretraining_tuning_curves.csv')
        pretraining_tc_all_run_df = pd.read_csv(pretraining_tc_file, header=0)
        headers = pretraining_tc_all_run_df.columns  # Use this for cell headers
    else:
        headers = training_tc_all_run_df.columns
    cell_headers = headers[2:]  # The headers for the 810 cells

    feature_rows = []
    num_cells = training_tc_all_run.shape[1] - 2  # Subtracting run_index and stage
    for run_index in range(num_runs):
        # Filtering tuning curves for run_index
        run_mask = training_tc_all_run[:, 0] == run_index
        tuning_curves_run = training_tc_all_run[run_mask, 1:]
        for stage in stages:        
            # Filtering tuning curves for stage
            stage_mask = tuning_curves_run[:, 0] == stage
            tuning_curves = tuning_curves_run[stage_mask, 1:]

            # Skip if there is no tc data for this stage
            if tuning_curves.shape[0] == 0:
                continue

            # Calculate tuning curve features
            features = compute_features(tuning_curves, num_cells, ori_list, ori_to_center_slope)

            # Prepare data for DataFrame insertion
            for feature_name, feature_values in features.items():
                row_data = {'run_index': run_index, 'stage': stage, 'feature': feature_name}
                row_data.update({cell: value for cell, value in zip(cell_headers, feature_values)})
                feature_rows.append(row_data)

    # Create DataFrame from rows and save to CSV
    tc_features_df = pd.DataFrame(feature_rows)
    output_filename = os.path.join(os.path.dirname(training_tc_file), filename)
    tc_features_df.to_csv(output_filename, index=False)

    return tc_features_df


def compute_features(tuning_curves, num_cells, ori_list, oris_to_calc_slope_at, d_theta_interp=0.2):
    """
    Computes tuning curve features for each cell.
    """
    # Initialize feature arrays
    features = {
        'fwhm': numpy.zeros(num_cells),
        'slope_55': numpy.zeros(num_cells),
        'slope_125': numpy.zeros(num_cells),
        'pref_ori': numpy.zeros(num_cells),
        'min': numpy.zeros(num_cells),
        'max': numpy.zeros(num_cells),
        'max_min_ratio': numpy.zeros(num_cells),
        'mean': numpy.zeros(num_cells),
        'std': numpy.zeros(num_cells),
        'slope_hm': numpy.zeros(num_cells)
    }

    for i in range(num_cells):
        tc_i = tuning_curves[:, i]

        # Perform cubic interpolation
        x_interp, y_interp = tc_cubic(ori_list, tc_i, d_theta_interp)

        # Full width half max
        features['fwhm'][i] = full_width_half_max(y_interp, d_theta=d_theta_interp)

        # Preferred orientation
        features['pref_ori'][i] = x_interp[numpy.argmax(y_interp)]

        # Gradient for slope calculations
        y_interp_scaled = y_interp/numpy.max(y_interp)
        grad_y_interp_scaled = numpy.gradient(y_interp_scaled, x_interp)

        # Slope at 55 and 125 degrees
        features['slope_55'][i] = grad_y_interp_scaled[numpy.argmin(numpy.abs(x_interp - oris_to_calc_slope_at[0]))]
        features['slope_125'][i] = grad_y_interp_scaled[numpy.argmin(numpy.abs(x_interp - oris_to_calc_slope_at[1]))]

        # Statistics: min, max, max-min ratio, mean, std
        features['min'][i] = numpy.min(y_interp)
        features['max'][i] = numpy.max(y_interp)
        features['max_min_ratio'][i] = (features['max'][i] - features['min'][i]) / features['max'][i] if features['max'][i] != 0 else 0
        features['mean'][i] = numpy.mean(y_interp)
        features['std'][i] = numpy.std(y_interp)

        # Slope at half-max
        half_max = (features['max'][i] + features['min'][i]) / 2
        features['slope_hm'][i] = grad_y_interp_scaled[numpy.argmin(numpy.abs(y_interp - half_max))]

    return features


def param_offset_correlations(folder, num_time_inds=2):
    """ Calculate the Pearson correlation coefficient between the offset threshold and the parameters."""
    
    # Helper function to calculate Pearson correlation
    def calculate_correlations(main_var, params, result_dict):
        for _, main_value in main_var.items():
            for param_key, param_values in params.items():
                corr, p_value = scipy.stats.pearsonr(main_value, param_values)
                result_dict[param_key] = [corr, p_value]
        return result_dict

    # Load the relative changes and drop items with NaN values
    rel_changes_train, _ = rel_change_for_runs(folder, num_indices=num_time_inds)
    
    # Separate offsets, losses, and params
    offsets_rel_change = {key: value for key, value in rel_changes_train.items() if key.endswith('_offset')}
    params_keys = ['J_', 'c', 'f_', 'kappa', 'EI_ratio_J_']
    params_rel_change = {key: value for key, value in rel_changes_train.items() if any([key.startswith(param) for param in params_keys])}
    
    # Correlation results dictionaries
    corr_psychometric_offset_param = {}
    corr_staircase_offset_param = {}
    
    # Calculate correlations for offset parameters
    for offset_key, offset_values in offsets_rel_change.items():
        result_dict = corr_psychometric_offset_param if offset_key == 'psychometric_offset' else corr_staircase_offset_param
        result_dict = calculate_correlations({offset_key: offset_values}, params_rel_change, result_dict)
    
    # Calculate correlations for loss parameters
    corr_loss_binary_param = {}
    corr_loss_binary_param = calculate_correlations({'loss_binary': rel_changes_train['loss_binary_cross_entr']}, params_rel_change, corr_loss_binary_param)

    return corr_psychometric_offset_param, corr_staircase_offset_param, corr_loss_binary_param, rel_changes_train


def MVPA_param_offset_correlations(folder, num_time_inds=3, x_labels=None):
    """ Calculate the Pearson correlation coefficient between the offset threshold, the parameter differences and the MVPA scores."""
    data, _ = rel_change_for_runs(folder, num_indices=num_time_inds)
    ##################### Correlate offset_th_diff with the combintation of the J_m and J_s, etc. #####################      
    offset_pars_corr = []
    offset_staircase_pars_corr = []
    if x_labels is None:
        x_labels = ['J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'f_E','f_I', 'cE_m', 'cI_m', 'cE_s', 'cI_s']
    for i in range(len(x_labels)):
        # Calculate the Pearson correlation coefficient and the p-value
        corr, p_value = scipy.stats.pearsonr(data['psychometric_offset'], data[x_labels[i]])
        offset_pars_corr.append({'corr': corr, 'p_value': p_value})
        corr, p_value = scipy.stats.pearsonr(data['staircase_offset'], data[x_labels[i]])
        offset_staircase_pars_corr.append({'corr': corr, 'p_value': p_value})
    
    # Load MVPA_scores and correlate them with the offset threshold and the parameter differences (samples are the different trainings)
    MVPA_scores = pd.read_csv(folder + '/MVPA_scores.csv').to_numpy() # num_trainings x layer x SGD_ind x ori_ind
    MVPA_scores_diff = MVPA_scores[:,:,1,:] - MVPA_scores[:,:,-1,:] # num_trainings x layer x ori_ind
    MVPA_offset_corr = []
    for i in range(MVPA_scores_diff.shape[1]):
        for j in range(MVPA_scores_diff.shape[2]):
            corr, p_value = scipy.stats.pearsonr(data['staircase_offset'], MVPA_scores_diff[:,i,j])
            MVPA_offset_corr.append({'corr': corr, 'p_value': p_value})
    MVPA_pars_corr = [] # (J_m_I,J_m_E,J_s_I,J_s_E,f_E,f_I,cE_m,cI_m,cE_s,cI_s) x ori_ind
    for j in range(MVPA_scores_diff.shape[2]):
        for i in range(MVPA_scores_diff.shape[1]):        
            if i==0:
                corr_m_J_I, p_val_m_J_I = scipy.stats.pearsonr(data['J_II_m']+data['J_EI_m'], MVPA_scores_diff[:,i,j])
                corr_m_J_E, p_val_m_J_E = scipy.stats.pearsonr(data['J_EE_m']+data['J_IE_m'], MVPA_scores_diff[:,i,j])
                corr_m_f_E, p_val_m_f_E = scipy.stats.pearsonr(data['f_E'], MVPA_scores_diff[:,i,j])
                corr_m_f_I, p_val_m_f_I = scipy.stats.pearsonr(data['f_I'], MVPA_scores_diff[:,i,j])
                corr_m_cE_m, p_val_m_cE_m = scipy.stats.pearsonr(data['cE_m'], MVPA_scores_diff[:,i,j])
                corr_m_cI_m, p_val_m_cI_m = scipy.stats.pearsonr(data['cI_m'], MVPA_scores_diff[:,i,j])
            if i==1:
                corr_s_J_I, p_val_s_J_I = scipy.stats.pearsonr(data['J_II_s']+data['J_EI_s'], MVPA_scores_diff[:,i,j])
                corr_s_J_E, p_val_s_J_E = scipy.stats.pearsonr(data['J_EE_s']+data['J_IE_s'], MVPA_scores_diff[:,i,j])                
                corr_s_f_E, p_val_s_f_E = scipy.stats.pearsonr(data['f_E'], MVPA_scores_diff[:,i,j])
                corr_s_f_I, p_val_s_f_I = scipy.stats.pearsonr(data['f_I'], MVPA_scores_diff[:,i,j])
                corr_s_cE_s, p_val_s_cE_s = scipy.stats.pearsonr(data['cE_s'], MVPA_scores_diff[:,i,j])
                corr_s_cI_s, p_val_s_cI_s = scipy.stats.pearsonr(data['cI_s'], MVPA_scores_diff[:,i,j])
            
        corr = [corr_m_J_E, corr_m_J_I, corr_s_J_E, corr_s_J_I, corr_m_f_E, corr_m_f_I, corr_m_cE_m, corr_m_cI_m, corr_s_f_E, corr_s_f_I, corr_s_cE_s, corr_s_cI_s]
        p_value = [p_val_m_J_E, p_val_m_J_I, p_val_s_J_E, p_val_s_J_I, p_val_m_f_E, p_val_m_f_I, p_val_m_cE_m, p_val_m_cI_m, p_val_s_f_E, p_val_s_f_I, p_val_s_cE_s, p_val_s_cI_s]
        MVPA_pars_corr.append({'corr': corr, 'p_value': p_value})
    # combine MVPA_offset_corr and MVPA_pars_corr into a single list
    MVPA_corrs = MVPA_offset_corr + MVPA_pars_corr

    return offset_pars_corr, offset_staircase_pars_corr, MVPA_corrs, data  # Returns a list of dictionaries for each training run

############################## helper functions for MVPA and Mahal distance analysis ##############################

def select_type_mid(r_mid, cell_type='E', phases=4):
    """Selects the excitatory or inhibitory cell responses. This function assumes that r_mid is 3D (trials x grid points x celltype and phase)"""
    if cell_type=='E':
        map_numbers = np.arange(1, 2 * phases, 2)-1 # 0,2,4,6
    else:
        map_numbers = np.arange(2, 2 * phases + 1, 2)-1 # 1,3,5,7
    
    out = np.zeros((r_mid.shape[0],r_mid.shape[1], int(r_mid.shape[2]/2)))
    for i in range(len(map_numbers)):
        out = out.at[:,:,i].set(r_mid[:,:,map_numbers[i]])

    return np.array(out)

vmap_select_type_mid = jax.vmap(select_type_mid, in_axes=(0, None, None))


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
    """Smooths a single trial of responses over grid points."""
    smoothed_data_trial = np.zeros((gridsize_Nx,gridsize_Nx,num_phases))
    for phase in range(num_phases):
        trial_response = X_trial[phase*num_grid_points:(phase+1)*num_grid_points]
        trial_response = trial_response.reshape(gridsize_Nx,gridsize_Nx)
        smoothed_data_trial_at_phase =  gaussian_filter_jax(trial_response, sigma = sigma)
        smoothed_data_trial=smoothed_data_trial.at[:,:, phase].set(smoothed_data_trial_at_phase)
    return smoothed_data_trial

vmap_smooth_trial = jax.vmap(smooth_trial, in_axes=(0, None, None, None, None))


def smooth_data(X, gridsize_Nx, sigma = 1):
    """ Smooth data matrix (trials x cells or single trial) over grid points. Data is reshaped into 9x9 grid before smoothing and then flattened again. """
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


def vmap_model_response(untrained_pars, ori, n_noisy_trials = 100, J_2x2_m = None, J_2x2_s = None, cE_m = None, cI_m = None, cE_s=None, cI_s=None, f_E = None, f_I = None, kappa=None):
    """Generate model response for a given orientation and noise level using vmap_evaluate_model_response."""
    # Generate noisy data
    ori_vec = np.repeat(ori, n_noisy_trials)
    jitter_vec = np.repeat(0, n_noisy_trials)
    x = untrained_pars.BW_image_jax_inp[4]
    y = untrained_pars.BW_image_jax_inp[5]
    alpha_channel = untrained_pars.BW_image_jax_inp[6]
    mask = untrained_pars.BW_image_jax_inp[7]
    
    # Generate data
    test_grating = BW_image_jit_noisy(untrained_pars.BW_image_jax_inp[0:4], x, y, alpha_channel, mask, ori_vec, jitter_vec)
    
    # Create middle and superficial SSN layers
    ssn_mid=SSN_mid(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_m)
    ssn_sup=SSN_sup(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_s, untrained_pars.dist_from_single_ori, untrained_pars.ori_dist, kappa=kappa)

    # Calculate fixed point for data    
    _, [r_mid, r_sup], _,  _, _ = vmap_evaluate_model_response(ssn_mid, ssn_sup, test_grating, untrained_pars.conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters)

    return r_mid, r_sup


def SGD_indices_at_stages(df, num_indices=2, peak_offset_flag=False):
    """Get the indices of the SGD steps at the end (and at the beginning if num_indices=3) of pretraining and at the end of training."""
    # get the number of rows in the dataframe
    num_SGD_steps = len(df)
    SGD_step_inds = numpy.zeros(num_indices, dtype=int)
    if num_indices>2:
        SGD_step_inds[0] = df.index[df['stage'] == 0][0] #index of when pretraining starts
        training_start = df.index[df['stage'] == 0][-1] + 1 #index of when training starts
        if peak_offset_flag:
            # get the index where max offset is reached 
            SGD_step_inds[1] = training_start + df['staircase_offset'][training_start:training_start+100].idxmax()
        else:    
            SGD_step_inds[1] = training_start
        SGD_step_inds[-1] = num_SGD_steps-1 #index of when training ends
        # Fill in the rest of the indices with equidistant points between end of pretraining and end of training
        for i in range(2,num_indices-1):
            SGD_step_inds[i] = int(SGD_step_inds[1] + (SGD_step_inds[-1]-SGD_step_inds[1])*(i-1)/(num_indices-2))
    else:
        SGD_step_inds[0] = df.index[df['stage'] > 0][0] # index of when training starts (first or second stages)
        SGD_step_inds[-1] = num_SGD_steps-1 #index of when training ends    
    return SGD_step_inds


################### Functions for MVPA and Mahalanobis distance analysis ###################

def select_response(responses, stage_ind, layer, ori):
    """
    Selects the response for a given sgd_step, layer and ori from the responses dictionary. If the dictionary has ref and target responses, it returns the difference between them.
    The response is the output from filtered_model_response or filtered_model_response_task functions.    
    """
    step_mask = responses['stage'] == stage_ind
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


def mahal(X,Y):
    """
    D2 = MAHAL(Y,X) returns the Mahalanobis distance (in squared units) of
    each observation (point) in Y from the sample data in X, i.e.,
    D2(I) = (Y(I,:)-MU) * SIGMA^(-1) * (Y(I,:)-MU)',
    where MU and SIGMA are the sample mean and covariance of the data in X.
    Rows of Y and X correspond to observations, and columns to variables.
    """

    rx, _ = X.shape
    ry, _ = Y.shape

    # Subtract the mean from X
    m = np.mean(X, axis=0)
    X_demean = X - m

    # Perform QR decomposition of X_demean
    Q, R = np.linalg.qr(X_demean, mode='reduced')

    # Solve for ri in the equation R' * ri = (Y-M) using least squares or directly if R is square and of full rank
    Y_demean=(Y-m).T
    ri = numpy.linalg.solve(R.T, Y_demean)

    # Calculate d as the sum of squares of ri, scaled by (rx-1)
    d = np.sum(ri**2, axis=0) * (rx-1)

    return np.sqrt(d)


def filtered_model_response(folder, run_ind, ori_list= np.asarray([55, 125, 0]), num_noisy_trials = 100, num_stage_inds = 2, r_noise=True, sigma_filter = 1, noise_std=1.0):
    """
    Calculate filtered model response for each orientation in ori_list and for each parameter set (that come from file_name at num_SGD_inds rows)
    """
    from parameters import ReadoutPars
    readout_pars = ReadoutPars()

    # Iterate overs SGD_step indices (default is before and after training)
    iloc_ind_vec=[0,-1,-1]
    if num_stage_inds==2:
        stages = [0,2]
    else:
        stages = [0,0,2]
    for stage_ind in range(len(stages)):
        stage = stages[stage_ind]
        # Load parameters from csv for given epoch
        _, trained_pars_stage2, untrained_pars = load_parameters(folder, run_index=run_ind, stage=stage, iloc_ind = iloc_ind_vec[stage_ind])
        # Get the parameters from the trained_pars dictionary and untreatned_pars class
        J_2x2_m, J_2x2_s, cE_m, cI_m, cE_s, cI_s, f_E, f_I, kappa = unpack_ssn_parameters(trained_pars_stage2, untrained_pars.ssn_pars)
        
        # Iterate over the orientations
        for ori in ori_list:
            # Calculate model response for each orientation
            r_mid, r_sup = vmap_model_response(untrained_pars, ori, num_noisy_trials, J_2x2_m, J_2x2_s, cE_m, cI_m, cE_s, cI_s, f_E, f_I, kappa)
            if r_noise:
                # Add noise to the responses
                noise_mid = generate_noise(num_noisy_trials, length = r_mid.shape[1], num_readout_noise = untrained_pars.num_readout_noise)
                r_mid = r_mid + noise_mid*np.sqrt(jax.nn.softplus(r_mid))
                noise_sup = generate_noise(num_noisy_trials, length = r_sup.shape[1], num_readout_noise = untrained_pars.num_readout_noise)
                r_sup = r_sup + noise_sup*np.sqrt(jax.nn.softplus(r_sup))

            # Smooth data for each celltype separately with Gaussian filter
            filtered_r_mid_EI= smooth_data(r_mid, untrained_pars.grid_pars.gridsize_Nx, sigma_filter)  #num_noisy_trials x 648
            filtered_r_mid_E=vmap_select_type_mid(filtered_r_mid_EI,'E',4)
            filtered_r_mid_I=vmap_select_type_mid(filtered_r_mid_EI,'I',4)
            filtered_r_mid=np.sum(0.8*filtered_r_mid_E + 0.2 *filtered_r_mid_I, axis=-1)# order of summing up phases and mixing I-E matters if we change to sum of squares!

            filtered_r_sup_EI= smooth_data(r_sup, untrained_pars.grid_pars.gridsize_Nx, sigma_filter)
            if filtered_r_sup_EI.ndim == 3:
                filtered_r_sup_E=filtered_r_sup_EI[:,:,0]
                filtered_r_sup_I=filtered_r_sup_EI[:,:,1]
            if filtered_r_sup_EI.ndim == 4:
                filtered_r_sup_E=filtered_r_sup_EI[:,:,:,0]
                filtered_r_sup_I=filtered_r_sup_EI[:,:,:,1]
            filtered_r_sup=0.8*filtered_r_sup_E + 0.2 *filtered_r_sup_I
            
            # Divide the responses by the std of the responses for equal noise effect ***
            filtered_r_mid = filtered_r_mid / np.std(filtered_r_mid)
            filtered_r_sup = filtered_r_sup / np.std(filtered_r_sup)

            # Concatenate all orientation responses
            if ori == ori_list[0] and stage_ind==0:
                filtered_r_mid_df = filtered_r_mid
                filtered_r_sup_df = filtered_r_sup
                ori_df = np.repeat(ori, num_noisy_trials)
                stage_df = np.repeat(stage_ind, num_noisy_trials)
            else:
                filtered_r_mid_df = np.concatenate((filtered_r_mid_df, filtered_r_mid))
                filtered_r_sup_df = np.concatenate((filtered_r_sup_df, filtered_r_sup))
                ori_df = np.concatenate((ori_df, np.repeat(ori, num_noisy_trials)))
                stage_df = np.concatenate((stage_df, np.repeat(stage_ind, num_noisy_trials)))

    # Get the responses from the center of the grid 
    min_ind = int((readout_pars.readout_grid_size[0] - readout_pars.readout_grid_size[1])/2)
    max_ind = int(readout_pars.readout_grid_size[0]) - min_ind
    filtered_r_mid_box_df = filtered_r_mid_df[:,min_ind:max_ind, min_ind:max_ind]
    filtered_r_sup_box_df = filtered_r_sup_df[:,min_ind:max_ind, min_ind:max_ind]

    # Add noise to the responses - scaled by the std of the responses
    filtered_r_mid_box_noisy_df= filtered_r_mid_box_df + numpy.random.normal(0, noise_std, filtered_r_mid_box_df.shape)
    filtered_r_sup_box_noisy_df= filtered_r_sup_box_df + numpy.random.normal(0, noise_std, filtered_r_sup_box_df.shape)

    output = dict(ori = ori_df, stage = stage_df, r_mid = filtered_r_mid_box_noisy_df, r_sup = filtered_r_sup_box_noisy_df)
    
    return output

def LMI_Mahal_df(num_training, num_layers, num_SGD_inds, mahal_train_control_mean, mahal_untrain_control_mean, mahal_within_train_mean, mahal_within_untrain_mean, train_SNR_mean, untrain_SNR_mean, LMI_across, LMI_within, LMI_ratio):
    """
    Create dataframes for Mahalanobis distance and LMI values
    """
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

    return df_mahal, df_LMI


######### Calculate MVPA and Mahalanobis distance for before pretraining, after pretraining and after training #########
def MVPA_Mahal_analysis(folder, num_runs, num_stages=2, r_noise = True, sigma_filter=1, num_noisy_trials=100, plot_flag=False):
    # Shared parameters
    ori_list = numpy.asarray([55, 125, 0])
    num_layers=2 # number of layers

    ####### Setup for the MVPA analysis #######
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)) # SVM classifier

    # Initialize the MVPA scores matrix
    MVPA_scores = numpy.zeros((num_runs, num_layers, num_stages, len(ori_list)-1))
    Mahal_scores = numpy.zeros((num_runs, num_layers, num_stages, len(ori_list)-1))

    ####### Setup for the Mahalanobis distance analysis #######
    num_PC_used=20 # number of principal components used for the analysis
    
    # Initialize arrays to store Mahalanobis distances and related metrics
    LMI_across = numpy.zeros((num_runs,num_layers,num_stages-1))
    LMI_within = numpy.zeros((num_runs,num_layers,num_stages-1))
    LMI_ratio = numpy.zeros((num_runs,num_layers,num_stages-1))
    
    mahal_within_train_all = numpy.zeros((num_runs,num_layers,num_stages, num_noisy_trials))
    mahal_within_untrain_all = numpy.zeros((num_runs,num_layers,num_stages, num_noisy_trials))
    mahal_train_control_all = numpy.zeros((num_runs,num_layers,num_stages, num_noisy_trials))
    mahal_untrain_control_all = numpy.zeros((num_runs,num_layers,num_stages, num_noisy_trials))
    train_SNR_all=numpy.zeros((num_runs,num_layers,num_stages, num_noisy_trials))
    untrain_SNR_all=numpy.zeros((num_runs,num_layers,num_stages, num_noisy_trials))

    mahal_train_control_mean = numpy.zeros((num_runs,num_layers,num_stages))
    mahal_untrain_control_mean = numpy.zeros((num_runs,num_layers,num_stages))
    mahal_within_train_mean = numpy.zeros((num_runs,num_layers,num_stages))
    mahal_within_untrain_mean = numpy.zeros((num_runs,num_layers,num_stages))
    train_SNR_mean = numpy.zeros((num_runs,num_layers,num_stages))
    untrain_SNR_mean = numpy.zeros((num_runs,num_layers,num_stages))
    
    # Define pca model
    pca = PCA(n_components=num_PC_used)
      
    # Iterate over the different parameter initializations (runs)
    for run_ind in range(num_runs):
        start_time=time.time()
                
        # Calculate num_noisy_trials filtered model response for each oris in ori list and for each parameter set (that come from file_name at num_stage_inds rows)
        r_mid_sup = filtered_model_response(folder, run_ind, ori_list= ori_list, num_noisy_trials = num_noisy_trials, num_stage_inds=num_stages,r_noise=r_noise, sigma_filter = sigma_filter)
        # Note: r_mid_sup is a dictionary with the oris and stages saved in them
        r_ori = r_mid_sup['ori']
        mesh_train_ = r_ori == ori_list[0] 
        mesh_untrain_ = r_ori == ori_list[1]
        mesh_control_ = r_ori == ori_list[2]
        # Iterate over the layers and stages
        num_PCA_plots= 6
        if plot_flag and run_ind<num_PCA_plots:
            # make grid of plots for each layer and stage
            fig, axs = plt.subplots(num_layers, num_stages+1, figsize=(5*(num_stages+2), 5*num_layers))
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

            for stage_ind in range(num_stages): 
                # Define filter to select the responses corresponding to stage_ind
                stage_mask = r_mid_sup['stage'] == stage_ind
                 
                # Separate data into orientation conditions
                train_data = r_pca[mesh_train_ & stage_mask,:]
                untrain_data = r_pca[mesh_untrain_ & stage_mask,:]
                control_data = r_pca[mesh_control_ & stage_mask,:]
                ############################# MVPA analysis #############################
                
                # MVPA for distinguishing trained or untrained orientations (55 and 125) and control orientation (0)
                train_control_data = numpy.concatenate((train_data, control_data))
                train_control_data = train_control_data.reshape(train_control_data.shape[0],-1)
                train_control_label = numpy.concatenate((numpy.zeros(num_noisy_trials), numpy.ones(num_noisy_trials)))
                
                untrain_control_data = numpy.concatenate((untrain_data, control_data))
                untrain_control_data = untrain_control_data.reshape(untrain_control_data.shape[0],-1)
                untrain_control_label = numpy.concatenate((numpy.zeros(num_noisy_trials), numpy.ones(num_noisy_trials)))
                              
                # fit the classifier for 10 randomly selected trial and test data and average the scores
                scores_untrain = []
                scores_train = []
                for i in range(10):
                    X_untrain, X_test_untrain, y_untrain, y_test_untrain = train_test_split(untrain_control_data, untrain_control_label, test_size=0.5, random_state=i)
                    X_train, X_test_train, y_train, y_test_train = train_test_split(train_control_data, train_control_label, test_size=0.5, random_state=i)
                    score_train = clf.fit(X_train, y_train).score(X_test_train, y_test_train)
                    score_untrain = clf.fit(X_untrain, y_untrain).score(X_test_untrain, y_test_untrain)
                    scores_untrain.append(score_untrain)
                    scores_train.append(score_train)
                MVPA_scores[run_ind,layer,stage_ind, 1] = np.mean(np.array(scores_untrain))
                MVPA_scores[run_ind,layer,stage_ind, 0] = np.mean(np.array(scores_train))

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
                stage_labels = ['prepre', 'pre', 'post']
                if plot_flag and run_ind < num_PCA_plots:
                    axs[layer,stage_ind].scatter(control_data[:,0], control_data[:,1], label='control '+stage_labels[stage_ind], color='tab:green', s=5, marker=symbols[stage_ind])
                    axs[layer,stage_ind].scatter(train_data[:,0], train_data[:,1], label='trained '+stage_labels[stage_ind], color='blue', s=5, marker=symbols[stage_ind])
                    axs[layer,stage_ind].scatter(untrain_data[:,0], untrain_data[:,1], label='untrained '+stage_labels[stage_ind], color='red', s=5, marker=symbols[stage_ind])
                    axs[layer,stage_ind].set_title(f'Layer {layer}, run {run_ind}')
                    # Add lines between the mean of the conditions and write the Euclidean distance between them
                    mean_control = numpy.mean(control_data, axis=0)
                    mean_train = numpy.mean(train_data, axis=0)
                    mean_untrain = numpy.mean(untrain_data, axis=0)
                    axs[layer,stage_ind].plot([mean_control[0], mean_train[0]], [mean_control[1], mean_train[1]], color='gray')
                    axs[layer,stage_ind].plot([mean_control[0], mean_untrain[0]], [mean_control[1], mean_untrain[1]], color='gray')
                    axs[layer,stage_ind].plot([mean_train[0], mean_untrain[0]], [mean_train[1], mean_untrain[1]], color='gray')
                    # add two lines of title, one with Eucledean distances and one with Mahalanobis distances
                    axs[layer,stage_ind].set_title(f'train:{numpy.linalg.norm(mean_control-mean_train):.2f},untrain:{numpy.linalg.norm(mean_control-mean_untrain):.2f} \n train:{np.mean(mahal_train_control):.2f},untrain:{np.mean(mahal_untrain_control):.2f} within: {np.mean(mahal_within_train):.2f}, {np.mean(mahal_within_untrain):.2f}')
                    axs[layer,stage_ind].legend()
                    
                # Save Mahal distances and ratios
                mahal_train_control_all[run_ind,layer,stage_ind,:] = mahal_train_control
                mahal_untrain_control_all[run_ind,layer,stage_ind,:] = mahal_untrain_control
                mahal_within_train_all[run_ind,layer,stage_ind,:] = mahal_within_train
                mahal_within_untrain_all[run_ind,layer,stage_ind,:] = mahal_within_untrain
                train_SNR_all[run_ind,layer,stage_ind,:] = mahal_train_control / mahal_within_train
                untrain_SNR_all[run_ind,layer,stage_ind,:] = mahal_untrain_control / mahal_within_untrain

                # Average over trials
                mahal_train_control_mean[run_ind,layer,stage_ind] = numpy.mean(mahal_train_control)
                mahal_untrain_control_mean[run_ind,layer,stage_ind] = numpy.mean(mahal_untrain_control)
                mahal_within_train_mean[run_ind,layer,stage_ind] = numpy.mean(mahal_within_train)
                mahal_within_untrain_mean[run_ind,layer,stage_ind] = numpy.mean(mahal_within_untrain)
                train_SNR_mean[run_ind,layer,stage_ind] = numpy.mean(train_SNR_all[run_ind,layer,stage_ind,:])
                untrain_SNR_mean[run_ind,layer,stage_ind] = numpy.mean(untrain_SNR_all[run_ind,layer,stage_ind,:])

            '''
            # 2D-PCA scatter plots where the three conditions have different colors
            if plot_flag and run_ind < num_PCA_plots:
                axs[layer,num_stages].bar([1,2,3],mahal_train_control_mean[run_ind,layer,:], color='blue', alpha=0.5)
                axs[layer,num_stages].bar([5,6,7],mahal_untrain_control_mean[run_ind,layer,:], color='red', alpha=0.5)
                axs[layer,num_stages].set_xticks([1,2,3,5,6,7])
                axs[layer,num_stages].set_xticklabels(['tr0', 'tr1', 'tr2', 'ut0', 'ut1', 'ut2'])
                fig.savefig(folder + f"/figures/PCA_{run_ind}")
                plt.close()
            '''

            # Calculate learning modulation indices (LMI)
            for stage_ind_ in range(num_stages-1):
                LMI_across[run_ind,layer,stage_ind_] = (mahal_train_control_mean[run_ind,layer,stage_ind_+1] - mahal_train_control_mean[run_ind,layer,stage_ind_]) - (mahal_untrain_control_mean[run_ind,layer,stage_ind_+1] - mahal_untrain_control_mean[run_ind,layer,stage_ind_] )
                LMI_within[run_ind,layer,stage_ind_] = (mahal_within_train_mean[run_ind,layer,stage_ind_+1] - mahal_within_train_mean[run_ind,layer,stage_ind_]) - (mahal_within_untrain_mean[run_ind,layer,stage_ind_+1] - mahal_within_untrain_mean[run_ind,layer,stage_ind_] )
                LMI_ratio[run_ind,layer,stage_ind_] = (train_SNR_mean[run_ind,layer,stage_ind_+1] - train_SNR_mean[run_ind,layer,stage_ind_]) - (untrain_SNR_mean[run_ind,layer,stage_ind_+1] - untrain_SNR_mean[run_ind,layer,stage_ind_] )
        
        # Print the results for the current run
        print(MVPA_scores[run_ind,:,:,0], 'trained vs control')
        print(MVPA_scores[run_ind,:,:,1], 'untrained vs control')
        print([np.mean(mahal_train_control_all[run_ind,0,:,:], axis = -1)] ,'train')
        print([np.mean(mahal_untrain_control_all[run_ind,0,:,:],axis=-1)],'untrain')

        print(f'runtime of run {run_ind}:',time.time()-start_time)

    ################# Create dataframes for the Mahalanobis distances and LMI #################
    #df_mahal, df_LMI = LMI_Mahal_df(num_training, num_layers, num_stage_inds, mahal_train_control_mean, mahal_untrain_control_mean, mahal_within_train_mean, mahal_within_untrain_mean, train_SNR_mean, untrain_SNR_mean, LMI_across, LMI_within, LMI_ratio)
    Mahal_scores[:,:,:,0] = mahal_train_control_mean
    Mahal_scores[:,:,:,1] = mahal_untrain_control_mean

    return MVPA_scores, Mahal_scores