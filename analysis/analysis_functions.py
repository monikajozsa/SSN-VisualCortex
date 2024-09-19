import pandas as pd
import jax.numpy as np
import numpy
import time
import scipy
import jax
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from training.model import vmap_evaluate_model_response, vmap_evaluate_model_response_mid
from training.SSN_classes import SSN_mid, SSN_sup
from training.training_functions import generate_noise
from util import load_parameters, filter_for_run_and_stage, unpack_ssn_parameters
from training.util_gabor import BW_image_jit_noisy, BW_image_jax_supp, BW_image_vmap

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
    data['EI_ratio_J_m'] = numpy.abs((data['J_II_m']+data['J_EI_m']))/numpy.abs((data['J_IE_m']+data['J_EE_m']))
    data['EI_ratio_J_s'] = numpy.abs((data['J_II_s']+data['J_EI_s']))/numpy.abs((data['J_IE_s']+data['J_EE_s']))
    data['EI_ratio_J_ms'] = numpy.abs((data['J_II_m']+data['J_EI_m']+data['J_II_s']+data['J_EI_s']))/numpy.abs((data['J_IE_m']+data['J_EE_m']+data['J_IE_s']+data['J_EE_s']))
    if num_indices == 3:
        pretraining_start = time_inds[0]
        training_start = time_inds[1]-1
        rel_change_pretrain = {key: calc_rel_change_supp(value, pretraining_start, training_start) for key, value in data.items()}
    else:
        training_start = time_inds[0]
        rel_change_pretrain = None

    rel_change_train = {key: calc_rel_change_supp(value, training_start, training_end) for key, value in data.items()}        
    
    return rel_change_train, rel_change_pretrain, time_inds

def rel_change_for_runs(folder, num_indices=3):
    """Calculate the relative changes in the parameters for all runs."""

    # Initialize the arrays to store the results in
    filepath = os.path.join(folder, 'pretraining_results.csv')
    df = pd.read_csv(filepath)
    num_runs = df['run_index'].iloc[-1]+1

    # Calculate the relative changes for all runs
    for i in range(num_runs):
        rel_change_train, rel_change_pretrain, _ = rel_change_for_run(folder, i, num_indices)
        if i == 0:
            rel_changes_train = {key: numpy.zeros(num_runs) for key in rel_change_train.keys()}
            rel_changes_pretrain = {key: numpy.zeros(num_runs) for key in rel_change_pretrain.keys()}
        for key, value in rel_change_train.items():
            rel_changes_train[key][i] = value
            rel_changes_pretrain[key][i] = rel_change_pretrain[key]
        
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
            stimuli = BW_image_vmap(BW_image_jax_inp[0:4], x, y, alpha_channel, mask, ori_vec, np.zeros(num_ori)) 
            for ori in range(num_ori):
                gabor_output[ori,phase_ind,0,grid_ind] = gabor_filters[phase_ind,0,grid_ind,:]@(stimuli[ori,:].T) # E cells
                gabor_output[ori,phase_ind,1,grid_ind] = gabor_filters[phase_ind,1,grid_ind,:]@(stimuli[ori,:].T) # I cells
    print('Time elapsed for gabor_output calculation:', time.time()-time_start)
     
    return gabor_output


def tuning_curve(untrained_pars, trained_pars, file_path=None, ori_vec=np.arange(0,180,6), training_stage=1, run_index=0, header = False):
    """ Calculate responses of middle and superficial layers to different orientations."""
    # Get the parameters from the trained_pars dictionary and untreatned_pars class
    ref_ori_saved = float(untrained_pars.stimuli_pars.ref_ori)
    J_2x2_m, J_2x2_s, cE_m, cI_m, cE_s, cI_s, f_E, f_I, kappa = unpack_ssn_parameters(trained_pars, untrained_pars.ssn_pars)

    x_map = untrained_pars.grid_pars.x_map
    y_map = untrained_pars.grid_pars.y_map
    ssn_pars = untrained_pars.ssn_pars

    ssn_mid=SSN_mid(ssn_pars=ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_m)
    
    num_ori = len(ori_vec)
    new_rows = []
    
    grid_size = x_map.shape[0]*x_map.shape[1]
    responses_mid_phase_match = numpy.zeros((len(ori_vec),grid_size*ssn_pars.phases*2))
    responses_sup_phase_match = numpy.zeros((len(ori_vec),grid_size*2))
    for i in range(x_map.shape[0]):
        for j in range(x_map.shape[1]):
            x0 = x_map[i, j]
            y0 = y_map[i, j]
            for phase_ind in range(ssn_pars.phases):
                # Generate stimulus
                phase = phase_ind * np.pi/2
                BW_image_jax_inp = BW_image_jax_supp(untrained_pars.stimuli_pars, x0=x0, y0=y0, phase=phase, full_grating=True)
                x = BW_image_jax_inp[4]
                y = BW_image_jax_inp[5]
                alpha_channel = BW_image_jax_inp[6]
                mask = BW_image_jax_inp[7]
                stimuli = BW_image_vmap(BW_image_jax_inp[0:4], x, y, alpha_channel, mask, ori_vec, np.zeros(num_ori))
                # Calculate model response for middle layer cells and save it to responses_mid_phase_match
                _, responses_mid,_, _, _,  _, _ = vmap_evaluate_model_response_mid(ssn_mid, stimuli, untrained_pars.conv_pars, cE_m, cI_m, untrained_pars.gabor_filters)
                mid_cell_ind_E = phase_ind*2*grid_size + i*x_map.shape[0]+j
                mid_cell_ind_I = phase_ind*2*grid_size + i*x_map.shape[0]+j+grid_size
                responses_mid_phase_match[:,mid_cell_ind_E]=responses_mid[:,mid_cell_ind_E] # E cell
                responses_mid_phase_match[:,mid_cell_ind_I]=responses_mid[:,mid_cell_ind_I] # I cell
                # Calculate model response for superficial layer cells and save it to responses_sup_phase_match
                if phase_ind==0:
                    # Superficial layer response per grid point
                    ssn_sup=SSN_sup(ssn_pars, untrained_pars.grid_pars, J_2x2_s, untrained_pars.dist_from_single_ori, untrained_pars.ori_dist, kappa)
                    _, [_, responses_sup],_, _, _, = vmap_evaluate_model_response(ssn_mid, ssn_sup, stimuli, untrained_pars.conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters)
                    sup_cell_ind = i*x_map.shape[0]+j
                    responses_sup_phase_match[:,sup_cell_ind]=responses_sup[:,sup_cell_ind]
                    responses_sup_phase_match[:,grid_size+sup_cell_ind]=responses_sup[:,grid_size+sup_cell_ind]

    # Save responses into csv file - overwrite the file if it already exists
    if file_path is not None:
        if os.path.exists(file_path) and header is not False:
            Warning('Tuning curve csv file will get multiple headers and will possibly have repeated rows!')
        # repeat training_stage run_index and expand dimension to add as the first two columns of the new_rows
        run_index_vec = numpy.repeat(run_index, len(ori_vec))
        training_stage_vec = numpy.repeat(training_stage, len(ori_vec))
        run_index_vec = numpy.expand_dims(run_index_vec, axis=1)
        training_stage_vec = numpy.expand_dims(training_stage_vec, axis=1)
        responses_combined=np.concatenate((responses_mid_phase_match, responses_sup_phase_match), axis=1)
        new_rows = numpy.concatenate((run_index_vec, training_stage_vec, responses_combined), axis=1)
        new_rows_df = pd.DataFrame(new_rows)
        new_rows_df.to_csv(file_path, mode='a', header=header, index=False, float_format='%.4f')

    untrained_pars.stimuli_pars.ref_ori = ref_ori_saved

    return responses_sup, responses_mid


def tc_slope(tuning_curve, x_axis, x1, x2, normalise=False):
    """ Calculates slope of (normalized if normalise=True) tuning_curve between points x1 and x2. tuning_curve is given at x_axis points. """
    #Remove baseline if normalising
    if normalise == True:
        tuning_curve = (tuning_curve - tuning_curve.min()) / tuning_curve.max()
    
    #Find indices corresponding to desired x values
    idx_1 = (np.abs(x_axis - x1)).argmin()
    idx_2 = (np.abs(x_axis - x2)).argmin()
    x1, x2 = x_axis[[idx_1, idx_2]]
     
    grad =np.abs((tuning_curve[idx_2] - tuning_curve[idx_1])/(x2-x1)) if x2 != x1 else None
    
    return grad


def full_width_half_max(vector, d_theta):
    """ Calculate width of tuning curve at half-maximum. This method should not be applied when tuning curve has multiple bumps. """
    # Remove baseline, calculate half-max
    vector = vector-vector.min()
    half_height = vector.max()/2

    # Get the number of points above half-max and multiply it by the delta angle to get width in angle
    points_above = len(vector[vector>half_height])
    distance = d_theta * points_above
    
    return distance


def tc_features(tuning_curve, ori_list=numpy.arange(0,180,6), expand_dims=False, ori_to_center_slope=[55, 125]):
    
    # Tuning curve of given cell indices
    num_cells = tuning_curve.shape[1]

    # Full width half height
    full_width_half_max_vec = numpy.zeros(num_cells) 
    d_theta = ori_list[1]-ori_list[0]
    for i in range(0, num_cells):
        full_width_half_max_vec[i] = full_width_half_max(tuning_curve[:,i], d_theta = d_theta)

    # Preferred orientation
    pref_ori = ori_list[np.argmax(tuning_curve, axis = 0)]

    # Norm slope
    avg_slope_vec =numpy.zeros((num_cells, len(ori_to_center_slope)))
    for i in range(num_cells):
        for j in range(len(ori_to_center_slope)):
            ori_ctr = ori_to_center_slope[j]
            avg_slope_vec[i,j] = tc_slope(tuning_curve[:, i], x_axis = ori_list, x1 = ori_ctr-3, x2 = ori_ctr+3, normalise =False)
    if expand_dims:
        avg_slope_vec = numpy.expand_dims(avg_slope_vec, axis=0)
        full_width_half_max_vec = numpy.expand_dims(full_width_half_max_vec, axis=0)
        pref_ori = numpy.expand_dims(pref_ori, axis=0)

    return avg_slope_vec, full_width_half_max_vec, pref_ori



def save_tc_features(folder_path, num_runs=1, ori_list=numpy.arange(0,180,6), ori_to_center_slope=[55, 125]):
    """ Calculate tuning curve features for all runs in the folder and save them into a csv file. """
    start_time = time.time()
    # Load training tuning curve data (no headers)
    tc_filename = os.path.join(folder_path, 'tuning_curves.csv')

    # Check if the first row contains any non-numeric values (indicating a header)
    first_row = pd.read_csv(tc_filename, nrows=1, header=None).iloc[0]
    if first_row.apply(lambda x: isinstance(x, str)).any():
        header = 0
    else:
        header = None

    training_tc_all_run_df = pd.read_csv(tc_filename, header=header)
    training_tc_all_run = training_tc_all_run_df.to_numpy()
    
    # Load pretraining tuning curve data (with headers)
    tc_filename = os.path.join(os.path.dirname(folder_path), 'pretraining_tuning_curves.csv')
    pretraining_tc_all_run_df = pd.read_csv(tc_filename, header=0)
    header = pretraining_tc_all_run_df.columns  # Use this for cell headers
    cell_headers = header[2:]  # The headers for the 810 cells
    pretraining_tc_all_run = pretraining_tc_all_run_df.to_numpy()

    tc_features_df = None

    for stage in range(3):
        if stage < 2:
            tuning_curves_all_run = pretraining_tc_all_run
        else:
            tuning_curves_all_run = training_tc_all_run
        for run_index in range(num_runs):
            mesh_run = tuning_curves_all_run[:,0] == run_index
            tuning_curves = tuning_curves_all_run[mesh_run, 1:]  # Skipping run_index
            mesh_stage = tuning_curves[:, 0] == stage
            tuning_curves = tuning_curves[mesh_stage, 1:]  # Skipping stage

            # Use tuning curve shape to get the number of cells and the delta angle
            num_cells = tuning_curves.shape[1]
            delta_oris = ori_list[1]-ori_list[0]

            # Initialize the feature vectors
            full_width_half_max_vec = numpy.zeros(num_cells)
            slope_55 = numpy.zeros(num_cells)
            slope_125 = numpy.zeros(num_cells)
            pref_ori = numpy.zeros(num_cells)
            min_tc = numpy.zeros(num_cells)
            max_tc = numpy.zeros(num_cells)
            max_min_ratio_tc = numpy.zeros(num_cells)
            mean_tc = numpy.zeros(num_cells)
            std_tc = numpy.zeros(num_cells)
            slope_hm_2nd_top = numpy.zeros(num_cells)
            slope_hm_top = numpy.zeros(num_cells)

            for i in range(num_cells):
                # Full width half max calculation
                full_width_half_max_vec[i] = full_width_half_max(tuning_curves[:, i], d_theta=delta_oris)

                # Preferred orientation
                pref_ori[i] = ori_list[np.argmax(tuning_curves[:, i])]

                # Slope calculations at specific orientations
                slope_55[i] = tc_slope(tuning_curves[:, i], x_axis=ori_list, x1=ori_to_center_slope[0] - 3, x2=ori_to_center_slope[0] + 3, normalise=False)
                slope_125[i] = tc_slope(tuning_curves[:, i], x_axis=ori_list, x1=ori_to_center_slope[1] - 3, x2=ori_to_center_slope[1] + 3, normalise=False)
                
                # Other tuning curve statistics: minimum, maximum, (max-min)/max, mean, std, slope at half-max level to the left and right of the peak
                min_tc[i] = numpy.min(tuning_curves[:, i])
                max_tc[i] = numpy.max(tuning_curves[:, i])
                max_min_ratio_tc[i] = (max_tc[i] - min_tc[i]) / max_tc[i] if max_tc[i] != 0 else 0
                mean_tc[i] = numpy.mean(tuning_curves[:, i])
                std_tc[i] = numpy.std(tuning_curves[:, i])
                    
                loc_max = int(numpy.argmax(tuning_curves[:,i], axis=0))
                num_oris = tuning_curves.shape[0]

                # Slope at half_max level: find point closest to full_width_half_max_vec to the left and right of the peak and calculate the slope                
                concatenated_ori_list = numpy.concatenate((ori_list, ori_list, ori_list))
                hm = (tuning_curves[:,i] - min_tc[i]).max()/2 + min_tc[i]
                concatenated_tc = numpy.concatenate((tuning_curves[:,i], tuning_curves[:,i], tuning_curves[:,i]))
                loc_hm_top = (numpy.argmin(numpy.abs(concatenated_tc[loc_max + num_oris : loc_max + int(num_oris*3/2)] - hm)) + loc_max) % num_oris
                loc_hm_2nd_top = (numpy.argmin(numpy.abs(concatenated_tc[loc_hm_top + 1 : loc_hm_top + num_oris] - hm)) + loc_hm_top + 1) % num_oris
                slope_hm_top[i] = tc_slope(concatenated_tc, x_axis = concatenated_ori_list, x1 = (loc_hm_top - 2) * delta_oris, x2 = (loc_hm_top + 2) * delta_oris, normalise =False)
                slope_hm_2nd_top[i] = tc_slope(concatenated_tc, x_axis = concatenated_ori_list, x1 = (loc_hm_2nd_top - 2) * delta_oris, x2 = (loc_hm_2nd_top + 2) * delta_oris, normalise =False)
        
            # Add each feature to the DataFrame
            for feature, values in zip(
                ['slope_55', 'slope_125', 'fwhm', 'pref_ori', 'min', 'max', 'max_min_ratio', 'mean', 'std', 'slope_hm_left', 'slope_hm_right'],
                [slope_55, slope_125, full_width_half_max_vec, pref_ori, min_tc, max_tc, max_min_ratio_tc, mean_tc, std_tc, slope_hm_2nd_top, slope_hm_top]
            ):
                row_data = {'run_index': run_index,'stage': stage,'feature': feature}
                row_data.update({cell: value for cell, value in zip(cell_headers, values)})
                if tc_features_df is None:
                    tc_features_df = pd.DataFrame([row_data])
                else:
                    tc_features_df = pd.concat([tc_features_df, pd.DataFrame([row_data])], ignore_index=True)
        print(f'Stage {stage} done in {time.time() - start_time} seconds')
    # Save the DataFrame to CSV and return it
    output_filename = os.path.join(folder_path, 'tuning_curve_features.csv')
    tc_features_df.to_csv(output_filename, index=False)

    return tc_features_df


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
    MVPA_scores = numpy.load(folder + '/MVPA_scores.npy') # num_trainings x layer x SGD_ind x ori_ind
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

    # Perform QR decomposition on C
    Q, R = np.linalg.qr(X_demean, mode='reduced')

    # Solve for ri in the equation R' * ri = (Y-M) using least squares or directly if R is square and of full rank
    Y_demean=(Y-m).T
    ri = numpy.linalg.solve(R.T, Y_demean)

    # Calculate d as the sum of squares of ri, scaled by (rx-1)
    d = np.sum(ri**2, axis=0) * (rx-1)

    return np.sqrt(d)


def filtered_model_response(folder, run_ind, ori_list= np.asarray([55, 125, 0]), num_noisy_trials = 100, num_stage_inds = 2, r_noise=True, sigma_filter = 1, plot_flag = False, noise_std=1.0):
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
        J_2x2_m, J_2x2_s, cE_m, cI_m, cE_s, cI_s, f_E, f_I, kappa=unpack_ssn_parameters(trained_pars_stage2, untrained_pars.ssn_pars)
        
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
