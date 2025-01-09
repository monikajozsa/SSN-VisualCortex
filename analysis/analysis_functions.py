import jax
import jax.numpy as jnp
from jax import vmap
import numpy
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats
from scipy.interpolate import interp1d

import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from training.model import vmap_evaluate_model_response, vmap_evaluate_model_response_mid
from training.SSN_classes import SSN_mid, SSN_sup
from training.training_functions import generate_noise
from util import load_parameters, filter_for_run_and_stage, unpack_ssn_parameters, check_header, csv_to_numpy, save_numpy_to_csv
from training.util_gabor import BW_image_jit_noisy, BW_image_jax_supp, BW_image_jit

############## Analysis functions ##########
def data_from_run(folder, run_index=0, num_indices=3):
    """ Read CSV files, filter them for run and return the combined dataframe together with the time indices where stages change."""
    
    pretrain_filepath = os.path.join(os.path.dirname(folder), 'pretraining_results.csv') # this reaches the pretraining_results.csv file in the same folder as the training_results.csv file
    train_filepath = os.path.join(folder, 'training_results.csv')
    df_pretrain = pd.read_csv(pretrain_filepath)
    df_train = pd.read_csv(train_filepath)
    df_pretrain_i = filter_for_run_and_stage(df_pretrain, run_index)
    df_train_i = filter_for_run_and_stage(df_train, run_index)
    if df_train_i.empty:
        print('No data for run_index = {}'.format(run_index))
        no_train_data = True
    else:
        no_train_data = False
    df_i = pd.concat((df_pretrain_i,df_train_i))
    df_i.reset_index(inplace=True)
    if df_i.empty:
        stage_time_inds = []
    else:
        stage_time_inds = SGD_indices_at_stages(df_i, num_indices)

    return df_i, stage_time_inds, no_train_data

def calc_rel_change_supp(variable, time_start, time_end):
    """ Calculate the relative change in a variable between two time points. """
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
    """ Calculate the relative changes in the parameters for a single run."""

    # Load the data for the given training index and drop irrelevant columns
    data, time_inds, no_train_data = data_from_run(folder, training_ind, num_indices)
    training_end = time_inds[-1]
    columns_to_drop = ['stage', 'SGD_steps'] 
    data = data.drop(columns=columns_to_drop)

    # Define additional columns
    data['J_I_m'] = numpy.abs(data['J_II_m'] + data['J_EI_m'])
    data['J_I_s'] = numpy.abs(data['J_II_s'] + data['J_EI_s'])
    data['J_E_m'] = numpy.abs(data['J_IE_m'] + data['J_EE_m'])
    data['J_E_s'] = numpy.abs(data['J_IE_s'] + data['J_EE_s'])
    data['EI_ratio_J_m'] = data['J_I_m'] / data['J_E_m']
    data['EI_ratio_J_s'] = data['J_I_s'] / data['J_E_s']
    data['EI_ratio_J_ms'] = (data['J_I_m'] + data['J_I_s']) / (data['J_E_m'] + data['J_E_s'])

    # Calculate relative changes for the pretraining 
    if num_indices == 3:
        pretraining_start = time_inds[0]
        training_start = time_inds[1]-1
        rel_change_pretrain = {}
        for key, value in data.items():
            item_rel_change = calc_rel_change_supp(value, pretraining_start, training_start)
            if item_rel_change is not None:
                rel_change_pretrain[key] = item_rel_change
        rel_change_pretrain.pop('index')
    else:
        training_start = time_inds[0]
        rel_change_pretrain = None
    
    # Calculate relative changes for the training
    if no_train_data:
        rel_change_train = None
    else:
        rel_change_train = {}
        for key, value in data.items():
            item_rel_change = calc_rel_change_supp(value, training_start, training_end)
            if item_rel_change is not None:
                rel_change_train[key] = item_rel_change    
        rel_change_train.pop('index')

    return rel_change_train, rel_change_pretrain, time_inds

def rel_change_for_runs(folder, num_time_inds=3, num_runs=None, excluded_runs=[]):
    """ Calculate the relative changes in parameters for all runs."""
    
    def load_existing_data(filepath):
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            return {key: df[key].to_numpy() for key in df.columns}
        return None

    def save_to_csv(data, filepath):
        if data is not None:
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
    
    # Determine number of runs
    if num_runs is None:
        pretrain_filepath = os.path.join(os.path.dirname(folder), 'pretraining_results.csv')
        pretrain_df = pd.read_csv(pretrain_filepath)
        num_runs = pretrain_df['run_index'].iloc[-1] + 1

    # Load pre-existing data if available
    rel_changes_pretrain = load_existing_data(os.path.join(os.path.dirname(folder), 'rel_changes_pretrain.csv'))
    if rel_changes_pretrain is None and num_time_inds == 3:
        pretrain_recalc = True
    else:
        pretrain_recalc = False
    rel_changes_train = load_existing_data(os.path.join(folder, 'rel_changes_train.csv'))

    # If data is missing, calculate relative changes
    if rel_changes_train is None or pretrain_recalc:
        successful_training_runs = []
        rel_changes_train = {}
        rel_changes_pretrain = {} if pretrain_recalc else rel_changes_pretrain

        for i in range(num_runs):
            rel_change_train, rel_change_pretrain, _ = rel_change_for_run(folder, i, num_time_inds)
            # Initialize storage arrays
            if rel_change_train is not None and (len(rel_changes_train.keys()) == 0):
                rel_changes_train = {key: numpy.zeros(num_runs) for key in rel_change_train.keys()}
            if pretrain_recalc and (len(rel_changes_pretrain.keys()) == 0):
                rel_changes_pretrain = {key: numpy.zeros(num_runs) for key in rel_change_pretrain.keys()}
            if pretrain_recalc:
                for key in rel_change_pretrain.keys():
                    rel_changes_pretrain[key][i] = rel_change_pretrain[key]
            if rel_change_train is None:
                continue
            successful_training_runs.append(i)
            for key in rel_changes_train.keys():
                rel_changes_train[key][i] = rel_change_train[key]
        
        # Filter rel_changes_train for successful runs and add run_index to the dictionaries 
        rel_changes_train = {key: rel_changes_train[key][successful_training_runs] for key in rel_changes_train.keys()}
        rel_changes_train['run_index'] = numpy.array(successful_training_runs)
        if pretrain_recalc:
            rel_changes_pretrain['run_index'] = numpy.arange(len(rel_changes_pretrain['acc']))
        
        # Save computed data to CSV files
        save_to_csv(rel_changes_train, os.path.join(folder, 'rel_changes_train.csv'))
        if pretrain_recalc:
            save_to_csv(rel_changes_pretrain, os.path.join(os.path.dirname(folder), 'rel_changes_pretrain.csv'))
    else:
        successful_training_runs = rel_changes_train['run_index']

    # Filter for included runs only
    included_runs = numpy.setdiff1d(successful_training_runs, excluded_runs)
    included_indices_train = [numpy.where(successful_training_runs == run)[0][0] for run in included_runs]
    included_indices_pretrain = [numpy.where(numpy.arange(num_runs) == run)[0][0] for run in included_runs]
    rel_changes_train = {key: rel_changes_train[key][included_indices_train] for key in rel_changes_train.keys()}
    if rel_changes_pretrain:
        rel_changes_pretrain = {key: rel_changes_pretrain[key][included_indices_pretrain] for key in rel_changes_pretrain.keys()}

    return rel_changes_train, rel_changes_pretrain


def pre_post_for_runs(folder, num_training, num_time_inds=3, excluded_runs=[]):
    """ Calculate the pre and post training values for runs that are not excluded. """
    included_runs = numpy.setdiff1d(numpy.arange(num_training), excluded_runs)
    for i, run_ind in enumerate(included_runs):
        df_i, stage_time_inds, no_train_data = data_from_run(folder, run_index=run_ind, num_indices=num_time_inds)
        train_end_ind = stage_time_inds[-1]
        if num_time_inds>2:
            pretrain_start_ind = stage_time_inds[0]
            train_start_ind = stage_time_inds[1]
        else:
            train_start_ind = stage_time_inds[0]
        if i==0:
            vals_pre = df_i.iloc[[train_start_ind]]
            vals_post = df_i.iloc[[train_end_ind]]
            if num_time_inds>2:
                vals_prepre = df_i.iloc[[pretrain_start_ind]]
            else:
                vals_prepre = None
        else:
            vals_pre=pd.concat([vals_pre,df_i.iloc[[train_start_ind]]], ignore_index=True)
            vals_post=pd.concat([vals_post, df_i.iloc[[train_end_ind]]], ignore_index=True)
            if num_time_inds>2:
                vals_prepre=pd.concat([vals_prepre, df_i.iloc[[pretrain_start_ind]]], ignore_index=True)

    return vals_prepre, vals_pre, vals_post


def gabor_tuning(untrained_pars, ori_vec=jnp.arange(0,180,6)):
    """ Calculate the responses of the gabor filters to stimuli with different orientations."""
    gabor_filters = untrained_pars.gabor_filters
    num_ori = len(ori_vec)
    # Getting the 'blank' alpha_channel and mask for a full image version stimuli with no background
    BW_image_jax_inp = BW_image_jax_supp(untrained_pars.stimuli_pars, x0 = 0, y0=0, phase=0.0, full_grating=True) 
    alpha_channel = BW_image_jax_inp[6]
    mask = BW_image_jax_inp[7]
    if len(gabor_filters.shape)==2:
        gabor_filters = jnp.reshape(gabor_filters, (untrained_pars.ssn_pars.phases,2,untrained_pars.grid_pars.gridsize_Nx **2,-1)) # the second dimension 2 is for I and E cells, the last dim is the image size
    
    # Initialize the gabor output array
    gabor_output = numpy.zeros((num_ori, untrained_pars.ssn_pars.phases,2,untrained_pars.grid_pars.gridsize_Nx **2))
    time_start = time.time()
    for grid_ind in range(gabor_filters.shape[2]):
        grid_ind_x = grid_ind//untrained_pars.grid_pars.gridsize_Nx # For the right order, it is important to match how gabors_demean is filled up in create_gabor_filters_ori_map, currently, it is [grid_size_1D*i+j,phases_ind,:], where x0 = x_map[i, j]
        grid_ind_y = grid_ind%untrained_pars.grid_pars.gridsize_Nx
        x0 = untrained_pars.grid_pars.x_map[grid_ind_x, grid_ind_y]
        y0 = untrained_pars.grid_pars.y_map[grid_ind_x, grid_ind_y]
        for phase_ind in range(gabor_filters.shape[0]):
            phase = phase_ind * jnp.pi/2
            BW_image_jax_inp = BW_image_jax_supp(untrained_pars.stimuli_pars, x0=x0, y0=y0, phase=phase, full_grating=True)
            x = BW_image_jax_inp[4]
            y = BW_image_jax_inp[5]
            stimuli = BW_image_jit(BW_image_jax_inp[0:4], x, y, alpha_channel, mask, ori_vec, jnp.zeros(num_ori)) 
            for ori in range(num_ori):
                gabor_output[ori,phase_ind,0,grid_ind] = gabor_filters[phase_ind,0,grid_ind,:]@(stimuli[ori,:].T) # E cells
                gabor_output[ori,phase_ind,1,grid_ind] = gabor_filters[phase_ind,1,grid_ind,:]@(stimuli[ori,:].T) # I cells
    print('Time elapsed for gabor_output calculation:', time.time()-time_start)
     
    return gabor_output


def tc_grid_point(inds_maps_flat, ssn_mid, ssn_sup, num_phases, untrained_pars, ori_vec, num_ori, grid_size, cE_m, cI_m, cE_s, cI_s, f_E, f_I, kappa_f = jnp.array([0.0, 0.0])):
    """ Calculate the responses of the middle and superficial layers to gratings at a single grid point - used for phase matched tuning curve calculation."""
    
    x_ind = inds_maps_flat[0]
    y_ind = inds_maps_flat[1]
    x_grid = inds_maps_flat[2]
    y_grid = inds_maps_flat[3]
    grid_size_1D = jnp.sqrt(grid_size).astype(int)
    
    # Initialize the arrays to store the responses
    responses_mid_phase_match = jnp.zeros((num_ori, 2, num_phases))
    responses_sup_phase_match = jnp.zeros((num_ori, 2))

    # Loop over the different phases
    for phase_ind in range(num_phases):
        phase = phase_ind * jnp.pi / 2
        
        # Generate stimulus
        BW_image_jax_inp = BW_image_jax_supp(untrained_pars.stimuli_pars, x0=x_grid, y0=y_grid, phase=phase, full_grating=True)
        x = BW_image_jax_inp[4]
        y = BW_image_jax_inp[5]
        alpha_channel = BW_image_jax_inp[6]
        mask = BW_image_jax_inp[7]
        
        stimuli = BW_image_jit(BW_image_jax_inp[0:4], x, y, alpha_channel, mask, ori_vec, jnp.zeros(num_ori))
        
        # Calculate model response for superficial layer cells (phase-invariant)
        if phase_ind == 0:
            _, [responses_mid, responses_sup], _, _, _, = vmap_evaluate_model_response(ssn_mid, ssn_sup, stimuli, untrained_pars.conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters, untrained_pars.dist_from_single_ori, kappa_f, untrained_pars.ssn_pars.kappa_range)
            # Fill in the responses_sup_phase_match array at the indices corresponding to the grid point
            sup_cell_ind = jnp.array(x_ind*grid_size_1D+y_ind).astype(int)
            responses_sup_phase_match = responses_sup_phase_match.at[:,0].set(responses_sup[:, sup_cell_ind]) # E cell
            responses_sup_phase_match = responses_sup_phase_match.at[:,1].set(responses_sup[:, grid_size+sup_cell_ind]) # I cell
        else:
            # Calculate model response for middle layer cells
            _, responses_mid, _, _, _, _, _ = vmap_evaluate_model_response_mid(ssn_mid, stimuli, untrained_pars.conv_pars, cE_m, cI_m, untrained_pars.gabor_filters)
        # Fill in the responses_mid_phase_match array at the indices corresponding to the grid point
        mid_cell_ind_E = jnp.array(phase_ind*2*grid_size + x_ind*grid_size_1D+y_ind).astype(int)
        mid_cell_ind_I = jnp.array(phase_ind*2*grid_size + x_ind*grid_size_1D+y_ind+grid_size).astype(int)
        responses_mid_phase_match = responses_mid_phase_match.at[:, 0, phase_ind].set(responses_mid[:, mid_cell_ind_E])
        responses_mid_phase_match = responses_mid_phase_match.at[:, 1, phase_ind].set(responses_mid[:, mid_cell_ind_I])
    
    return responses_mid_phase_match, responses_sup_phase_match

# Vectorizing over the flattened indices of the grid
vmapped_tc_grid_point = vmap(tc_grid_point, in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None))


def tuning_curve(untrained_pars, trained_pars, file_path=None, ori_vec=jnp.arange(0,180,6), training_stage=1, run_index=0, header = False):
    """ Calculate responses of middle and superficial layers to gratings (of full images without added noise) with different orientations."""
    if trained_pars is None:
        return None, None
    # Initialize the arrays to store the responses
    grid_size = untrained_pars.grid_pars.gridsize_Nx ** 2
    num_phases = untrained_pars.ssn_pars.phases
    num_ori = len(ori_vec)
    responses_sup_phase_match_2D = numpy.zeros((num_ori,grid_size*2))
    responses_mid_phase_match_2D = numpy.zeros((num_ori,grid_size*num_phases*2))

    # Get the parameters from the trained_pars dictionary and untreatned_pars class
    ref_ori_saved = float(untrained_pars.stimuli_pars.ref_ori)
    J_2x2_m, J_2x2_s, cE_m, cI_m, cE_s, cI_s, f_E, f_I, kappa_Jsup,_,_ = unpack_ssn_parameters(trained_pars, untrained_pars.ssn_pars)
    ssn_pars = untrained_pars.ssn_pars        
    x_map = untrained_pars.grid_pars.x_map
    y_map = untrained_pars.grid_pars.y_map
    grid_size = x_map.shape[0]*x_map.shape[1]
    
    # Define the SSN layers
    ssn_mid = SSN_mid(ssn_pars, untrained_pars.grid_pars, J_2x2_m, untrained_pars.dist_from_single_ori)
    ssn_sup = SSN_sup(ssn_pars, untrained_pars.grid_pars, J_2x2_s, untrained_pars.dist_from_single_ori, untrained_pars.ori_dist, kappa_Jsup)
    
    # Flatten the grid indices and the x, y coordinates for vmap    
    num_rows, num_cols = x_map.shape
    i_indices, j_indices = numpy.meshgrid(jnp.arange(num_rows), jnp.arange(num_cols), indexing='ij')
    inds_maps_flat_0 = i_indices.flatten()  # i index (row index)
    inds_maps_flat_1 = j_indices.flatten()  # j index (column index)
    inds_maps_flat_2 = x_map.flatten()
    inds_maps_flat_3 = y_map.flatten()
    inds_maps_flat = jnp.stack((inds_maps_flat_0, inds_maps_flat_1, inds_maps_flat_2, inds_maps_flat_3), axis=1)

    responses_mid_phase_match, responses_sup_phase_match = vmapped_tc_grid_point(inds_maps_flat, ssn_mid, ssn_sup, ssn_pars.phases, untrained_pars, ori_vec, num_ori, grid_size, cE_m, cI_m, cE_s, cI_s, f_E, f_I)
    
    # rearrange responses to 2D, where first dim is oris and second dim is cells
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
        
        responses_combined=jnp.concatenate((responses_mid_phase_match_2D, responses_sup_phase_match_2D), axis=1)
        new_rows = numpy.concatenate((run_index_vec, training_stage_vec, responses_combined), axis=1)
        new_rows_df = pd.DataFrame(new_rows)
        new_rows_df.to_csv(file_path, mode='a', header=header, index=False, float_format='%.4f')

    # Set the reference orientation back to the original value
    untrained_pars.stimuli_pars.ref_ori = ref_ori_saved

    return responses_sup_phase_match_2D, responses_mid_phase_match_2D


def save_tc_features(training_tc_file, num_runs=1, ori_list=jnp.arange(0,180,6), ori_to_center_slope=[55, 125], stages=[1, 2], filename='tuning_curve_features.csv'):
    """ Calls compute_features for each stage and run index and saves the results into a CSV file. """
    output_filename = os.path.join(os.path.dirname(training_tc_file), filename)
    if os.path.exists(output_filename):
        print('File already exists. Please delete it before running save_tc_features function if you want to overwrite it.')
        tc_features_df = pd.read_csv(output_filename)
        return tc_features_df
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
        # Skip if there is no tc data for this run
        if numpy.sum(run_mask) == 0:
            continue
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
    tc_features_df.to_csv(output_filename, index=False)

    return tc_features_df


def compute_features(tuning_curves, num_cells, ori_list, oris_to_calc_slope_at, d_theta_interp=0.2):
    """ Computes tuning curve features for each cell."""
    def full_width_half_max(vector, d_theta):
        """ Calculate width of tuning curve at half-maximum. This method should not be applied when tuning curve has multiple bumps. """
        # Remove baseline, calculate half-max
        vector = vector-vector.min()
        half_height = vector.max()/2

        # Get the number of points above half-max and multiply it by the delta angle to get width in angle
        point_inds = numpy.arange(len(vector))
        point_inds_above = point_inds[vector>half_height]
        
        # check if points_above is consecutive
        point_above_diff = point_inds_above[1:] - point_inds_above[0:-1]
        if not numpy.all(point_above_diff==1):
            # Get longest consecutive sequence of points above half-max
            point_inds_above = numpy.split(point_inds_above, numpy.where(numpy.diff(point_inds_above) != 1)[0]+1)
            # concatenate the point_inds_above[0] and point_inds_above[-1] if point_inds_above[0][0] == 0 and point_inds_above[-1][-1] == len(vector)-1
            if point_inds_above[0][0] == 0 and point_inds_above[-1][-1] == len(vector)-1:
                point_inds_above[0] = numpy.concatenate([point_inds_above[0], point_inds_above[-1]])
                point_inds_above.pop()
            num_points_above = numpy.max([len(x) for x in point_inds_above])
        else:
            num_points_above = len(point_inds_above)
        distance = d_theta * num_points_above
        
        return distance

    def tc_cubic(x,y, d_theta=0.5):
        """ Cubic interpolation of tuning curve data. """

        # add first value as last value to make the interpolation periodic
        if 360 not in x:
            x = numpy.append(x, 360)
            y = numpy.append(y, y[0])
        
        mask = ~jnp.isnan(y)
        if numpy.sum(mask) < len(y)*0.9:
            print('Warning: More than 10% of the tuning curve data is missing.')

        # Create cubic interpolation object
        cubic_interpolator = interp1d(x[mask], y[mask], kind='cubic')

        # Create new x values and get interpolated values
        x_interp = numpy.arange(0, max(x), d_theta)
        y_interp = cubic_interpolator(x_interp)

        return x_interp, y_interp

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

        # Preferred orientation - accounting for the double bumps
        peak_loc = y_interp >= numpy.max(y_interp) * 0.95
        # if peak_loc has an interval in (0, 180) and also in (180, 360), take the one in (0, 180)
        if numpy.any(peak_loc[:len(peak_loc)//2]) and numpy.any(peak_loc[len(peak_loc)//2:]):
            peak_loc[len(peak_loc)//2:] = False
        # Get the indices of the peak locations instead of the boolean values
        peak_indices = numpy.where(peak_loc)[0]
        # Find the max index within the peak region (first bump)
        max_peak_index = peak_indices[numpy.argmax(y_interp[peak_indices])]
        # get the max index with the peak_loc
        features['pref_ori'][i] = x_interp[max_peak_index]

        # Gradient for slope calculations
        if numpy.max(y_interp) == 0:
            y_interp_scaled = y_interp
        else:
            y_interp_scaled = y_interp/numpy.max(y_interp)
        grad_y_interp_scaled = numpy.abs(numpy.gradient(y_interp_scaled, x_interp))

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
        half_max_ind = numpy.argmin(numpy.abs(y_interp - half_max))
        # If half max is at the edge of the tuning curve, get the second or third closest point
        if half_max_ind == 0 or half_max_ind == len(y_interp) - 1:
            # get the second closest point to the half-max point
            half_max_ind = numpy.argsort(numpy.abs(y_interp - half_max))[1]
            if half_max_ind == 0 or half_max_ind == len(y_interp) - 1:
                # get the third closest point to the half-max point
                half_max_ind = numpy.argsort(numpy.abs(y_interp - half_max))[2]
        features['slope_hm'][i] = grad_y_interp_scaled[half_max_ind]

    return features


def param_offset_correlations(folder, num_time_inds=2, excluded_runs=[]):
    """ Calculate the Pearson correlation coefficient between the offset threshold and the parameters."""
    
    # Helper function to calculate Pearson correlation
    def calculate_correlations(main_var, params, result_dict):
        """ Calculate the Pearson correlation coefficient between the main variable and the parameters."""
        for _, main_value in main_var.items():
            for param_key, param_values in params.items():
                corr, p_value = scipy.stats.pearsonr(main_value, param_values)
                result_dict[param_key] = [corr, p_value]
        return result_dict

    # Load the relative changes and drop items with NaN values
    rel_changes_train, _ = rel_change_for_runs(folder, num_time_inds=num_time_inds, excluded_runs=excluded_runs)
    
    # Separate offsets, losses, and params
    offsets_rel_change = {key: value for key, value in rel_changes_train.items() if key.endswith('_offset')}
    params_keys = ['J_', 'c', 'f_', 'kappa_Jsup', 'EI_ratio_J_']
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
    data, _ = rel_change_for_runs(folder, num_time_inds=num_time_inds)
    ##################### Correlate offset_th_diff with the combintation of the J_m and J_s, etc. #####################      
    offset_staircase_pars_corr = []
    offset_psychometric_pars_corr = []
    if x_labels is None:
        x_labels = ['J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'f_E','f_I', 'cE_m', 'cI_m', 'cE_s', 'cI_s']
    for i in range(len(x_labels)):
        # Calculate the Pearson correlation coefficient and the p-value
        corr, p_value = scipy.stats.pearsonr(data['staircase_offset'], data[x_labels[i]])
        offset_staircase_pars_corr.append({'corr': corr, 'p_value': p_value})
        corr, p_value = scipy.stats.pearsonr(data['psychometric_offset'], data[x_labels[i]])
        offset_psychometric_pars_corr.append({'corr': corr, 'p_value': p_value})
    
    # Load MVPA_scores and correlate them with the offset threshold and the parameter differences (samples are the different trainings)
    MVPA_scores = csv_to_numpy(folder + '/MVPA_scores.csv') # num_trainings x layer x SGD_ind x ori_ind
    MVPA_scores_diff = MVPA_scores[:,:,-2,:] - MVPA_scores[:,:,-1,:] # num_trainings x layer x ori_ind
    MVPA_psychometric_offset_corr = []
    for i in range(MVPA_scores_diff.shape[1]):
        for j in range(MVPA_scores_diff.shape[2]):
            corr, p_value = scipy.stats.pearsonr(data['psychometric_offset'], MVPA_scores_diff[:,i,j])
            MVPA_psychometric_offset_corr.append({'corr': corr, 'p_value': p_value})
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
    MVPA_corrs = MVPA_psychometric_offset_corr + MVPA_pars_corr

    return offset_staircase_pars_corr, offset_psychometric_pars_corr, MVPA_corrs, data  # Returns a list of dictionaries for each training run

############################## helper functions for MVPA and Mahal distance analysis ##############################

def select_type_mid(r_mid, cell_type='E', phases=4):
    """ Selects the excitatory or inhibitory cell responses. This function assumes that r_mid is 3D (trials x grid points x celltype and phase)"""
    if cell_type=='E':
        map_numbers = jnp.arange(1, 2 * phases, 2)-1 # 0,2,4,6
    else:
        map_numbers = jnp.arange(2, 2 * phases + 1, 2)-1 # 1,3,5,7
    
    out = jnp.zeros((r_mid.shape[0],r_mid.shape[1], int(r_mid.shape[2]/2)))
    for i in range(len(map_numbers)):
        out = out.at[:,:,i].set(r_mid[:,:,map_numbers[i]])

    return jnp.array(out)

vmap_select_type_mid = jax.vmap(select_type_mid, in_axes=(0, None, None))

def gaussian_filter_jax(image, sigma: float):
    """ Applies Gaussian filter to a 2D JAX array (image)."""
    def gaussian_kernel(size: int, sigma: float):
        """ Generates a 2D Gaussian kernel."""
        x = jnp.arange(-size // 2 + 1., size // 2 + 1.)
        y = jnp.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = jnp.meshgrid(x, y)
        kernel = jnp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / jnp.sum(kernel)
        return kernel
    
    size = int(jnp.ceil(3 * sigma) * 2 + 1)  # Kernel size
    kernel = gaussian_kernel(size, sigma)
    smoothed_image = jax.scipy.signal.convolve2d(image, kernel, mode='same')
    return smoothed_image


def smooth_trial(X_trial, num_phases, gridsize_Nx, sigma, num_grid_points):
    """ Smooths a single trial of responses over grid points."""
    smoothed_data_trial = jnp.zeros((gridsize_Nx,gridsize_Nx,num_phases))
    for phase in range(num_phases):
        trial_response = X_trial[phase*num_grid_points:(phase+1)*num_grid_points]
        trial_response = trial_response.reshape(gridsize_Nx,gridsize_Nx)
        smoothed_data_trial_at_phase =  gaussian_filter_jax(trial_response, sigma = sigma)
        smoothed_data_trial=smoothed_data_trial.at[:,:, phase].set(smoothed_data_trial_at_phase)
    return smoothed_data_trial

vmap_smooth_trial = jax.vmap(smooth_trial, in_axes=(0, None, None, None, None))


def vmap_model_response(untrained_pars, ori, n_noisy_trials = 100, J_2x2_m = None, J_2x2_s = None, cE_m = None, cI_m = None, cE_s=None, cI_s=None, f_E = None, f_I = None, kappa_Jsup=jnp.array([[0.0,0.0],[0.0,0.0]]), kappa_Jmid=jnp.array([[0.0,0.0],[0.0,0.0]]), kappa_f=jnp.array([0.0,0.0])):
    """ Generate model response for a given orientation and noise level using vmap_evaluate_model_response."""
    # Generate noisy data
    ori_vec = jnp.repeat(ori, n_noisy_trials)
    jitter_vec = jnp.repeat(0, n_noisy_trials)
    x = untrained_pars.BW_image_jax_inp[4]
    y = untrained_pars.BW_image_jax_inp[5]
    alpha_channel = untrained_pars.BW_image_jax_inp[6]
    mask = untrained_pars.BW_image_jax_inp[7]
    
    # Generate data
    test_grating = BW_image_jit_noisy(untrained_pars.BW_image_jax_inp[0:4], x, y, alpha_channel, mask, ori_vec, jitter_vec)
    
    # Create middle and superficial SSN layers
    ssn_mid=SSN_mid(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_m, untrained_pars.dist_from_single_ori, kappa_Jmid)
    ssn_sup=SSN_sup(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_s, untrained_pars.dist_from_single_ori, untrained_pars.ori_dist, kappa_Jsup=kappa_Jsup)

    # Calculate fixed point for data    
    _, [r_mid, r_sup], _,  _, _ = vmap_evaluate_model_response(ssn_mid, ssn_sup, test_grating, untrained_pars.conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters, untrained_pars.dist_from_single_ori, kappa_f, untrained_pars.ssn_pars.kappa_range)

    return r_mid, r_sup


def SGD_indices_at_stages(df, num_indices=2, peak_offset_flag=False):
    """ Get the indices of the SGD steps at the end (and at the beginning if num_indices=3) of pretraining and at the end of training."""
    # get the number of rows in the dataframe
    num_SGD_steps = len(df)
    SGD_step_inds = numpy.zeros(num_indices, dtype=int)
    stage2_inds = df.index[df['stage'] == 2]
    if num_indices>2:
        SGD_step_inds[0] = df.index[df['stage'] == 0][0] #index of when pretraining starts
        if len(stage2_inds) > 0:
            training_start = stage2_inds[0] #index of when training starts (second stage)
        else: # This case happens when training came back with NA and there is no data from stage 2
            training_start = len(df.index[df['stage']])-1
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
        if len(stage2_inds) > 0:
            SGD_step_inds[0] = df.index[df['stage'] == 2][0] # index of when training starts (first or second stages)
            SGD_step_inds[-1] = num_SGD_steps-1 #index of when training ends    
        else:
            # This case happens when training came back with NA and there is no data from stage 2
            SGD_step_inds[0] = num_SGD_steps-2 #index of when training starts
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
    m = jnp.mean(X, axis=0)
    X_demean = X - m

    # Perform QR decomposition of X_demean
    Q, R = jnp.linalg.qr(X_demean, mode='reduced')

    # Solve for ri in the equation R' * ri = (Y-M) using least squares or directly if R is square and of full rank
    Y_demean=(Y-m).T
    ri = numpy.linalg.solve(R.T, Y_demean)

    # Calculate d as the sum of squares of ri, scaled by (rx-1)
    d = jnp.sum(ri**2, axis=0) * (rx-1)

    return jnp.sqrt(d)


def filtered_model_response(folder, run_ind, ori_list= jnp.asarray([55, 125, 0]), num_noisy_trials = 100, num_stage_inds = 2, r_noise=True, sigma_filter = 1, noise_std=1.0):
    """
    Calculate filtered model response for each orientation in ori_list and for each parameter set (that come from file_name at num_SGD_inds rows)
    """
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

    from parameters import ReadoutPars
    readout_pars = ReadoutPars()

    # Iterate overs SGD_step indices (default is before and after training)
    iloc_ind_vec=[0,-1,-1]
    if num_stage_inds==2:
        stages = [1,2]
    else:
        stages = [0,1,2]
    for stage_ind in range(len(stages)):
        stage = stages[stage_ind]
        # Load parameters from csv for given epoch
        _, trained_pars_stage2, untrained_pars = load_parameters(folder, run_index=run_ind, stage=stage, iloc_ind = iloc_ind_vec[stage_ind])
        # Get the parameters from the trained_pars dictionary and untreatned_pars class
        J_2x2_m, J_2x2_s, cE_m, cI_m, cE_s, cI_s, f_E, f_I, kappa_Jsup, kappa_Jmid, kappa_f = unpack_ssn_parameters(trained_pars_stage2, untrained_pars.ssn_pars)
        
        # Iterate over the orientations
        for ori in ori_list:
            # Calculate model response for each orientation
            r_mid, r_sup = vmap_model_response(untrained_pars, ori, num_noisy_trials, J_2x2_m, J_2x2_s, cE_m, cI_m, cE_s, cI_s, f_E, f_I, kappa_Jsup, kappa_Jmid, kappa_f)
            if r_noise:
                # Add noise to the responses
                noise_mid = generate_noise(num_noisy_trials, length = r_mid.shape[1], num_readout_noise = untrained_pars.num_readout_noise)
                r_mid = r_mid + noise_mid*jnp.sqrt(jax.nn.softplus(r_mid))
                noise_sup = generate_noise(num_noisy_trials, length = r_sup.shape[1], num_readout_noise = untrained_pars.num_readout_noise)
                r_sup = r_sup + noise_sup*jnp.sqrt(jax.nn.softplus(r_sup))

            # Smooth data for each celltype separately with Gaussian filter
            filtered_r_mid_EI= smooth_data(r_mid, untrained_pars.grid_pars.gridsize_Nx, sigma_filter)  #num_noisy_trials x 648
            filtered_r_mid_E=vmap_select_type_mid(filtered_r_mid_EI,'E',4)
            filtered_r_mid_I=vmap_select_type_mid(filtered_r_mid_EI,'I',4)
            filtered_r_mid=jnp.sum(0.8*filtered_r_mid_E + 0.2 *filtered_r_mid_I, axis=-1)# order of summing up phases and mixing I-E matters if we change to sum of squares!

            filtered_r_sup_EI= smooth_data(r_sup, untrained_pars.grid_pars.gridsize_Nx, sigma_filter)
            if filtered_r_sup_EI.ndim == 3:
                filtered_r_sup_E=filtered_r_sup_EI[:,:,0]
                filtered_r_sup_I=filtered_r_sup_EI[:,:,1]
            if filtered_r_sup_EI.ndim == 4:
                filtered_r_sup_E=filtered_r_sup_EI[:,:,:,0]
                filtered_r_sup_I=filtered_r_sup_EI[:,:,:,1]
            filtered_r_sup=0.8*filtered_r_sup_E + 0.2 *filtered_r_sup_I
            
            # Divide the responses by the std of the responses for equal noise effect ***
            filtered_r_mid = filtered_r_mid / jnp.std(filtered_r_mid)
            filtered_r_sup = filtered_r_sup / jnp.std(filtered_r_sup)

            # Concatenate all orientation responses
            if ori == ori_list[0] and stage_ind==0:
                filtered_r_mid_df = filtered_r_mid
                filtered_r_sup_df = filtered_r_sup
                ori_df = jnp.repeat(ori, num_noisy_trials)
                stage_df = jnp.repeat(stage_ind, num_noisy_trials)
            else:
                filtered_r_mid_df = jnp.concatenate((filtered_r_mid_df, filtered_r_mid))
                filtered_r_sup_df = jnp.concatenate((filtered_r_sup_df, filtered_r_sup))
                ori_df = jnp.concatenate((ori_df, jnp.repeat(ori, num_noisy_trials)))
                stage_df = jnp.concatenate((stage_df, jnp.repeat(stage_ind, num_noisy_trials)))

    # Get the responses from the center of the grid 
    min_ind = int((readout_pars.readout_grid_size[0] - readout_pars.readout_grid_size[1])/2)
    max_ind = int(readout_pars.readout_grid_size[0]) - min_ind
    filtered_r_mid_box_df = filtered_r_mid_df[:,min_ind:max_ind, min_ind:max_ind]
    filtered_r_sup_box_df = filtered_r_sup_df[:,min_ind:max_ind, min_ind:max_ind]

    # Add noise to the responses (numpy.std(filtered_r) is about 0.3)
    filtered_r_mid_box_noisy_df= filtered_r_mid_box_df + numpy.random.normal(0, noise_std, filtered_r_mid_box_df.shape)
    filtered_r_sup_box_noisy_df= filtered_r_sup_box_df + numpy.random.normal(0, noise_std, filtered_r_sup_box_df.shape)

    output = dict(ori = ori_df, stage = stage_df, r_mid = filtered_r_mid_box_noisy_df, r_sup = filtered_r_sup_box_noisy_df)
    
    return output


######### Calculate MVPA and Mahalanobis distance for before pretraining, after pretraining and after training #########
def MVPA_Mahal_analysis(folder, num_runs, num_stages=2, r_noise = True, sigma_filter=1, num_noisy_trials=100, filtered_r_noise_std=1.0, excluded_runs=[]):
    """ Calculate MVPA and Mahalanobis distance for each run and layer at different stages of training."""
    # Shared parameters
    ori_list = numpy.asarray([55, 125, 0])
    num_layers=2 # number of layers

    ####### Setup for the MVPA analysis #######
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)) # SVM classifier

    # Initialize the MVPA scores matrix
    valid_runs = numpy.setdiff1d(numpy.arange(num_runs), excluded_runs)
    num_valid_runs = len(valid_runs)
    MVPA_scores = numpy.zeros((num_valid_runs, num_layers, num_stages, len(ori_list)-1))
    Mahal_scores = numpy.zeros((num_valid_runs, num_layers, num_stages, len(ori_list)-1))

    ####### Setup for the Mahalanobis distance analysis #######
    num_PC_used=20 # number of principal components used for the analysis
    
    # Initialize arrays to store Mahalanobis distances and related metrics
    LMI_across = numpy.zeros((num_valid_runs,num_layers,num_stages-1))
    LMI_within = numpy.zeros((num_valid_runs,num_layers,num_stages-1))
    LMI_ratio = numpy.zeros((num_valid_runs,num_layers,num_stages-1))
    
    mahal_within_train_all = numpy.zeros((num_valid_runs,num_layers,num_stages, num_noisy_trials))
    mahal_within_untrain_all = numpy.zeros((num_valid_runs,num_layers,num_stages, num_noisy_trials))
    mahal_train_control_all = numpy.zeros((num_valid_runs,num_layers,num_stages, num_noisy_trials))
    mahal_untrain_control_all = numpy.zeros((num_valid_runs,num_layers,num_stages, num_noisy_trials))
    train_SNR_all=numpy.zeros((num_valid_runs,num_layers,num_stages, num_noisy_trials))
    untrain_SNR_all=numpy.zeros((num_valid_runs,num_layers,num_stages, num_noisy_trials))

    mahal_train_control_mean = numpy.zeros((num_valid_runs,num_layers,num_stages))
    mahal_untrain_control_mean = numpy.zeros((num_valid_runs,num_layers,num_stages))
    mahal_within_train_mean = numpy.zeros((num_valid_runs,num_layers,num_stages))
    mahal_within_untrain_mean = numpy.zeros((num_valid_runs,num_layers,num_stages))
    train_SNR_mean = numpy.zeros((num_valid_runs,num_layers,num_stages))
    untrain_SNR_mean = numpy.zeros((num_valid_runs,num_layers,num_stages))
    
    # Define pca model
    pca = PCA(n_components=num_PC_used)
      
    # Iterate over the different parameter initializations (runs)
    for i, run_ind in enumerate(valid_runs):
        start_time=time.time()
                
        # Calculate num_noisy_trials filtered model response for each oris in ori list and for each parameter set (that come from file_name at num_stage_inds rows)
        r_mid_sup = filtered_model_response(folder, run_ind, ori_list = ori_list, num_noisy_trials = num_noisy_trials, num_stage_inds = num_stages, r_noise = r_noise, sigma_filter = sigma_filter, noise_std = filtered_r_noise_std)
        # Note: r_mid_sup is a dictionary with the oris and stages saved in them
        r_ori = r_mid_sup['ori']
        mesh_train_ = r_ori == ori_list[0] 
        mesh_untrain_ = r_ori == ori_list[1]
        mesh_control_ = r_ori == ori_list[2]
        # Iterate over the layers and stages
        for layer in range(num_layers):
            if layer == 0:
                r_l = r_mid_sup['r_mid']
            else:
                r_l = r_mid_sup['r_sup']
            r_l = r_l.reshape((r_l.shape[0], -1))
            if numpy.any(numpy.isnan(r_l)):
                print('Warning: NaNs in the responses')
                continue
            else:
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
                    for j in range(10):
                        X_untrain, X_test_untrain, y_untrain, y_test_untrain = train_test_split(untrain_control_data, untrain_control_label, test_size=0.5, random_state=j)
                        X_train, X_test_train, y_train, y_test_train = train_test_split(train_control_data, train_control_label, test_size=0.5, random_state=j)
                        score_train = clf.fit(X_train, y_train).score(X_test_train, y_test_train)
                        score_untrain = clf.fit(X_untrain, y_untrain).score(X_test_untrain, y_test_untrain)
                        scores_untrain.append(score_untrain)
                        scores_train.append(score_train)
                    MVPA_scores[i,layer,stage_ind, 1] = jnp.mean(jnp.array(scores_untrain))
                    MVPA_scores[i,layer,stage_ind, 0] = jnp.mean(jnp.array(scores_train))

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
    
                    # Save Mahal distances and ratios
                    mahal_train_control_all[i,layer,stage_ind,:] = mahal_train_control
                    mahal_untrain_control_all[i,layer,stage_ind,:] = mahal_untrain_control
                    mahal_within_train_all[i,layer,stage_ind,:] = mahal_within_train
                    mahal_within_untrain_all[i,layer,stage_ind,:] = mahal_within_untrain
                    train_SNR_all[i,layer,stage_ind,:] = mahal_train_control / mahal_within_train
                    untrain_SNR_all[i,layer,stage_ind,:] = mahal_untrain_control / mahal_within_untrain

                    # Average over trials
                    mahal_train_control_mean[i,layer,stage_ind] = numpy.mean(mahal_train_control)
                    mahal_untrain_control_mean[i,layer,stage_ind] = numpy.mean(mahal_untrain_control)
                    mahal_within_train_mean[i,layer,stage_ind] = numpy.mean(mahal_within_train)
                    mahal_within_untrain_mean[i,layer,stage_ind] = numpy.mean(mahal_within_untrain)
                    train_SNR_mean[i,layer,stage_ind] = numpy.mean(train_SNR_all[i,layer,stage_ind,:])
                    untrain_SNR_mean[i,layer,stage_ind] = numpy.mean(untrain_SNR_all[i,layer,stage_ind,:])

                # Calculate learning modulation indices (LMI)
                for stage_ind_ in range(num_stages-1):
                    LMI_across[i,layer,stage_ind_] = (mahal_train_control_mean[i,layer,stage_ind_+1] - mahal_train_control_mean[i,layer,stage_ind_]) - (mahal_untrain_control_mean[i,layer,stage_ind_+1] - mahal_untrain_control_mean[i,layer,stage_ind_] )
                    LMI_within[i,layer,stage_ind_] = (mahal_within_train_mean[i,layer,stage_ind_+1] - mahal_within_train_mean[i,layer,stage_ind_]) - (mahal_within_untrain_mean[i,layer,stage_ind_+1] - mahal_within_untrain_mean[i,layer,stage_ind_] )
                    LMI_ratio[i,layer,stage_ind_] = (train_SNR_mean[i,layer,stage_ind_+1] - train_SNR_mean[i,layer,stage_ind_]) - (untrain_SNR_mean[i,layer,stage_ind_+1] - untrain_SNR_mean[i,layer,stage_ind_] )
        
        print(f'runtime of run {run_ind}:',time.time()-start_time)

    ################# Create dataframes for the Mahalanobis distances and LMI #################
    Mahal_scores[:,:,:,0] = mahal_train_control_mean
    Mahal_scores[:,:,:,1] = mahal_untrain_control_mean

    return MVPA_scores, Mahal_scores


def MVPA_anova(folder, file_name='MVPA_scores.csv'):
    """ Perform two-way ANOVA on MVPA or Mahalanobis scores to measure the effect of SGD steps and orientation. """
    
    # Load the data from the CSV
    scores = csv_to_numpy(os.path.join(folder, file_name))  # Assume this returns a 4D numpy array
    
    num_trainings, num_layers, num_sgd_inds, num_ori_inds = scores.shape
    
    # Prepare dataset
    data = []
    for training in range(num_trainings):
        for layer in range(num_layers):
            for sgd in range(1,num_sgd_inds):
                for ori in range(num_ori_inds):
                    data.append({
                        'Score': scores[training, layer, sgd, ori],  # The score value for each training, layer, sgd, and ori
                        'SGD_step': sgd,                             # The SGD step as a factor
                        'Ori': ori,                                  # The orientation factor
                        'Layer': layer                               # The layer factor
                    })
    
    # Convert the list to a DataFrame
    df = pd.DataFrame(data)
    
    # Perform two-way ANOVA for each layer
    for layer in range(num_layers):
        layer_data = df[df['Layer'] == layer]  # Filter data by the current layer
        
        # Perform two-way ANOVA using an OLS model
        model = ols('Score ~ C(SGD_step) + C(Ori) + C(SGD_step):C(Ori)', data=layer_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2) # *** ValueError array must not contain infs or NaNs
        
        print(f"ANOVA results for Layer {layer}:")
        print(anova_table)
        '''
        # paired t-test on MVPA_score diff 
        print(f"Paired t-test for Layer {layer}:")
        ori_tr = 0
        trained = numpy.array(layer_data[(layer_data['Ori'] == ori_tr) & (layer_data['SGD_step'] == 2)]['Score']) - numpy.array(layer_data[(layer_data['Ori'] == ori_tr) & (layer_data['SGD_step'] == 1)]['Score'])
        ori_ut = 1
        untrained = numpy.array(layer_data[(layer_data['Ori'] == ori_ut) & (layer_data['SGD_step'] == 2)]['Score']) - numpy.array(layer_data[(layer_data['Ori'] == ori_ut) & (layer_data['SGD_step'] == 1)]['Score'])
        t_stat, p_val = scipy.stats.ttest_rel(trained, untrained)
        print(f"t-statistic: {t_stat}, p-value: {p_val}")
        print(f'55 mean:{numpy.mean(trained)}, std:{numpy.std(trained)}')
        print(f'125 mean:{numpy.mean(untrained)}, std:{numpy.std(untrained)}')
        '''


def make_exclude_run_csv(folder, num_runs, offset_condition=True):
    """ Create a csv file with the indices of the runs that should be excluded from the analysis based on missing data and the offset condition."""
    excluded_inds = []
    keys_metrics = ['psychometric_offset', 'staircase_offset']
    for run_index in range(num_runs):
        df, _, no_train_data  = data_from_run(folder, run_index=run_index, num_indices=3)
        if no_train_data:
            excluded_inds.append(run_index)
        else:
            if offset_condition:
                if any(df[keys_metrics[0]][-11:-1] > 10)  and sum(df[keys_metrics[1]][-11:-1] > 9.9) > 8:
                    excluded_inds.append(run_index)
    # Save excluded_inds into a csv file
    excluded_inds_df = pd.DataFrame(excluded_inds)
    excluded_inds_df.to_csv(folder + '/excluded_runs.csv', index=False, header=False)

    return excluded_inds

########## CALCULATE TUNING CURVES ############
def main_tuning_curves(folder_path, num_training, start_time_in_main, stage_inds = range(3), tc_ori_list = numpy.arange(0,180,6), add_header=True, filename=None):
    """ Calculate tuning curves for the different runs and different stages in each run """
    from parameters import GridPars, SSNPars
    grid_pars, ssn_pars = GridPars(), SSNPars()
    # Define the filename for the tuning curves 
    if filename is not None:
        tc_file_path = os.path.join(folder_path, filename)
    else:
        tc_file_path = os.path.join(folder_path, 'tuning_curves.csv')
    if os.path.exists(tc_file_path):
        print(f'Tuning curves already exist in {tc_file_path}.')
    else:      
        if add_header:
            # Define the header for the tuning curves
            tc_headers = []
            tc_headers.append('run_index')
            tc_headers.append('training_stage')
            # Headers for middle layer cells - order matches the gabor filters
            type_str = ['_E_','_I_']
            for phase_ind in range(ssn_pars.phases):
                for type_ind in range(2):
                    for i in range(grid_pars.gridsize_Nx**2):
                        tc_header = 'G'+ str(i+1) + type_str[type_ind] + 'Ph' + str(phase_ind) + '_M'
                        tc_headers.append(tc_header)
            # Headers for superficial layer cells
            for type_ind in range(2):
                for i in range(grid_pars.gridsize_Nx**2):
                    tc_header = 'G'+str(i+1) + type_str[type_ind] +'S'
                    tc_headers.append(tc_header)
        else:
            tc_headers = False

        # Loop over the different runs
        iloc_ind_vec = [0,-1,-1]
        stages = [0,1,2]
        for i in range(num_training):    
            # Loop over the different stages (before pretraining, after pretraining, after training) and calculate and save tuning curves
            for stage_ind in stage_inds:
                _, trained_pars_dict, untrained_pars = load_parameters(folder_path, run_index=i, stage=stages[stage_ind], iloc_ind=iloc_ind_vec[stage_ind])
                _, _ = tuning_curve(untrained_pars, trained_pars_dict, tc_file_path, ori_vec=tc_ori_list, training_stage=stage_ind, run_index=i, header=tc_headers)
                tc_headers = False
            if i%10==0:    
                print(f'Finished calculating tuning curves for training {i} in {time.time()-start_time_in_main} seconds')


########## CALCULATE MVPA SCORES AND MAHALANOBIS DISTANCES ############
def main_MVPA(folder, num_runs, num_stages=2, sigma_filter=5, r_noise=True, num_noisy_trials=100, excluded_runs=[]):
    """ Calculate MVPA scores for before pretraining, after pretraining and after training - score should increase for trained ori more than for other two oris especially in superficial layer. """
    
    if not os.path.exists(folder +'/MVPA_scores.csv'):
        MVPA_scores, Mahal_scores = MVPA_Mahal_analysis(folder,num_runs=num_runs, num_stages=num_stages, r_noise = r_noise, sigma_filter=sigma_filter, num_noisy_trials=num_noisy_trials, excluded_runs=excluded_runs)
        _ = save_numpy_to_csv(MVPA_scores, folder + '/MVPA_scores.csv')
        _ = save_numpy_to_csv(Mahal_scores, folder + '/Mahal_scores.csv')
    else:
        MVPA_scores = csv_to_numpy(folder +'/MVPA_scores.csv')
        Mahal_scores = csv_to_numpy(folder +'/Mahal_scores.csv')