import pandas as pd
import numpy
import os

import jax.numpy as np

from util import sep_exponentiate
from util_gabor import BW_image_jit, BW_image_jit_noisy, BW_image_vmap
from SSN_classes import SSN_mid, SSN_sup
from model import evaluate_model_response


def tuning_curves(untrained_pars, trained_pars, tuning_curves_filename=None, ori_vec=np.arange(0,180,6)):
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
            trained_pars[new_key] = sep_exponentiate(trained_pars[key])
    
    ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=trained_pars['J_2x2_m'])
    
    N_ori = len(ori_vec)
    new_rows = []
    x = untrained_pars.BW_image_jax_inp[5]
    y = untrained_pars.BW_image_jax_inp[6]
    alpha_channel = untrained_pars.BW_image_jax_inp[7]
    mask = untrained_pars.BW_image_jax_inp[8]
    background = untrained_pars.BW_image_jax_inp[9]
    
    train_data = BW_image_jit(untrained_pars.BW_image_jax_inp[0:5], x, y, alpha_channel, mask, background, ori_vec, np.zeros(N_ori))
    for i in range(N_ori):
        ssn_sup=SSN_sup(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=trained_pars['J_2x2_s'], p_local=untrained_pars.ssn_layer_pars.p_local_s, oris=untrained_pars.oris, s_2x2=untrained_pars.ssn_layer_pars.s_2x2_s, sigma_oris = untrained_pars.ssn_layer_pars.sigma_oris, ori_dist = untrained_pars.ori_dist, train_ori = untrained_pars.stimuli_pars.ref_ori)
        _, _, [_,_], [_,_], [_,_,_,_], [r_mid_i, r_sup_i] = evaluate_model_response(ssn_mid, ssn_sup, train_data[i,:], untrained_pars.conv_pars, trained_pars['c_E'], trained_pars['c_I'], trained_pars['f_E'], trained_pars['f_I'], untrained_pars.gabor_filters)
        if i==0:
            responses_mid = numpy.zeros((N_ori,len(r_mid_i)))
            responses_sup = numpy.zeros((N_ori,len(r_sup_i)))
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


def full_width_half_max(vector, d_theta):
    vector = vector - vector.min()
    half_height = vector.max() / 2
    points_above = len(vector[vector > half_height])

    distance = d_theta * points_above

    return distance


def sort_neurons(ei_indices, close_far_indices):
    empty_list = []
    for i in ei_indices:
        if i in close_far_indices:
            empty_list.append(i)

    return np.asarray([empty_list])


def close_far_indices(train_ori, ssn):
    close_indices = []
    far_indices = []

    upper_range = train_ori + 90 / 2
    print(upper_range)

    for i in range(len(ssn.ori_vec)):
        if 0 < ssn.ori_vec[i] <= upper_range:
            close_indices.append(i)
        else:
            far_indices.append(i)

    return np.asarray([close_indices]), np.asarray([far_indices])


def sort_close_far_EI(ssn, train_ori):
    close, far = close_far_indices(55, ssn)
    close = close.squeeze()
    far = far.squeeze()
    e_indices = np.where(ssn.tau_vec == ssn.tauE)[0]
    i_indices = np.where(ssn.tau_vec == ssn.tauI)[0]

    e_close = sort_neurons(e_indices, close)
    e_far = sort_neurons(e_indices, far)
    i_close = sort_neurons(i_indices, close)
    i_far = sort_neurons(i_indices, far)

    return e_close, e_far, i_close, i_far
