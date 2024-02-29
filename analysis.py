import pandas as pd
import numpy
import csv
import os
import copy 

import jax.numpy as np

from util import BW_Grating, sep_exponentiate
from SSN_classes import SSN_mid, SSN_sup
from model import evaluate_model_response

def create_grating(stimuli_pars, batch_size): # *** use Bw_image_jit in tuning curve and then we do not need this function!
    '''
    Create input stimuli gratings.
    Input:
       stimuli pars
       batch_size - batch size
    
    Output:
        dictionary containing stimuli gratings
    '''
    
    #initialise empty arrays
    ref_ori = stimuli_pars.ref_ori
    data_dict = {'input':[]}

    for _ in range(batch_size):
        #create grating
        ref = BW_Grating(ori_deg = ref_ori, jitter=0, stimuli_pars = stimuli_pars).BW_image().ravel()

        data_dict['input'].append(ref)
        
    data_dict['input'] = np.asarray(data_dict['input'])

    return data_dict

def tuning_curves(constant_pars, trained_pars, tuning_curves_filename=None, ori_vec=range(0,180,6)):
    '''
    Calculate responses of middle and superficial layers to different orientations.
    '''
    ref_ori_saved = float(constant_pars.stimuli_pars.ref_ori)
    for key in list(trained_pars.keys()):  # Use list to make a copy of keys to avoid RuntimeError
        # Check if key starts with 'log'
        if key.startswith('log'):
            # Create a new key by removing 'log' prefix
            new_key = key[4:]
            # Exponentiate the values and assign to the new key
            trained_pars[new_key] = sep_exponentiate(trained_pars[key])
    
    ssn_mid=SSN_mid(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, J_2x2=trained_pars['J_2x2_m'])
    
    N_ori = len(ori_vec)
    new_rows = []
    for i in range(N_ori):
        constant_pars.stimuli_pars.ref_ori = ori_vec[i]
        train_data = create_grating(constant_pars.stimuli_pars, 1)
        ssn_sup=SSN_sup(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, J_2x2=trained_pars['J_2x2_s'], p_local=constant_pars.ssn_layer_pars.p_local_s, oris=constant_pars.oris, s_2x2=constant_pars.ssn_layer_pars.s_2x2_s, sigma_oris = constant_pars.ssn_layer_pars.sigma_oris, ori_dist = constant_pars.ori_dist, train_ori = constant_pars.stimuli_pars.ref_ori)
        _, _, [_,_], [_,_], [_,_,_,_], [r_mid_i, r_sup_i] = evaluate_model_response(ssn_mid, ssn_sup, train_data['input'], constant_pars.conv_pars, trained_pars['c_E'], trained_pars['c_I'], trained_pars['f_E'], trained_pars['f_I'], constant_pars.gabor_filters)
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

    constant_pars.stimuli_pars.ref_ori = ref_ori_saved

    return responses_sup, responses_mid


def param_ratios(results_file):
    results = pd.read_csv(results_file, header=0)

    if "J_EE" in results.columns:
        Js = results[["J_EE", "J_EI", "J_IE", "J_II"]]
        Js = Js.to_numpy()
        print("J ratios = ", np.array((Js[-1, :] / Js[0, :] - 1) * 100, dtype=int))

    if "s_EE" in results.columns:
        ss = results[["s_EE", "s_EI", "s_IE", "s_II"]]
        ss = ss.to_numpy()
        print("s ratios = ", np.array((ss[-1, :] / ss[0, :] - 1) * 100, dtype=int))

    if "c_E" in results.columns:
        cs = results[["c_E", "c_I"]]
        cs = cs.to_numpy()
        print("c ratios = ", np.array((cs[-1, :] / cs[0, :] - 1) * 100, dtype=int))

    if "sigma_orisE" in results.columns:
        sigma_oris = results[["sigma_orisE", "sigma_orisI"]]
        sigma_oris = sigma_oris.to_numpy()
        print(
            "sigma_oris ratios = ",
            np.array((sigma_oris[-1, :] / sigma_oris[0, :] - 1) * 100, dtype=int),
        )

    if "kappa_preE" in results.columns:
        kappa_pre = results[["kappa_preE", "kappa_preI"]]
        kappa_pre = kappa_pre.to_numpy()
        print(
            "kappa_pre ratios = ",
            np.array((kappa_pre[-1, :] / kappa_pre[0, :] - 1) * 100, dtype=int),
        )

    if "kappa_postE" in results.columns:
        kappa_post = results[["kappa_postE", "kappa_postI"]]
        kappa_post = kappa_post.to_numpy()
        print(
            "kappa_post ratios = ",
            np.array((kappa_post[-1, :] / kappa_post[0, :] - 1) * 100, dtype=int),
        )

    if "sigma_oris" in results.columns:
        sigma_oris = results[["sigma_oris"]]
        sigma_oris = sigma_oris.to_numpy()

        print(
            "sigma_oris ratios = ",
            np.array((sigma_oris[-1, :] / sigma_oris[0, :] - 1) * 100, dtype=int),
        )


def param_ratios_two_layer(results_file, epoch=None, percent_acc=0.85):
    results = pd.read_csv(results_file, header=0)

    if epoch == None:
        accuracies = list(results["val_accuracy"][:20].values)
        count = 9
        while np.asarray(accuracies).mean() < percent_acc:
            count += 1
            del accuracies[0]
            if count > len(results["val_accuracy"]):
                break
            else:
                accuracies.append(results["val_accuracy"][count])

        epoch = results["epoch"][count]
        epoch_index = results[results["epoch"] == epoch].index
        print(epoch, epoch_index)

    if epoch == -1:
        epoch_index = epoch
    else:
        epoch_index = results[results["epoch"] == epoch].index

    if "J_EE_m" in results.columns:
        Js = results[["J_EE_m", "J_EI_m", "J_IE_m", "J_II_m"]]
        Js = Js.to_numpy()
        print("J_m ratios = ", np.array((Js[epoch_index, :] / Js[0, :] - 1) * 100))

    if "J_EE_s" in results.columns:
        Js = results[["J_EE_s", "J_EI_s", "J_IE_s", "J_II_s"]]
        Js = Js.to_numpy()
        print("J_s ratios = ", np.array((Js[epoch_index, :] / Js[0, :] - 1) * 100))

    if "s_EE_m" in results.columns:
        ss = results[["s_EE_m", "s_EI_m", "s_IE_m", "s_II_m"]]
        ss = ss.to_numpy()
        print(
            "s_m ratios = ",
            np.array(
                (ss[epoch_index, :] / ss[0, :] - 1) * 100,
            ),
        )

    if "s_EE_s" in results.columns:
        ss = results[["s_EE_s", "s_EI_s", "s_IE_s", "s_II_s"]]
        ss = ss.to_numpy()
        print("s_s ratios = ", np.array((ss[epoch_index, :] / ss[0, :] - 1) * 100))

    if "c_E" in results.columns:
        cs = results[["c_E", "c_I"]]
        cs = cs.to_numpy()
        print("c ratios = ", np.array((cs[epoch_index, :] / cs[0, :] - 1) * 100))

    if "sigma_orisE" in results.columns:
        sigma_oris = results[["sigma_orisE", "sigma_orisI"]]
        sigma_oris = sigma_oris.to_numpy()
        print(
            "sigma_oris ratios = ",
            np.array((sigma_oris[epoch_index, :] / sigma_oris[0, :] - 1) * 100),
        )

    if "sigma_oris" in results.columns:
        sigma_oris = results[["sigma_oris"]]
        sigma_oris = sigma_oris.to_numpy()
        print(
            "sigma_oris ratios = ",
            np.array((sigma_oris[epoch_index, :] / sigma_oris[0, :] - 1) * 100),
        )

    if "f_E" in results.columns:
        fs = results[["f_E", "f_I"]]
        fs = fs.to_numpy()
        print("f ratios = ", np.array((fs[epoch_index, :] / fs[0, :] - 1) * 100))

    if "kappa_preE" in results.columns:
        kappas = results[["kappa_preE", "kappa_preI", "kappa_postE", "kappa_postI"]]
        kappas = kappas.to_numpy()
        print("kappas = ", kappas[epoch_index, :])


def assemble_pars(all_pars, matrix=True):
    """
    Take parameters from csv file and

    """
    pre_train = np.asarray(all_pars.iloc[0].tolist())
    post_train = np.asarray(all_pars.iloc[-1].tolist())

    if matrix == True:
        matrix_pars = lambda Jee, Jei, Jie, Jii: np.array([[Jee, Jei], [Jie, Jii]])

        pre_train = matrix_pars(*pre_train)
        post_train = matrix_pars(*post_train)

    return pre_train, post_train


def accuracies(all_acc, p=0.75):
    """
    Print accuracies and jitters that give have a probability p of having initial accuracy betwen 0.45-0.55
    """

    acc_to_save = []
    for x in range(len(all_acc)):
        acc = all_acc[x][2]
        if ((0.45 < acc) & (acc < 0.55)).sum() / len(acc) > p:
            print(all_acc[x][0], all_acc[x][1])
            acc_to_save.append([all_acc[x][0], all_acc[x][1]])

    return acc_to_save


def obtain_regular_indices(ssn, number=8, test_oris=None):
    """
    Function takes SSN network and outputs linearly separated list of orientation indices
    """

    array = ssn.ori_map[2:7, 2:7]
    array = array.ravel()

    if test_oris:
        pass
    else:
        test_oris = np.linspace(array.min(), array.max(), number)
    indices = []

    for test_ori in test_oris:
        idx = (np.abs(array - test_ori)).argmin()
        indices.append(idx)

    testing_angles = [array[idx] for idx in indices]
    print(testing_angles)

    return indices


def obtain_min_max_indices(ssn, fp):
    idx = (ssn.ori_vec > 45) * (ssn.ori_vec < 65)
    indices = np.where(idx)
    responses_45_65 = fp[indices]
    j_s = []
    max_min_indices = np.concatenate(
        [np.argsort(responses_45_65)[:3], np.argsort(responses_45_65)[-3:]]
    )

    for i in max_min_indices:
        j = indices[0][i]
        j_s.append(j)

    return j_s


def label_neuron(index):
    labels = ["E_ON", "I_ON", "E_OFF", "I_OFF"]
    return labels[int(np.floor(index / 81))]


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


def create_grating_single(stimuli_pars, n_trials = 10):

    all_stimuli = []
    jitter_val = stimuli_pars.jitter_val
    ref_ori = stimuli_pars.ref_ori

    for i in range(0, n_trials):
        jitter = numpy.random.uniform(-jitter_val, jitter_val, 1)

        #create reference grating
        ref = BW_Grating(ori_deg = ref_ori, jitter=jitter, stimuli_pars = stimuli_pars).BW_image().ravel()
        all_stimuli.append(ref)
    
    return np.vstack([all_stimuli])


def response_matrix(J_2x2_m, J_2x2_s, kappa_pre, kappa_post, c_E, c_I, f_E, f_I, constant_pars, tuning_pars, radius_list, ori_list, trained_ori):
    '''
    Construct a response matrix of sizze n_orientations x n_neurons x n_radii
    '''
    #Initialize ssn
    ssn_mid=SSN_mid(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_m, filter_pars=constant_pars.filter_pars, J_2x2=J_2x2_m, gE = constant_pars.gE[0], gI=constant_pars.gI[0], ori_map = constant_pars.ssn_ori_map)
    ssn_sup=SSN_sup(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_s, J_2x2=J_2x2_s, s_2x2=constant_pars.s_2x2, sigma_oris = constant_pars.sigma_oris, ori_map = constant_pars.ssn_ori_map, train_ori = trained_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)

    responses_sup = []
    responses_mid = []
    conv_pars = constant_pars.conv_pars
    
    for i in range(len(ori_list)):
        
        #Find responses at different stimuli radii
        x_response_sup, x_response_mid = surround_suppression(ssn_mid, ssn_sup, tuning_pars, conv_pars, radius_list, c_E, c_I, f_E, f_I, ref_ori = ori_list[i])
        print(x_response_sup.shape)
        #inputs.append(SSN_input)
        responses_sup.append(x_response_sup)
        responses_mid.append(x_response_mid)
        
    return np.stack(responses_sup, axis = 2), np.stack(responses_mid, axis = 2)


def surround_suppression(ssn_mid, ssn_sup, tuning_pars, conv_pars, radius_list, c_E, c_I, f_E, f_I, ref_ori):    
    
    '''
    Produce matrix response for given two layer ssn network given a list of varying stimuli radii
    '''
    
    all_responses_sup = []
    all_responses_mid = []
    
    tuning_pars.ref_ori = ref_ori
   
    print(ref_ori) #create stimuli in the function just input radii)
    for radii in radius_list:
        
        tuning_pars.outer_radius = radii
        tuning_pars.inner_radius = radii*(2.5/3)
        
        stimuli = create_grating_single(n_trials = 1, stimuli_pars = tuning_pars)
        stimuli = stimuli.squeeze()
        
        #stimuli = np.load('/mnt/d/ABG_Projects_Backup/ssn_modelling/ssn-simulator/debugging/new_stimuli.npy')
        
        r_sup, _, _, [max_E_mid, max_I_mid, max_E_sup, max_I_sup], [fp_mid, fp_sup] = evaluate_model_response(ssn_mid, ssn_sup, stimuli, conv_pars, c_E, c_I, f_E, f_I)
        
         
        all_responses_sup.append(fp_sup.ravel())
        all_responses_mid.append(fp_mid.ravel())
        print('Mean population response {} (max in population {}), centre neurons {}'.format(fp_sup.mean(), fp_sup.max(), fp_sup.mean()))
    
    return np.vstack(all_responses_sup), np.vstack(all_responses_mid)
