import pandas as pd
import numpy
from numpy import random
import copy
import jax.numpy as np

from util import take_log, create_grating_pairs
from SSN_classes import SSN_mid, SSN_sup
from model import evaluate_model_response

def perturb_params(param_dict, percent = 0.1):
    param_perturbed = copy.deepcopy(param_dict)
    for key, param_array in param_dict.items():
        if type(param_array) == float:
            random_mtx = random.uniform(low=-1, high=1)
        else:
            random_mtx = random.uniform(low=-1, high=1, size=param_array.shape)
            random_mtx = np.array(random_mtx)
        param_perturbed[key] = param_array + percent * param_array * random_mtx
    return param_perturbed

def randomize_params(readout_pars, ssn_layer_pars, constant_pars, percent=0.1):
    #define the parameters that get perturbed
    pars_stage1 = dict(w_sig=readout_pars.w_sig, b_sig=readout_pars.b_sig)
    temp_w_sig=numpy.random.normal(scale = 0.25, size=(len(readout_pars.w_sig),)) / numpy.sqrt(len(readout_pars.w_sig))
    pars_stage1['w_sig']=np.array(temp_w_sig)
    pars_stage2_nolog = dict(J_m_temp=ssn_layer_pars.J_2x2_m, J_s_temp=ssn_layer_pars.J_2x2_s, c_E_temp=ssn_layer_pars.c_E, c_I_temp=ssn_layer_pars.c_I, f_E_temp=ssn_layer_pars.f_E, f_I_temp=ssn_layer_pars.f_I)
    
    # Perturb parameters under conditions for J_mid and convergence of the differential equations of the model
    i=0
    cond1 = False
    cond2 = False
    cond3 = False
    cond4 = False
    cond5 = False
    tol_th = 1e-1
    while not (cond1 and cond2 and cond3 and cond4 and cond5):
        params_perturbed = perturb_params(pars_stage2_nolog, percent)
        cond1 = np.abs(params_perturbed['J_m_temp'][0,0]*params_perturbed['J_m_temp'][1,1]) + tol_th < np.abs(params_perturbed['J_m_temp'][1,0]*params_perturbed['J_m_temp'][0,1])
        cond2 = np.abs(params_perturbed['J_m_temp'][0,1]*ssn_layer_pars.gI_m) + tol_th < np.abs(params_perturbed['J_m_temp'][1,1]*ssn_layer_pars.gE_m)
        # checking the convergence of the differential equations of the model
        ssn_mid=SSN_mid(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, J_2x2=params_perturbed['J_m_temp'])
        ssn_sup=SSN_sup(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, J_2x2=params_perturbed['J_s_temp'], p_local=constant_pars.ssn_layer_pars.p_local_s, oris=constant_pars.oris, s_2x2=constant_pars.ssn_layer_pars.s_2x2_s, sigma_oris = constant_pars.ssn_layer_pars.sigma_oris, ori_dist = constant_pars.ori_dist, train_ori = constant_pars.stimuli_pars.ref_ori)
        train_data = create_grating_pairs(constant_pars.stimuli_pars, 1)
        r_ref,_, [_, _], [avg_dx_mid, avg_dx_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'], constant_pars.conv_pars, params_perturbed['c_E_temp'], params_perturbed['c_I_temp'], params_perturbed['f_E_temp'], params_perturbed['f_I_temp'], constant_pars.gabor_filters)
        cond3 = not numpy.any(numpy.isnan(r_ref))
        cond4 = avg_dx_mid + avg_dx_sup < 50
        cond5 = min([max_E_mid, max_I_mid, max_E_sup, max_I_sup])>10 and max([max_E_mid, max_I_mid, max_E_sup, max_I_sup])<101
        if i>20:
            print("Perturbed parameters violate inequality conditions or lead to divergence in diff equation.")
            break
        else:
            i = i+1

    pars_stage2 = dict(
        log_J_2x2_m= take_log(params_perturbed['J_m_temp']),
        log_J_2x2_s= take_log(params_perturbed['J_s_temp']),
        c_E=params_perturbed['c_E_temp'],
        c_I=params_perturbed['c_I_temp'],
        f_E=params_perturbed['f_E_temp'],
        f_I=params_perturbed['f_I_temp'],
    )

    return pars_stage1, pars_stage2


def load_pretrained_parameters(file_path, readout_grid_size=5, iloc_ind=-1):

    # Get the last row of the given csv file
    df = pd.read_csv(file_path)
    selected_row = df.iloc[iloc_ind]

    # Extract matrices from dataframe
    w_sig_keys = [f'w_sig_{i}' for i in range(1, readout_grid_size*readout_grid_size+1)] 
    J_m_keys = ['logJ_m_EE','logJ_m_EI','logJ_m_IE','logJ_m_II'] 
    J_s_keys = ['logJ_s_EE','logJ_s_EI','logJ_s_IE','logJ_s_II']
    J_m_values = selected_row[J_m_keys].values.reshape(2, 2)
    J_s_values = selected_row[J_s_keys].values.reshape(2, 2)
    w_sig_values = selected_row[w_sig_keys].values

    pars_stage1 = dict(w_sig=w_sig_values, b_sig=selected_row['b_sig'])

    pars_stage2 = dict(
        log_J_2x2_m = J_m_values,
        log_J_2x2_s = J_s_values,
        c_E=selected_row['c_E'],
        c_I=selected_row['c_I'],
        f_E=selected_row['f_E'],
        f_I=selected_row['f_I'],
    )
    return pars_stage1, pars_stage2
