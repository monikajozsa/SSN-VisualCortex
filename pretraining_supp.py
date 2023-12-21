import pandas as pd
import numpy
from numpy import random
import copy

from util import take_log

def perturb_params(param_dict, percent = 0.1):
    param_perturbed = copy.copy(param_dict)
    for key, param_array in param_dict.items():
        if type(param_array) == float:
            random_mtx = random.uniform(low=-1, high=1)
        else:
            random_mtx = random.uniform(low=-1, high=1, size=param_array.shape)
        param_perturbed[key] = param_array + percent * param_array * random_mtx
    return param_perturbed

def randomize_params(readout_pars, ssn_layer_pars, percent=0.1):
    #define the parameters that get perturbed
    pars_stage1 = dict(w_sig=readout_pars.w_sig, b_sig=readout_pars.b_sig)
    pars_stage2_nolog = dict(J_m_temp=ssn_layer_pars.J_2x2_m, J_s_temp=ssn_layer_pars.J_2x2_s, c_E_temp=ssn_layer_pars.c_E, c_I_temp=ssn_layer_pars.c_I, f_E_temp=numpy.exp(ssn_layer_pars.f_E), f_I_temp=numpy.exp(ssn_layer_pars.f_I))
    
    pars_stage1 = perturb_params(pars_stage1, percent)

    # Perturb parameters under conditions for J_mid
    i=0
    cond1 = False
    cond2 = False
    while not cond1 and not cond2:
        params_perturbed = perturb_params(pars_stage2_nolog, percent)
        cond1 = numpy.abs(params_perturbed['J_m_temp'][0,0]*params_perturbed['J_m_temp'][1,1]) < numpy.abs(params_perturbed['J_m_temp'][1,0]*params_perturbed['J_m_temp'][0,1])
        cond2 = numpy.abs(params_perturbed['J_m_temp'][0,1]*ssn_layer_pars.gI_m) < numpy.abs(params_perturbed['J_m_temp'][1,1]*ssn_layer_pars.gE_m)
        if i>10:
            print("Perturbed parameters violate inequality conditions")
            break
        else:
            i = i+1

    pars_stage2 = dict(
        log_J_2x2_m= take_log(params_perturbed['J_m_temp']),
        log_J_2x2_s= take_log(params_perturbed['J_s_temp']),
        c_E=params_perturbed['c_E_temp'],
        c_I=params_perturbed['c_I_temp'],
        f_E=numpy.log(params_perturbed['f_E_temp']),
        f_I=numpy.log(params_perturbed['f_I_temp']),
    )

    return pars_stage1, pars_stage2


def load_pretrained_parameters(file_path, readout_grid_size):

    # Get the last row of the given csv file
    df = pd.read_csv(file_path)
    last_row = df.iloc[-1]

    # Extract matrices from dataframe
    w_sig_keys = [f'w_sig{i}' for i in range(1, readout_grid_size*readout_grid_size+1)] 
    J_m_keys = ['logJ_m_EE','logJ_m_EI','logJ_m_IE','logJ_m_II'] 
    J_s_keys = ['logJ_s_EE','logJ_s_EI','logJ_s_IE','logJ_s_II']
    J_m_values = last_row[J_m_keys].values.reshape(2, 2)
    J_s_values = last_row[J_s_keys].values.reshape(2, 2)
    w_sig_values = last_row[w_sig_keys].values.reshape(9, 9)

    pars_stage1 = dict(w_sig=w_sig_values, b_sig=last_row['b_sig'])

    pars_stage2 = dict(
        log_J_2x2_m = J_m_values,
        log_J_2x2_s = J_s_values,
        c_E=last_row['c_E_temp'],
        c_I=last_row['c_I_temp'],
        f_E=last_row['f_E_temp'],
        f_I=last_row['f_I_temp'],
    )
    return pars_stage1, pars_stage2
