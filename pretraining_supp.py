import pandas as pd
import numpy
from numpy import random


def disturb_params(param_dict, percent = 0.1):
    param_disturbed = param_dict
    for key, param_array in param_dict.items():
        if type(param_array) == float:
            random_mtx = random.uniform(low=-1, high=1)
        else:
            random_mtx = random.uniform(low=-1, high=1, size=param_array.shape)
        param_disturbed[key] = param_array + percent * param_array * random_mtx
    return param_disturbed

def randomize_params(ssn_layer_pars,stimuli_pars, percent=0.1):
    params2disturb = dict(c_E_temp=ssn_layer_pars.c_E, c_I_temp=ssn_layer_pars.c_I, J_s_temp=ssn_layer_pars.J_2x2_s, J_m_temp=ssn_layer_pars.J_2x2_m, 
                      kappa_pre_temp = ssn_layer_pars.kappa_pre, kappa_post_temp = ssn_layer_pars.kappa_post, f_E_temp=numpy.exp(ssn_layer_pars.f_E), f_I_temp=numpy.exp(ssn_layer_pars.f_I))
    params_disturbed = disturb_params(params2disturb, percent)
    #ssn_layer_pars.c_E = params_disturbed['c_E_temp']
    #ssn_layer_pars.c_I = params_disturbed['c_I_temp']
    ssn_layer_pars.J_2x2_s = params_disturbed['J_s_temp']
    #ssn_layer_pars.J_2x2_m = params_disturbed['J_m_temp']
    #ssn_layer_pars.kappa_pre = params_disturbed['kappa_pre_temp']
    #ssn_layer_pars.kappa_post = params_disturbed['kappa_post_temp']
    #ssn_layer_pars.f_E = numpy.log(params_disturbed['f_E_temp'])
    #ssn_layer_pars.f_I = numpy.log(params_disturbed['f_I_temp'])

    stimuli_pars.ref_ori = random.uniform(low=0, high=180)
    #stimuli_pars.offset = random.uniform(low=4, high=5)

def get_trained_params(results_file=None):
    if results_file is None:
        from parameters import ssn_layer_pars
        J_2x2_m = ssn_layer_pars.J_2x2_m
        J_2x2_s = ssn_layer_pars.J_2x2_s
        c_E = ssn_layer_pars.c_E
        c_I = ssn_layer_pars.c_I
        f_E = ssn_layer_pars.f_E
        f_I = ssn_layer_pars.f_I
        kappa_pre = ssn_layer_pars.kappa_pre
        kappa_post = ssn_layer_pars.kappa_post
    else:
        df = pd.read_csv(results_file, header=0)
        num_epochs = len(df)
        J_m_names = ['J_EE_m','J_EI_m','J_IE_m','J_II_m']
        J_m = [df[var].values for var in J_m_names]
        J_2x2_m = numpy.stack(J_m, axis=-1).reshape(num_epochs,2, 2)
        J_s_names = ['J_EE_s','J_EI_s','J_IE_s','J_II_s']
        J_s = [df[var].values for var in J_s_names]
        J_2x2_s = numpy.stack(J_s, axis=-1).reshape(num_epochs,2, 2)
        c_E = df['c_E'][num_epochs-1]
        c_I = df['c_I'][num_epochs-1]
        f_E = df['f_E'][num_epochs-1]
        f_I = df['f_I'][num_epochs-1]
        kappa_pre = df['kappe_pre'][num_epochs-1]
        kappa_post = df['kappe_post'][num_epochs-1]
    return J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I, kappa_pre, kappa_post

# apply get_trained_params and update the parameters
def update_params(results_file, ssn_layer_pars):
    J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I, kappa_pre, kappa_post = get_trained_params(results_file)
    ssn_layer_pars.c_E = c_E
    ssn_layer_pars.c_I = c_I
    ssn_layer_pars.J_2x2_s = J_2x2_s
    ssn_layer_pars.J_2x2_m = J_2x2_m
    ssn_layer_pars.kappa_pre = kappa_pre
    ssn_layer_pars.kappa_post = kappa_post
    ssn_layer_pars.f_E = f_E
    ssn_layer_pars.f_I = f_I

    return ssn_layer_pars
