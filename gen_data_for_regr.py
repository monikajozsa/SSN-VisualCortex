import pickle
import numpy
import time
import jax.numpy as np
from jax import vmap

from parameters import pretrain_pars, training_pars, stimuli_pars
from util import create_grating_pretraining, sep_exponentiate
from util_gabor import BW_image_jax_supp
from SSN_classes import SSN_mid, SSN_sup
from model import evaluate_model_response

def gen_stim(pretrain_pars, training_pars, stimuli_pars, file_name='stim_data_dict.pkl', N=1000):
    # Allocate dictionary for the stimuli
    data_dict = {
        "ref": [],
        "target": [],
        "label": []
    }

    # Generate stimulus data for setting w_sig and b_sig based on regression (pretraining) and logistic regression (training)
    start_time=time.time()
    for i in range(N):
        data_i = create_grating_pretraining(pretrain_pars, training_pars.batch_size, BW_image_jax_supp(stimuli_pars))
        
        data_dict['ref'].append(data_i['ref'])
        data_dict['target'].append(data_i['target'])
        data_dict['label'].append(data_i['label'])
        if numpy.round(i/(N/10)) == i/(N/10):
            print(time.time()-start_time)
            print(i)

    # Save the dictionary to a file
    with open(file_name, 'wb') as file:
        pickle.dump(data_dict, file)

gen_stim(pretrain_pars, training_pars, stimuli_pars)

vmap_evaluate_model_response = vmap(evaluate_model_response, in_axes = (None, None, 0, None, None, None, None, None, None) )

def linregression_sig_layer(ssn_layer_pars_dict, constant_pars, file_name='stim_data_dict.pkl', N=100):
    
    # This function calculates optimal w_sig and b_sig based on N samples of stimuli (where each has a batch_size).
    
    # Load the dictionary from the file
    with open(file_name, 'rb') as file:
        loaded_stim = pickle.load(file)

    log_J_2x2_m = ssn_layer_pars_dict['log_J_2x2_m']
    log_J_2x2_s = ssn_layer_pars_dict['log_J_2x2_s']
    c_E = ssn_layer_pars_dict['c_E']
    c_I = ssn_layer_pars_dict['c_I']
    f_E = np.exp(ssn_layer_pars_dict['f_E'])
    f_I = np.exp(ssn_layer_pars_dict['f_I'])
    if 'kappa_pre' in ssn_layer_pars_dict:
        kappa_pre = np.tanh(ssn_layer_pars_dict['kappa_pre'])
        kappa_post = np.tanh(ssn_layer_pars_dict['kappa_post'])
    else:
        kappa_pre = constant_pars.ssn_layer_pars.kappa_pre
        kappa_post = constant_pars.ssn_layer_pars.kappa_post
    p_local_s = constant_pars.ssn_layer_pars.p_local_s
    s_2x2 = constant_pars.ssn_layer_pars.s_2x2_s
    sigma_oris = constant_pars.ssn_layer_pars.sigma_oris
    ref_ori = constant_pars.stimuli_pars.ref_ori
    
    J_2x2_m = sep_exponentiate(log_J_2x2_m)
    J_2x2_s = sep_exponentiate(log_J_2x2_s)   

    conv_pars = constant_pars.conv_pars
    # Create middle and superficial SSN layers *** this is something that would be great to change - to call the ssn classes from inside the training
    ssn_mid=SSN_mid(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, J_2x2=J_2x2_m)
    ssn_sup=SSN_sup(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, J_2x2=J_2x2_s, p_local=p_local_s, oris=constant_pars.oris, s_2x2=s_2x2, sigma_oris = sigma_oris, ori_dist = constant_pars.ori_dist, train_ori = ref_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)
    
    Ng = constant_pars.grid_pars.gridsize_Nx
    batch_size=constant_pars.training_pars.batch_size
    X=np.zeros(( N*batch_size, Ng*Ng))
    Y=X=np.zeros(( N*batch_size, 1))
    for i in range(N):
        data_ref=loaded_stim['ref'][i]
        data_target=loaded_stim['target'][i]
        r_ref, _, [_, _], [_, _],[_, _, _, _], [_,_] = vmap_evaluate_model_response(ssn_mid, ssn_sup, data_ref, conv_pars, c_E, c_I, f_E, f_I, constant_pars.gabor_filters)
        r_target, _, [_, _], [_, _],[_, _, _, _], [_, _]= vmap_evaluate_model_response(ssn_mid, ssn_sup, data_target, conv_pars, c_E, c_I, f_E, f_I, constant_pars.gabor_filters)
        X[i*batch_size:(i+1)*batch_size,:] = r_ref-r_target
        Y[i*batch_size:(i+1)*batch_size,:] = loaded_stim['label'][i]
    
    coefficients = np.polyfit(X, Y, 1)

    return coefficients
    #slope, intercept = coefficients[0], coefficients[1]
#coeffs = linregression_sig_layer(ssn_layer_pars_dict, constant_pars, file_name='stim_data_dict.pkl', N=100)
