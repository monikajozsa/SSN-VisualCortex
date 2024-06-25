import numpy
from numpy import random
import copy
import jax.numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from util import take_log, create_grating_training, sep_exponentiate, create_grating_pretraining
from util_gabor import init_untrained_pars
from SSN_classes import SSN_mid, SSN_sup
from model import evaluate_model_response, vmap_evaluate_model_response
from training_functions import mean_training_task_acc_test

def perturb_params_supp_old(param_dict, percent = 0.1):
    '''Perturb all values in a dictionary by a percentage of their values. The perturbation is done by adding a uniformly sampled random value to the original value.'''
    param_perturbed = copy.deepcopy(param_dict)
    for key, param_array in param_dict.items():
        if type(param_array) == float:
            random_mtx = random.uniform(low=-1, high=1)
        else:
            random_mtx = random.uniform(low=-1, high=1, size=param_array.shape)
        param_perturbed[key] = param_array + percent * param_array * random_mtx
    return param_perturbed

def perturb_params_supp(param_dict, perturb_pars):
    '''Perturb all values in a dictionary by a percentage of their values. The perturbation is done by adding a uniformly sampled random value to the original value.'''
    param_perturbed = copy.deepcopy(param_dict)
    attributes = dir(perturb_pars)
    
    for key, param_array in param_dict.items():
        matching_attributes = [attr for attr in attributes if attr.startswith(key[0])]
        param_range = getattr(perturb_pars, matching_attributes[0])
        if type(param_array) == float:
            random_sample = random.uniform(low=param_range[0], high=param_range[1])
            param_perturbed[key] = random_sample
        else:
            # handling the J_2x2_m and J_2x2_s matrices
            random_sample = random.uniform(low=param_range[0], high=param_range[1], size=param_array.shape)
            JEE = 2*random_sample[0,0]
            JEI = -random_sample[0,1]
            JIE = 2*random_sample[1,0]
            JII = -random_sample[1,0]
            param_perturbed[key] = np.array([[JEE, JEI],[JIE, JII]])
            #np.array([[4.4, -1.66], [5, -1.24]])
        
    return param_perturbed


def perturb_params(readout_pars, trained_pars, untrained_pars, perturb_pars, logistic_regr=True, trained_pars_dict=None, num_init=0, start_time=time.time()):
    # define the parameters to perturb
    if trained_pars_dict is None:
        trained_pars_dict = dict(J_2x2_m=trained_pars.J_2x2_m, J_2x2_s=trained_pars.J_2x2_s)
        for key, value in vars(trained_pars).items():
            if key == 'c_E' or key == 'c_I' or key == 'f_E' or key == 'f_I':
                trained_pars_dict[key] = value
        
    # Perturb parameters under conditions for J_mid and convergence of the differential equations of the model
    i=0
    cond1 = False # parameter inequality on Jm
    cond2 = False # parameter inequality on Jm and gE, gI

    while not (cond1 and cond2):
        perturbed_pars = perturb_params_supp(trained_pars_dict, perturb_pars)
        cond1 = np.abs(perturbed_pars['J_2x2_m'][0,0]*perturbed_pars['J_2x2_m'][1,1])*1.1 < np.abs(perturbed_pars['J_2x2_m'][1,0]*perturbed_pars['J_2x2_m'][0,1])
        cond2 = np.abs(perturbed_pars['J_2x2_m'][0,1]*untrained_pars.filter_pars.gI_m)*1.1 < np.abs(perturbed_pars['J_2x2_m'][1,1]*untrained_pars.filter_pars.gE_m)   
        i = i+1
    
    # Calculate model response to check the convergence of the differential equations
    ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=perturbed_pars['J_2x2_m'])
    ssn_sup=SSN_sup(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=perturbed_pars['J_2x2_s'], p_local=untrained_pars.ssn_pars.p_local_s, oris=untrained_pars.oris, s_2x2=untrained_pars.ssn_pars.s_2x2_s, sigma_oris = untrained_pars.ssn_pars.sigma_oris, ori_dist = untrained_pars.ori_dist)
    train_data = create_grating_training(untrained_pars.stimuli_pars, batch_size=5, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp) 
    pretrain_data = create_grating_pretraining(untrained_pars.pretrain_pars, batch_size=5, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp)
    if 'c_E' in perturbed_pars:
        c_E = perturbed_pars['c_E']
        c_I = perturbed_pars['c_I']
    else:
        c_E = untrained_pars.ssn_pars.c_E
        c_I = untrained_pars.ssn_pars.c_I
    if 'f_E' in perturbed_pars:
        f_E = perturbed_pars['f_E']
        f_I = perturbed_pars['f_I']
    else:
        f_E = untrained_pars.ssn_pars.f_E
        f_I = untrained_pars.ssn_pars.f_I
    [r_train,_],_ , [avg_dx_mid, avg_dx_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = vmap_evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'], untrained_pars.conv_pars,c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)
    [r_pretrain,_],_, [avg_dx_mid, avg_dx_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = vmap_evaluate_model_response(ssn_mid, ssn_sup, pretrain_data['ref'], untrained_pars.conv_pars,c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)
    cond3 = bool((avg_dx_mid + avg_dx_sup < 50).all())
    cond4 = min([float(np.min(max_E_mid)), float(np.min(max_I_mid)), float(np.min(max_E_sup)), float(np.min(max_I_sup))])>5 and max([float(np.max(max_E_mid)), float(np.max(max_I_mid)), float(np.max(max_E_sup)), float(np.max(max_I_sup))])<151
    cond5 = not numpy.any(np.isnan(r_pretrain))
    cond6 = not numpy.any(np.isnan(r_train))
    if not (cond3 and cond4 and cond5 and cond6):
        if num_init>500:
            print(" ########### Perturbed parameters violate conditions even after 200 sampling. ###########")
        else:
            num_init = num_init+i
            print(f'Perturbed parameters {num_init} times',[bool(cond1),bool(cond2),cond3,cond4,cond5,cond6])
            pars_stage1, perturbed_pars_log, untrained_pars = perturb_params(readout_pars, trained_pars, untrained_pars, perturb_pars, logistic_regr, trained_pars_dict, num_init, start_time)
    else:
        print(f'Parameters found that satisfy conditions in {time.time() - start_time:.2f} seconds')
        # Take log of the J and f parameters (if f_I, f_E are in the perturbed parameters)
        perturbed_pars_log = dict(
            log_J_2x2_m= take_log(perturbed_pars['J_2x2_m']),
            log_J_2x2_s= take_log(perturbed_pars['J_2x2_s']))
        for key, vale in perturbed_pars.items():
            if key=='c_E' or key=='c_I':
                perturbed_pars_log[key] = vale
            elif key=='f_E' or key=='f_I':
                perturbed_pars_log['log_'+key] = np.log(vale)

        # Optimize readout parameters by using log-linear regression
        if logistic_regr:
            pars_stage1 = readout_pars_from_regr(readout_pars, perturbed_pars_log, untrained_pars)
        else:
            pars_stage1 = dict(w_sig=readout_pars.w_sig, b_sig=readout_pars.b_sig)
        pars_stage1['w_sig'] = (pars_stage1['w_sig'] / np.std(pars_stage1['w_sig']) ) * 0.25 / int(np.sqrt(len(pars_stage1['w_sig']))) # get the same std as before - see param

        # Perturb learning rate
        untrained_pars.training_pars.eta = random.uniform(perturb_pars.eta_range[0], perturb_pars.eta_range[1])
        
    return pars_stage1, perturbed_pars_log, untrained_pars


def readout_pars_from_regr(readout_pars, trained_pars_dict, untrained_pars, N=1000):
    '''
    This function sets readout_pars based on N sample data using linear regression. This method is to initialize w_sig, b_sig optimally (given limited data) for a set of perturbed trained_pars_dict parameters (to be trained).
    '''
    # Generate stimuli and label data for setting w_sig and b_sig based on linear regression (pretraining)
    data = create_grating_pretraining(untrained_pars.pretrain_pars, N, untrained_pars.BW_image_jax_inp, numRnd_ori1=N)
    
    #### Get model response for stimuli data['ref'] and data['target'] ####
    # 1. extract trained and untrained parameters
    J_2x2_m = sep_exponentiate(trained_pars_dict['log_J_2x2_m'])
    J_2x2_s = sep_exponentiate(trained_pars_dict['log_J_2x2_s'])

    if 'c_E' in trained_pars_dict:
        c_E = trained_pars_dict['c_E']
        c_I = trained_pars_dict['c_I']
    else:
        c_E = untrained_pars.ssn_pars.c_E
        c_I = untrained_pars.ssn_pars.c_I
        
    if 'log_f_E' in trained_pars_dict:  
        f_E = np.exp(trained_pars_dict['log_f_E'])
        f_I = np.exp(trained_pars_dict['log_f_I'])
    else:
        f_E = untrained_pars.ssn_pars.f_E
        f_I = untrained_pars.ssn_pars.f_I

    p_local_s = untrained_pars.ssn_pars.p_local_s
    s_2x2 = untrained_pars.ssn_pars.s_2x2_s
    sigma_oris = untrained_pars.ssn_pars.sigma_oris
    ref_ori = untrained_pars.stimuli_pars.ref_ori
    conv_pars = untrained_pars.conv_pars

    # 2. define middle layer and superficial layer SSN
    ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_m)
    ssn_sup=SSN_sup(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_s, p_local=p_local_s, oris=untrained_pars.oris, s_2x2=s_2x2, sigma_oris = sigma_oris, ori_dist = untrained_pars.ori_dist)
    
    # Run reference and target through two layer model
    [r_ref, _], _,_, _, _ = vmap_evaluate_model_response(ssn_mid, ssn_sup, data['ref'], conv_pars, c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)
    [r_target, _], _, _, _, _= vmap_evaluate_model_response(ssn_mid, ssn_sup, data['target'], conv_pars, c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)

    X = r_ref-r_target
    y = data['label']
    
    # Perform logistic regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    log_reg = LogisticRegression(max_iter=100)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy of logistic regression', accuracy)
    
    # Set the readout parameters based on the results of the logistic regression
    readout_pars_opt = {'b_sig': 0, 'w_sig': np.zeros(len(X[0]))}
    readout_pars_opt['b_sig'] = float(log_reg.intercept_)
    w_sig = log_reg.coef_.T
    w_sig = w_sig.squeeze()
    
    if untrained_pars.pretrain_pars.is_on:
        readout_pars_opt['w_sig'] = w_sig
    else:
        readout_pars_opt['w_sig'] = w_sig[untrained_pars.middle_grid_ind]

    # check if the readout parameters solve the task in the correct direction
    test_offset_vec = numpy.array([2, 5, 10, 18]) 
    acc_mean, _, _ = mean_training_task_acc_test(trained_pars_dict, readout_pars_opt, untrained_pars, True, test_offset_vec)
    if np.sum(acc_mean<0.5)>0.5*len(acc_mean):
        # flip the sign of w_sig and b_sig if the logistic regression is solving the flipped task
        readout_pars_opt['w_sig'] = -w_sig
        readout_pars_opt['b_sig'] = -readout_pars_opt['b_sig']
        acc_mean_flipped, _, _ = mean_training_task_acc_test(trained_pars_dict, readout_pars_opt, untrained_pars, True, test_offset_vec)
        print('accuracy of logistic regression before and after flipping the w_sig and b_sig', np.mean(acc_mean), np.mean(acc_mean_flipped))

    return readout_pars_opt


def create_initial_parameters_df(initial_parameters, perturbed_parameters, perturbed_eta):
    '''
    This function creates or appens a dataframe with the initial parameters for the model. The dataframe includes the perturbed traine dparameters, and the perturbed learning rate eta.
    '''
    # Take log of the J and f parameters (if f_I, f_E are in the perturbed parameters)
    J_2x2_m = sep_exponentiate(perturbed_parameters['log_J_2x2_m'])
    J_2x2_s = sep_exponentiate(perturbed_parameters['log_J_2x2_s'])
    # Create a dictionary with the new perturbed parameters
    new_vals_dict = dict(J_m_EE=J_2x2_m[0,0], J_m_EI=J_2x2_m[1,0], J_m_IE=J_2x2_m[0,1], J_m_II=J_2x2_m[1,1],
                        J_s_EE=J_2x2_s[0,0], J_s_EI=J_2x2_s[1,0], J_s_IE=J_2x2_s[0,1], J_s_II=J_2x2_s[1,1],
                        eta=perturbed_eta)
    if 'log_f_E' in perturbed_parameters:
        new_vals_dict['f_E'] = np.exp(perturbed_parameters['log_f_E'])
        new_vals_dict['f_I'] = np.exp(perturbed_parameters['log_f_I'])
    if 'c_E' in perturbed_parameters:
        new_vals_dict['c_E'] = perturbed_parameters['c_E']
        new_vals_dict['c_I'] = perturbed_parameters['c_I']

    # Create a dataframe with the initial parameters
    if initial_parameters is None:
        initial_parameters_df = pd.DataFrame(new_vals_dict, index=[0])
    else:
        initial_parameters_df = pd.concat([initial_parameters, pd.DataFrame(new_vals_dict, index=[0])])  
    
    return initial_parameters_df