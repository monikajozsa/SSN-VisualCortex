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
from training_functions import mean_training_task_acc_test, task_acc_test

def perturb_params_supp_old(param_dict, percent = 0.1):
    '''Perturb all values in a dictionary by a percentage of their values. The perturbation is done by adding a uniformly sampled random value to the original value.'''
    param_randomized = copy.deepcopy(param_dict)
    for key, param_array in param_dict.items():
        if type(param_array) == float:
            random_mtx = random.uniform(low=-1, high=1)
        else:
            random_mtx = random.uniform(low=-1, high=1, size=param_array.shape)
        param_randomized[key] = param_array + percent * param_array * random_mtx
    return param_randomized

def randomize_params_supp(param_dict, randomize_pars):
    '''Randomize all values in a dictionary by a percentage of their values. The randomization is done by uniformly sampling random values from predefined ranges.'''
    param_randomized = copy.deepcopy(param_dict)
    attributes = dir(randomize_pars)
    
    for key, param_array in param_dict.items():
        matching_attributes = [attr for attr in attributes if attr.startswith(key[0])]
        param_range = getattr(randomize_pars, matching_attributes[0])
        if type(param_array) == float:
            random_sample = random.uniform(low=param_range[0], high=param_range[1])
            param_randomized[key] = random_sample
        else:
            # handling the J_2x2_m and J_2x2_s matrices
            JEE = random.uniform(low=param_range[0][0], high=param_range[0][1])
            JEI = -random.uniform(low=param_range[1][0], high=param_range[1][1])
            JIE = random.uniform(low=param_range[2][0], high=param_range[2][1])
            JII = -random.uniform(low=param_range[3][0], high=param_range[3][1])
            param_randomized[key] = np.array([[JEE, JEI],[JIE, JII]])
            #np.array([[4.4, -1.66], [5, -1.24]])
        
    return param_randomized


def randomize_params(readout_pars, trained_pars, untrained_pars, randomize_pars, logistic_regr=True, trained_pars_dict=None, num_init=0, start_time=time.time()):
    # define the parameters to randomize
    if trained_pars_dict is None:
        trained_pars_dict = dict(J_2x2_m=trained_pars.J_2x2_m, J_2x2_s=trained_pars.J_2x2_s)
        for key, value in vars(trained_pars).items():
            if key == 'c_E' or key == 'c_I' or key == 'f_E' or key == 'f_I':
                trained_pars_dict[key] = value
        
    # Initialize parameters such that conditions are satisfied for J_mid and convergence of the differential equations of the model
    i=0
    cond_ineq1 = False # parameter inequality on Jm
    cond_ineq2 = False # parameter inequality on Jm and gE, gI

    while not (cond_ineq1 and cond_ineq2):
        randomized_pars = randomize_params_supp(trained_pars_dict, randomize_pars)
        cond_ineq1 = np.abs(randomized_pars['J_2x2_m'][0,0]*randomized_pars['J_2x2_m'][1,1])*1.1 < np.abs(randomized_pars['J_2x2_m'][1,0]*randomized_pars['J_2x2_m'][0,1])
        # randomize gE and gI
        untrained_pars.filter_pars.gI_m =  random.uniform(low=randomize_pars.g_range[0], high=randomize_pars.g_range[1])
        untrained_pars.filter_pars.gE_m =  random.uniform(low=randomize_pars.g_range[0], high=randomize_pars.g_range[1])
        cond_ineq2 = np.abs(randomized_pars['J_2x2_m'][0,1]*untrained_pars.filter_pars.gI_m)*1.1 < np.abs(randomized_pars['J_2x2_m'][1,1]*untrained_pars.filter_pars.gE_m)   
        i = i+1
    
    # Calculate model response to check the convergence of the differential equations
    ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=randomized_pars['J_2x2_m'])
    ssn_sup=SSN_sup(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=randomized_pars['J_2x2_s'], p_local=untrained_pars.ssn_pars.p_local_s, oris=untrained_pars.oris, s_2x2=untrained_pars.ssn_pars.s_2x2_s, sigma_oris = untrained_pars.ssn_pars.sigma_oris, ori_dist = untrained_pars.ori_dist)
    train_data = create_grating_training(untrained_pars.stimuli_pars, batch_size=5, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp) 
    pretrain_data = create_grating_pretraining(untrained_pars.pretrain_pars, batch_size=5, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp)
    if 'c_E' in randomized_pars:
        c_E = randomized_pars['c_E']
        c_I = randomized_pars['c_I']
    else:
        c_E = untrained_pars.ssn_pars.c_E
        c_I = untrained_pars.ssn_pars.c_I
    if 'f_E' in randomized_pars:
        f_E = randomized_pars['f_E']
        f_I = randomized_pars['f_I']
    else:
        f_E = untrained_pars.ssn_pars.f_E
        f_I = untrained_pars.ssn_pars.f_I
    [r_train,_],_ ,[avg_dx_mid, avg_dx_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], [mean_E_mid, mean_I_mid, mean_E_sup, mean_I_sup] = vmap_evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'], untrained_pars.conv_pars,c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)
    [r_pretrain,_],_, _,_,_ = vmap_evaluate_model_response(ssn_mid, ssn_sup, pretrain_data['ref'], untrained_pars.conv_pars,c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)
    cond_dx = bool((avg_dx_mid + avg_dx_sup < 50).all())
    cond_rmax = min([float(np.min(max_E_mid)), float(np.min(max_I_mid)), float(np.min(max_E_sup)), float(np.min(max_I_sup))])>10 and max([float(np.max(max_E_mid)), float(np.max(max_I_mid)), float(np.max(max_E_sup)), float(np.max(max_I_sup))])<151
    cond_rmean = min([float(np.min(mean_E_mid)), float(np.min(mean_I_mid)), float(np.min(mean_E_sup)), float(np.min(mean_I_sup))])>5 and max([float(np.max(mean_E_mid)), float(np.max(mean_I_mid)), float(np.max(mean_E_sup)), float(np.max(mean_I_sup))])<80
    cond_r_pretrain = not numpy.any(np.isnan(r_pretrain))
    cond_r_train = not numpy.any(np.isnan(r_train))
    if not (cond_dx and cond_rmax and cond_rmean and cond_r_pretrain and cond_r_train):
        if num_init>1000:
            raise Exception(" ########### Randomized parameters violate conditions even after 1000 sampling. ###########")
        else:
            num_init = num_init+i
            print(f'Randomized parameters {num_init} times',[bool(cond_ineq1),bool(cond_ineq2),cond_dx,cond_rmax,cond_rmean,cond_r_pretrain,cond_r_train])
            optimized_readout_pars, randomized_pars_log, untrained_pars = randomize_params(readout_pars, trained_pars, untrained_pars, randomize_pars, logistic_regr, trained_pars_dict, num_init, start_time)
    else:
        print(f'Parameters found that satisfy conditions in {time.time() - start_time:.2f} seconds')
        # Take log of the J and f parameters (if f_I, f_E are in the randomized parameters)
        randomized_pars_log = dict(
            log_J_2x2_m= take_log(randomized_pars['J_2x2_m']),
            log_J_2x2_s= take_log(randomized_pars['J_2x2_s']))
        for key, vale in randomized_pars.items():
            if key=='c_E' or key=='c_I':
                randomized_pars_log[key] = vale
            elif key=='f_E' or key=='f_I':
                randomized_pars_log['log_'+key] = np.log(vale)

        # Optimize readout parameters by using log-linear regression
        if logistic_regr:
            optimized_readout_pars = readout_pars_from_regr(readout_pars, randomized_pars_log, untrained_pars)
        else:
            optimized_readout_pars = dict(w_sig=readout_pars.w_sig, b_sig=readout_pars.b_sig)
        optimized_readout_pars['w_sig'] = (optimized_readout_pars['w_sig'] / np.std(optimized_readout_pars['w_sig']) ) * 0.25 / int(np.sqrt(len(optimized_readout_pars['w_sig']))) # get the same std as before - see param

        # Randomize learning rate
        untrained_pars.training_pars.eta = random.uniform(randomize_pars.eta_range[0], randomize_pars.eta_range[1])

    return optimized_readout_pars, randomized_pars_log, untrained_pars


def readout_pars_from_regr(readout_pars, trained_pars_dict, untrained_pars, N=1000):
    '''
    This function sets readout_pars based on N sample data using linear regression. This method is to initialize w_sig, b_sig optimally (given limited data) for a set of randomized trained_pars_dict parameters (to be trained).
    '''
    # Generate stimuli and label data for setting w_sig and b_sig based on linear regression (pretraining)
    data = create_grating_pretraining(untrained_pars.pretrain_pars, N, untrained_pars.BW_image_jax_inp, numRnd_ori1=N)
    # Adding 10% of training data to avoind weird phenomena like returning the opposite direction for the fine discrimination task
    #data_training = create_grating_training(untrained_pars.stimuli_pars, int(N/10), untrained_pars.BW_image_jax_inp)
    #data = {'ref': np.concatenate((data_pretraining['ref'], data_training['ref'])), 'target': np.concatenate((data_pretraining['target'], data_training['target'])), 'label': np.concatenate((data_pretraining['label'], data_training['label']))}
    
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
    acc_train, _ = task_acc_test( trained_pars_dict, readout_pars_opt, untrained_pars, True, 4)
    acc_pretrain, _ = task_acc_test( trained_pars_dict, readout_pars_opt, untrained_pars, True, None, pretrain_task=True)
    print('accuracy of logistic regression on training and pretraining data', acc_train, acc_pretrain)

    return readout_pars_opt


def create_initial_parameters_df(initial_parameters, randomized_parameters, randomized_eta):
    '''
    This function creates or appens a dataframe with the initial parameters for the model. The dataframe includes the randomized traine dparameters, and the randomized learning rate eta.
    '''
    # Take log of the J and f parameters (if f_I, f_E are in the randomized parameters)
    J_2x2_m = sep_exponentiate(randomized_parameters['log_J_2x2_m'])
    J_2x2_s = sep_exponentiate(randomized_parameters['log_J_2x2_s'])
    # Create a dictionary with the new randomized parameters
    new_vals_dict = dict(J_m_EE=J_2x2_m[0,0], J_m_EI=J_2x2_m[1,0], J_m_IE=J_2x2_m[0,1], J_m_II=J_2x2_m[1,1],
                        J_s_EE=J_2x2_s[0,0], J_s_EI=J_2x2_s[1,0], J_s_IE=J_2x2_s[0,1], J_s_II=J_2x2_s[1,1],
                        eta=randomized_eta)
    if 'log_f_E' in randomized_parameters:
        new_vals_dict['f_E'] = np.exp(randomized_parameters['log_f_E'])
        new_vals_dict['f_I'] = np.exp(randomized_parameters['log_f_I'])
    if 'c_E' in randomized_parameters:
        new_vals_dict['c_E'] = randomized_parameters['c_E']
        new_vals_dict['c_I'] = randomized_parameters['c_I']

    # Create a dataframe with the initial parameters
    if initial_parameters is None:
        initial_parameters_df = pd.DataFrame(new_vals_dict, index=[0])
    else:
        initial_parameters_df = pd.concat([initial_parameters, pd.DataFrame(new_vals_dict, index=[0])])  
    
    return initial_parameters_df