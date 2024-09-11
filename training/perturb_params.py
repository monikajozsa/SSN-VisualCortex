import numpy
from numpy import random
import jax.numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from util import take_log, create_grating_training, create_grating_pretraining, unpack_ssn_parameters, cosdiff_ring
from util_gabor import update_untrained_pars, save_orimap
from SSN_classes import SSN_mid, SSN_sup
from model import vmap_evaluate_model_response, vmap_evaluate_model_response_mid
from training_functions import task_acc_test

def fill_attribute_list(class_to_fill, attr_list, value_list):
    """Fill the attributes of a class with the given values."""
    i=0
    for attr in attr_list:
        if hasattr(class_to_fill, attr):
            setattr(class_to_fill, attr, value_list[i])

    return class_to_fill

def randomize_mid_params(randomize_pars, readout_pars, num_calls=0, untrained_pars=None, J_2x2_m=None, cE_m=None, cI_m=None, ssn_mid=None, train_data=None, pretrain_data=None):
    """Randomize the middle layer parameters of the model and check if the inequality and response conditions are satisfied."""
    if num_calls > 300:
        raise Exception(f'More than {num_calls} calls to initialize middle layer parameters.')

    i=0
    cond_ineq1 = False # parameter inequality on Jm
    cond_ineq2 = False # parameter inequality on Jm and gE, gI
    J_range=randomize_pars.J_range
    g_range=randomize_pars.g_range
    while not (cond_ineq1 and cond_ineq2):
        J_EE_m, J_EI_m_nosign, J_IE_m, J_II_m_nosign = random.uniform(low=[J_range[0][0], J_range[1][0], J_range[2][0], J_range[3][0]],high=[J_range[0][1], J_range[1][1], J_range[2][1], J_range[3][1]])
        gE_m, gI_m = random.uniform(low=[g_range[0], g_range[0]], high=[g_range[1], g_range[1]])
        cond_ineq1 = np.abs(J_EE_m*J_II_m_nosign*1.1 < J_EI_m_nosign*J_IE_m)
        cond_ineq2 = np.abs(J_EI_m_nosign*gI_m)*1.1 < np.abs(J_II_m_nosign*gE_m)
        i = i+1
        if i>200:
            raise Exception(" ########### Randomized parameters violate conditions 1 or 2 after 200 sampling. ###########")
    print(f'Parameters found that satisfy inequalities in {i} steps')
    J_2x2_m = np.array([[J_EE_m, -J_EI_m_nosign],[J_IE_m, -J_II_m_nosign]])
    
    ##### Initialize untrained parameters if they are not given #####
    if untrained_pars is None:
        from util_gabor import init_untrained_pars
        from parameters import GridPars, FilterPars, StimuliPars, SSNPars, ConvPars, TrainingPars, LossPars, PretrainingPars
        grid_pars, filter_pars, stimuli_pars, ssn_pars = GridPars(), FilterPars(), StimuliPars(), SSNPars()
        conv_pars, training_pars, loss_pars, pretraining_pars = ConvPars(), TrainingPars(), LossPars(), PretrainingPars()

        # Randomize eta and overwrite g (and J_m if untrained) in the instances of parameter classes
        eta = random.uniform(low=randomize_pars.eta_range[0], high=randomize_pars.eta_range[1])
        training_pars.eta = eta
        filter_pars.gE_m = gE_m
        filter_pars.gI_m = gI_m
        ssn_pars = fill_attribute_list(ssn_pars, ['J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m'], [J_EE_m, -J_EI_m_nosign, J_IE_m, -J_II_m_nosign])

        # Initialize untrained parameters
        untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                    loss_pars, training_pars, pretraining_pars, readout_pars)
    else:
        untrained_pars = update_untrained_pars(untrained_pars, readout_pars, gE_m, gI_m)
        
    ##### Check conditions on the middle layer response: call function recursively if they are not satisfied and return values if they are #####
    # 1. Calculate middle layer responses
    if train_data is None:
        train_data = create_grating_training(untrained_pars.stimuli_pars, batch_size=10, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp) 
        pretrain_data = create_grating_pretraining(untrained_pars.pretrain_pars, batch_size=10, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp)
    ssn_mid = SSN_mid(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_m)
    c_range= randomize_pars.c_range
    cE_m, cI_m = random.uniform(low=[c_range[0], c_range[0]], high=[c_range[1], c_range[1]])
    r_train_mid,_ ,avg_dx_mid, max_E_mid, max_I_mid, mean_E_mid, mean_I_mid = vmap_evaluate_model_response_mid(ssn_mid, train_data['ref'], untrained_pars.conv_pars, cE_m, cI_m, untrained_pars.gabor_filters)
    # Check if response has nan
    cond_r_train_mid = not numpy.any(np.isnan(r_train_mid))
    if cond_r_train_mid:
        r_pretrain_mid,_ ,_, _, _, _, _ = vmap_evaluate_model_response_mid(ssn_mid, pretrain_data['ref'], untrained_pars.conv_pars, cE_m, cI_m, untrained_pars.gabor_filters)
        
        # 2. Evaluate middle layer conditions
        cond_dx_mid = bool((avg_dx_mid < 50).all())
        rmean_min_mid = min([float(np.min(mean_E_mid)), float(np.min(mean_I_mid))])
        rmean_max_mid = max([float(np.max(mean_E_mid)), float(np.max(mean_I_mid))])    
        rmax_min_mid = min([float(np.min(max_E_mid)), float(np.min(max_I_mid))])
        rmax_max_mid = max(float(np.max(max_E_mid)), float(np.max(max_I_mid)))
        cond_rmax_mid = rmax_min_mid>10 and rmax_max_mid<151
        cond_rmean_mid = rmean_min_mid>5 and  rmean_max_mid<80
        cond_r_pretrain_mid = not numpy.any(np.isnan(r_pretrain_mid))        

        if not (cond_dx_mid and cond_rmax_mid and cond_rmean_mid and cond_r_pretrain_mid):
            # RECURSIVE FUNCTION CALL
            num_calls=num_calls+1
            return randomize_mid_params(randomize_pars, readout_pars, num_calls, untrained_pars, J_2x2_m, cE_m, cI_m, ssn_mid, train_data, pretrain_data)
        else:
            return untrained_pars, J_2x2_m, cE_m, cI_m, ssn_mid, train_data, pretrain_data
    else:
        # RECURSIVE FUNCTION CALL
        num_calls=num_calls+1
        return randomize_mid_params(randomize_pars, readout_pars, num_calls, untrained_pars, J_2x2_m, cE_m, cI_m, ssn_mid, train_data, pretrain_data)
    

def randomize_params(folder, run_index, untrained_pars=None, logistic_regr=True, num_mid_calls=0, num_calls=0, start_time=time.time(), J_2x2_m=None, cE_m=None, cI_m=None, ssn_mid=None, train_data=None, pretrain_data=None):
    """Randomize the required initial parameters of the model and optimize the readout parameters using logistic regression. 
    The randomization is done by uniformly sampling random values from predefined ranges."""

    ##### Initialize middle layer parameters such that inequality conditions and response conditions are satisfied #####        
    from parameters import RandomizePars, ReadoutPars
    randomize_pars, readout_pars = RandomizePars(), ReadoutPars()
    if num_calls==0 or num_calls>100:
        untrained_pars, J_2x2_m, cE_m, cI_m, ssn_mid, train_data, pretrain_data = randomize_mid_params(randomize_pars, readout_pars, untrained_pars=untrained_pars)
        num_calls = 0
        num_mid_calls = num_mid_calls + 1
        print(f'Middle layer parameters found that satisfy conditions in {time.time() - start_time:.2f} seconds.')

    ##### Initialize superficial layer parameters such that response conditions are satisfied #####
    J_range=randomize_pars.J_range
    J_EE_s, J_EI_s_nosign, J_IE_s, J_II_s_nosign = random.uniform(low=[J_range[0][0], J_range[1][0], J_range[2][0], J_range[3][0]],high=[J_range[0][1], J_range[1][1], J_range[2][1], J_range[3][1]])
    J_2x2_s = np.array([[J_EE_s, -J_EI_s_nosign],[J_IE_s, -J_II_s_nosign]])
    ssn_sup = SSN_sup(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_s, untrained_pars.dist_from_single_ori, untrained_pars.ori_dist)
    c_range= randomize_pars.c_range  
    f_range= randomize_pars.f_range  
    cE_s, cI_s = random.uniform(low=[c_range[0], c_range[0]], high=[c_range[1], c_range[1]])
    f_E, f_I = random.uniform(low=[f_range[0], f_range[0]], high=[f_range[1], f_range[1]])
    [r_train_sup,_],_ ,[_, avg_dx_sup],[_, _, max_E_sup, max_I_sup], [_, _, mean_E_sup, mean_I_sup] = vmap_evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'], untrained_pars.conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters)
    [r_pretrain_sup,_],_, _,_,_ = vmap_evaluate_model_response(ssn_mid, ssn_sup, pretrain_data['ref'], untrained_pars.conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters)
    
    # 2. Evaluate conditions   
    cond_dx_sup = bool((avg_dx_sup < 50).all())
    rmax_min_sup = min([float(np.min(max_E_sup)), float(np.min(max_I_sup))])
    rmax_max_sup = max([float(np.max(max_E_sup)), float(np.max(max_I_sup))])
    cond_rmax_sup = rmax_min_sup>10 and rmax_max_sup<151
    rmean_min_sup = min([ float(np.min(mean_E_sup)), float(np.min(mean_I_sup))])
    rmean_max_sup = max([float(np.max(mean_E_sup)), float(np.max(mean_I_sup))])
    cond_rmean_sup = rmean_min_sup>5 and  rmean_max_sup<80
    cond_r_pretrain_sup = not numpy.any(np.isnan(r_pretrain_sup))
    cond_r_train_sup = not numpy.any(np.isnan(r_train_sup))

    # 3. Resample if conditions do not hold
    if not (cond_dx_sup and cond_rmax_sup and cond_rmean_sup and cond_r_pretrain_sup and cond_r_train_sup):
        if (num_mid_calls-1)*100+num_calls>1000:
            raise Exception(f" ########### Randomized parameters violate conditions after {num_calls} sampling. ###########")
        else:
            num_calls = num_calls + 1
            print(f'Randomized superficial parameters {num_calls} time(s)',[cond_dx_sup,cond_rmax_sup,cond_rmean_sup,cond_r_pretrain_sup,cond_r_train_sup])
            # RECURSIVE FUNCTION CALL
            return randomize_params(folder, run_index, untrained_pars, logistic_regr, num_mid_calls, num_calls, start_time, J_2x2_m, cE_m, cI_m, ssn_mid, train_data, pretrain_data)
    else:
        # 4. If conditions hold, calculate logarithms of f and J parameters and optimize readout parameters
        print(f'Conditions rmax', rmax_min_sup, rmax_max_sup, ' rmean ', rmean_min_sup, rmean_max_sup, ' dx ', avg_dx_sup)
        print(f'Parameters found that satisfy conditions in {time.time() - start_time:.2f} seconds')
        # Take log of the J and f parameters (if f_I, f_E are in the randomized parameters)
        log_J_2x2_m = take_log(J_2x2_m)
        log_J_2x2_s = take_log(J_2x2_s)
        randomized_pars_log = dict(
            log_J_EE_m = log_J_2x2_m[0,0],
            log_J_EI_m = log_J_2x2_m[0,1],
            log_J_IE_m = log_J_2x2_m[1,0],
            log_J_II_m = log_J_2x2_m[1,1],
            log_J_EE_s = log_J_2x2_s[0,0],
            log_J_EI_s = log_J_2x2_s[0,1],
            log_J_IE_s = log_J_2x2_s[1,0],
            log_J_II_s = log_J_2x2_s[1,1],
            cE_m = cE_m,
            cI_m = cI_m,
            cE_s = cE_s,    
            cI_s = cI_s,
            log_f_E = np.log(f_E),
            log_f_I = np.log(f_I))

        # Optimize readout parameters by using logistic regression
        if logistic_regr:
            optimized_readout_pars = readout_pars_from_regr(randomized_pars_log, untrained_pars)
        else:
            optimized_readout_pars = dict(w_sig=readout_pars.w_sig, b_sig=readout_pars.b_sig)
        optimized_readout_pars['w_sig'] = (optimized_readout_pars['w_sig'] / np.std(optimized_readout_pars['w_sig']) ) * 0.25 / int(np.sqrt(len(optimized_readout_pars['w_sig']))) # get the same std as before - see param
        
        # Update c,f or J values if they are in the untrained_pars.ssn_pars
        attrib_keys = ['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s']
        attrib_vals = [cE_m, cI_m, cE_s, cI_s, f_E, f_I, J_EE_s, -J_EI_s_nosign, J_IE_s, -J_II_s_nosign]
        untrained_pars.ssn_pars = fill_attribute_list(untrained_pars.ssn_pars, attrib_keys, attrib_vals)

        save_orimap(untrained_pars, run_index, folder_to_save=folder)

        return optimized_readout_pars, randomized_pars_log, untrained_pars


def readout_pars_from_regr(trained_pars_dict, untrained_pars, N=1000):
    """
    This function sets readout_pars based on N sample data using logistic regression. This method is to initialize w_sig, b_sig optimally (given limited data) for a set of randomized trained_pars_dict parameters (to be trained).
    """
    # Generate stimuli and label data for setting w_sig and b_sig based on logistic regression (pretraining)
    data = create_grating_pretraining(untrained_pars.pretrain_pars, N, untrained_pars.BW_image_jax_inp, numRnd_ori1=N)

    # Extract trained and untrained parameters
    J_2x2_m, J_2x2_s, cE_m, cI_m, cE_s, cI_s, f_E, f_I, _= unpack_ssn_parameters(trained_pars_dict, untrained_pars.ssn_pars, return_kappa=False)

    # Define middle and superficial layers
    ssn_mid=SSN_mid(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_m)
    
    ssn_sup=SSN_sup(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_s, untrained_pars.dist_from_single_ori, untrained_pars.ori_dist)

    # Run reference and target data through the two layer model
    conv_pars = untrained_pars.conv_pars
    [r_ref, _], _,_, _, _ = vmap_evaluate_model_response(ssn_mid, ssn_sup, data['ref'], conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters)
    [r_target, _], _, _, _, _= vmap_evaluate_model_response(ssn_mid, ssn_sup, data['target'], conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters)
    
    # Define data for logistic regression
    from training_functions import generate_noise
    import jax
    noise_ref = generate_noise(batch_size = N, length = len(untrained_pars.oris), num_readout_noise = untrained_pars.num_readout_noise)
    noise_target = generate_noise(batch_size = N, length = len(untrained_pars.oris), num_readout_noise = untrained_pars.num_readout_noise)
    noisy_r_ref = r_ref+noise_ref*np.sqrt(jax.nn.softplus(r_ref))
    noisy_r_target = r_target+noise_target*np.sqrt(jax.nn.softplus(r_target))
    X = noisy_r_ref - noisy_r_target
    y = data['label']

    # Perform logistic regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=False)
    log_reg = LogisticRegression(max_iter=100)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy of logistic regression on test data', accuracy)
        
    # Set the readout parameters based on the results of the logistic regression
    readout_pars_opt = {'b_sig': 0, 'w_sig': np.zeros(len(X[0]))}
    readout_pars_opt['b_sig'] = float(log_reg.intercept_)
    w_sig = log_reg.coef_.T
    w_sig = w_sig.squeeze()

    if untrained_pars.pretrain_pars.is_on:
        readout_pars_opt['w_sig'] = w_sig
    else:
        readout_pars_opt['w_sig'] = w_sig[untrained_pars.middle_grid_ind]

    # Check how well the optimized readout parameters solve the tasks
    acc_train, _ = task_acc_test( trained_pars_dict, readout_pars_opt, untrained_pars, True, 4)
    acc_pretrain, _ = task_acc_test( trained_pars_dict, readout_pars_opt, untrained_pars, jit_on=True, test_offset=None, batch_size=300, pretrain_task=True)
    print('accuracy of training and pretraining with task_acc_test', acc_train, acc_pretrain)

    return readout_pars_opt


def create_initial_parameters_df(folder_path, initial_parameters, pretrained_parameters, randomized_eta, randomized_gE, randomized_gI, run_index=0, stage=0):
    """
    This function creates or appends a dataframe with the initial parameters for pretraining or training (stage 0 or 1).
    """
    # Take log of the J and f parameters (if f_I, f_E are in the randomized parameters)
    J_EE_m = np.exp(pretrained_parameters['log_J_EE_m'])
    J_EI_m = -np.exp(pretrained_parameters['log_J_EI_m'])
    J_IE_m = np.exp(pretrained_parameters['log_J_IE_m'])
    J_II_m = -np.exp(pretrained_parameters['log_J_II_m'])
    J_EE_s = np.exp(pretrained_parameters['log_J_EE_s'])
    J_EI_s = -np.exp(pretrained_parameters['log_J_EI_s'])
    J_IE_s = np.exp(pretrained_parameters['log_J_IE_s'])
    J_II_s = -np.exp(pretrained_parameters['log_J_II_s'])
    f_E = np.exp(pretrained_parameters['log_f_E'])
    f_I = np.exp(pretrained_parameters['log_f_I'])
    cE_m = pretrained_parameters['cE_m']
    cI_m = pretrained_parameters['cI_m']
    cE_s = pretrained_parameters['cE_s']
    cI_s = pretrained_parameters['cI_s']
    
    # Create the new row of initial_parameters as a dictionary with the new randomized parameters and the given other parameters (Note that EI is I to E connection)
    init_vals_dict = dict(run_index=run_index, stage=stage, f_E = f_E, f_I = f_I, cE_m = cE_m, cI_m = cI_m, cE_s = cE_s, cI_s = cI_s,
                        J_EE_m=J_EE_m, J_EI_m=J_EI_m, J_IE_m=J_IE_m, J_II_m=J_II_m, J_EE_s=J_EE_s, J_EI_s=J_EI_s, J_IE_s=J_IE_s, J_II_s=J_II_s,
                        eta=randomized_eta, gE = randomized_gE, gI= randomized_gI)

    # Create a dataframe with the initial parameters
    if initial_parameters is None:
        initial_parameters_df = pd.DataFrame(init_vals_dict, index=[0])
    else:
        initial_parameters_df = pd.concat([initial_parameters, pd.DataFrame(init_vals_dict, index=[0])])  
    
    # Save initial_parameters_df to file
    initial_parameters_df.to_csv(os.path.join(folder_path, 'initial_parameters.csv'), index=False)

    return initial_parameters_df