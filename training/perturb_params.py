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

from util import take_log, create_grating_training, sep_exponentiate, create_grating_pretraining, unpack_ssn_parameters
from util_gabor import update_untrained_pars
from SSN_classes import SSN_mid, SSN_sup
from model import vmap_evaluate_model_response
from training_functions import task_acc_test


def randomize_params_supp(param_dict, randomize_pars, attributes_ranges=None):
    '''Randomize all values in param_dict dictionary by uniformly sampling from predefined ranges in randomize_pars.'''
    if attributes_ranges is None:
        attributes_ranges = [f for f in dir(randomize_pars) if not callable(getattr(randomize_pars,f)) and not f.startswith('__')]

    for key, param_array in param_dict.items():
        matching_attributes = [attr for attr in attributes_ranges if attr.startswith(key[0])]
        param_range = getattr(randomize_pars, matching_attributes[0]) # Note that this would not work for gE and gI ranges as their first letter is the same but those are not sampled with this function
        if isinstance(param_array, (float,np.floating, numpy.floating)):
            random_sample = random.uniform(low=param_range[0], high=param_range[1])
            param_dict[key] = random_sample
        else:
            # handling the J_2x2_m and J_2x2_s matrices
            JEE, JEI, JIE, JII = random.uniform(
                low=[param_range[0][0], param_range[1][0], param_range[2][0], param_range[3][0]],
                high=[param_range[0][1], param_range[1][1], param_range[2][1], param_range[3][1]])
            param_dict[key] = np.array([[JEE, -JEI],[JIE, -JII]])
        
    return param_dict


def randomize_params(folder, run_index, untrained_pars=None, logistic_regr=True, num_init=0, start_time=time.time()):
    '''Randomize the required initial parameters of the model and optimize the readout parameters using logistic regression. 
    The randomization is done by uniformly sampling random values from predefined ranges.'''
        
    from parameters import PretrainedSSNPars, RandomizePars, ReadoutPars
    pretrained_pars, randomize_pars, readout_pars = PretrainedSSNPars(), RandomizePars(), ReadoutPars()
    pretrained_pars_keys = {attr: getattr(pretrained_pars, attr) for attr in dir(pretrained_pars) if not callable(getattr(pretrained_pars,attr)) and not attr.startswith('__')}

    ##### Initialize untrained parameters if they are not given #####
    if untrained_pars is None:
        from util_gabor import init_untrained_pars
        from parameters import GridPars, FilterPars, StimuliPars, SSNPars, ConvPars, TrainingPars, LossPars, PretrainingPars
        grid_pars, filter_pars, stimuli_pars, ssn_pars = GridPars(), FilterPars(), StimuliPars(), SSNPars()
        conv_pars, training_pars, loss_pars, pretraining_pars = ConvPars(), TrainingPars(), LossPars(), PretrainingPars()

        # Randomize gE and gI (feedforward weights from stimuli and mid layer) and learning rate 
        training_pars.eta = random.uniform(randomize_pars.eta_range[0], randomize_pars.eta_range[1])
        filter_pars.gE_m = random.uniform(low=randomize_pars.gE_range[0], high=randomize_pars.gE_range[1])
        filter_pars.gI_m = random.uniform(low=randomize_pars.gI_range[0], high=randomize_pars.gI_range[1])
        # Initialize untrained parameters
        untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                    loss_pars, training_pars, pretraining_pars, readout_pars, run_index, folder_to_save=folder)

    randomized_pars_dict = {}
    # Loop through each attribute and assign values
    for attr in pretrained_pars_keys:
        randomized_pars_dict[attr] = getattr(pretrained_pars, attr)
        
    ##### Initialize parameters such inequality conditions are satisfied #####
    i=0
    cond_ineq1 = False # parameter inequality on Jm
    cond_ineq2 = False # parameter inequality on Jm and gE, gI
    attributes_ranges = [f for f in dir(randomize_pars) if not callable(getattr(randomize_pars,f)) and not f.startswith('__')]
    while not (cond_ineq1 and cond_ineq2):
        randomized_pars = randomize_params_supp(randomized_pars_dict, randomize_pars, attributes_ranges)
        cond_ineq1 = np.abs(randomized_pars['J_2x2_m'][0,0]*randomized_pars['J_2x2_m'][1,1])*1.1 < np.abs(randomized_pars['J_2x2_m'][1,0]*randomized_pars['J_2x2_m'][0,1])
        cond_ineq2 = np.abs(randomized_pars['J_2x2_m'][0,1]*untrained_pars.filter_pars.gI_m)*1.1 < np.abs(randomized_pars['J_2x2_m'][1,1]*untrained_pars.filter_pars.gE_m)
        if not cond_ineq2:
            gE_m = random.uniform(low=randomize_pars.gE_range[0], high=randomize_pars.gE_range[1])
            gI_m = random.uniform(low=randomize_pars.gI_range[0], high=randomize_pars.gI_range[1])
            untrained_pars = update_untrained_pars(untrained_pars,readout_pars, gE_m, gI_m)
        print(cond_ineq1, cond_ineq2)
        i = i+1
        if i>200:
            raise Exception(" ########### Randomized parameters violate conditions 1 or 2 after 200 sampling. ###########")
    
    ##### Accept initialization if conditions on the model response are also satisfied #####
    # 1. Calculate model responses
    ssn_mid = SSN_mid(untrained_pars.ssn_pars, untrained_pars.grid_pars, randomized_pars['J_2x2_m'])
    ssn_sup = SSN_sup(untrained_pars.ssn_pars, untrained_pars.grid_pars, randomized_pars['J_2x2_s'], untrained_pars.oris, untrained_pars.ori_dist)
    train_data = create_grating_training(untrained_pars.stimuli_pars, batch_size=5, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp) 
    pretrain_data = create_grating_pretraining(untrained_pars.pretrain_pars, batch_size=5, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp)
    c_E = randomized_pars['c_E']
    c_I = randomized_pars['c_I']
    f_E = randomized_pars['f_E']
    f_I = randomized_pars['f_I']
    [r_train,_],_ ,[avg_dx_mid, avg_dx_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], [mean_E_mid, mean_I_mid, mean_E_sup, mean_I_sup] = vmap_evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'], untrained_pars.conv_pars,c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)
    [r_pretrain,_],_, _,_,_ = vmap_evaluate_model_response(ssn_mid, ssn_sup, pretrain_data['ref'], untrained_pars.conv_pars,c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)
    
    # 2. Evaluate conditions
    cond_dx = bool((avg_dx_mid + avg_dx_sup < 50).all())
    rmax_min_to_check = min([float(np.min(max_E_mid)), float(np.min(max_I_mid)), float(np.min(max_E_sup)), float(np.min(max_I_sup))])
    rmax_max_to_check = max([float(np.max(max_E_mid)), float(np.max(max_I_mid)), float(np.max(max_E_sup)), float(np.max(max_I_sup))])
    cond_rmax = rmax_min_to_check>10 and rmax_max_to_check<151
    rmean_min_to_check = min([float(np.min(mean_E_mid)), float(np.min(mean_I_mid)), float(np.min(mean_E_sup)), float(np.min(mean_I_sup))])
    rmean_max_to_check =  max([float(np.max(mean_E_mid)), float(np.max(mean_I_mid)), float(np.max(mean_E_sup)), float(np.max(mean_I_sup))])
    cond_rmean = rmean_min_to_check>5 and  rmean_max_to_check<80
    cond_r_pretrain = not numpy.any(np.isnan(r_pretrain))
    cond_r_train = not numpy.any(np.isnan(r_train))

    # 3. Resample if conditions do not hold
    if not (cond_dx and cond_rmax and cond_rmean and cond_r_pretrain and cond_r_train):
        if num_init>2000:
            raise Exception(" ########### Randomized parameters violate conditions after 2000 sampling. ###########")
        else:
            num_init = num_init+i
            print(f'Randomized parameters {num_init} times',[cond_dx,cond_rmax,cond_rmean,cond_r_pretrain,cond_r_train])
            # RECURSIVE CALL
            optimized_readout_pars, randomized_pars_log, untrained_pars = randomize_params(folder, run_index, untrained_pars = untrained_pars, logistic_regr=logistic_regr, num_init=num_init, start_time=start_time)
    else:
        # 4. If conditions hold, calculate logarithms of f and J parameters and optimize readout parameters
        print(f'Conditions rmax', rmax_min_to_check, rmax_max_to_check, ' rmean ', rmean_min_to_check, rmean_max_to_check, ' dx ',avg_dx_mid + avg_dx_sup)
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
            optimized_readout_pars = readout_pars_from_regr(randomized_pars_log, untrained_pars)
        else:
            optimized_readout_pars = dict(w_sig=readout_pars.w_sig, b_sig=readout_pars.b_sig)
        optimized_readout_pars['w_sig'] = (optimized_readout_pars['w_sig'] / np.std(optimized_readout_pars['w_sig']) ) * 0.25 / int(np.sqrt(len(optimized_readout_pars['w_sig']))) # get the same std as before - see param
        
        # Update c,f or J values if they are in the untrained_pars.ssn_pars
        if hasattr(untrained_pars.ssn_pars, 'c_E'):
            untrained_pars.ssn_pars.c_E = randomized_pars['c_E']
            untrained_pars.ssn_pars.c_I = randomized_pars['c_I']

        if hasattr(untrained_pars.ssn_pars, 'f_E'):
            untrained_pars.ssn_pars.f_E = randomized_pars['f_E']
            untrained_pars.ssn_pars.f_I = randomized_pars['f_I']

        if hasattr(untrained_pars.ssn_pars, 'J_2x2_s'):
            untrained_pars.ssn_pars.J_2x2_s = randomized_pars['J_2x2_s']

        if hasattr(untrained_pars.ssn_pars, 'J_2x2_m'):
            untrained_pars.ssn_pars.J_2x2_m = randomized_pars['J_2x2_m']

    return optimized_readout_pars, randomized_pars_log, untrained_pars


def readout_pars_from_regr(trained_pars_dict, untrained_pars, N=1000):
    '''
    This function sets readout_pars based on N sample data using linear regression. This method is to initialize w_sig, b_sig optimally (given limited data) for a set of randomized trained_pars_dict parameters (to be trained).
    '''
    # Generate stimuli and label data for setting w_sig and b_sig based on linear regression (pretraining)
    data = create_grating_pretraining(untrained_pars.pretrain_pars, N, untrained_pars.BW_image_jax_inp, numRnd_ori1=N)
    
    # Extract trained and untrained parameters
    J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I, _= unpack_ssn_parameters(trained_pars_dict, untrained_pars, return_kappa=False)

    # Define middle and superficial layers
    ssn_mid=SSN_mid(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_m)
    ssn_sup=SSN_sup(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_s, untrained_pars.oris, untrained_pars.ori_dist)
    
    # Run reference and target data through the two layer model
    conv_pars = untrained_pars.conv_pars
    [r_ref, _], _,_, _, _ = vmap_evaluate_model_response(ssn_mid, ssn_sup, data['ref'], conv_pars, c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)
    [r_target, _], _, _, _, _= vmap_evaluate_model_response(ssn_mid, ssn_sup, data['target'], conv_pars, c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)

    # Get data for logistic regression
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

    # Check how well the optimized readout parameters solve the tasks
    acc_train, _ = task_acc_test( trained_pars_dict, readout_pars_opt, untrained_pars, True, 4)
    acc_pretrain, _ = task_acc_test( trained_pars_dict, readout_pars_opt, untrained_pars, True, None, batch_size=300, pretrain_task=True)
    print('accuracy of logistic regression on training and pretraining data', acc_train, acc_pretrain)

    return readout_pars_opt


def create_initial_parameters_df(folder_path, initial_parameters, pretrained_parameters, randomized_eta, randomized_gE, randomized_gI, run_index=0, stage=0):
    '''
    This function creates or appends a dataframe with the initial parameters for pretraining or training (stage 0 or 1).
    '''
    # Take log of the J and f parameters (if f_I, f_E are in the randomized parameters)
    J_2x2_m = sep_exponentiate(pretrained_parameters['log_J_2x2_m'])
    J_2x2_s = sep_exponentiate(pretrained_parameters['log_J_2x2_s'])
    f_E = np.exp(pretrained_parameters['log_f_E'])
    f_I = np.exp(pretrained_parameters['log_f_I'])
    c_E = pretrained_parameters['c_E']
    c_I = pretrained_parameters['c_I']
    
    # Create the new row of initial_parameters as a dictionary with the new randomized parameters and the given other parameters (Note that EI is I to E connection)
    init_vals_dict = dict(run_index=run_index, stage=stage, f_E = f_E, f_I = f_I, c_E = c_E, c_I = c_I,
                        J_m_EE=J_2x2_m[0,0], J_m_EI=J_2x2_m[0,1], J_m_IE=J_2x2_m[1,0], J_m_II=J_2x2_m[1,1],
                        J_s_EE=J_2x2_s[0,0], J_s_EI=J_2x2_s[0,1], J_s_IE=J_2x2_s[1,0], J_s_II=J_2x2_s[1,1],
                        eta=randomized_eta, gE = randomized_gE, gI= randomized_gI)

    # Create a dataframe with the initial parameters
    if initial_parameters is None:
        initial_parameters_df = pd.DataFrame(init_vals_dict, index=[0])
    else:
        initial_parameters_df = pd.concat([initial_parameters, pd.DataFrame(init_vals_dict, index=[0])])  
    
    # Save initial_parameters_df to file
    initial_parameters_df.to_csv(os.path.join(folder_path, 'initial_parameters.csv'), index=False)

    return initial_parameters_df