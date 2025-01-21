import numpy
from numpy import random
import pandas as pd
import time
import os
import sys
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))

from util import load_parameters, take_log, create_grating_training, create_grating_pretraining, unpack_ssn_parameters, filter_for_run_and_stage
from util_gabor import update_untrained_pars, save_orimap
from training_functions import train_ori_discr, task_acc_test, generate_noise, mean_training_task_acc_test, offset_at_baseline_acc
from SSN_classes import SSN_mid, SSN_sup
from model import vmap_evaluate_model_response, vmap_evaluate_model_response_mid

def fill_attribute_list(class_to_fill, attr_list, value_list):
    """Fill the attributes of a class with the given values."""
    i=0
    for attr in attr_list:
        if hasattr(class_to_fill, attr):
            setattr(class_to_fill, attr, value_list[i])

    return class_to_fill


def randomize_mid_params(randomize_pars, readout_pars, num_calls=0, untrained_pars=None, J_2x2_m=None, cE_m=None, cI_m=None, ssn_mid=None, train_data=None, pretrain_data=None, verbose = True):
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
        cond_ineq1 = jnp.abs(J_EE_m*J_II_m_nosign)*1.1 < abs(J_EI_m_nosign*J_IE_m)
        cond_ineq2 = jnp.abs(J_EI_m_nosign*gI_m)*1.1 < jnp.abs(J_II_m_nosign*gE_m)
        i = i+1
        if i>200:
            raise Exception(" ########### Randomized parameters violate conditions 1 or 2 after 200 sampling. ###########")
    if verbose:
        print(f'Parameters found that satisfy inequalities in {i} steps')
    J_2x2_m = jnp.array([[J_EE_m, -J_EI_m_nosign],[J_IE_m, -J_II_m_nosign]])
    
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
        untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, loss_pars, training_pars, pretraining_pars, readout_pars)
    else:
        untrained_pars = update_untrained_pars(untrained_pars, readout_pars, gE_m, gI_m)
        
    ##### Check conditions on the middle layer response: call function recursively if they are not satisfied and return values if they are #####
    # 1. Calculate middle layer responses
    if train_data is None:
        train_data = create_grating_training(untrained_pars.stimuli_pars, batch_size=10, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp) 
        pretrain_data = create_grating_pretraining(untrained_pars.pretrain_pars, batch_size=10, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp)
    ssn_mid = SSN_mid(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_m, untrained_pars.dist_from_single_ori)
    c_range= randomize_pars.c_range
    cE_m, cI_m = random.uniform(low=[c_range[0], c_range[0]], high=[c_range[1], c_range[1]])
    r_train_mid,_ ,avg_dx_mid, max_E_mid, max_I_mid, mean_E_mid, mean_I_mid = vmap_evaluate_model_response_mid(ssn_mid, train_data['ref'], untrained_pars.conv_pars, cE_m, cI_m, untrained_pars.gabor_filters)
    # Check if response has nan
    cond_r_train_mid = not numpy.any(jnp.isnan(r_train_mid))
    if cond_r_train_mid:
        r_pretrain_mid,_ ,_, _, _, _, _ = vmap_evaluate_model_response_mid(ssn_mid, pretrain_data['ref'], untrained_pars.conv_pars, cE_m, cI_m, untrained_pars.gabor_filters)
        
        # 2. Evaluate middle layer conditions
        cond_dx_mid = bool((avg_dx_mid < 50).all())
        rmean_min_mid = min([float(jnp.min(mean_E_mid)), float(jnp.min(mean_I_mid))])
        rmean_max_mid = max([float(jnp.max(mean_E_mid)), float(jnp.max(mean_I_mid))])    
        rmax_min_mid = min([float(jnp.min(max_E_mid)), float(jnp.min(max_I_mid))])
        rmax_max_mid = max(float(jnp.max(max_E_mid)), float(jnp.max(max_I_mid)))
        cond_rmax_mid = rmax_min_mid>10 and rmax_max_mid<151
        cond_rmean_mid = rmean_min_mid>5 and  rmean_max_mid<80
        cond_r_pretrain_mid = not numpy.any(jnp.isnan(r_pretrain_mid))        

        if not (cond_dx_mid and cond_rmax_mid and cond_rmean_mid and cond_r_pretrain_mid):
            # RECURSIVE FUNCTION CALL
            num_calls=num_calls+1
            return randomize_mid_params(randomize_pars, readout_pars, num_calls, untrained_pars, J_2x2_m, cE_m, cI_m, ssn_mid, train_data, pretrain_data, verbose = verbose)
        else:
            return untrained_pars, J_2x2_m, cE_m, cI_m, ssn_mid, train_data, pretrain_data
    else:
        # RECURSIVE FUNCTION CALL
        num_calls=num_calls+1
        return randomize_mid_params(randomize_pars, readout_pars, num_calls, untrained_pars, J_2x2_m, cE_m, cI_m, ssn_mid, train_data, pretrain_data, verbose = verbose)


def randomize_params_old(folder, run_index, untrained_pars=None, logistic_regr=True, trained_pars_dict=None, num_init=0, start_time=time.time(), verbose=True):
    def randomize_params_supp(param_dict, randomize_pars):
        import copy
        '''Randomize all values in a dictionary by a percentage of their values. The randomization is done by uniformly sampling random values from predefined ranges.'''
        param_randomized = copy.deepcopy(param_dict)
        attributes = dir(randomize_pars)
        
        for key, param_array in param_dict.items():
            matching_attributes = [attr for attr in attributes if attr.startswith(key[0])]
            param_range = getattr(randomize_pars, matching_attributes[0])
            if key.startswith('J_EE'):
                # handling the J_2x2_m and J_2x2_s matrices
                param_randomized[key] = random.uniform(low=param_range[0][0], high=param_range[0][1])
            elif key.startswith('J_EI'):
                param_randomized[key] = -random.uniform(low=param_range[1][0], high=param_range[1][1])
            elif key.startswith('J_IE'):
                param_randomized[key] = random.uniform(low=param_range[2][0], high=param_range[2][1])
            elif key.startswith('J_II'):
                param_randomized[key] = -random.uniform(low=param_range[3][0], high=param_range[3][1])
                #Clara's settings: np.array([[4.4, -1.66], [5, -1.24]])
            else:
                random_sample = random.uniform(low=param_range[0], high=param_range[1])
                param_randomized[key] = random_sample
                
        return param_randomized
    
    if untrained_pars is None: # Initialize untrained parameters if they are not given - trained SSN parameters will be 0 by default
        from util_gabor import init_untrained_pars
        from parameters import GridPars, FilterPars, StimuliPars, SSNPars, ConvPars, TrainingPars, LossPars, PretrainingPars, ReadoutPars
        grid_pars, filter_pars, stimuli_pars, ssn_pars = GridPars(), FilterPars(), StimuliPars(), SSNPars()
        conv_pars, training_pars, loss_pars, pretraining_pars, readout_pars = ConvPars(), TrainingPars(), LossPars(), PretrainingPars(), ReadoutPars()

        # Initialize untrained parameters
        untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, loss_pars, training_pars, pretraining_pars, readout_pars)
    
    from parameters import RandomizePars, ReadoutPars, TrainedSSNPars
    randomize_pars, readout_pars, trained_pars = RandomizePars(), ReadoutPars(), TrainedSSNPars()
    # define the parameters to randomize
    if trained_pars_dict is None: 
        trained_pars_dict = dict(J_EE_m=trained_pars.J_EE_m, J_IE_m=trained_pars.J_IE_m, J_EI_m=trained_pars.J_EI_m, J_II_m=trained_pars.J_II_m, J_EE_s=trained_pars.J_EE_s, J_IE_s=trained_pars.J_IE_s, J_EI_s=trained_pars.J_EI_s, J_II_s=trained_pars.J_II_s)
        for key, value in vars(trained_pars).items():
            if key.startswith('cE_') or key.startswith('cI_') or key.startswith('f_'):
                trained_pars_dict[key] = value    
        
    # Initialize parameters such that conditions are satisfied for J_mid and convergence of the differential equations of the model
    i=0
    cond_ineq1 = False # parameter inequality on Jm
    cond_ineq2 = False # parameter inequality on Jm and gE, gI

    while not (cond_ineq1 and cond_ineq2):
        randomized_pars = randomize_params_supp(trained_pars_dict, randomize_pars)
        if untrained_pars.ssn_pars.couple_c_ms:
            randomized_pars['cE_s'] = randomized_pars['cE_m']
            randomized_pars['cI_s'] = randomized_pars['cI_m']
        cond_ineq1 = jnp.abs(randomized_pars['J_EE_m']*randomized_pars['J_II_m'])*1.1 < jnp.abs(randomized_pars['J_EI_m']*randomized_pars['J_IE_m'])
        # randomize gE and gI
        untrained_pars.filter_pars.gI_m = random.uniform(low=randomize_pars.g_range[0], high=randomize_pars.g_range[1])
        untrained_pars.filter_pars.gE_m = random.uniform(low=randomize_pars.g_range[0], high=randomize_pars.g_range[1])
        cond_ineq2 = jnp.abs(randomized_pars['J_EI_m']*untrained_pars.filter_pars.gI_m)*1.1 < jnp.abs(randomized_pars['J_II_m']*untrained_pars.filter_pars.gE_m)   
        i = i+1
    
    # Calculate model response to check the convergence of the differential equations
    J_2x2_m = jnp.array([[randomized_pars['J_EE_m'], randomized_pars['J_EI_m']],[randomized_pars['J_IE_m'], randomized_pars['J_II_m']]])
    J_2x2_s = jnp.array([[randomized_pars['J_EE_s'], randomized_pars['J_EI_s']],[randomized_pars['J_IE_s'], randomized_pars['J_II_s']]])
    ssn_mid=SSN_mid(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_m, untrained_pars.dist_from_single_ori)
    ssn_sup=SSN_sup(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_s, untrained_pars.dist_from_single_ori, untrained_pars.ori_dist)
    train_data = create_grating_training(untrained_pars.stimuli_pars, batch_size=5, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp) 
    pretrain_data = create_grating_pretraining(untrained_pars.pretrain_pars, batch_size=5, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp)
    if 'cE_m' in randomized_pars:
        cE_m = randomized_pars['cE_m']
        cI_m = randomized_pars['cI_m']
    else:
        cE_m = untrained_pars.ssn_pars.cE_m
        cI_m = untrained_pars.ssn_pars.cI_m
    if 'cE_s' in randomized_pars:
        cE_s = randomized_pars['cE_s']
        cI_s = randomized_pars['cI_s']
    else:
        cE_s = untrained_pars.ssn_pars.cE_s
        cI_s = untrained_pars.ssn_pars.cI_s
    if 'f_E' in randomized_pars:
        f_E = randomized_pars['f_E']
        f_I = randomized_pars['f_I']
    else:
        f_E = untrained_pars.ssn_pars.f_E
        f_I = untrained_pars.ssn_pars.f_I
    [r_train,_],_ ,[avg_dx_mid, avg_dx_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], [mean_E_mid, mean_I_mid, mean_E_sup, mean_I_sup] = vmap_evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'], untrained_pars.conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters, untrained_pars.dist_from_single_ori, jnp.array([0.0,0.0]), untrained_pars.ssn_pars.kappa_range)
    [r_pretrain,_],_, _,_,_ = vmap_evaluate_model_response(ssn_mid, ssn_sup, pretrain_data['ref'], untrained_pars.conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters, untrained_pars.dist_from_single_ori, jnp.array([0.0,0.0]), untrained_pars.ssn_pars.kappa_range)
    cond_dx = bool((avg_dx_mid + avg_dx_sup < 50).all())
    cond_rmax = min([float(jnp.min(max_E_mid)), float(jnp.min(max_I_mid)), float(jnp.min(max_E_sup)), float(jnp.min(max_I_sup))])>10 and max([float(jnp.max(max_E_mid)), float(jnp.max(max_I_mid)), float(jnp.max(max_E_sup)), float(jnp.max(max_I_sup))])<151
    cond_rmean = min([float(jnp.min(mean_E_mid)), float(jnp.min(mean_I_mid)), float(jnp.min(mean_E_sup)), float(jnp.min(mean_I_sup))])>5 and max([float(jnp.max(mean_E_mid)), float(jnp.max(mean_I_mid)), float(jnp.max(mean_E_sup)), float(jnp.max(mean_I_sup))])<80
    cond_r_pretrain = not numpy.any(jnp.isnan(r_pretrain))
    cond_r_train = not numpy.any(jnp.isnan(r_train))
    if not (cond_dx and cond_rmax and cond_rmean and cond_r_pretrain and cond_r_train):
        if num_init>2000:
            raise Exception(" ########### Randomized parameters violate conditions even after 1000 sampling. ###########")
        else:
            num_init = num_init+i
            print(f'Randomized parameters {num_init} times',[bool(cond_ineq1),bool(cond_ineq2),cond_dx,cond_rmax,cond_rmean,cond_r_pretrain,cond_r_train])
            optimized_readout_pars, randomized_pars_log, untrained_pars = randomize_params_old(folder, run_index, untrained_pars, logistic_regr, trained_pars_dict, num_init, start_time)
    else:
        print(f'Parameters found that satisfy conditions in {time.time() - start_time:.2f} seconds')
        # Take log of the J and f parameters (if f_I, f_E are in the randomized parameters)
        log_J_2x2_m= take_log(J_2x2_m)
        log_J_2x2_s= take_log(J_2x2_s)
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
            log_f_E = jnp.log(f_E),
            log_f_I = jnp.log(f_I))

        # Optimize readout parameters by using log-linear regression
        if logistic_regr:
            optimized_readout_pars = readout_pars_from_regr(randomized_pars_log, untrained_pars)
        else:
            optimized_readout_pars = dict(w_sig=readout_pars.w_sig, b_sig=readout_pars.b_sig)
        optimized_readout_pars['w_sig'] = (optimized_readout_pars['w_sig'] / jnp.std(optimized_readout_pars['w_sig']) ) * 0.25 / int(jnp.sqrt(len(optimized_readout_pars['w_sig']))) # get the same std as before - see param

        # Randomize learning rate
        untrained_pars.training_pars.eta = random.uniform(randomize_pars.eta_range[0], randomize_pars.eta_range[1])

        save_orimap(untrained_pars.oris, run_index, folder_to_save=folder)
    
    return optimized_readout_pars, randomized_pars_log, untrained_pars


def randomize_params(folder, run_index, untrained_pars=None, logistic_regr=True, num_mid_calls=0, num_calls=0, start_time=time.time(), J_2x2_m=None, cE_m=None, cI_m=None, ssn_mid=None, train_data=None, pretrain_data=None, verbose = True):
    """Randomize the required initial parameters of the model and optimize the readout parameters using logistic regression. 
    The randomization is done by uniformly sampling random values from predefined ranges."""

    ##### Initialize middle layer parameters such that inequality conditions and response conditions are satisfied #####        
    from parameters import RandomizePars, ReadoutPars
    randomize_pars, readout_pars = RandomizePars(), ReadoutPars()
    if num_calls==0 or num_calls>100:
        untrained_pars, J_2x2_m, cE_m, cI_m, ssn_mid, train_data, pretrain_data = randomize_mid_params(randomize_pars, readout_pars, untrained_pars=untrained_pars, verbose = verbose)
        num_calls = 0
        num_mid_calls = num_mid_calls + 1
        if verbose:
            print(f'Middle layer parameters found that satisfy conditions in {time.time() - start_time:.2f} seconds.')

    ##### Initialize superficial layer parameters such that response conditions are satisfied #####
    J_range=randomize_pars.J_range
    J_EE_s, J_EI_s_nosign, J_IE_s, J_II_s_nosign = random.uniform(low=[J_range[0][0], J_range[1][0], J_range[2][0], J_range[3][0]],high=[J_range[0][1], J_range[1][1], J_range[2][1], J_range[3][1]])
    J_2x2_s = jnp.array([[J_EE_s, -J_EI_s_nosign],[J_IE_s, -J_II_s_nosign]])
    ssn_sup = SSN_sup(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_s, untrained_pars.dist_from_single_ori, untrained_pars.ori_dist)
    
    if untrained_pars.ssn_pars.couple_c_ms:
        cE_s = cE_m
        cI_s = cI_m
    else:
        c_range= randomize_pars.c_range 
        cE_s, cI_s = random.uniform(low=[c_range[0], c_range[0]], high=[c_range[1], c_range[1]])
    f_range= randomize_pars.f_range
    f_E, f_I = random.uniform(low=[f_range[0], f_range[0]], high=[f_range[1], f_range[1]])
    [r_train_sup,_],_ ,[avg_dx_mid, avg_dx_sup],[_, _, max_E_sup, max_I_sup], [_, _, mean_E_sup, mean_I_sup] = vmap_evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'], untrained_pars.conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters, untrained_pars.dist_from_single_ori, jnp.array([0.0,0.0]), untrained_pars.ssn_pars.kappa_range)
    [r_pretrain_sup,_], _, [avg_dx_pretrain_mid, avg_dx_pretrain_sup],_,_ = vmap_evaluate_model_response(ssn_mid, ssn_sup, pretrain_data['ref'], untrained_pars.conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters, untrained_pars.dist_from_single_ori, jnp.array([0.0,0.0]), untrained_pars.ssn_pars.kappa_range)
    
    # 2. Evaluate conditions   
    cond_dx_train = bool((avg_dx_mid < 100).all()) and bool((avg_dx_sup < 100).all())
    cond_dx_pretrain = bool((avg_dx_pretrain_mid < 200).all()) and bool((avg_dx_pretrain_sup < 200).all())
    cond_dx = cond_dx_train and cond_dx_pretrain
    rmax_min_sup = min([float(jnp.min(max_E_sup)), float(jnp.min(max_I_sup))])
    rmax_max_sup = max([float(jnp.max(max_E_sup)), float(jnp.max(max_I_sup))])
    cond_rmax_sup = rmax_min_sup>10 and rmax_max_sup<151
    rmean_min_sup = min([ float(jnp.min(mean_E_sup)), float(jnp.min(mean_I_sup))])
    rmean_max_sup = max([float(jnp.max(mean_E_sup)), float(jnp.max(mean_I_sup))])
    cond_rmean_sup = rmean_min_sup>5 and  rmean_max_sup<80
    cond_r_pretrain_sup = not numpy.any(jnp.isnan(r_pretrain_sup))
    cond_r_train_sup = not numpy.any(jnp.isnan(r_train_sup))

    # 3. Resample if conditions do not hold
    if not (cond_dx and cond_rmax_sup and cond_rmean_sup and cond_r_pretrain_sup and cond_r_train_sup):
        if (num_mid_calls-1)*100+num_calls>1000:
            raise Exception(f" ########### Randomized parameters violate conditions after {num_calls} sampling. ###########")
        else:
            num_calls = num_calls + 1
            if verbose:
                print(f'Randomized superficial parameters {num_calls} time(s)',[cond_dx, cond_rmax_sup,cond_rmean_sup,cond_r_pretrain_sup,cond_r_train_sup])
            # RECURSIVE FUNCTION CALL
            return randomize_params(folder, run_index, untrained_pars, logistic_regr, num_mid_calls, num_calls, start_time, J_2x2_m, cE_m, cI_m, ssn_mid, train_data, pretrain_data, verbose=verbose)
    else:
        # 4. If conditions hold, calculate logarithms of f and J parameters and optimize readout parameters
        if verbose:
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
            log_f_E = jnp.log(f_E),
            log_f_I = jnp.log(f_I))

        # Optimize readout parameters by using logistic regression
        if logistic_regr:
            optimized_readout_pars = readout_pars_from_regr(randomized_pars_log, untrained_pars, verbose=verbose)
        else:
            optimized_readout_pars = dict(w_sig=readout_pars.w_sig, b_sig=readout_pars.b_sig)
        optimized_readout_pars['w_sig'] = (optimized_readout_pars['w_sig'] / jnp.std(optimized_readout_pars['w_sig']) ) * 0.25 / int(jnp.sqrt(len(optimized_readout_pars['w_sig']))) # get the same std as before - see param
        
        # Update c,f or J values if they are in the untrained_pars.ssn_pars
        attrib_keys = ['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s']
        attrib_vals = [cE_m, cI_m, cE_s, cI_s, f_E, f_I, J_EE_s, -J_EI_s_nosign, J_IE_s, -J_II_s_nosign]
        untrained_pars.ssn_pars = fill_attribute_list(untrained_pars.ssn_pars, attrib_keys, attrib_vals)

        save_orimap(untrained_pars.oris, run_index, folder_to_save=folder)

        return optimized_readout_pars, randomized_pars_log, untrained_pars


def readout_pars_from_regr(trained_pars_dict, untrained_pars, N=1000, for_training=False, verbose=True):
    """
    This function sets readout_pars based on N sample data using logistic regression. This method is to initialize w_sig, b_sig optimally (given limited data) for a set of randomized trained_pars_dict parameters (to be trained).
    """
    # Generate stimuli and label data for setting w_sig and b_sig based on logistic regression (pretraining)
    data = create_grating_pretraining(untrained_pars.pretrain_pars, N, untrained_pars.BW_image_jax_inp, numRnd_ori1=N)

    # Extract trained and untrained parameters
    J_2x2_m, J_2x2_s, cE_m, cI_m, cE_s, cI_s, f_E, f_I, _, _, _= unpack_ssn_parameters(trained_pars_dict, untrained_pars.ssn_pars, return_kappas=False)

    # Define middle and superficial layers
    ssn_mid=SSN_mid(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_m, untrained_pars.dist_from_single_ori) 

    ssn_sup=SSN_sup(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_s, untrained_pars.dist_from_single_ori, untrained_pars.ori_dist)

    # Run reference and target data through the two layer model
    conv_pars = untrained_pars.conv_pars
    [r_sup_ref, r_mid_ref], _,_, _, _ = vmap_evaluate_model_response(ssn_mid, ssn_sup, data['ref'], conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters, untrained_pars.dist_from_single_ori, jnp.array([0.0,0.0]), untrained_pars.ssn_pars.kappa_range)
    [r_sup_target, r_mid_target], _, _, _, _= vmap_evaluate_model_response(ssn_mid, ssn_sup, data['target'], conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters, untrained_pars.dist_from_single_ori, jnp.array([0.0,0.0]), untrained_pars.ssn_pars.kappa_range)
    
    if for_training:
        # Readout configurations, when there is an additional logistic regression at the beginning of training
        sup_mid_contrib = untrained_pars.sup_mid_readout_contrib
        middle_grid_ind = untrained_pars.middle_grid_ind
        if sup_mid_contrib[0] == 0 or sup_mid_contrib[1] == 0:
            r_ref = sup_mid_contrib[0] * r_sup_ref[:,middle_grid_ind] + sup_mid_contrib[1] * r_mid_ref[:,middle_grid_ind]
            r_target = sup_mid_contrib[0] * r_sup_target[:,middle_grid_ind] + sup_mid_contrib[1] * r_mid_target[:,middle_grid_ind]
            noise_ref = generate_noise(batch_size = N, length = len(middle_grid_ind), num_readout_noise = untrained_pars.num_readout_noise)
            noise_target = generate_noise(batch_size = N, length = len(middle_grid_ind), num_readout_noise = untrained_pars.num_readout_noise)
        else: # concatenate the two layers
            r_ref = jnp.concatenate((sup_mid_contrib[1] * r_mid_ref[:,middle_grid_ind], sup_mid_contrib[0] * r_sup_ref[:,middle_grid_ind]), axis=1)
            r_target = jnp.concatenate((sup_mid_contrib[1] * r_mid_target[:,middle_grid_ind], sup_mid_contrib[0] * r_sup_target[:,middle_grid_ind]), axis=1)
            noise_ref = generate_noise(batch_size = N, length = 2*len(middle_grid_ind), num_readout_noise = untrained_pars.num_readout_noise)
            noise_target = generate_noise(batch_size = N, length = 2*len(middle_grid_ind), num_readout_noise = untrained_pars.num_readout_noise)
    else:
        r_ref = r_sup_ref
        r_target = r_sup_target
        noise_ref = generate_noise(batch_size = N, length = len(untrained_pars.oris), num_readout_noise = untrained_pars.num_readout_noise)
        noise_target = generate_noise(batch_size = N, length = len(untrained_pars.oris), num_readout_noise = untrained_pars.num_readout_noise)
    
    noisy_r_ref = r_ref+noise_ref*jnp.sqrt(jax.nn.softplus(r_ref))
    noisy_r_target = r_target+noise_target*jnp.sqrt(jax.nn.softplus(r_target))
    X = noisy_r_ref - noisy_r_target
    y = data['label']

    # Perform logistic regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=False)
    log_reg = LogisticRegression(max_iter=100)
    log_reg.fit(X_train, y_train)
    if verbose:
        y_pred = log_reg.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('accuracy of logistic regression on test data', accuracy)
        
    # Set the readout parameters based on the results of the logistic regression
    readout_pars_opt = {'b_sig': 0, 'w_sig': jnp.zeros(len(X[0]))}
    readout_pars_opt['b_sig'] = float(log_reg.intercept_)
    w_sig = log_reg.coef_.T
    w_sig = w_sig.squeeze()

    if untrained_pars.pretrain_pars.is_on or for_training:
        readout_pars_opt['w_sig'] = w_sig
    else:
        readout_pars_opt['w_sig'] = w_sig[untrained_pars.middle_grid_ind]

    # Check how well the optimized readout parameters solve the tasks
    if verbose:
        acc_train, _ = task_acc_test( trained_pars_dict, readout_pars_opt, untrained_pars, True, 4)
        acc_pretrain, _ = task_acc_test( trained_pars_dict, readout_pars_opt, untrained_pars, jit_on=True, test_offset=None, batch_size=300, pretrain_task=True)
        print('accuracy of training and pretraining with task_acc_test', acc_train, acc_pretrain)

    return readout_pars_opt


def create_initial_parameters_df(folder_path, initial_parameters, pretrained_parameters, randomized_eta, randomized_gE, randomized_gI, run_index=0, stage=0):
    """
    This function creates or appends a dataframe with the initial parameters for pretraining or training (stage 0 or 1).
    """
    # Take log of the J and f parameters (if f_I, f_E are in the randomized parameters)
    J_EE_m = jnp.exp(pretrained_parameters['log_J_EE_m'])
    J_EI_m = -jnp.exp(pretrained_parameters['log_J_EI_m'])
    J_IE_m = jnp.exp(pretrained_parameters['log_J_IE_m'])
    J_II_m = -jnp.exp(pretrained_parameters['log_J_II_m'])
    J_EE_s = jnp.exp(pretrained_parameters['log_J_EE_s'])
    J_EI_s = -jnp.exp(pretrained_parameters['log_J_EI_s'])
    J_IE_s = jnp.exp(pretrained_parameters['log_J_IE_s'])
    J_II_s = -jnp.exp(pretrained_parameters['log_J_II_s'])
    f_E = jnp.exp(pretrained_parameters['log_f_E'])
    f_I = jnp.exp(pretrained_parameters['log_f_I'])
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


def exclude_runs(folder_path, input_vector):
    """Exclude runs from the analysis by removing them from the CSV files - file modifications only happen within folder_path folders."""
    # Read the original CSV file
    df_pretraining_results = pd.read_csv(os.path.join(folder_path, 'pretraining_results.csv') )
    df_orimap = pd.read_csv(os.path.join(folder_path,'orimap.csv'))
    df_init_params = pd.read_csv(os.path.join(folder_path,'initial_parameters.csv'))
    
    # Save the original dataframe as results_complete.csv in the folder_path folder
    df_pretraining_results.to_csv(os.path.join(folder_path,'pretraining_results_complete.csv'), index=False)
    df_orimap.to_csv(os.path.join(folder_path,'orimap_complete.csv'), index=False)
    df_init_params.to_csv(os.path.join(folder_path,'initial_parameters_complete.csv'), index=False)
    
    # Exclude rows where 'runs' column is in the input_vector
    df_pretraining_results_filtered = df_pretraining_results[~df_pretraining_results['run_index'].isin(input_vector)]
    df_orimap_filtered = df_orimap[~df_orimap['run_index'].isin(input_vector)]
    df_init_params_filtered = df_init_params[~df_init_params['run_index'].isin(input_vector)]

    # Adjust the 'run_index' column
    df_orimap_filtered['run_index'] = range(len(df_orimap_filtered))
    df_init_params_filtered.loc[df_init_params_filtered['stage']==0, 'run_index'] = range(len(df_orimap_filtered)) # ChainedAssignmentError: behaviour will change in pandas 3.0
    df_init_params_filtered.loc[df_init_params_filtered['stage']==1, 'run_index'] = range(len(df_orimap_filtered))
    for i in range(df_pretraining_results_filtered['run_index'].max() + 1):
        if i not in input_vector:
            shift_val = sum(x < i for x in input_vector)
            df_pretraining_results_filtered.loc[df_pretraining_results_filtered['run_index'] == i, 'run_index'] = i - shift_val             
    
    # Save the filtered dataframes as csv files in the folder_path folder
    df_pretraining_results_filtered.to_csv(os.path.join(folder_path,'pretraining_results.csv'), index=False)
    df_orimap_filtered.to_csv(os.path.join(folder_path,'orimap.csv'), index=False)
    df_init_params_filtered.to_csv(os.path.join(folder_path,'initial_parameters.csv'), index=False)


############### PRETRAINING ###############
def main_pretraining(folder_path, num_training, initial_parameters=None, starting_time_in_main=0, verbose=True):
    """ Initialize parameters randomly and run pretraining on the general orientation discrimination task """
    def create_readout_init(data, log_regr, layer, sup_only, pretrained_readout_pars_dict, psychometric_offset, i, N):
        """ Populate the data dictionary with readout parameters """
        data['run_index'].append(i)
        data['log_regr'].append(log_regr)
        data['layer'].append(layer)
        data['sup_only'].append(sup_only)
        for j in range(N):
            data[f'w_sig_{j}'].append(pretrained_readout_pars_dict['w_sig'][j])
        data['b_sig'].append(pretrained_readout_pars_dict['b_sig'])
        data['psychometric_offset'].append(psychometric_offset)

    # Run num_training number of pretraining + training
    num_FailedRuns = 0
    i = 0
    
    run_indices=[]
    while i < num_training and num_FailedRuns < 20:

        ##### RANDOM INITIALIZATION #####
        numpy.random.seed(i + num_FailedRuns)
        
        ##### Randomize readout_pars, trained_pars, eta such that they satisfy certain conditions #####
        readout_pars_opt_dict, pretrain_pars_rand_dict, untrained_pars = randomize_params_old(folder_path, i, verbose = verbose)

        ##### Save initial parameters into initial_parameters variable #####
        initial_parameters = create_initial_parameters_df(folder_path, initial_parameters, pretrain_pars_rand_dict, untrained_pars.training_pars.eta, untrained_pars.filter_pars.gE_m,untrained_pars.filter_pars.gI_m, run_index = i, stage =0)

        ##### PRETRAINING ON GENERAL ORIENTAION DISCRIMINATION TASK #####
        results_filename = os.path.join(folder_path,'pretraining_results.csv')
        training_output_df = train_ori_discr(
                readout_pars_opt_dict,
                pretrain_pars_rand_dict,
                untrained_pars,
                stage = 0,
                results_filename=results_filename,
                jit_on=True,
                offset_step = 0.1,
                run_index = i,
                verbose = verbose
            )
        
        # Handle the case when pretraining failed (possible reason can be the divergence of ssn diff equations)
        if training_output_df is None:
            print('######### Stopped run {} because of NaN values  - num failed runs = {} #########'.format(i, num_FailedRuns))
            num_FailedRuns = num_FailedRuns + 1
            continue  
        
        ##### STAGE 1: SGD algorithm for readout parameters ######
        pretrained_readout_pars_dict_no_log_regr, trained_pars_dict, untrained_pars, _, _ = load_parameters(folder_path, run_index=i, stage=1, iloc_ind=-1, for_training=True)
        training_output_df = train_ori_discr(
                pretrained_readout_pars_dict_no_log_regr,
                trained_pars_dict,
                untrained_pars,
                stage = 1,
                results_filename=results_filename,
                jit_on=True,
                offset_step = 0.1,
                run_index = i,
                verbose = verbose
            )

        ##### LOGISTIC REGRESSION FOR READOUT PARAMETERS #####
        test_offset_vec = numpy.array([1, 3, 7, 12, 20])
        num_samples = 50000
        pretrained_readout_pars_dict_no_log_regr, trained_pars_dict, untrained_pars, offset, _ = load_parameters(folder_path, run_index=i, stage=1, iloc_ind=-1, for_training=True)
        N=len(pretrained_readout_pars_dict_no_log_regr['w_sig'])
        # Initialize the dictionary to store data for the CSV
        if os.path.exists(os.path.join(folder_path, 'init_readout_params.csv')):
            data = pd.read_csv(os.path.join(folder_path, 'init_readout_params.csv')).to_dict(orient='list')
        else:
            data = {'run_index': [], 'log_regr': [],'layer': [], 'sup_only': []}
            for j in range(N):
                data[f'w_sig_{j}'] = []
            data['b_sig'] = []
            data['psychometric_offset'] = []
            
        create_readout_init(data, 0, 1, 1, pretrained_readout_pars_dict_no_log_regr, offset, i, N)    
        # Add the optimized readout parameters for sup_only and mixed readout cases
        untrained_pars.sup_mid_readout_contrib[0]=1
        pretrained_readout_pars_dict = readout_pars_from_regr(trained_pars_dict, untrained_pars, num_samples, for_training=True)
        # Calculate psychometric offset threshold
        acc_mean, acc_std, _, _ = mean_training_task_acc_test(trained_pars_dict, pretrained_readout_pars_dict, untrained_pars, True, test_offset_vec, sample_size = 5)
        psychometric_offset = offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= untrained_pars.pretrain_pars.acc_th)

        create_readout_init(data, 1, 1, 1, pretrained_readout_pars_dict, float(psychometric_offset), i, N)
        # fill up the dictionary with pretrained_readout_pars_dict['b_sig'] pretrained_readout_pars_dict['w_sig'][i], layer =1 and sup_only=1 and log_regr = 1
        untrained_pars.sup_mid_readout_contrib=[0.5, 0.5]
        pretrained_readout_pars_dict = readout_pars_from_regr(trained_pars_dict, untrained_pars, num_samples, for_training=True)
        acc_mean, acc_std, _, _ = mean_training_task_acc_test(trained_pars_dict, pretrained_readout_pars_dict, untrained_pars, True, test_offset_vec, sample_size = 5)
        psychometric_offset = offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= untrained_pars.pretrain_pars.acc_th)
        # NOTE: pretrained_readout_pars_dict['w_sig'] is 2N-long in the mixed readout case with middle layer (layer 0) being the first N and superficial layer (layer 1) being the last N
        create_readout_init(data, 1, 0, 0, pretrained_readout_pars_dict, float(psychometric_offset), i, N)
        pretrained_readout_pars_dict['w_sig'] = pretrained_readout_pars_dict['w_sig'][N:]
        create_readout_init(data, 1, 1, 0, pretrained_readout_pars_dict, float(psychometric_offset), i, N)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(folder_path, 'init_readout_params.csv'), index=False)

        ##### Save final values into initial_parameters as initial parameters for training stage #####
        _, pretrained_pars_dict, untrained_pars = load_parameters(folder_path, run_index = i, stage = 1, iloc_ind = -1)
        initial_parameters = create_initial_parameters_df(folder_path, initial_parameters, pretrained_pars_dict, untrained_pars.training_pars.eta, untrained_pars.filter_pars.gE_m,untrained_pars.filter_pars.gI_m, run_index = i, stage =1)
        
        run_indices.append(i)
        i = i + 1
        print('Runtime of {} pretraining'.format(i), time.time()-starting_time_in_main, ' Estimated time left from pretraining', (time.time()-starting_time_in_main)*(num_training-i)/i)
        print('number of failed runs = ', num_FailedRuns)
    
    # Read pretraining_results.csv file, go over runs and check if the last psychometric_offset within that run is in the range pretraining_pars.offset_threshold. If not, then add to the excluded_run_inds.
    run_indices = [i for i in range(num_training)]
    _, _, untrained_pars = load_parameters(folder_path, run_index = num_training-1, stage = 1, iloc_ind = -1)
    exclude_run_inds = []
    for j in run_indices:
        df = pd.read_csv(os.path.join(folder_path,'pretraining_results.csv'))
        df_j = filter_for_run_and_stage(df, j, stage=1)        
        last_non_nan = df_j['psychometric_offset'].last_valid_index()
        if numpy.isnan(df_j['psychometric_offset'].iloc[last_non_nan]):
            exclude_run_inds.append(j)
        elif df_j['psychometric_offset'].iloc[last_non_nan] < untrained_pars.pretrain_pars.offset_threshold[0] or df_j['psychometric_offset'].iloc[last_non_nan] > untrained_pars.pretrain_pars.offset_threshold[1]:
            exclude_run_inds.append(j)
    # Save excluded runs to a file
    with open(os.path.join(folder_path, 'excluded_runs_from_pretraining.csv'), 'w') as f:
        for item in exclude_run_inds:
            f.write("%s\n" % item)

    # Exclude runs with indices exclude_run_inds from the initial_parameters, orimap and pretraining_results files
    exclude_runs(folder_path, exclude_run_inds)
    print('Number of runs excluded from pretraining:', len(exclude_run_inds), 'out of', len(run_indices))

    return len(run_indices)-len(exclude_run_inds)