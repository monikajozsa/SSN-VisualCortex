#import matplotlib.pyplot as plt
import gc
import jax
import jax.numpy as jnp
from jax import vmap
import optax
import pandas as pd
import numpy
from scipy.stats import mannwhitneyu, linregress
import copy
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from util import create_grating_training, sigmoid, create_grating_pretraining, unpack_ssn_parameters
from training.SSN_classes import SSN_mid, SSN_sup
from training.model import evaluate_model_response


def has_plateaued(loss, p_threshold=0.05, window_size = 10):
    """
    Check if the loss has plateaued by performing a t-test on the last 20 values.

    Parameters:
    loss: A vector of loss values over time.
    p_threshold (float): threshold p-value for a ttest on the mean of the last two windows
    window_size (int): size of the windows for which we calculate the ttest

    Returns:
    int: 1 if the loss has plateaued, 0 otherwise.
    """
    if len(loss) < 2*window_size:
        return False

    # Check if the mean of the last window_size values is significantly different from the previous window_size values
    _, p_value = mannwhitneyu(loss[-window_size:-1], loss[-2*window_size:-window_size])
    
    return int(p_value > p_threshold)


def append_parameter_lists(trained_pars_dict, ssn_pars, key_list, value_list, as_log=False, flatten=False):
    """
    Appends the value list by the new values that is in either the trained_pars_dict or the ssn_pars.
    """
    new_values = []
    for key in key_list:
        if key in trained_pars_dict.keys():
            # no need to handle log because when as_log is True (only for J and f), the trained_pars_dict stores the log values
            new_values.append(float(trained_pars_dict[key]))
        else:
            if as_log:
                if key.startswith('log'):
                    nolog_key=key[4:]
                    new_values.append(float(jnp.log(jnp.abs(getattr(ssn_pars, nolog_key)))))
                else:
                    new_values.append(float(getattr(ssn_pars, key)))
            else:
                new_values.append(float(getattr(ssn_pars, key)))
    if len(new_values) > 1:
        value_list.append(jnp.array(new_values))
    else:
        value_list.append(new_values[0])
    
    return value_list


def train_ori_discr(
    readout_pars_dict,
    trained_pars_dict,
    untrained_pars,
    threshold = 0.75,
    offset_step = 0.25,
    results_filename=None,
    jit_on=True,
    run_index = 0
):
    """
    Trains a two-layer SSN network model in a pretraining and a training stage. It first trains the SSN layer parameters and the readout parameters for a general orientation discrimination task. Then, it further trains the SSN layer parameters for a fine orientation discrimination task.
    
    Parameters:
    - readout_pars_dict (dict): Parameters for the readout layer.
    - trained_pars_dict (dict): Parameters for the SSN layer.
    - untrained_pars (class): Includes grid_pars, stimuli_pars, conn_pars_m, 
                                  conn_pars_s, filter_pars, ssn_ori_map, ssn_pars, 
                                  conv_pars, loss_pars, training_pars.
    - results_filename (str, optional): Filename for saving results.
    - jit_on (bool): If True, enables JIT compilation for performance improvement.
    """
    # Unpack training_pars and stimuli_pars from untrained_pars
    ssn_pars = untrained_pars.ssn_pars
    training_pars = untrained_pars.training_pars
    stimuli_pars = untrained_pars.stimuli_pars
    pretrain_on = untrained_pars.pretrain_pars.is_on
    pretrain_offset_threshold = untrained_pars.pretrain_pars.offset_threshold
    shuffle_labels = untrained_pars.training_pars.shuffle_labels

    # Initialise optimizer and set first stage accuracy threshold
    optimizer = optax.adam(training_pars.eta)
    if pretrain_on:
        opt_state_ssn = optimizer.init(trained_pars_dict)
        opt_state_readout = optimizer.init(readout_pars_dict)
        training_loss_val_and_grad = training_loss_val_and_grad_ssn_readout # train both readout pars and ssn layers pars
    else:
        opt_state_ssn = optimizer.init(trained_pars_dict)
        training_loss_val_and_grad = training_loss_val_and_grad_ssn_only # train only ssn layers pars

    # Define SGD_steps indices and offsets where training task accuracy is calculated
    if pretrain_on:
        numSGD_steps = untrained_pars.pretrain_pars.SGD_steps
        min_acc_check_ind = untrained_pars.pretrain_pars.min_acc_check_ind
        acc_check_ind = jnp.arange(0, numSGD_steps, untrained_pars.pretrain_pars.acc_check_freq)
        acc_check_ind = acc_check_ind[(acc_check_ind > min_acc_check_ind) | (acc_check_ind <2)] # by leaving in 0, we make sure that it is not empty as we refer to it later
        pretrain_stage_1_acc_th = untrained_pars.pretrain_pars.pretrain_stage_1_acc_th
        stage_list = [0,1]
    else:
        numSGD_steps = training_pars.SGD_steps
        stage_list = [2]
    test_offset_vec = numpy.array([1, 3, 7, 12, 20])  # offset values to define offset threshold where given accuracy is achieved

    # Define SGD_steps indices where losses an accuracy are validated
    val_steps = jnp.arange(0, numSGD_steps, training_pars.validation_freq)
    print_steps = jnp.arange(0, numSGD_steps, 20) # make this such that they are from val_steps - at the moment, printing happens within the validation steps

    print(
        "SGD_step: {} ¦ learning rate: {} ¦ batch size {}".format(
            numSGD_steps, training_pars.eta, training_pars.batch_size,
        )
    )

    if results_filename:
        print("Will be saving results to ", results_filename)
    else:
        print("#### NOT SAVING! ####")

    start_time = time.time()
    
    ######## Pretraining: One-stage, Training: Two-stage, where 1) parameters of sigmoid layer 2) parameters of SSN layers #############
    pretrain_stop_flag = False
    for stage in stage_list:
        # Skip stage 1 in pretraining if pretrain_stop_flag is True
        if pretrain_stop_flag:
            break
        if stage == 1:
            # Reinitialise optimizer and reset the argnum to take gradient wrt readout pars
            opt_state_readout = optimizer.init(readout_pars_dict)
            training_loss_val_and_grad = training_loss_val_and_grad_readout_only           

        # STOCHASTIC GRADIENT DESCENT LOOP
        for SGD_step in range(numSGD_steps):
            # i) Calculate model loss, accuracy, gradient
            train_loss, train_loss_all, train_acc, _, _, train_max_rate, train_mean_rate, grad = loss_and_grad_ori_discr(stage, trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, training_loss_val_and_grad, shuffle_labels=shuffle_labels)

            if stage == 1 and SGD_step == 0:
                if train_acc > pretrain_stage_1_acc_th:
                    print("Early stop of stage 1: accuracy {} reached target {}".format(
                        train_acc, pretrain_stage_1_acc_th)
                    )
                    # Store final step index and exit loop
                    pretrain_stop_flag = True
                    break

            if jax.numpy.isnan(train_loss):
                print('NaN values in loss at step', SGD_step)
                return None
            
            # Calculate psychometric_offset if the SGD step is in the acc_check_ind vector
            if pretrain_on and SGD_step in acc_check_ind:
                acc_mean, acc_std, _, _ = mean_training_task_acc_test(trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, test_offset_vec, sample_size = 5)
                # fit log-linear curve to acc_mean_max and test_offset_vec and find where it crosses baseline_acc=0.794
                psychometric_offset = offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= untrained_pars.pretrain_pars.acc_th)
                if SGD_step in print_steps:
                    print('Mean accuracies:',acc_mean, 'for offsets', test_offset_vec)
                    print('estimated psychometric threshold:', psychometric_offset)

            # ii) Store parameters and metrics (append or initialize lists)
            if 'stages' in locals():
                stages.append(stage)
                train_losses_all.append(train_loss_all.ravel())
                train_accs.append(train_acc)
                train_max_rates.append(train_max_rate)
                train_mean_rates.append(train_mean_rate)
                append_parameter_lists(trained_pars_dict, ssn_pars, ['log_J_EE_m', 'log_J_EI_m', 'log_J_IE_m', 'log_J_II_m'], log_J_2x2_m, as_log=True)
                append_parameter_lists(trained_pars_dict, ssn_pars, ['log_J_EE_s', 'log_J_EI_s', 'log_J_IE_s', 'log_J_II_s'], log_J_2x2_s, as_log=True)
                append_parameter_lists(trained_pars_dict, ssn_pars, ['cE_m'], cE_m)
                append_parameter_lists(trained_pars_dict, ssn_pars, ['cI_m'], cI_m)
                append_parameter_lists(trained_pars_dict, ssn_pars, ['cE_s'], cE_s)
                append_parameter_lists(trained_pars_dict, ssn_pars, ['cI_s'], cI_s)
                append_parameter_lists(trained_pars_dict, ssn_pars, ['log_f_E'], log_f_E, as_log=True)
                append_parameter_lists(trained_pars_dict, ssn_pars, ['log_f_I'], log_f_I, as_log=True)
                if 'kappa_Jsup' in trained_pars_dict.keys():
                    tanh_kappa_Jsup = jnp.tanh(trained_pars_dict['kappa_Jsup'])
                    kappas_Jsup.append(tanh_kappa_Jsup.ravel())
                elif hasattr(ssn_pars, 'kappa_Jsup'):
                    tanh_kappa_Jsup = jnp.tanh(ssn_pars.kappa_Jsup)
                    kappas_Jsup.append(tanh_kappa_Jsup.ravel())
                if 'kappa_Jmid' in trained_pars_dict.keys():
                    tanh_kappa_Jmid = jnp.tanh(trained_pars_dict['kappa_Jmid'])
                    kappas_Jmid.append(tanh_kappa_Jmid.ravel())
                elif hasattr(ssn_pars, 'kappa_Jmid'):
                    tanh_kappa_Jmid = jnp.tanh(ssn_pars.kappa_Jmid)
                    kappas_Jmid.append(tanh_kappa_Jmid.ravel())
                if 'kappa_f' in trained_pars_dict.keys():
                    tanh_kappa_f = jnp.tanh(trained_pars_dict['kappa_f'])
                    kappas_f.append(tanh_kappa_f.ravel())
                elif hasattr(ssn_pars, 'kappa_f'):
                    tanh_kappa_f = jnp.tanh(ssn_pars.kappa_f)
                    kappas_f.append(tanh_kappa_f.ravel())
                if pretrain_on:
                    w_sigs.append(readout_pars_dict['w_sig'])
                    b_sigs.append(readout_pars_dict['b_sig'])
                else:
                    staircase_offsets.append(stimuli_pars.offset)
            else:
                stages=[stage]
                train_losses_all=[train_loss_all.ravel()]
                train_accs=[train_acc]
                train_max_rates=[train_max_rate]
                train_mean_rates=[train_mean_rate]
                log_J_2x2_m, log_J_2x2_s, cE_m, cI_m, cE_s, cI_s, log_f_E, log_f_I, kappas_Jsup, kappas_Jmid, kappas_f = unpack_ssn_parameters(trained_pars_dict, ssn_pars, as_log_list=True) 
                if pretrain_on:
                    w_sigs = [readout_pars_dict['w_sig']]
                    b_sigs = [readout_pars_dict['b_sig']]
                else:
                    staircase_offsets=[stimuli_pars.offset]

            if pretrain_on and SGD_step in acc_check_ind:
                if 'psychometric_offsets' in locals():
                    psychometric_offsets.append(float(psychometric_offset))
                    acc_means.append(acc_mean)
                    acc_stds.append(acc_std)
                else:
                    psychometric_offsets=[float(psychometric_offset)]
                    acc_means=[acc_mean]
                    acc_stds=[acc_std]

            # ii) Early stopping during pre-training and training
            # Check for early stopping during pre-training: break out from SGD_step loop and stages loop (using a flag)
            if pretrain_on:
                if SGD_step > untrained_pars.pretrain_pars.min_stop_ind and len(psychometric_offsets)>2:
                    pretrain_stop_flag = all(jnp.array(psychometric_offsets[-2:]) > pretrain_offset_threshold[0]) and all(jnp.array(psychometric_offsets[-2:]) < pretrain_offset_threshold[1])
                if pretrain_stop_flag:
                    print('Stopping pretraining: desired accuracy achieved for training task.')
                    break
                
            # Check for stopping criterium for stage 1 in pretraining
            if pretrain_on and stage==1:
                if SGD_step == 0:
                    avg_acc = train_acc
                else:
                    avg_acc = jnp.mean(jnp.asarray(train_accs[-min(SGD_step,3):]))
                if avg_acc > pretrain_stage_1_acc_th:
                    print("Early stop of stage 1: accuracy {} reached target {}".format(
                            avg_acc, pretrain_stage_1_acc_th)
                    )
                    # Store final step index and exit first training loop
                    pretrain_stop_flag = True
                    break

            # iii) Staircase algorithm during training: 3-down 1-up adjustment rule for the offset
            if not pretrain_on and stage==2:
                if train_acc < threshold:
                    temp_threshold=0
                    stimuli_pars.offset =  min(stimuli_pars.offset + offset_step, stimuli_pars.max_train_offset)
                else:
                    temp_threshold=1
                    if SGD_step > 2 and jnp.sum(jnp.asarray(threshold_variables[-3:])) == 3:
                        stimuli_pars.offset =  stimuli_pars.offset - offset_step
                if 'threshold_variables' in locals():
                    threshold_variables.append(temp_threshold)
                else:
                    threshold_variables=[temp_threshold]

            # iv) Loss and accuracy validation + printing results    
            if SGD_step in val_steps:
                #### Calculate loss and accuracy on new validation data set
                val_acc_vec, val_loss_vec = task_acc_test(trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, stimuli_pars.offset)
                acc_vec_125, loss_vec_125 = task_acc_test(trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, 125.0)
            
                val_loss = jnp.mean(val_loss_vec)
                val_acc = jnp.mean(val_acc_vec)
                acc_125 = jnp.mean(acc_vec_125)
                if 'val_accs' in locals():
                    val_accs.append(val_acc)
                    val_losses.append(val_loss)
                    accs_125.append(acc_125)
                else:
                    val_accs=[val_acc]
                    val_losses=[val_loss]
                    accs_125=[acc_125]
                
                if not pretrain_on:
                    acc_mean, acc_std,_, _ = mean_training_task_acc_test(trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, test_offset_vec, sample_size = 5)
                    # fit log-linear curve to acc_mean_max and test_offset_vec and find where it crosses baseline_acc=0.794
                    psychometric_offset = offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= untrained_pars.pretrain_pars.acc_th)
                    if 'psychometric_offsets' in locals():
                        psychometric_offsets.append(float(psychometric_offset))
                    else:
                        psychometric_offsets=[float(psychometric_offset)]
                    SGD_step_time = time.time() - start_time
                    if SGD_step in print_steps:
                        print("Stage: {}¦ Train loss: {:.3f} ¦ Val loss: {:.3f} ¦ Train accuracy: {:.3f} ¦ Val accuracy: {:.3f} ¦ Readout loss: {:.3f}  ¦ Dx loss: {:.3f} ¦ Rmax loss: {:.3f}  ¦ Rmean loss: {:.3f}  ¦ SGD step: {} ¦ Psy. offset: {} ¦ Runtime: {:.4f} ".format(
                            stage, train_loss, val_loss, train_acc, val_acc, train_loss_all[0].item(), train_loss_all[1].item(),  train_loss_all[2].item(),  train_loss_all[3].item(), SGD_step, psychometric_offset, SGD_step_time))
                    if 'acc_means' in locals():
                        acc_means.append(acc_mean)
                        acc_stds.append(acc_std)
                    else:
                        acc_means=[acc_mean]
                        acc_stds=[acc_std]
                    # Early stopping criteria for training - if accuracies in multiple relevant offsets did not change
                    if SGD_step > training_pars.min_stop_ind:
                        acc_means_np = numpy.array(acc_means)
                        # If accuracy is stable for the last three offsets in test_offset_vec, then stop training
                        if has_plateaued(acc_means_np[:,0], window_size=10) and has_plateaued(acc_means_np[:,1], window_size=10) and has_plateaued(acc_means_np[:,2], window_size=10):
                            break
                
            # v) Parameter update. Note that training has one-stage, pre-training has two-stages, where the second stage (stage 1) is skipped if the accuracy satisfies a minimum threshold criteria
            if pretrain_on:
                # linear regression to check if acc_mean is decreasing (happens when pretraining goes on while solving the flipped training task)
                acc_mean_slope, _, _, p_value, _ = linregress(range(len(acc_mean)), acc_mean)
                # During pretraining, if the validation accuracy is low and accuracy is decreasing as the offset increases, then flip the readout parameters
                train_acc_test, _ = task_acc_test(trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, test_offset= 4.0, pretrain_task= False) 
                if stage==0 and (acc_mean_slope < -0.01) and (p_value < 0.05) and train_acc_test<0.5:
                    # Flip the center readout parameters if validation accuracy is low (which corresponds to training task) and training task accuracy is decreasing as offset increases
                    readout_pars_dict['w_sig'] = readout_pars_dict['w_sig'].at[untrained_pars.middle_grid_ind].set(-readout_pars_dict['w_sig'][untrained_pars.middle_grid_ind])
                    readout_pars_dict['b_sig'] = -readout_pars_dict['b_sig']
                    val_acc =  1-val_acc
                    # Update the first moments in the opt_state_readout
                    m = opt_state_readout[0]  # First moment vector
                    m[1]['b_sig']=-m[1]['b_sig']
                    m[1]['w_sig']=m[1]['w_sig'].at[untrained_pars.middle_grid_ind].set(-m[1]['w_sig'][untrained_pars.middle_grid_ind])
                    
                    # Print out the changes in accuracy
                    pretrain_acc_test, _ = task_acc_test(trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, test_offset= None, pretrain_task= True)                  
                    print('Flipping readout parameters. Pretrain acc', pretrain_acc_test,'Training acc:', train_acc_test)
                else:
                    if stage == 0:
                        # Update ssn layer parameters 
                        updates_ssn, opt_state_ssn = optimizer.update(grad[0], opt_state_ssn)
                        trained_pars_dict = optax.apply_updates(trained_pars_dict, updates_ssn)
                        # Update readout parameters
                        updates_readout, opt_state_readout = optimizer.update(grad[1], opt_state_readout)
                        readout_pars_dict = optax.apply_updates(readout_pars_dict, updates_readout)
                    else:
                        # Update readout parameters for stage 1
                        updates_readout, opt_state_readout = optimizer.update(grad, opt_state_readout)
                        readout_pars_dict = optax.apply_updates(readout_pars_dict, updates_readout)
            else:                                
                # Update ssn layer parameters
                updates_ssn, opt_state_ssn = optimizer.update(grad, opt_state_ssn)
                trained_pars_dict = optax.apply_updates(trained_pars_dict, updates_ssn)
            
        # Clear python cache data
        gc.collect()
           
    ############# SAVING and RETURN OUTPUT #############

    # Define SGD_steps indices for training and validation
    if pretrain_on:
        SGD_steps = jnp.arange(0, len(stages))
        val_accs = jnp.array(val_accs)
        accs_125 = jnp.array(accs_125)
        val_steps_stage_0_and_1 = jnp.concatenate([val_steps, numSGD_steps+val_steps])
        val_SGD_steps = val_steps_stage_0_and_1[0:len(val_accs)]
        acc_check_ind_stage_0_and_1 = jnp.concatenate([acc_check_ind, numSGD_steps+acc_check_ind])
        acc_check_ind = acc_check_ind_stage_0_and_1[0:len(psychometric_offsets)]
        step_indices = dict(SGD_steps=SGD_steps, val_SGD_steps=val_SGD_steps, acc_check_ind=acc_check_ind)
    else:
        SGD_steps = jnp.arange(0, len(stages))
        val_SGD_steps = val_steps[0:len(val_accs)]
        step_indices = dict(SGD_steps=SGD_steps, val_SGD_steps=val_SGD_steps)
        
    # Create DataFrame and save the DataFrame to a CSV file
    if stage < 2:  
        df_train, df_val = make_dataframe(stages, step_indices, train_accs, val_accs, accs_125, train_losses_all, val_losses, train_max_rates, train_mean_rates, log_J_2x2_m, log_J_2x2_s, cE_m, cI_m, cE_s, cI_s, log_f_E, log_f_I, b_sigs, w_sigs, None, psychometric_offsets, acc_means=acc_means, acc_stds=acc_stds, test_offset_vec=test_offset_vec)
    else:
        df_train, df_val = make_dataframe(stages, step_indices, train_accs, val_accs, accs_125, train_losses_all, val_losses, train_max_rates, train_mean_rates, log_J_2x2_m, log_J_2x2_s, cE_m, cI_m, cE_s, cI_s, log_f_E, log_f_I, staircase_offsets=staircase_offsets, psychometric_offsets=psychometric_offsets, kappas_Jsup=kappas_Jsup, kappas_Jmid=kappas_Jmid, kappas_f=kappas_f, acc_means=acc_means, acc_stds=acc_stds, test_offset_vec=test_offset_vec)
    # insert run index as the first column 
    df_train.insert(0, 'run_index', run_index) 
    df_val.insert(0, 'run_index', run_index)
    if results_filename:
        file_exists = os.path.isfile(results_filename)
        df_train.to_csv(results_filename, mode='a', header=not file_exists, index=False)
        df_val.to_csv(results_filename[:-4] + '_val' + results_filename[-4:], mode='a', header=not file_exists, index=False)

    # Clear jax cash data
    jax.clear_caches()

    return df_train


def loss_and_grad_ori_discr(stage, trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, training_loss_val_and_grad, shuffle_labels=False):
    """
    Top level function to calculate losses, accuracies and other relevant metrics. It generates noises and training data and then applies the function training_loss_val_and_grad.

    Args:
        trained_pars_dict (dict): Dictionary containing parameters for the SSN layer.
        readout_pars_dict (dict): Dictionary containing parameters for the readout layer.
        untrained_pars (object): Object containing various parameters that are untrained.
        jit_on (bool): Flag indicating whether just-in-time (JIT) compilation is enabled.
        training_loss_val_and_grad (function): Function for calculating losses and gradients.

    Returns:
        tuple: Tuple containing loss, all_losses, accuracy, sig_input, sig_output, max_rates, and gradients.
    """

    # Create stimulus for middle layer: train_data is a dictionary with keys 'ref', 'target' and 'label'
    if stage == 0:
        pretrain_pars=untrained_pars.pretrain_pars
        # Generate training data and noise that is added to the output of the model
        noise_ref = generate_noise(pretrain_pars.batch_size, readout_pars_dict["w_sig"].shape[0], num_readout_noise = untrained_pars.num_readout_noise)
        noise_target = generate_noise(pretrain_pars.batch_size, readout_pars_dict["w_sig"].shape[0], num_readout_noise = untrained_pars.num_readout_noise)
        train_data = create_grating_pretraining(untrained_pars.pretrain_pars, pretrain_pars.batch_size, untrained_pars.BW_image_jax_inp, numRnd_ori1=pretrain_pars.batch_size)
    else:
        training_pars=untrained_pars.training_pars
        if untrained_pars.training_pars.pretraining_task:
            noise_ref = generate_noise(training_pars.batch_size, readout_pars_dict["w_sig"].shape[0], num_readout_noise = untrained_pars.num_readout_noise)
            noise_target = generate_noise(training_pars.batch_size, readout_pars_dict["w_sig"].shape[0], num_readout_noise = untrained_pars.num_readout_noise)
            train_data = create_grating_pretraining(untrained_pars.pretrain_pars, training_pars.batch_size, untrained_pars.BW_image_jax_inp, numRnd_ori1=training_pars.batch_size)
        else:
            # Generate noise that is added to the output of the model
            noise_ref = generate_noise(training_pars.batch_size, readout_pars_dict["w_sig"].shape[0], num_readout_noise = untrained_pars.num_readout_noise)
            noise_target = generate_noise(training_pars.batch_size, readout_pars_dict["w_sig"].shape[0], num_readout_noise = untrained_pars.num_readout_noise)
            train_data = create_grating_training(untrained_pars.stimuli_pars, training_pars.batch_size, untrained_pars.BW_image_jax_inp, shuffle_labels= shuffle_labels)

    # Calculate gradient, loss and accuracy
    [loss, [all_losses, accuracy, sig_input, sig_output, max_rates, mean_rates]], grad = training_loss_val_and_grad(
        trained_pars_dict,
        readout_pars_dict,
        untrained_pars,
        train_data,
        noise_ref,
        noise_target,
        jit_on
    )
    return loss, all_losses, accuracy, sig_input, sig_output, max_rates, mean_rates, grad


def loss_ori_discr(trained_pars_dict, readout_pars_dict, untrained_pars, train_data, noise_ref, noise_target): 
    """
    Bottom level function to calculate losses, accuracies and other relevant metrics. It applies evaluate_model_response to training data, adds noise to the model output and calculates the relevant metrics.

    Args:
        trained_pars_dict (dict): Dictionary containing parameters for the SSN layer.
        readout_pars_dict (dict): Dictionary containing parameters for the readout layer.
        untrained_pars (object): Object containing various parameters that are untrained.
        train_data: Dictionary containing reference and target training data and corresponding labels
        noise_ref: additive noise for the model output reference training data
        noise_target: additive noise for the model output from target training data

    Returns:
        tuple: Tuple containing loss, all_losses, accuracy, sig_input, sig_output, max_rates
    Note that this is the only function where f_E and f_I are exponentiated.
    """
    
    pretraining = untrained_pars.pretrain_pars.is_on
    loss_pars = untrained_pars.loss_pars
    conv_pars = untrained_pars.conv_pars
    
    # Create middle and superficial SSN layers
    if pretraining:
        J_2x2_m, J_2x2_s, cE_m, cI_m, cE_s, cI_s, f_E, f_I, kappa_Jsup, kappa_Jmid, kappa_f = unpack_ssn_parameters(trained_pars_dict, untrained_pars.ssn_pars, return_kappas=False)
        ssn_sup=SSN_sup(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_s, untrained_pars.dist_from_single_ori, untrained_pars.ori_dist)
    else:
        J_2x2_m, J_2x2_s, cE_m, cI_m, cE_s, cI_s, f_E, f_I, kappa_Jsup, kappa_Jmid, kappa_f = unpack_ssn_parameters(trained_pars_dict, untrained_pars.ssn_pars)
        ssn_sup=SSN_sup(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_s, untrained_pars.dist_from_single_ori, untrained_pars.ori_dist, kappa_Jsup)
    if 'kappa_Jmid' in trained_pars_dict.keys():
        ssn_mid=SSN_mid(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_m, untrained_pars.dist_from_single_ori, kappa_Jmid)
    else:
        ssn_mid=SSN_mid(untrained_pars.ssn_pars, untrained_pars.grid_pars, J_2x2_m, untrained_pars.dist_from_single_ori)
    
    # Run reference and target through the model
    [r_sup_ref, r_mid_ref], _, [avg_dx_ref_mid, avg_dx_ref_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], [mean_E_mid, mean_I_mid, mean_E_sup, mean_I_sup] = evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'], conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters, untrained_pars.dist_from_single_ori, kappa_f, untrained_pars.ssn_pars.kappa_range)
    [r_sup_target,r_mid_target],_, [avg_dx_target_mid, avg_dx_target_sup], _, _= evaluate_model_response(ssn_mid, ssn_sup, train_data['target'], conv_pars, cE_m, cI_m, cE_s, cI_s, f_E, f_I, untrained_pars.gabor_filters, untrained_pars.dist_from_single_ori, kappa_f, untrained_pars.ssn_pars.kappa_range)
    
    # Select the middle grid and sum the contribution from the middle and the superficial layer
    if pretraining:
        middle_grid_ind = jnp.arange(untrained_pars.grid_pars.gridsize_Nx ** 2)
    else:
        middle_grid_ind = untrained_pars.middle_grid_ind
    
    sup_mid_contrib = untrained_pars.sup_mid_readout_contrib
    if sup_mid_contrib[0] == 0 or sup_mid_contrib[1] == 0:
        r_ref_box = sup_mid_contrib[0] * r_sup_ref[middle_grid_ind] + sup_mid_contrib[1] * r_mid_ref[middle_grid_ind]
        r_target_box = sup_mid_contrib[0] * r_sup_target[middle_grid_ind] + sup_mid_contrib[1] * r_mid_target[middle_grid_ind]
    else: # concatenate the two layers
        r_ref_box = jnp.concatenate((sup_mid_contrib[1] * r_mid_ref[middle_grid_ind], sup_mid_contrib[0] * r_sup_ref[middle_grid_ind]))
        r_target_box = jnp.concatenate((sup_mid_contrib[1] * r_mid_target[middle_grid_ind], sup_mid_contrib[0] * r_sup_target[middle_grid_ind]))
    
    # Add noise
    r_ref_box = r_ref_box + noise_ref*jnp.sqrt(jax.nn.softplus(r_ref_box))
    r_target_box = r_target_box + noise_target*jnp.sqrt(jax.nn.softplus(r_target_box))
    
    # Define output from sigmoid layer and calculate losses
    w_sig = readout_pars_dict['w_sig']
    b_sig = readout_pars_dict['b_sig']
    # i) Multiply (reference - target) by sigmoid layer weights, add bias and apply sigmoid funciton
    sig_input = jnp.dot(w_sig, (r_ref_box - r_target_box)) + b_sig     
    sig_output = sigmoid(sig_input)
    # ii) Calculate readout loss and the predicted label
    loss_readout = binary_crossentropy_loss(train_data['label'], sig_output)
    pred_label = jnp.round(sig_output)
    # ii) Calculate other loss terms
    # Loss for max mean rates deviation from baseline
    Rmax_E = loss_pars.Rmax_E
    Rmax_I = loss_pars.Rmax_I
    Rmean_E = loss_pars.Rmean_E
    Rmean_I = loss_pars.Rmean_I
    max_E_mid, max_I_mid, max_E_sup, max_I_sup
    loss_rmax_mid= jnp.mean(jnp.maximum(0, (max_E_mid/Rmax_E - 1)) + jnp.maximum(0, (max_I_mid/Rmax_I - 1)))
    loss_rmax_sup = jnp.mean(jnp.maximum(0, (max_E_sup/Rmax_E - 1)) + jnp.maximum(0, (max_I_sup/Rmax_I - 1)))
    loss_r_max = loss_pars.lambda_r_max*(loss_rmax_mid+loss_rmax_sup) # optionally, we could use leaky relu here
    loss_rmean_mid = jnp.mean((mean_E_mid/Rmean_E[0] - 1) ** 2 + (mean_I_mid/Rmean_I[0] - 1) ** 2)
    loss_rmean_sup = jnp.mean((mean_E_sup/Rmean_E[1] - 1) ** 2 + (mean_I_sup/Rmean_I[1] - 1) ** 2)
    loss_r_mean = loss_pars.lambda_r_mean*(loss_rmean_mid + loss_rmean_sup)
    loss_dx_max = loss_pars.lambda_dx*jnp.mean(jnp.array([avg_dx_ref_mid, avg_dx_target_mid, avg_dx_ref_sup, avg_dx_target_sup])**2)
    loss_w = loss_pars.lambda_w*(jnp.linalg.norm(w_sig)**2)
    loss_b = loss_pars.lambda_b*(b_sig**2)   
    loss = loss_readout + loss_w + loss_b +  loss_dx_max + loss_r_max + loss_r_mean
    all_losses = jnp.vstack((loss_readout, loss_dx_max, loss_r_max, loss_r_mean, loss_w, loss_b, loss))
    
    return loss, all_losses, pred_label, sig_input, sig_output,  [max_E_mid, max_I_mid, max_E_sup, max_I_sup], [mean_E_mid, mean_I_mid, mean_E_sup, mean_I_sup]

# Parallelize orientation discrimination task and jit the parallelized function
vmap_ori_discrimination = vmap(loss_ori_discr, in_axes = (None, None, None, {'ref':0, 'target':0, 'label':0}, 0, 0) )
jit_ori_discrimination = jax.jit(vmap_ori_discrimination, static_argnums = [2])


def batch_loss_ori_discr(trained_pars_dict, readout_pars_dict, untrained_pars, train_data, noise_ref, noise_target, jit_on=True):
    """
    Middle level function to calculate losses, accuracies and other relevant metrics to a batch of training data and output noise. It uses either the vmap-ed or vmap-ed and jit-ed version of loss_ori_discr depending on jit_on.

    Args:
        trained_pars_dict (dict): Dictionary containing parameters for the SSN layer.
        readout_pars_dict (dict): Dictionary containing parameters for the readout layer.
        untrained_pars (object): Object containing various parameters that are untrained.
        train_data: Dictionary containing reference and target training data and corresponding labels
        noise_ref: additive noise for the model output reference training data
        noise_target: additive noise for the model output from target training data
        jit_on (bool): Flag indicating whether just-in-time (JIT) compilation is enabled.

    Returns:
        tuple: Tuple containing loss, all_losses, accuracy, sig_input, sig_output, max_rates
    """
    #Run orientation discrimination task
    if jit_on:
        total_loss, all_losses, pred_label, sig_input, sig_output, max_rates, mean_rates = jit_ori_discrimination(trained_pars_dict, readout_pars_dict, untrained_pars, train_data, noise_ref, noise_target)
    else:
        total_loss, all_losses, pred_label, sig_input, sig_output, max_rates, mean_rates = vmap_ori_discrimination(trained_pars_dict, readout_pars_dict, untrained_pars, train_data, noise_ref, noise_target)
    
    # Average total loss within a batch (across trials)
    loss= jnp.mean(total_loss)
    
    # Average individual losses within a batch (accross trials)
    all_losses = jnp.mean(all_losses, axis = 0)
    
    # Find maximum rates  within a batch (across trials)
    max_rates = [item.max() for item in max_rates]

    # Calculate mean rates within a batch (across trials)
    mean_rates = [item.mean() for item in mean_rates]
    
    # Calculate the proportion of labels that are predicted well (within a batch) 
    true_accuracy = jnp.sum(train_data['label'] == pred_label)/len(train_data['label'])

    return loss, [all_losses, true_accuracy, sig_input, sig_output, max_rates, mean_rates]


training_loss_val_and_grad_ssn_readout = jax.value_and_grad(batch_loss_ori_discr, argnums=[0, 1], has_aux=True)
training_loss_val_and_grad_readout_only = jax.value_and_grad(batch_loss_ori_discr, argnums=1, has_aux=True)
training_loss_val_and_grad_ssn_only = jax.value_and_grad(batch_loss_ori_discr, argnums=0, has_aux=True)

############### Other supporting functions ###############

def binary_crossentropy_loss(n, x):
    """Loss function calculating binary cross entropy. n is the true label and x is the predicted label."""
    return -(n * jnp.log(x) + (1 - n) * jnp.log(1 - x))


def generate_noise(batch_size, length, num_readout_noise=125, dt_readout = 0.2):
    """
    This function creates batch_size number of vectors of neural noise where each vector is of length = length. 
    """
    sig_noise = 1/jnp.sqrt(num_readout_noise * dt_readout)
    return sig_noise*numpy.random.randn(batch_size, length)


####### Functions for testing training task accuracy for different offsets and finding the offset value where it crosses baseline accuracy 
def task_acc_test(trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, test_offset, batch_size=300, pretrain_task= False):
    """
    This function tests the accuracy of the training orientation discrimination task given a set of parameters across different stimulus offsets.
    
    Parameters:
    - trained_pars_dict: Dictionary of parameters for the SSN layers.
    - readout_pars_dict: Dictionary of parameters for the readout layer.
    - untrained_pars: Untrained parameters used across the model.
    - jit_on: Flag to turn on/off JIT compilation.
    - test_offset: An offset to test the model performance on.
    - batch_size: Number of samples to test the model performance.
    - pretrain_task: Flag to indicate whether the model is tested on the pretraining task or the training task.
    
    Returns:
    - acc: mean true accuracy
    - loss: mean loss
    
    """
    # Create copies of offset, pretrain_pars.is_on and readout_pars_dict because their values may change in this function
    offset_saved = untrained_pars.stimuli_pars.offset
    pretrain_is_on_saved = untrained_pars.pretrain_pars.is_on
    
    if pretrain_is_on_saved and not pretrain_task:
        readout_pars_dict_copy = copy.deepcopy(readout_pars_dict)
        readout_pars_dict_copy['w_sig'] = readout_pars_dict_copy['w_sig'][untrained_pars.middle_grid_ind]
    else:
        readout_pars_dict_copy = copy.deepcopy(readout_pars_dict)
        
    # Generate noise that is added to the output of the model
    noise_ref = generate_noise(batch_size = batch_size, length = readout_pars_dict_copy["w_sig"].shape[0], num_readout_noise = untrained_pars.num_readout_noise)
    noise_target = generate_noise(batch_size = batch_size, length = readout_pars_dict_copy["w_sig"].shape[0], num_readout_noise = untrained_pars.num_readout_noise)
    
    # Create stimulus for middle layer: train_data is a dictionary with keys 'ref', 'target' and 'label'
    if pretrain_task:
        train_data = create_grating_pretraining(untrained_pars.pretrain_pars, batch_size, untrained_pars.BW_image_jax_inp, numRnd_ori1=batch_size)
    else:
        untrained_pars.stimuli_pars.offset = test_offset
        train_data = create_grating_training(untrained_pars.stimuli_pars, batch_size, untrained_pars.BW_image_jax_inp)
    
    # Calculate loss and accuracy
    if pretrain_task:
        loss, [_, acc, _, _, _, _] = batch_loss_ori_discr(trained_pars_dict, readout_pars_dict_copy, untrained_pars, train_data, noise_ref, noise_target, jit_on)
    else:
        untrained_pars.pretrain_pars.is_on=False # this is to get the accuracy metric that corresponds to the training task
        loss, [_, acc, _, _, _, _] = batch_loss_ori_discr(trained_pars_dict, readout_pars_dict_copy, untrained_pars, train_data, noise_ref, noise_target, jit_on)
        # Restore the original values of pretrain_is_on and offset
        untrained_pars.pretrain_pars.is_on = pretrain_is_on_saved
        untrained_pars.stimuli_pars.offset = offset_saved
        
    return acc, loss


def mean_training_task_acc_test(trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, offset_vec, sample_size = 1):
    """
    This function runs training_task_acc_test sample_size times to get accuracies and losses for the training task for a given parameter set. It averages the accuracies accross independent samples of accuracies.
    """
    # Initialize arrays to store loss, accuracy, and max rates
    N = len(offset_vec)
    accuracy = numpy.zeros((N,sample_size))
    loss = numpy.zeros((N,sample_size))
    
    # For the all the offsets, calculate fine discrimination accuracy sample_size times
    for i in range(N):
        for j in range(sample_size):
            temp_acc, temp_loss = task_acc_test(trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, offset_vec[i], batch_size=100, pretrain_task= False)
            accuracy[i,j] = temp_acc
            loss[i,j] = temp_loss
        
    # Calculate mean loss and accuracy
    accuracy_mean = jnp.mean(accuracy, axis=1)
    accuracy_std = jnp.std(accuracy, axis=1)

    return accuracy_mean, accuracy_std, accuracy, loss


def offset_at_baseline_acc(accuracies, offset_vec=[2, 4, 6, 9, 12, 15, 20], x_vals=numpy.linspace(1, 90, 300), baseline_acc=0.794):
    """
    This function fits a log-linear curve to x=offset_vec, y=accuracies data and returns the x value, where the curve crosses baseline_acc.
    """
        
    # Define x values for the log-linear curve fit
    offset_vec = numpy.array(offset_vec)
    offset_vec[offset_vec == 0] = jnp.finfo(float).eps
    log_offset_vec = jnp.log(offset_vec)
    # If accuracies is a matrix, then repeat log_offset_vec and flatten both
    if len(numpy.shape(accuracies)) > 1:
        log_offset_vec = numpy.repeat(log_offset_vec, numpy.shape(accuracies)[1])
        accuracies = accuracies.flatten()
        log_offset_vec = log_offset_vec.flatten()
    # Fit a log-linear learning curve
    a, b = jnp.polyfit(log_offset_vec, accuracies, 1)

    # Evaluate curve at x_vals
    x_vals[x_vals == 0] = jnp.finfo(float).eps
    log_x_vals = jnp.log(x_vals)
    y_vals = a * log_x_vals + b

    # Find where y_vals cross baseline_acc
    if y_vals[-1] < baseline_acc:
        offsets_at_bl_acc = jnp.array(180.0)
    else:
        # Calculate the midpoint of the interval where y_vals crosses baseline_acc
        sign_change_ind = jnp.where(jnp.diff(jnp.sign(y_vals - baseline_acc)))[0]
        offsets_at_bl_acc = (x_vals[sign_change_ind] + x_vals[sign_change_ind + 1]) / 2

    if offsets_at_bl_acc.size ==0:
        mask = accuracies < baseline_acc
        first_index = jnp.argmax(mask)
        offsets_at_bl_acc = offset_vec[first_index]
    elif offsets_at_bl_acc.size > 1:
        offsets_at_bl_acc = offsets_at_bl_acc[0]

    return offsets_at_bl_acc


####### Function for creating DataFrame from training results #######
def make_dataframe(stages, step_indices, train_accs, val_accs, accs_125, train_losses_all, val_losses, train_max_rates, train_mean_rates, log_J_2x2_m, log_J_2x2_s, cE_m, cI_m, cE_s, cI_s, log_f_E, log_f_I, b_sigs=None, w_sigs=None, staircase_offsets=None, psychometric_offsets=None, kappas_Jsup=None, kappas_Jmid=None, kappas_f=None, acc_means=None, acc_stds=None, test_offset_vec=None):
    """ This function collects different variables from training results into a dataframe."""
    from parameters import ReadoutPars
    readout_pars = ReadoutPars()
    # Create an empty DataFrame and initialize it with stages, SGD steps, and training accuracies
    df_train = pd.DataFrame({
        'stage': stages,
        'SGD_steps': step_indices['SGD_steps'],
        'acc': train_accs
    })
    
    df_train['val_acc'] = None
    df_train['val_loss'] = None
    df_train['accs_125'] = None
    df_train.loc[step_indices['val_SGD_steps'],'val_acc'] = val_accs
    df_train.loc[step_indices['val_SGD_steps'],'val_loss'] = val_losses
    df_train.loc[step_indices['val_SGD_steps'],'accs_125'] = accs_125
    
    train_max_rates = jnp.vstack(jnp.asarray(train_max_rates))
    train_mean_rates = jnp.vstack(jnp.asarray(train_mean_rates))
    log_J_2x2_m = jnp.stack(log_J_2x2_m)
    log_J_2x2_s = jnp.stack(log_J_2x2_s)
    train_losses_all = jnp.stack(train_losses_all)

    # Add different types of training and validation losses to df
    loss_names = ['loss_binary_cross_entr', 'loss_dx_max', 'loss_r_max', 'loss_r_mean', 'loss_w_sig', 'loss_b_sig', 'loss_all']
    for i in range(len(train_losses_all[0])):
        df_train[loss_names[i]]=train_losses_all[:,i]
    
    # Add max rates data to df
    max_rates_names = ['maxr_E_mid', 'maxr_I_mid', 'maxr_E_sup', 'maxr_I_sup']
    for i in range(len(train_max_rates[0])):
        df_train[max_rates_names[i]]=train_max_rates[:,i]

    # Add mean rates data to df
    mean_rates_names = ['meanr_E_mid', 'meanr_I_mid', 'meanr_E_sup', 'meanr_I_sup']
    for i in range(len(train_mean_rates[0])):
        df_train[mean_rates_names[i]]=train_mean_rates[:,i]
    
    # Add parameters that are trained in two stages during training and in one stage during pretraining
    log_J_m_names = ['log_J_EE_m', 'log_J_EI_m', 'log_J_IE_m', 'log_J_II_m']
    log_J_s_names = ['log_J_EE_s', 'log_J_EI_s', 'log_J_IE_s', 'log_J_II_s']
    J_m_names = ['J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m']
    J_s_names = ['J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s']
    J_2x2_m=jnp.transpose(jnp.array([jnp.exp(log_J_2x2_m[:,0]),-jnp.exp(log_J_2x2_m[:,1]),jnp.exp(log_J_2x2_m[:,2]),-jnp.exp(log_J_2x2_m[:,3])]))
    J_2x2_s=jnp.transpose(jnp.array([jnp.exp(log_J_2x2_s[:,0]),-jnp.exp(log_J_2x2_s[:,1]),jnp.exp(log_J_2x2_s[:,2]),-jnp.exp(log_J_2x2_s[:,3])]))
    for i in range(len(log_J_2x2_m[0])):
        df_train[log_J_m_names[i]] = log_J_2x2_m[:,i]
    for i in range(len(log_J_2x2_s[0])):
        df_train[log_J_s_names[i]] = log_J_2x2_s[:,i]
    for i in range(len(log_J_2x2_m[0])):
        df_train[J_m_names[i]] = J_2x2_m[:,i]
    for i in range(len(log_J_2x2_s[0])):
        df_train[J_s_names[i]] = J_2x2_s[:,i]
    df_train['cE_m']=cE_m
    df_train['cI_m']=cI_m
    df_train['cE_s']=cE_s
    df_train['cI_s']=cI_s
    df_train['log_f_E']=log_f_E
    df_train['log_f_I']=log_f_I
    df_train['f_E']=[jnp.exp(log_f_E[i]) for i in range(len(log_f_E))]
    df_train['f_I']=[jnp.exp(log_f_I[i]) for i in range(len(log_f_I))]

    # Distinguish psychometric and staircase offsets
    df_train['psychometric_offset']=None
    max_stages = max(1,max(stages))
    if max_stages==1:
        psychometric_offsets=jnp.hstack(psychometric_offsets)
        df_train.loc[step_indices['acc_check_ind'],'psychometric_offset']=psychometric_offsets
    else:   
        df_train.loc[step_indices['val_SGD_steps'],'psychometric_offset']=psychometric_offsets
        step_indices['acc_check_ind'] = step_indices['val_SGD_steps']
        df_train['staircase_offset']= staircase_offsets
    # save w_sigs when pretraining is on
    if max_stages==1:
        w_sigs = jnp.stack(w_sigs)
        # Create a new DataFrame from the weight_data dictionary and concatenate it with the existing DataFrame
        weight_data = {}
        if w_sigs.shape[1] == readout_pars.readout_grid_size[0] ** 2:
            middle_grid_inds = range(readout_pars.readout_grid_size[0]**2)
        else:
            middle_grid_inds = readout_pars.middle_grid_ind
        w_sig_keys = [f'w_sig_{middle_grid_inds[i]}' for i in range(len(middle_grid_inds))] 
        for i in range(w_sigs.shape[1]):      
            weight_data[w_sig_keys[i]] = w_sigs[:,i]
        weight_df = pd.DataFrame(weight_data)
        df_train = pd.concat([df_train, weight_df], axis=1)

        # Add b_sig to the DataFrame
        df_train['b_sig'] = b_sigs

    # Add kappa_pre and kappa_post to the DataFrame
    if max_stages==2 and kappas_Jsup is not None:
        kappas_Jsup_np=jnp.asarray(kappas_Jsup)
        kappa_Jsup_names = ['kappa_Jsup_EE_pre', 'kappa_Jsup_EI_pre', 'kappa_Jsup_IE_pre', 'kappa_Jsup_II_pre', 'kappa_Jsup_EE_post', 'kappa_Jsup_EI_post', 'kappa_Jsup_IE_post', 'kappa_Jsup_II_post']
        for i in range(len(kappas_Jsup_np[0])):
            df_train[kappa_Jsup_names[i]] = kappas_Jsup_np[:,i]
    
    if max_stages==2 and kappas_Jmid is not None:
        kappas_Jmid_np=jnp.asarray(kappas_Jmid)
        kappa_Jmid_names = ['kappa_Jmid_EE_pre', 'kappa_Jmid_IE_pre', 'kappa_Jmid_EE_post', 'kappa_Jmid_IE_post']
        for i in range(len(kappas_Jmid_np[0])):
            df_train[kappa_Jmid_names[i]] = kappas_Jmid_np[:,i]
    
    if max_stages==2 and kappas_f is not None:
        kappas_f_np=jnp.asarray(kappas_f)
        kappa_f_names = ['kappa_f_E', 'kappa_f_I']
        for i in range(len(kappas_f_np[0])):
            df_train[kappa_f_names[i]] = kappas_f_np[:,i]

    if acc_means is not None:
        df_val = df_train.copy()
        # create offset_{i} keys for each offset in test_offset_vec
        for i in range(len(test_offset_vec)):
            acc_mean_new_col = f'acc_mean_offset_{test_offset_vec[i]}'
            acc_std_new_col = f'acc_std_offset_{test_offset_vec[i]}'
            df_val[acc_mean_new_col] = None
            df_val[acc_std_new_col] = None
            df_val.loc[step_indices['acc_check_ind'],acc_mean_new_col] = jnp.asarray(acc_means)[:,i]
            df_val.loc[step_indices['acc_check_ind'],acc_std_new_col] = jnp.asarray(acc_stds)[:,i]
        df_val['psychometric_offset']=None
        df_val.loc[step_indices['acc_check_ind'], 'psychometric_offset'] = psychometric_offsets  
        df_val = df_val.loc[step_indices['acc_check_ind']].copy()
    else:
        df_val = df_train.loc[step_indices['acc_check_ind']].copy()

    return df_train, df_val
