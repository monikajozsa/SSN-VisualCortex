#import matplotlib.pyplot as plt
import jax
import jax.numpy as np
from jax import vmap
import optax
import time
import pandas as pd
import numpy
import os
import copy

from util import create_grating_training, sep_exponentiate, sigmoid, create_grating_pretraining
from SSN_classes import SSN_mid, SSN_sup
from model import evaluate_model_response


def train_ori_discr(
    readout_pars_dict,
    ssn_layer_pars_dict,
    untrained_pars,
    threshold = 0.75,
    offset_step = 0.25,
    results_filename=None,
    jit_on=True
):
    """
    Trains a two-layer SSN network model in two distinct stages.
    
    Stage 1: Trains the readout layer parameters until either the accuracy 
    threshold (first_stage_acc_th) is met or a specified number of SGD_steps (training_pars.SGD_steps) is reached.
    
    Stage 2: Trains the SSN layer parameters for a fixed number of SGD_steps (training_pars.SGD_steps).
    
    Parameters:
    - readout_pars_dict (dict): Parameters for the readout layer.
    - ssn_layer_pars_dict (dict): Parameters for the SSN layer.
    - untrained_pars (class): Includes grid_pars, stimuli_pars, conn_pars_m, 
                                  conn_pars_s, filter_pars, ssn_ori_map, ssn_pars, 
                                  ssn_layer_pars, conv_pars, loss_pars, training_pars.
    - results_filename (str, optional): Filename for saving results.
    - jit_on (bool): If True, enables JIT compilation for performance improvement.
    """
     # Unpack training_pars and stimuli_pars from untrained_pars
    training_pars = untrained_pars.training_pars
    stimuli_pars = untrained_pars.stimuli_pars
    pretrain_on = untrained_pars.pretrain_pars.is_on
    pretrain_offset_threshold = untrained_pars.pretrain_pars.offset_threshold

    # Define indices of sigmoid layer weights to save
    if pretrain_on:
        w_indices_to_save = untrained_pars.middle_grid_ind
    else:
        w_indices_to_save = numpy.array([i for i in range(untrained_pars.readout_grid_size[1] ** 2)])

    # Initialise optimizer and set first stage accuracy threshold
    optimizer = optax.adam(training_pars.eta)
    if pretrain_on:
        opt_state_ssn = optimizer.init(ssn_layer_pars_dict)
        opt_state_readout = optimizer.init(readout_pars_dict)
        training_loss_val_and_grad = jax.value_and_grad(batch_loss_ori_discr, argnums=[0,1], has_aux=True)
    else:
        opt_state_readout = optimizer.init(readout_pars_dict)
        training_loss_val_and_grad = jax.value_and_grad(batch_loss_ori_discr, argnums=1, has_aux=True)

    # Define SGD_steps indices and offsets where training task accuracy is calculated
    if pretrain_on:
        numSGD_steps = untrained_pars.pretrain_pars.SGD_steps
        min_acc_check_ind = untrained_pars.pretrain_pars.min_acc_check_ind
        acc_check_ind = np.arange(0, numSGD_steps, untrained_pars.pretrain_pars.acc_check_freq)
        acc_check_ind = acc_check_ind[(acc_check_ind > min_acc_check_ind) | (acc_check_ind <2)] # by leaving in 0, we make sure that it is not empty as we refer to it later
        test_offset_vec = numpy.array([2, 5, 10, 18])  # numpy.array([2, 4, 6, 9, 12, 15, 20]) # offsets that help us define accuracy for training task during pretraining
        numStages = 1
    else:
        numSGD_steps = training_pars.SGD_steps
        first_stage_acc_th = training_pars.first_stage_acc_th
        numStages = 2
    
    # Define SGD_steps indices where losses an accuracy are validated
    val_steps = np.arange(0, numSGD_steps, training_pars.validation_freq)
    first_stage_final_step = numSGD_steps - 1

    print(
        "SGD_step: {} ¦ learning rate: {} ¦ batch size {}".format(
            numSGD_steps, training_pars.eta, training_pars.batch_size,
        )
    )

    if results_filename:
        print("Saving results to ", results_filename)
    else:
        print("#### NOT SAVING! ####")

    start_time = time.time()
    
    ######## Pretraining: One-stage, Training: Two-stage, where 1) parameters of sigmoid layer 2) parameters of SSN layers #############
    pretrain_stop_flag = False
    if not pretrain_stop_flag:
        for stage in range(1,numStages+1):
            if stage == 2:
                # Reinitialise optimizer and reset the argnum to take gradient of
                opt_state_ssn = optimizer.init(ssn_layer_pars_dict)
                training_loss_val_and_grad = jax.value_and_grad(batch_loss_ori_discr, argnums=0, has_aux=True)

            # STOCHASTIC GRADIENT DESCENT LOOP
            for SGD_step in range(numSGD_steps):
                # i) Calculate model loss, accuracy, gradient
                train_loss, train_loss_all, train_acc, _, _, train_max_rate, grad = loss_and_grad_ori_discr(ssn_layer_pars_dict, readout_pars_dict, untrained_pars, jit_on, training_loss_val_and_grad)
                if jax.numpy.isnan(train_loss):
                    return None, None
                
                if not pretrain_on and stage==1 and SGD_step == 0:
                    if train_acc > first_stage_acc_th:
                        print("Early stop: accuracy {} reached target {} for stage 1 training".format(
                                train_acc, first_stage_acc_th)
                        )
                        # Store final step index and exit first training loop (-1 because we did do not save this step)
                        first_stage_final_step = SGD_step -1
                        break

                # ii) Store parameters and metrics 
                if 'stages' in locals():
                    train_losses_all.append(train_loss_all.ravel())
                    train_accs.append(train_acc)
                    train_max_rates.append(train_max_rate)
                    if pretrain_on:
                        stages.append(stage-1)
                    else:
                        stages.append(stage)
                else:
                    train_losses_all=[train_loss_all.ravel()]
                    train_accs=[train_acc]
                    train_max_rates=[train_max_rate]
                    if pretrain_on:
                        stages=[stage-1]
                    else:
                        stages=[stage]
                if 'log_J_2x2_m' in locals():
                        log_J_2x2_m.append(ssn_layer_pars_dict['log_J_2x2_m'].ravel())
                        log_J_2x2_s.append(ssn_layer_pars_dict['log_J_2x2_s'].ravel())
                        c_E.append(ssn_layer_pars_dict['c_E'])
                        c_I.append(ssn_layer_pars_dict['c_I'])
                        log_f_E.append(ssn_layer_pars_dict['log_f_E'])
                        log_f_I.append(ssn_layer_pars_dict['log_f_I'])
                else:
                    log_J_2x2_m = [ssn_layer_pars_dict['log_J_2x2_m'].ravel()]
                    log_J_2x2_s = [ssn_layer_pars_dict['log_J_2x2_s'].ravel()]
                    c_E = [ssn_layer_pars_dict['c_E']]
                    c_I = [ssn_layer_pars_dict['c_I']]
                    log_f_E = [ssn_layer_pars_dict['log_f_E']]
                    log_f_I = [ssn_layer_pars_dict['log_f_I']]
                w_sig_temp=readout_pars_dict['w_sig']
                if 'w_sigs' in locals():
                    w_sigs.append(w_sig_temp[w_indices_to_save])
                    b_sigs.append(readout_pars_dict['b_sig'])
                else:
                    w_sigs = [w_sig_temp[w_indices_to_save]]
                    b_sigs = [readout_pars_dict['b_sig']]
                if 'offsets' in locals():
                    offsets.append(stimuli_pars.offset)
                else:
                    offsets=[stimuli_pars.offset]

                # ii) Early stopping during pre-training and training
                # Check for early stopping during pre-training
                if pretrain_on and SGD_step in acc_check_ind:
                    acc_mean, _, _ = mean_training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, untrained_pars, jit_on, test_offset_vec)
                    # early stopping happens even if the training task is solved for the flipped problem (w_sig flips it back in stage 1 in a few SGD steps)
                    if np.sum(acc_mean<0.5)>0.5*len(acc_mean):
                        acc_mean = 1-acc_mean
                    # fit log-linear curve to acc_mean_max and test_offset_vec and find where it crosses baseline_acc=0.794
                    offset_at_bl_acc = offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= pretrain_offset_threshold)
                    if SGD_step==acc_check_ind[0] and stage==1:
                        offsets_at_bl_acc=[float(offset_at_bl_acc)]
                    else:
                        offsets_at_bl_acc.append(float(offset_at_bl_acc))
                    print('Baseline acc is achieved at offset:', offset_at_bl_acc, ' for step ', SGD_step)

                    # Stop pretraining: break out from SGD_step loop and stages loop (using a flag)
                    if len(offsets_at_bl_acc)>2: # we stop pretraining even if the training task is solved for the pretraining
                        pretrain_stop_flag = all(np.array(offsets_at_bl_acc[-2:]) < pretrain_offset_threshold)
                        offsets_at_bl_acc[-1]=np.mean(offsets_at_bl_acc[-2:])
                    if pretrain_stop_flag:
                        print('Desired accuracy achieved during pretraining.')
                        first_stage_final_step = SGD_step
                        break
                    
                # Check for early stopping during training
                if not pretrain_on and stage==1:
                    if SGD_step == 0:
                        avg_acc = train_acc
                    else:
                        avg_acc = np.mean(np.asarray(train_accs[-min(SGD_step,10):]))
                    if avg_acc > first_stage_acc_th:
                        print("Early stop: accuracy {} reached target {} for stage 1 training".format(
                                avg_acc, first_stage_acc_th)
                        )
                        # Store final step index and exit first training loop
                        first_stage_final_step = SGD_step
                        break

                # iii) Staircase algorithm during training: 3-down 1-up adjustment rule for the offset
                if stage==2 and not pretrain_on:
                    if train_acc < threshold:
                        temp_threshold=0
                        stimuli_pars.offset =  stimuli_pars.offset + offset_step
                    else:
                        temp_threshold=1
                        if SGD_step > 2 and np.sum(np.asarray(threshold_variables[-3:])) == 3:
                            stimuli_pars.offset =  stimuli_pars.offset - offset_step
                    if 'threshold_variables' in locals():
                        threshold_variables.append(temp_threshold)
                    else:
                        threshold_variables=[temp_threshold]

                # iv) Loss and accuracy validation + printing results    
                if SGD_step in val_steps:
                    #### Calculate loss and accuracy - *** could be switched to mean loss and acc easily
                    val_acc_vec, val_loss_vec = training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, untrained_pars, jit_on, stimuli_pars.offset)
                
                    val_loss = np.mean(val_loss_vec)
                    val_acc = np.mean(val_acc_vec)
                    if 'val_accs' in locals():
                        val_accs.append(val_acc)
                        val_losses.append(val_loss)
                    else:
                        val_accs=[val_acc]
                        val_losses=[val_loss]
                        
                    SGD_step_time = time.time() - start_time
                    print("Stage: {}¦ Readout loss: {:.3f}  ¦ Tot training loss: {:.3f} ¦ Val loss: {:.3f} ¦ Train accuracy: {:.3f} ¦ Val accuracy: {:.3f} ¦ SGD step: {} ¦ Runtime: {:.4f} ".format(
                        stage, train_loss_all[0].item(), train_loss, val_loss, train_acc, val_acc, SGD_step, SGD_step_time
                    ))
                
                # v) Parameter update. Note that pre-training is one-stage, training is two-stage
                if numStages==1:
                    # Update readout parameters
                    updates_ssn, opt_state_ssn = optimizer.update(grad[0], opt_state_ssn)
                    ssn_layer_pars_dict = optax.apply_updates(ssn_layer_pars_dict, updates_ssn)
                    # Update ssn layer parameters       
                    updates_readout, opt_state_readout = optimizer.update(grad[1], opt_state_readout)
                    readout_pars_dict = optax.apply_updates(readout_pars_dict, updates_readout)
                else:
                    if stage == 1:
                        # Update readout parameters
                        updates_readout, opt_state_readout = optimizer.update(grad, opt_state_readout)
                        readout_pars_dict = optax.apply_updates(readout_pars_dict, updates_readout)
                    else:                    
                        # Update ssn layer parameters
                        updates_ssn, opt_state_ssn = optimizer.update(grad, opt_state_ssn)
                        ssn_layer_pars_dict = optax.apply_updates(ssn_layer_pars_dict, updates_ssn)
                        
    ############# SAVING and RETURN OUTPUT #############
    
    # Define SGD_steps indices for training and validation
    if pretrain_on:
        SGD_steps = np.arange(0, len(stages))
        val_SGD_steps = val_steps[0:len(val_accs)]
        offsets = None
        acc_check_ind = acc_check_ind[0:len(offsets_at_bl_acc)]
        step_indices = dict(SGD_steps=SGD_steps, val_SGD_steps=val_SGD_steps, acc_check_ind=acc_check_ind)
    else:
        SGD_steps = np.arange(0, len(stages))
        val_SGD_steps_stage1 = val_steps[val_steps < first_stage_final_step + 1]
        val_SGD_steps_stage2 = np.arange(first_stage_final_step + 1, first_stage_final_step + 1 + numSGD_steps, training_pars.validation_freq)
        val_SGD_steps = np.hstack([val_SGD_steps_stage1, val_SGD_steps_stage2])
        val_SGD_steps = val_SGD_steps[0:len(val_accs)]
        offsets_at_bl_acc = None
        step_indices = dict(SGD_steps=SGD_steps, val_SGD_steps=val_SGD_steps)
        
    # Create DataFrame and save the DataFrame to a CSV file
        
    df = make_dataframe(stages, step_indices, train_accs, val_accs, train_losses_all, val_losses, train_max_rates, b_sigs, w_sigs, log_J_2x2_m, log_J_2x2_s, c_E, c_I, log_f_E, log_f_I, offsets, offsets_at_bl_acc)

    if results_filename:
        file_exists = os.path.isfile(results_filename)
        df.to_csv(results_filename, mode='a', header=not file_exists, index=False)

    return df, first_stage_final_step


def loss_and_grad_ori_discr(ssn_layer_pars_dict, readout_pars_dict, untrained_pars, jit_on, training_loss_val_and_grad):
    """
    Top level function to calculate losses, accuracies and other relevant metrics. It generates noises and training data and then applies the function training_loss_val_and_grad.

    Args:
        ssn_layer_pars_dict (dict): Dictionary containing parameters for the SSN layer.
        readout_pars_dict (dict): Dictionary containing parameters for the readout layer.
        untrained_pars (object): Object containing various parameters that are untrained.
        jit_on (bool): Flag indicating whether just-in-time (JIT) compilation is enabled.
        training_loss_val_and_grad (function): Function for calculating losses and gradients.

    Returns:
        tuple: Tuple containing loss, all_losses, accuracy, sig_input, sig_output, max_rates, and gradients.
    """

    # Create stimulus for middle layer: train_data is a dictionary with keys 'ref', 'target' and 'label'
    if untrained_pars.pretrain_pars.is_on:
        pretrain_pars=untrained_pars.pretrain_pars
        # Generate training data and noise that is added to the output of the model
        noise_ref = generate_noise(pretrain_pars.batch_size, readout_pars_dict["w_sig"].shape[0], N_readout = untrained_pars.N_readout_noise)
        noise_target = generate_noise(pretrain_pars.batch_size, readout_pars_dict["w_sig"].shape[0], N_readout = untrained_pars.N_readout_noise)
        train_data = create_grating_pretraining(untrained_pars.pretrain_pars, pretrain_pars.batch_size, untrained_pars.BW_image_jax_inp, numRnd_ori1=pretrain_pars.batch_size)
    else:
        training_pars=untrained_pars.training_pars
        # Generate noise that is added to the output of the model
        noise_ref = generate_noise(training_pars.batch_size, readout_pars_dict["w_sig"].shape[0], N_readout = untrained_pars.N_readout_noise)
        noise_target = generate_noise(training_pars.batch_size, readout_pars_dict["w_sig"].shape[0], N_readout = untrained_pars.N_readout_noise)
        train_data = create_grating_training(untrained_pars.stimuli_pars, training_pars.batch_size, untrained_pars.BW_image_jax_inp)

    # Calculate gradient, loss and accuracy
    [loss, [all_losses, accuracy, sig_input, sig_output, max_rates]], grad = training_loss_val_and_grad(
        ssn_layer_pars_dict,
        readout_pars_dict,
        untrained_pars,
        train_data,
        noise_ref,
        noise_target,
        jit_on
    )
    return loss, all_losses, accuracy, sig_input, sig_output, max_rates, grad


def loss_ori_discr(ssn_layer_pars_dict, readout_pars_dict, untrained_pars, train_data, noise_ref, noise_target): 
    """
    Bottom level function to calculate losses, accuracies and other relevant metrics. It applies evaluate_model_response to training data, adds noise to the model output and calculates the relevant metrics.

    Args:
        ssn_layer_pars_dict (dict): Dictionary containing parameters for the SSN layer.
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
    w_sig = readout_pars_dict['w_sig']
    b_sig = readout_pars_dict['b_sig']
    c_E = ssn_layer_pars_dict['c_E']
    c_I = ssn_layer_pars_dict['c_I']
    f_E = np.exp(ssn_layer_pars_dict['log_f_E'])
    f_I = np.exp(ssn_layer_pars_dict['log_f_I'])
    J_2x2_m = sep_exponentiate(ssn_layer_pars_dict['log_J_2x2_m'])
    J_2x2_s = sep_exponentiate(ssn_layer_pars_dict['log_J_2x2_s']) 

    kappa_pre = untrained_pars.ssn_layer_pars.kappa_pre
    kappa_post = untrained_pars.ssn_layer_pars.kappa_post    
    p_local_s = untrained_pars.ssn_layer_pars.p_local_s
    s_2x2 = untrained_pars.ssn_layer_pars.s_2x2_s
    sigma_oris = untrained_pars.ssn_layer_pars.sigma_oris
    ref_ori = untrained_pars.stimuli_pars.ref_ori
    loss_pars = untrained_pars.loss_pars
    conv_pars = untrained_pars.conv_pars

    # Create middle and superficial SSN layers *** this is something that would be great to call from outside the SGD loop and only refresh the params that change (and what rely on them such as W)
    ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_m)
    ssn_sup=SSN_sup(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_s, p_local=p_local_s, oris=untrained_pars.oris, s_2x2=s_2x2, sigma_oris = sigma_oris, ori_dist = untrained_pars.ori_dist, train_ori = ref_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)
    
    #Run reference and targetthrough two layer model
    r_ref, _, [r_max_ref_mid, r_max_ref_sup], [avg_dx_ref_mid, avg_dx_ref_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'], conv_pars, c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)
    r_target, _, [r_max_target_mid, r_max_target_sup], [avg_dx_target_mid, avg_dx_target_sup], _, _= evaluate_model_response(ssn_mid, ssn_sup, train_data['target'], conv_pars, c_E, c_I, f_E, f_I, untrained_pars.gabor_filters)

    if pretraining:
        # Readout is from all neurons in sup layer
        r_ref_box = r_ref
        r_target_box = r_target
    else:
        # Select the middle grid
        r_ref_box = r_ref[untrained_pars.middle_grid_ind]
        r_target_box = r_target[untrained_pars.middle_grid_ind]       
    
    # Add noise
    r_ref_box = r_ref_box + noise_ref*np.sqrt(jax.nn.softplus(r_ref_box))
    r_target_box = r_target_box + noise_target*np.sqrt(jax.nn.softplus(r_target_box))
    
    # Define losses
    # i) Multiply (reference - target) by sigmoid layer weights, add bias and apply sigmoid funciton
    sig_input = np.dot(w_sig, (r_ref_box - r_target_box)) + b_sig     
    sig_output = sigmoid(sig_input)
    # ii) Calculate readout loss and the predicted label
    loss_readout = binary_crossentropy_loss(train_data['label'], sig_output)
    pred_label = np.round(sig_output)
    # ii) Calculate other loss terms
    loss_dx_max = loss_pars.lambda_dx*np.mean(np.array([avg_dx_ref_mid, avg_dx_target_mid, avg_dx_ref_sup, avg_dx_target_sup])**2)
    loss_r_max =  loss_pars.lambda_r_max*np.mean(np.array([r_max_ref_mid, r_max_target_mid, r_max_ref_sup, r_max_target_sup]))
    loss_w = loss_pars.lambda_w*(np.linalg.norm(w_sig)**2)
    loss_b = loss_pars.lambda_b*(b_sig**2)   
    loss = loss_readout + loss_w + loss_b +  loss_dx_max + loss_r_max
    all_losses = np.vstack((loss_readout, loss_dx_max, loss_r_max, loss_w, loss_b, loss))
    
    return loss, all_losses, pred_label, sig_input, sig_output,  [max_E_mid, max_I_mid, max_E_sup, max_I_sup]

# Parallelize orientation discrimination task and jit the parallelized function
vmap_ori_discrimination = vmap(loss_ori_discr, in_axes = (None, None, None, {'ref':0, 'target':0, 'label':0}, 0, 0) )
jit_ori_discrimination = jax.jit(vmap_ori_discrimination, static_argnums = [2])


def batch_loss_ori_discr(ssn_layer_pars_dict, readout_pars_dict, untrained_pars, train_data, noise_ref, noise_target, jit_on=True):
    """
    Middle level function to calculate losses, accuracies and other relevant metrics to a batch of training data and output noise. It uses either the vmap-ed or vmap-ed and jit-ed version of loss_ori_discr depending on jit_on.

    Args:
        ssn_layer_pars_dict (dict): Dictionary containing parameters for the SSN layer.
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
        total_loss, all_losses, pred_label, sig_input, sig_output, max_rates = jit_ori_discrimination(ssn_layer_pars_dict, readout_pars_dict, untrained_pars, train_data, noise_ref, noise_target)
    else:
        total_loss, all_losses, pred_label, sig_input, sig_output, max_rates = vmap_ori_discrimination(ssn_layer_pars_dict, readout_pars_dict, untrained_pars, train_data, noise_ref, noise_target)
    
    # Average total loss within a batch (across trials)
    loss= np.mean(total_loss)
    
    # Average individual losses within a batch (accross trials)
    all_losses = np.mean(all_losses, axis = 0)
    
    #Find maximum rates  within a batch (across trials)
    max_rates = [item.max() for item in max_rates]
    
    # Calculate the proportion of labels that are predicted well (within a batch) 
    true_accuracy = np.sum(train_data['label'] == pred_label)/len(train_data['label'])

    return loss, [all_losses, true_accuracy, sig_input, sig_output, max_rates]


############### Other supporting functions ###############

def binary_crossentropy_loss(n, x):
    '''
    Loss function calculating binary cross entropy
    '''
    return -(n * np.log(x) + (1 - n) * np.log(1 - x))


def generate_noise(batch_size, length, N_readout=125, dt_readout = 0.2):
    '''
    Creates vectors of neural noise. Function creates batch_size number of vectors, each vector of length = length. 
    '''
    sig_noise = 1/np.sqrt(N_readout * dt_readout)
    return sig_noise*numpy.random.randn(batch_size, length)


####### Functions for testing training task accuracy for different offsets and finding the offset value where it crosses baseline accuracy 
def training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, untrained_pars, jit_on, test_offset):
    '''
    This function tests the accuracy of the training orientation discrimination task given a set of parameters across different stimulus offsets.
    
    Parameters:
    - ssn_layer_pars_dict: Dictionary of parameters for the SSN layers.
    - readout_pars_dict: Dictionary of parameters for the readout layer.
    - untrained_pars: Untrained parameters used across the model.
    - stage: The stage of training readout pars or ssn layer pars.
    - jit_on: Flag to turn on/off JIT compilation.
    - offset_vec: A list of offsets to test the model performance.
    
    Returns:
    - loss: mean and std of losses for each offset over sample_size samples.
    - true_accuracy: mean and std of true accuracies for each offset  over sample_size samples.
    '''
    # Create copies of stimuli and readout_pars_dict because their 
    pretrain_is_on_saved = untrained_pars.pretrain_pars.is_on
    untrained_pars.pretrain_pars.is_on=False # this is to get the accuracy metric that corresponds to the training task and not the regression task
      
    if pretrain_is_on_saved:
        readout_pars_dict_copy = copy.deepcopy(readout_pars_dict)
        readout_pars_dict_copy['w_sig'] = readout_pars_dict_copy['w_sig'][untrained_pars.middle_grid_ind]
    else:
        readout_pars_dict_copy = copy.deepcopy(readout_pars_dict)
    
    # Iterate over each offset by using vmap outside of the function
    offset_saved = untrained_pars.stimuli_pars.offset
    untrained_pars.stimuli_pars.offset = test_offset
    
    # Generate noise that is added to the output of the model
    batch_size=300
    noise_ref = generate_noise(batch_size = batch_size, length = readout_pars_dict_copy["w_sig"].shape[0], N_readout = untrained_pars.N_readout_noise)
    noise_target = generate_noise(batch_size = batch_size, length = readout_pars_dict_copy["w_sig"].shape[0], N_readout = untrained_pars.N_readout_noise)
    
    # Create stimulus for middle layer: train_data is a dictionary with keys 'ref', 'target' and 'label'
    untrained_pars.stimuli_pars.jitter_val=0
    untrained_pars.stimuli_pars.std=0
    train_data = create_grating_training(untrained_pars.stimuli_pars, batch_size, untrained_pars.BW_image_jax_inp)
    
    # Calculate loss and accuracy
    loss, [_, acc, _, _, _] = batch_loss_ori_discr(ssn_layer_pars_dict, readout_pars_dict_copy, untrained_pars, train_data, noise_ref, noise_target, jit_on)

    untrained_pars.pretrain_pars.is_on = pretrain_is_on_saved
    untrained_pars.stimuli_pars.offset = offset_saved
    
    return acc, loss

def mean_training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, untrained_pars, jit_on, offset_vec, sample_size = 1):
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
            temp_acc, temp_loss = training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, untrained_pars, jit_on, offset_vec[i])
            accuracy[i,j] = temp_acc
            loss[i,j] = temp_loss
        
    # Calculate mean loss and accuracy
    accuracy_mean = np.mean(accuracy, axis=1)

    return accuracy_mean, accuracy, loss


def offset_at_baseline_acc(acc_vec, offset_vec=[2, 4, 6, 9, 12, 15, 20], x_vals=numpy.linspace(1, 90, 300), baseline_acc=0.794):
    '''
    This function fits a log-linear curve to x=offset_vec, y=acc_vec data and returns the x value, where the curve crosses baseline_acc.
    '''
        
    # Fit a log-linear learning curve
    offset_vec = numpy.array(offset_vec)
    offset_vec[offset_vec == 0] = np.finfo(float).eps
    log_offset_vec = np.log(offset_vec)
    a, b = np.polyfit(log_offset_vec, acc_vec, 1)

    # Evaluate curve at x_vals
    x_vals[x_vals == 0] = np.finfo(float).eps
    log_x_vals = np.log(x_vals)
    y_vals = a * log_x_vals + b

    # Find where y_vals cross baseline_acc
    if y_vals[-1] < baseline_acc:
        offsets_at_bl_acc = np.array(180.0)
    else:
        # Calculate the midpoint of the interval where y_vals crosses baseline_acc
        sign_change_ind = np.where(np.diff(np.sign(y_vals - baseline_acc)))[0]
        offsets_at_bl_acc = (x_vals[sign_change_ind] + x_vals[sign_change_ind + 1]) / 2
    
    if offsets_at_bl_acc.size ==0:
        mask = acc_vec < baseline_acc
        first_index = np.argmax(mask)
        offsets_at_bl_acc = offset_vec[first_index]

    return offsets_at_bl_acc


####### Function for creating DataFrame
def make_dataframe(stages, step_indices, train_accs, val_accs, train_losses_all, val_losses, train_max_rates, b_sigs,w_sigs, log_J_2x2_m, log_J_2x2_s, c_E, c_I, log_f_E, log_f_I, offsets=None, offsets_at_bl_acc=None):
    ''' This function collects different variables from training results into a dataframe.'''
    # Create an empty DataFrame and initialize it with stages, SGD steps, and training accuracies
    df = pd.DataFrame({
        'stage': stages,
        'SGD_steps': step_indices['SGD_steps'],
        'acc': train_accs
    })

    train_max_rates = np.vstack(np.asarray(train_max_rates))
    w_sigs = np.stack(w_sigs)
    log_J_2x2_m = np.stack(log_J_2x2_m)
    log_J_2x2_s = np.stack(log_J_2x2_s)
    train_losses_all = np.stack(train_losses_all)

    # Add validation accuracies at specified SGD steps
    df['val_acc'] = None
    df.loc[step_indices['val_SGD_steps'], 'val_acc'] = val_accs

    # Add different types of training and validation losses to df
    loss_names = ['loss_binary_cross_entr', 'loss_dx_max', 'loss_r_max', 'loss_w_sig', 'loss_b_sig', 'loss_all']
    for i in range(len(train_losses_all[0])):
        df[loss_names[i]]=train_losses_all[:,i]
    
    df['val_loss']=None
    df.loc[step_indices['val_SGD_steps'], 'val_loss']=val_losses

    # Add max rates data to df
    max_rates_names = ['maxr_E_mid', 'maxr_I_mid', 'maxr_E_sup', 'maxr_I_sup']
    for i in range(len(train_max_rates[0])):
        df[max_rates_names[i]]=train_max_rates[:,i]
    
    # Add parameters that are trained in two stages during training and in one stage during pretraining
    max_stages = max(1,max(stages))
    log_J_m_names = ['log_J_m_EE', 'log_J_m_EI', 'log_J_m_IE', 'log_J_m_II']
    log_J_s_names = ['log_J_s_EE', 'log_J_s_EI', 'log_J_s_IE', 'log_J_s_II']
    J_m_names = ['J_m_EE', 'J_m_EI', 'J_m_IE', 'J_m_II']
    J_s_names = ['J_s_EE', 'J_s_EI', 'J_s_IE', 'J_s_II']
    J_2x2_m=np.transpose(np.array([np.exp(log_J_2x2_m[:,0]),-np.exp(log_J_2x2_m[:,1]),np.exp(log_J_2x2_m[:,2]),-np.exp(log_J_2x2_m[:,3])]))
    J_2x2_s=np.transpose(np.array([np.exp(log_J_2x2_s[:,0]),-np.exp(log_J_2x2_s[:,1]),np.exp(log_J_2x2_s[:,2]),-np.exp(log_J_2x2_s[:,3])]))
    for i in range(len(log_J_2x2_m[0])):
        df[log_J_m_names[i]] = log_J_2x2_m[:,i]
    for i in range(len(log_J_2x2_s[0])):
        df[log_J_s_names[i]] = log_J_2x2_s[:,i]
    for i in range(len(log_J_2x2_m[0])):
        df[J_m_names[i]] = J_2x2_m[:,i]
    for i in range(len(log_J_2x2_s[0])):
        df[J_s_names[i]] = J_2x2_s[:,i]
    df['c_E']=c_E
    df['c_I']=c_I
    df['log_f_E']=log_f_E
    df['log_f_I']=log_f_I
    df['f_E']=[np.exp(log_f_E[i]) for i in range(len(log_f_E))]
    df['f_I']=[np.exp(log_f_I[i]) for i in range(len(log_f_I))]

    if max_stages==1:    
        df['offset']=None
        offsets_at_bl_acc=np.hstack(offsets_at_bl_acc)
        df.loc[step_indices['acc_check_ind'],'offset']=offsets_at_bl_acc
    else:    
        # Add offset to df if staircase was in place
        if offsets is not None:
            df['offset']= offsets
            
    for i in range(len(w_sigs[0])):
        weight_name = f'w_sig_{i+1}'
        df[weight_name] =  w_sigs[:,i]
    df['b_sig'] = b_sigs
    return df
