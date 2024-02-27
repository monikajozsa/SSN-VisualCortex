import jax
import jax.numpy as np
from jax import vmap
import optax
import time
import pandas as pd
import numpy
import os
import copy

from util import create_grating_pairs, sep_exponentiate, sigmoid, create_grating_pretraining
from SSN_classes import SSN_mid, SSN_sup
from model import evaluate_model_response


def train_ori_discr(
    readout_pars_dict,
    ssn_layer_pars_dict,
    constant_pars,
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
    - readout_parameters (dict): Parameters for the readout layer.
    - ssn_layer_parameters (dict): Parameters for the SSN layer.
    - constant_parameters (dict): Includes grid_pars, stimuli_pars, conn_pars_m, 
                                  conn_pars_s, filter_pars, ssn_ori_map, ssn_pars, 
                                  ssn_layer_pars, conv_pars, loss_pars, training_pars.
    - results_filename (str, optional): Filename for saving results.
    - jit_on (bool): If True, enables JIT compilation for performance improvement.
    """
     # Unpack training_pars and stimuli_pars from constant_pars
    training_pars = constant_pars.training_pars
    stimuli_pars = constant_pars.stimuli_pars
    pretrain_on = constant_pars.pretrain_pars.is_on
    pretrain_offset_threshold = constant_pars.pretrain_pars.offset_threshold

    # Define indices of sigmoid layer weights to save
    if pretrain_on:
        w_indices_to_save = constant_pars.middle_grid_ind
    else:
        w_indices_to_save = numpy.array([i for i in range(constant_pars.readout_grid_size[1] ** 2)])

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
        numSGD_steps = training_pars.SGD_steps[0]
        min_acc_check_ind = constant_pars.pretrain_pars.min_acc_check_ind
        acc_check_ind = np.arange(0, numSGD_steps, constant_pars.pretrain_pars.acc_check_freq)
        acc_check_ind = acc_check_ind[(acc_check_ind > min_acc_check_ind) | (acc_check_ind <2)] # by leaving in 0, we make sure that it is not empty as we refer to it later
        test_offset_vec = numpy.array([2, 4, 6, 9, 12, 15, 20]) # offsets that help us define accuracy for training task during pretraining
        numStages = constant_pars.pretrain_pars.numStages
    else:
        numSGD_steps = training_pars.SGD_steps[1]
        first_stage_acc_th = training_pars.first_stage_acc_th
        numStages = 2
    
    # Define SGD_steps indices where losses an accuracy are validated
    val_steps = np.arange(0, numSGD_steps, training_pars.validation_freq)
    first_stage_final_step = numSGD_steps

    print(
        "SGD_step: {} ¦ learning rate: {} ¦ sig_noise: {} ¦ batch size {}".format(
            numSGD_steps, training_pars.eta, training_pars.sig_noise, training_pars.batch_size,
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
                train_loss, train_loss_all, train_acc, _, _, train_max_rate, grad = loss_and_grad_ori_discr(ssn_layer_pars_dict, readout_pars_dict, constant_pars, jit_on, training_loss_val_and_grad)
                if jax.numpy.isnan(train_loss):
                    return None, None

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
                        f_E.append(ssn_layer_pars_dict['f_E'])
                        f_I.append(ssn_layer_pars_dict['f_I'])
                else:
                    log_J_2x2_m = [ssn_layer_pars_dict['log_J_2x2_m'].ravel()]
                    log_J_2x2_s = [ssn_layer_pars_dict['log_J_2x2_s'].ravel()]
                    c_E = [ssn_layer_pars_dict['c_E']]
                    c_I = [ssn_layer_pars_dict['c_I']]
                    f_E = [ssn_layer_pars_dict['f_E']]
                    f_I = [ssn_layer_pars_dict['f_I']]
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
                    acc_mean, _, _ = mean_training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, constant_pars, jit_on, test_offset_vec)
                    # fit log-linear curve to acc_mean and test_offset_vec and find where it crosses baseline_acc=0.794
                    offset_at_bl_acc = offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= constant_pars.pretrain_pars.acc_th)
                    if SGD_step==acc_check_ind[0] and stage==1:
                        offsets_at_bl_acc=[float(offset_at_bl_acc)]
                    else:
                        offsets_at_bl_acc.append(float(offset_at_bl_acc))
                    print('Baseline acc is achieved at offset:', offset_at_bl_acc, ' for step ', SGD_step)

                    # Stop pretraining: nreak out from STG_step loop and stages loop (using a flag)
                    if len(offsets_at_bl_acc)>3:
                        pretrain_stop_flag = all(np.array(offsets_at_bl_acc[-3:]) < pretrain_offset_threshold)
                    else:
                        pretrain_stop_flag = True if offset_at_bl_acc < pretrain_offset_threshold else False
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
                    val_loss_vec, val_acc_vec = training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, constant_pars, jit_on, stimuli_pars.offset)
                
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
        val_SGD_steps_stage2 = np.arange(first_stage_final_step, first_stage_final_step + numSGD_steps, training_pars.validation_freq)
        val_SGD_steps = np.hstack([val_SGD_steps_stage1, val_SGD_steps_stage2])
        val_SGD_steps = val_SGD_steps[0:len(val_accs)]
        offsets_at_bl_acc = None
        step_indices = dict(SGD_steps=SGD_steps, val_SGD_steps=val_SGD_steps)
        
    # Create DataFrame and save the DataFrame to a CSV file
        
    df = make_dataframe(stages, step_indices, train_accs, val_accs, train_losses_all, val_losses, train_max_rates, b_sigs, w_sigs, log_J_2x2_m, log_J_2x2_s, c_E, c_I, f_E, f_I, offsets, offsets_at_bl_acc)

    if results_filename:
        file_exists = os.path.isfile(results_filename)
        df.to_csv(results_filename, mode='a', header=not file_exists, index=False)

    return df, first_stage_final_step


def loss_and_grad_ori_discr(ssn_layer_pars_dict, readout_pars_dict, constant_pars, jit_on, training_loss_val_and_grad):

    # Create stimulus for middle layer: train_data is a dictionary with keys 'ref', 'target' and 'label'
    training_pars=constant_pars.training_pars
    if constant_pars.pretrain_pars.is_on:
        # Generate noise that is added to the output of the model
        noise_ref = generate_noise(
            training_pars.sig_noise, training_pars.batch_size[0], readout_pars_dict["w_sig"].shape[0]
        )
        noise_target = generate_noise(
            training_pars.sig_noise, training_pars.batch_size[0], readout_pars_dict["w_sig"].shape[0]
        )        
        train_data = create_grating_pretraining(constant_pars.pretrain_pars, training_pars.batch_size[0], constant_pars.BW_image_jax_inp, numRnd_ori1=training_pars.batch_size[0])
    else:
        # Generate noise that is added to the output of the model
        noise_ref = generate_noise(
            training_pars.sig_noise, training_pars.batch_size[1], readout_pars_dict["w_sig"].shape[0]
        )
        noise_target = generate_noise(
            training_pars.sig_noise, training_pars.batch_size[1], readout_pars_dict["w_sig"].shape[0]
        )
        train_data = create_grating_pairs(constant_pars.stimuli_pars, training_pars.batch_size[1], constant_pars.BW_image_jax_inp)

    # Calculate gradient, loss and accuracy
    [loss, [all_losses, accuracy, sig_input, sig_output, max_rates]], grad = training_loss_val_and_grad(
        ssn_layer_pars_dict,
        readout_pars_dict,
        constant_pars,
        train_data,
        noise_ref,
        noise_target,
        jit_on
    )
    return loss, all_losses, accuracy, sig_input, sig_output, max_rates, grad


def loss_ori_discr(ssn_layer_pars_dict, readout_pars_dict, constant_pars, train_data, noise_ref, noise_target): 
    
    '''
    Orientation discrimanation task ran using SSN two-layer model.The reference and target are run through the two layer model individually. 
    Inputs:
        individual parameters - having taken logs of differentiable parameters
        noise_type: select different noise models
    Outputs:
        losses to take gradient with respect to
        sig_input, x: I/O values for sigmoid layer
    '''
    pretraining = constant_pars.pretrain_pars.is_on
    log_J_2x2_m = ssn_layer_pars_dict['log_J_2x2_m']
    log_J_2x2_s = ssn_layer_pars_dict['log_J_2x2_s']
    c_E = ssn_layer_pars_dict['c_E']
    c_I = ssn_layer_pars_dict['c_I']
    f_E = np.exp(ssn_layer_pars_dict['f_E'])
    f_I = np.exp(ssn_layer_pars_dict['f_I'])
    kappa_pre = constant_pars.ssn_layer_pars.kappa_pre
    kappa_post = constant_pars.ssn_layer_pars.kappa_post
    w_sig = readout_pars_dict['w_sig']
    b_sig = readout_pars_dict['b_sig']
    p_local_s = constant_pars.ssn_layer_pars.p_local_s
    s_2x2 = constant_pars.ssn_layer_pars.s_2x2_s
    sigma_oris = constant_pars.ssn_layer_pars.sigma_oris
    ref_ori = constant_pars.stimuli_pars.ref_ori
    
    J_2x2_m = sep_exponentiate(log_J_2x2_m)
    J_2x2_s = sep_exponentiate(log_J_2x2_s)   

    loss_pars = constant_pars.loss_pars
    conv_pars = constant_pars.conv_pars

    # Create middle and superficial SSN layers *** this is something that would be great to call from outside the SGD loop and only refresh the params that change (and what rely on them such as W)
    ssn_mid=SSN_mid(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, J_2x2=J_2x2_m)
    ssn_sup=SSN_sup(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, J_2x2=J_2x2_s, p_local=p_local_s, oris=constant_pars.oris, s_2x2=s_2x2, sigma_oris = sigma_oris, ori_dist = constant_pars.ori_dist, train_ori = ref_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)
    
    #Run reference and targetthrough two layer model
    r_ref, _, [r_max_ref_mid, r_max_ref_sup], [avg_dx_ref_mid, avg_dx_ref_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'], conv_pars, c_E, c_I, f_E, f_I, constant_pars.gabor_filters)
    r_target, _, [r_max_target_mid, r_max_target_sup], [avg_dx_target_mid, avg_dx_target_sup], _, _= evaluate_model_response(ssn_mid, ssn_sup, train_data['target'], conv_pars, c_E, c_I, f_E, f_I, constant_pars.gabor_filters)

    if pretraining:
        # Readout is from all neurons in sup layer
        r_ref_box = r_ref
        r_target_box = r_target
    else:
        # Select the middle grid
        r_ref_box = r_ref[constant_pars.middle_grid_ind]
        r_target_box = r_target[constant_pars.middle_grid_ind]       
    
    # Add noise
    r_ref_box = r_ref_box + noise_ref*np.sqrt(jax.nn.softplus(r_ref_box))
    r_target_box = r_target_box + noise_target*np.sqrt(jax.nn.softplus(r_target_box))
    
    # Define losses
    # i) Multiply (reference - target) by sigmoid layer weights, add bias and apply sigmoid funciton
    sig_input = np.dot(w_sig, (r_ref_box - r_target_box)) + b_sig     
    sig_output = sigmoid(sig_input)
    # ii) Calculate readout loss and the predicted label
    loss_readout = binary_loss(train_data['label'], sig_output)
    pred_label = np.round(sig_output)
    # ii) Calculate other loss terms
    loss_dx_max = loss_pars.lambda_dx*np.mean(np.array([avg_dx_ref_mid, avg_dx_target_mid, avg_dx_ref_sup, avg_dx_target_sup])**2)
    loss_r_max =  loss_pars.lambda_r_max*np.mean(np.array([r_max_ref_mid, r_max_target_mid, r_max_ref_sup, r_max_target_sup]))
    loss_w = loss_pars.lambda_w*(np.linalg.norm(w_sig)**2)
    loss_b = loss_pars.lambda_b*(b_sig**2)   
    loss = loss_readout + loss_w + loss_b +  loss_dx_max + loss_r_max
    all_losses = np.vstack((loss_readout, loss_dx_max, loss_r_max, loss_w, loss_b, loss))
    
    return loss, all_losses, pred_label, sig_input, sig_output,  [max_E_mid, max_I_mid, max_E_sup, max_I_sup]

#Parallelize orientation discrimination task and jit the parallelized function
vmap_ori_discrimination = vmap(loss_ori_discr, in_axes = (None, None, None, {'ref':0, 'target':0, 'label':0}, 0, 0) )
jit_ori_discrimination = jax.jit(vmap_ori_discrimination, static_argnums = [2])


def batch_loss_ori_discr(ssn_layer_pars_dict, readout_pars_dict, constant_pars, train_data, noise_ref, noise_target, jit_on=True):
    '''
    Run orientation discrimination task on given batch of data. Returns losses averaged over the trials within the batch. Function over which the gradient is taken.
    '''    
    #Run orientation discrimination task
    if jit_on:
        total_loss, all_losses, pred_label, sig_input, sig_output, max_rates = jit_ori_discrimination(ssn_layer_pars_dict, readout_pars_dict, constant_pars, train_data, noise_ref, noise_target)
    else:
        total_loss, all_losses, pred_label, sig_input, sig_output, max_rates = vmap_ori_discrimination(ssn_layer_pars_dict, readout_pars_dict, constant_pars, train_data, noise_ref, noise_target)
    
    #Total loss to take gradient with respect to 
    loss= np.mean(total_loss)
    
    #Find mean of different losses
    all_losses = np.mean(all_losses, axis = 0)
    
    #Find maximum rates across trials
    max_rates = [item.max() for item in max_rates]
    
    #Calculate accuracy 
    true_accuracy = np.sum(train_data['label'] == pred_label)/len(train_data['label'])  
    
    return loss, [all_losses, true_accuracy, sig_input, sig_output, max_rates]


############### Other supporting functions 

def binary_loss(n, x):
    '''
    Loss function calculating binary cross entropy
    '''
    return -(n * np.log(x) + (1 - n) * np.log(1 - x))


def generate_noise(sig_noise,  batch_size, length):
    '''
    Creates vectors of neural noise. Function creates N vectors, where N = batch_size, each vector of length = length. 
    '''
    return sig_noise*numpy.random.randn(batch_size, length)


####### Functions for testing training task accuracy for different offsets and finding the offset value where it crosses baseline accuracy 
def training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, constant_pars, jit_on, test_offset):
    '''
    This function tests the accuracy of the training orientation discrimination task given a set of parameters across different stimulus offsets.
    
    Parameters:
    - ssn_layer_pars_dict: Dictionary of parameters for the SSN layers.
    - readout_pars_dict: Dictionary of parameters for the readout layer.
    - constant_pars: Constant parameters used across the model.
    - stage: The stage of training readout pars or ssn layer pars.
    - jit_on: Flag to turn on/off JIT compilation.
    - offset_vec: A list of offsets to test the model performance.
    
    Returns:
    - loss: mean and std of losses for each offset over sample_size samples.
    - true_accuracy: mean and std of true accuracies for each offset  over sample_size samples.
    '''
    # Create copies of stimuli and readout_pars_dict because their 
    pretrain_is_on_saved = constant_pars.pretrain_pars.is_on
    constant_pars.pretrain_pars.is_on=False # this is to get the accuracy metric that corresponds to the training task and not the regression task
      
    if pretrain_is_on_saved:
        readout_pars_dict_copy = copy.deepcopy(readout_pars_dict)
        readout_pars_dict_copy['w_sig'] = readout_pars_dict_copy['w_sig'][constant_pars.middle_grid_ind]
        #batch_size = constant_pars.training_pars.batch_size[0]
    else:
        readout_pars_dict_copy = copy.deepcopy(readout_pars_dict)
        #batch_size = constant_pars.training_pars.batch_size[1]
    
    # Iterate over each offset by using vmap outside of the function
    offset_saved = constant_pars.stimuli_pars.offset
    constant_pars.stimuli_pars.offset = test_offset
    
    # Generate noise that is added to the output of the model
    noise_ref = generate_noise(
        constant_pars.training_pars.sig_noise, 200, readout_pars_dict_copy["w_sig"].shape[0]
    )
    noise_target = generate_noise(
        constant_pars.training_pars.sig_noise, 200, readout_pars_dict_copy["w_sig"].shape[0]
    )
    
    # Create stimulus for middle layer: train_data is a dictionary with keys 'ref', 'target' and 'label'
    train_data = create_grating_pairs(constant_pars.stimuli_pars, 200, constant_pars.BW_image_jax_inp)

    # Calculate loss and accuracy
    loss, [_, acc, _, _, _] = batch_loss_ori_discr(ssn_layer_pars_dict, readout_pars_dict_copy, constant_pars, train_data, noise_ref, noise_target, jit_on)

    constant_pars.pretrain_pars.is_on = pretrain_is_on_saved
    constant_pars.stimuli_pars.offset = offset_saved
    
    return loss, acc

def mean_training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, constant_pars, jit_on, offset_vec, sample_size = 3):
    # Initialize arrays to store loss, accuracy, and max rates
    N = len(offset_vec)
    accuracy = numpy.zeros((N,sample_size))
    loss = numpy.zeros((N,sample_size))
    
    accuracy_mean = numpy.zeros((N))

    # For the all the offsets, calculate fine discrimination accuracy sample_size times
    for i in range(N):
        for j in range(sample_size):
            temp_loss, temp_acc = training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, constant_pars, jit_on, offset_vec[i])
            accuracy[i,j] = temp_acc
            loss[i,j] = temp_loss
        
    # Calculate mean loss and accuracy
    accuracy_mean = np.mean(accuracy, axis=1)

    return accuracy_mean, accuracy, loss


def offset_at_baseline_acc(acc_vec, offset_vec=[2, 4, 6, 9, 12, 15, 20], x_vals=numpy.linspace(1, 90, 300), baseline_acc=0.794):
    '''
    This function fits a log-linear curve to x=offset_vec, y=acc_vec data and returns the x value, where the curve crosses baseline_acc.
    '''
    
    # #Fit cubic learning curve and find curve values y_vals at x_vals / this version used std as weights
    # acc_mean = acc_vec[:,0]
    # acc_mean[acc_mean < 0.001] = 0.001
    # acc_weights = 1/acc_vec[:,1]
    # coefficients = numpy.polyfit(offset_vec, acc_mean, 2, w = acc_weights)
    # y_vals = sum(c * x_vals ** i for i, c in enumerate(reversed(coefficients)))
    
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
        offsets_at_bl_acc = 180
    else:
        # Calculate the midpoint of the interval where y_vals crosses baseline_acc
        sign_change_ind = np.where(np.diff(np.sign(y_vals - baseline_acc)))[0]
        offsets_at_bl_acc = (x_vals[sign_change_ind] + x_vals[sign_change_ind + 1]) / 2

    return offsets_at_bl_acc


####### Function for creating DataFrame
def make_dataframe(stages, step_indices, train_accs, val_accs, train_losses_all, val_losses, train_max_rates, b_sigs,w_sigs, log_J_2x2_m, log_J_2x2_s, c_E, c_I, f_E, f_I, offsets=None, offsets_at_bl_acc=None):
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
    loss_names = ['loss_binary', 'loss_dx_max', 'loss_r_max', 'loss_w_sig', 'loss_b_sig', 'loss_all']
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
    J_m_names = ['logJ_m_EE', 'logJ_m_EI', 'logJ_m_IE', 'logJ_m_II']
    J_s_names = ['logJ_s_EE', 'logJ_s_EI', 'logJ_s_IE', 'logJ_s_II']
    for i in range(len(w_sigs[0])):
        weight_name = f'w_sig_{i+1}'
        df[weight_name] =  w_sigs[:,i]
    df['b_sig'] = b_sigs
    for i in range(len(log_J_2x2_m[0])):
        df[J_m_names[i]] = log_J_2x2_m[:,i]
    for i in range(len(log_J_2x2_s[0])):
        df[J_s_names[i]] = log_J_2x2_s[:,i]
    df['c_E']=c_E
    df['c_I']=c_I
    df['f_E']=f_E
    df['f_I']=f_I

    if max_stages==1:    
        df['offset']=None
        offsets_at_bl_acc=np.hstack(offsets_at_bl_acc)
        df.loc[step_indices['acc_check_ind'],'offset']=offsets_at_bl_acc
    else:    
        # Add offset to df if staircase was in place
        if offsets is not None:
            df['offset']= offsets
            
    return df
