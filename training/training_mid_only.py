#import matplotlib.pyplot as plt
import numpy
import jax
import jax.numpy as np
from jax import vmap
import optax
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from util import sep_exponentiate, sigmoid, create_grating_training, create_grating_pretraining, take_log
from training.SSN_classes import SSN_mid
from training.model import vmap_evaluate_model_response_mid, evaluate_model_response_mid
from training.training_functions import loss_and_grad_ori_discr, binary_crossentropy_loss, offset_at_baseline_acc, mean_training_task_acc_test, training_task_acc_test
from training.perturb_params import perturb_params_supp

def train_ori_discr_mid(
    readout_pars_dict,
    trained_pars_dict,
    untrained_pars,
    threshold = 0.75,
    offset_step = 0.25,
    results_filename=None,
    jit_on=True
):
    """
    Trains a one-layer SSN network model in a pretraining and a training stage. It first trains the SSN layer parameters and the readout parameters for a general orientation discrimination task. Then, it further trains the SSN layer parameters for a fine orientation discrimination task.
    
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
        opt_state_ssn = optimizer.init(trained_pars_dict)
        opt_state_readout = optimizer.init(readout_pars_dict)
        training_loss_val_and_grad = jax.value_and_grad(batch_loss_ori_discr_mid, argnums=[0,1], has_aux=True)
    else:
        opt_state_readout = optimizer.init(readout_pars_dict)
        training_loss_val_and_grad = jax.value_and_grad(batch_loss_ori_discr_mid, argnums=1, has_aux=True)

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
                opt_state_ssn = optimizer.init(trained_pars_dict)
                training_loss_val_and_grad = jax.value_and_grad(batch_loss_ori_discr_mid, argnums=0, has_aux=True)

            # STOCHASTIC GRADIENT DESCENT LOOP
            for SGD_step in range(numSGD_steps):
                # i) Calculate model loss, accuracy, gradient
                train_loss, train_loss_all, train_acc, _, _, train_max_rate, grad = loss_and_grad_ori_discr(trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, training_loss_val_and_grad)
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
                        log_J_2x2_m.append(trained_pars_dict['log_J_2x2_m'].ravel())
                        if 'c_E' in trained_pars_dict.keys():
                            c_E.append(trained_pars_dict['c_E'])
                            c_I.append(trained_pars_dict['c_I'])
                        else:
                            c_E.append(untrained_pars.ssn_pars.c_E)
                            c_I.append(untrained_pars.ssn_pars.c_I)
                else:
                    log_J_2x2_m = [trained_pars_dict['log_J_2x2_m'].ravel()]
                    if 'c_E' in trained_pars_dict.keys():
                        c_E = [trained_pars_dict['c_E']]
                        c_I = [trained_pars_dict['c_I']]
                    else:
                        c_E = [untrained_pars.ssn_pars.c_E]
                        c_I = [untrained_pars.ssn_pars.c_I]

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
                    acc_mean, _, _ = mean_training_task_acc_test(trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, test_offset_vec, loss_functioon_mid_only=batch_loss_ori_discr_mid)
                    ## fit log-linear curve to acc_mean_max and test_offset_vec and find where it crosses baseline_acc=0.794
                    offset_at_bl_acc = offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= untrained_pars.pretrain_pars.acc_th)
                    if SGD_step==acc_check_ind[0] and stage==1:
                        offsets_th=[float(offset_at_bl_acc)]
                    else:
                        offsets_th.append(float(offset_at_bl_acc))
                    print('Baseline acc is achieved at offset:', offset_at_bl_acc, ' for step ', SGD_step)

                    # Stop pretraining: break out from SGD_step loop and stages loop (using a flag)
                    if len(offsets_th)>1: # we stop pretraining even if the training task is solved for the pretraining
                        pretrain_stop_flag = all(np.array(offsets_th[-2:]) < pretrain_offset_threshold)
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
                    val_acc_vec, val_loss_vec = training_task_acc_test(trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, stimuli_pars.offset, loss_functioon_mid_only=batch_loss_ori_discr_mid)
                
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
                    trained_pars_dict = optax.apply_updates(trained_pars_dict, updates_ssn)
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
                        trained_pars_dict = optax.apply_updates(trained_pars_dict, updates_ssn)
                        
    ############# SAVING and RETURN OUTPUT #############
    
    # Define SGD_steps indices for training and validation
    if pretrain_on:
        SGD_steps = np.arange(0, len(stages))
        val_SGD_steps = val_steps[0:len(val_accs)]
        offsets = None
        acc_check_ind = acc_check_ind[0:len(offsets_th)]
        step_indices = dict(SGD_steps=SGD_steps, val_SGD_steps=val_SGD_steps, acc_check_ind=acc_check_ind)
    else:
        SGD_steps = np.arange(0, len(stages))
        val_SGD_steps_stage1 = val_steps[val_steps < first_stage_final_step + 1]
        val_SGD_steps_stage2 = np.arange(first_stage_final_step + 1, first_stage_final_step + 1 + numSGD_steps, training_pars.validation_freq)
        val_SGD_steps = np.hstack([val_SGD_steps_stage1, val_SGD_steps_stage2])
        val_SGD_steps = val_SGD_steps[0:len(val_accs)]
        offsets_th = None
        step_indices = dict(SGD_steps=SGD_steps, val_SGD_steps=val_SGD_steps)
        
    # Create DataFrame and save the DataFrame to a CSV file
        
    df = make_dataframe_mid(stages, step_indices, train_accs, val_accs, train_losses_all, val_losses, train_max_rates, b_sigs, w_sigs, log_J_2x2_m, c_E, c_I, offsets, offsets_th)

    if results_filename:
        file_exists = os.path.isfile(results_filename)
        df.to_csv(results_filename, mode='a', header=not file_exists, index=False)

    return df, first_stage_final_step


def loss_ori_discr_mid(trained_pars_dict, readout_pars_dict, untrained_pars, train_data, noise_ref, noise_target): 
    """
    Bottom level function to calculate losses, accuracies and other relevant metrics. It applies evaluate_model_response_mid to training data, adds noise to the model output and calculates the relevant metrics.

    Args:
        trained_pars_dict (dict): Dictionary containing parameters for the SSN layer.
        readout_pars_dict (dict): Dictionary containing parameters for the readout layer.
        untrained_pars (object): Object containing various parameters that are untrained.
        train_data: Dictionary containing reference and target training data and corresponding labels
        noise_ref: additive noise for the model output reference training data
        noise_target: additive noise for the model output from target training data

    Returns:
        tuple: Tuple containing loss, all_losses, accuracy, sig_input, sig_output, max_rates
    """
    
    pretraining = untrained_pars.pretrain_pars.is_on
    w_sig = readout_pars_dict['w_sig']
    b_sig = readout_pars_dict['b_sig']

    if 'c_E' in trained_pars_dict:
        c_E = trained_pars_dict['c_E']
        c_I = trained_pars_dict['c_I']
    else:
        c_E = untrained_pars.ssn_pars.c_E
        c_I = untrained_pars.ssn_pars.c_I

    J_2x2_m = sep_exponentiate(trained_pars_dict['log_J_2x2_m'])

    loss_pars = untrained_pars.loss_pars
    conv_pars = untrained_pars.conv_pars

    #Run reference and targetthrough two layer model
    ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_m)
    
    r_ref, r_max_ref_mid, avg_dx_ref_mid, max_E_mid, max_I_mid, _ = evaluate_model_response_mid(ssn_mid, train_data['ref'], conv_pars, c_E, c_I, untrained_pars.gabor_filters)
    r_target, r_max_target_mid, avg_dx_target_mid, max_E_mid, max_I_mid, _= evaluate_model_response_mid(ssn_mid, train_data['target'], conv_pars, c_E, c_I, untrained_pars.gabor_filters)

    if pretraining:
        # Readout is from all neurons in middle layer
        r_ref_box = r_ref
        r_target_box = r_target
    else:
        # Select the ceter grid of the middle layer
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
    loss_dx_max = loss_pars.lambda_dx*np.mean(np.array([avg_dx_ref_mid, avg_dx_target_mid])**2)
    loss_r_max =  loss_pars.lambda_r_max*np.mean(np.array([r_max_ref_mid, r_max_target_mid]))
    loss_w = loss_pars.lambda_w*(np.linalg.norm(w_sig)**2)
    loss_b = loss_pars.lambda_b*(b_sig**2)   
    loss = loss_readout + loss_w + loss_b +  loss_dx_max + loss_r_max
    all_losses = np.vstack((loss_readout, loss_dx_max, loss_r_max, loss_w, loss_b, loss))
    
    return loss, all_losses, pred_label, sig_input, sig_output,  [max_E_mid, max_I_mid]

# Parallelize orientation discrimination task and jit the parallelized function
vmap_ori_discr_mid = vmap(loss_ori_discr_mid, in_axes = (None, None, None, {'ref':0, 'target':0, 'label':0}, 0, 0) )
jit_ori_discr_mid = jax.jit(vmap_ori_discr_mid, static_argnums = [2])


def batch_loss_ori_discr_mid(trained_pars_dict, readout_pars_dict, untrained_pars, train_data, noise_ref, noise_target, jit_on=True):
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
        total_loss, all_losses, pred_label, sig_input, sig_output, max_rates = jit_ori_discr_mid(trained_pars_dict, readout_pars_dict, untrained_pars, train_data, noise_ref, noise_target)
    else:
        total_loss, all_losses, pred_label, sig_input, sig_output, max_rates = vmap_ori_discr_mid(trained_pars_dict, readout_pars_dict, untrained_pars, train_data, noise_ref, noise_target)
    
    # Average total loss within a batch (across trials)
    loss= np.mean(total_loss)
    
    # Average individual losses within a batch (accross trials)
    all_losses = np.mean(all_losses, axis = 0)
    
    #Find maximum rates  within a batch (across trials)
    max_rates = [item.max() for item in max_rates]
    
    # Calculate the proportion of labels that are predicted well (within a batch) 
    true_accuracy = np.sum(train_data['label'] == pred_label)/len(train_data['label'])

    return loss, [all_losses, true_accuracy, sig_input, sig_output, max_rates]


def make_dataframe_mid(stages, step_indices, train_accs, val_accs, train_losses_all, val_losses, train_max_rates, b_sigs,w_sigs, log_J_2x2_m, c_E, c_I, offsets=None, offsets_at_bl_acc=None):
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
    max_rates_names = ['maxr_E_mid', 'maxr_I_mid']
    for i in range(len(train_max_rates[0])):
        df[max_rates_names[i]]=train_max_rates[:,i]
    
    # Add parameters that are trained in two stages during training and in one stage during pretraining
    max_stages = max(1,max(stages))
    log_J_m_names = ['log_J_m_EE', 'log_J_m_EI', 'log_J_m_IE', 'log_J_m_II']
    J_m_names = ['J_m_EE', 'J_m_EI', 'J_m_IE', 'J_m_II']
    J_2x2_m=np.transpose(np.array([np.exp(log_J_2x2_m[:,0]),-np.exp(log_J_2x2_m[:,1]),np.exp(log_J_2x2_m[:,2]),-np.exp(log_J_2x2_m[:,3])]))
    for i in range(len(log_J_2x2_m[0])):
        df[log_J_m_names[i]] = log_J_2x2_m[:,i]
    for i in range(len(log_J_2x2_m[0])):
        df[J_m_names[i]] = J_2x2_m[:,i]
    df['c_E']=c_E
    df['c_I']=c_I

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


def load_parameters_mid(file_path, readout_grid_size=5, iloc_ind=-1, trained_pars_keys=['log_J_2x2_m', 'c_E', 'c_I']):

    # Get the last row of the given csv file
    df = pd.read_csv(file_path)
    selected_row = df.iloc[int(iloc_ind)]

    # Extract stage 1 parameters from df
    w_sig_keys = [f'w_sig_{i}' for i in range(1, readout_grid_size*readout_grid_size+1)] 
    w_sig_values = selected_row[w_sig_keys].values
    pars_stage1 = dict(w_sig=w_sig_values, b_sig=selected_row['b_sig'])

    # Extract stage 2 parameters from df
    log_J_m_keys = ['log_J_m_EE','log_J_m_EI','log_J_m_IE','log_J_m_II'] 
    J_m_values = selected_row[log_J_m_keys].values.reshape(2, 2)
    # Create a dictionary with the trained parameters
    pars_stage2 = dict(
        log_J_2x2_m = J_m_values
    )
    if 'c_E' in trained_pars_keys:
        pars_stage2['c_E'] = selected_row['c_E']
        pars_stage2['c_I'] = selected_row['c_I']
    

    offsets  = df['offset'].dropna().reset_index(drop=True)
    offset_last = offsets[len(offsets)-1]

    return pars_stage1, pars_stage2, offset_last


def perturb_params_mid(readout_pars, trained_pars, untrained_pars, percent=0.1, orimap_filename=None, logistic_regr=True):
    # define the parameters to perturb
    trained_pars_dict = dict(J_2x2_m=trained_pars.J_2x2_m)
    for key, value in vars(trained_pars).items():
        if key == 'c_E' or key == 'c_I':
            trained_pars_dict[key] = value
       
    # Perturb parameters under conditions for J_mid and convergence of the differential equations of the model
    i=0
    cond1 = False
    cond2 = False
    cond3 = False
    cond4 = False
    cond5 = False

    while not (cond1 and cond2 and cond3 and cond4 and cond5):
        perturbed_pars = perturb_params_supp(trained_pars_dict, percent)
        cond1 = np.abs(perturbed_pars['J_2x2_m'][0,0]*perturbed_pars['J_2x2_m'][1,1])*1.1 < np.abs(perturbed_pars['J_2x2_m'][1,0]*perturbed_pars['J_2x2_m'][0,1])
        cond2 = np.abs(perturbed_pars['J_2x2_m'][0,1]*untrained_pars.filter_pars.gI_m)*1.1 < np.abs(perturbed_pars['J_2x2_m'][1,1]*untrained_pars.filter_pars.gE_m)
        
        # Calculate model response to check the convergence of the differential equations
        ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=perturbed_pars['J_2x2_m'])
        train_data = create_grating_training(untrained_pars.stimuli_pars, batch_size=1, BW_image_jit_inp_all=untrained_pars.BW_image_jax_inp)
        if 'c_E' in perturbed_pars:
            c_E = perturbed_pars['c_E']
            c_I = perturbed_pars['c_I']
        else:
            c_E = untrained_pars.ssn_pars.c_E
            c_I = untrained_pars.ssn_pars.c_I
        r_ref, avg_dx_mid, max_E_mid, max_I_mid, _,_,_ = vmap_evaluate_model_response_mid(ssn_mid, train_data['ref'], untrained_pars.conv_pars,c_E, c_I, untrained_pars.gabor_filters)
        cond3 = not numpy.any(numpy.isnan(r_ref))
        cond4 = avg_dx_mid  < 50
        cond5 = min([max_E_mid, max_I_mid])>10 and max([max_E_mid, max_I_mid])<101
        if i>50:
            print(" ########### Perturbed parameters violate conditions even after 50 sampling. ###########")
            break
        else:
            if orimap_filename is not None:
                # regenerate orimap
                untrained_pars =  init_untrained_pars(untrained_pars.grid_pars, untrained_pars.stimuli_pars, untrained_pars.filter_pars, untrained_pars.ssn_pars, untrained_pars.conv_pars, 
                    untrained_pars.loss_pars, untrained_pars.training_pars, untrained_pars.pretrain_pars, readout_pars, orimap_filename)
            i = i+1

    # Take log of the J parameters
    perturbed_pars_log = dict(log_J_2x2_m= take_log(perturbed_pars['J_2x2_m']))
    for key, vale in perturbed_pars.items():
        if key=='c_E' or key=='c_I':
            perturbed_pars_log[key] = vale

    # Optimize readout parameters by using log-linear regression
    if logistic_regr:
        pars_stage1 = readout_pars_from_regr_mid(readout_pars, perturbed_pars_log, untrained_pars)
    else:
        pars_stage1 = dict(w_sig=readout_pars.w_sig, b_sig=readout_pars.b_sig)
    pars_stage1['w_sig'] = (pars_stage1['w_sig'] / np.std(pars_stage1['w_sig']) ) * 0.25 / int(np.sqrt(len(pars_stage1['w_sig']))) # get the same std as before - see param

    # Perturb learning rate
    untrained_pars.training_pars.eta = untrained_pars.training_pars.eta + percent * untrained_pars.training_pars.eta * numpy.random.uniform(-1, 1)
    
    return pars_stage1, perturbed_pars_log, untrained_pars

def readout_pars_from_regr_mid(readout_pars, trained_pars_dict, untrained_pars, N=1000):
    '''
    This function sets readout_pars based on N sample data using linear regression. This method is to initialize w_sig, b_sig optimally (given limited data) for a set of perturbed trained_pars_dict parameters (to be trained).
    '''
    # Generate stimuli and label data for setting w_sig and b_sig based on linear regression (pretraining)
    data = create_grating_pretraining(untrained_pars.pretrain_pars, N, untrained_pars.BW_image_jax_inp, numRnd_ori1=N)
    
    #### Get model response for stimuli data['ref'] and data['target'] ####
    # 1. extract trained and untrained parameters
    J_2x2_m = sep_exponentiate(trained_pars_dict['log_J_2x2_m'])

    if 'c_E' in trained_pars_dict:
        c_E = trained_pars_dict['c_E']
        c_I = trained_pars_dict['c_I']
    else:
        c_E = untrained_pars.ssn_pars.c_E
        c_I = untrained_pars.ssn_pars.c_I

    conv_pars = untrained_pars.conv_pars

    # 2. define middle layer and superficial layer SSN
    ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_m)
    
    # Run reference and target through two layer model
    r_ref, _, _, _, _, _, _  = vmap_evaluate_model_response_mid(ssn_mid, data['ref'], conv_pars, c_E, c_I,  untrained_pars.gabor_filters)
    r_target, _, _, _, _, _, _= vmap_evaluate_model_response_mid(ssn_mid, data['target'], conv_pars, c_E, c_I, untrained_pars.gabor_filters)

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
    readout_pars_opt = {key: None for key in vars(readout_pars)}
    readout_pars_opt['b_sig'] = float(log_reg.intercept_)
    w_sig = log_reg.coef_.T
    w_sig = w_sig.squeeze()
    
    if untrained_pars.pretrain_pars.is_on:
        readout_pars_opt['w_sig'] = w_sig
    else:
        readout_pars_opt['w_sig'] = w_sig[untrained_pars.middle_grid_ind]

    # check if the readout parameters solve the task in the correct direction
    test_offset_vec = numpy.array([2, 5, 10, 18]) 
    acc_mean, _, _ = mean_training_task_acc_test(trained_pars_dict, readout_pars_opt, untrained_pars, True, test_offset_vec, loss_functioon_mid_only=batch_loss_ori_discr_mid)
    if np.sum(acc_mean<0.5)>0.5*len(acc_mean):
        # flip the sign of w_sig and b_sig if the logistic regression is solving the flipped task
        readout_pars_opt['w_sig'] = -w_sig
        readout_pars_opt['b_sig'] = -readout_pars_opt['b_sig']
        acc_mean_flipped, _, _ = mean_training_task_acc_test(trained_pars_dict, readout_pars_opt, untrained_pars, True, test_offset_vec, loss_functioon_mid_only=batch_loss_ori_discr_mid)
        print('accuracy of logistic regression before and after flipping the w_sig and b_sig', acc_mean, acc_mean_flipped)

    return readout_pars_opt

############### MAIN CODE FOR TRAINING MIDDLE LAYER ONLY ###############
import numpy
from util_gabor import init_untrained_pars
from util import save_code
from perturb_params import perturb_params, create_initial_parameters_df
from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    readout_pars,
    ssn_pars,
    trained_pars,
    conv_pars,
    training_pars,
    loss_pars,
    pretrain_pars # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
)

# Checking that pretrain_pars.is_on is on
if not pretrain_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to True in parameters.py to run training with pretraining!')
# check if untrained_pars.ssn_pars.f_I is defined
if not hasattr(ssn_pars, 'f_I'):
    raise ValueError('Define f_I and f_E in ssn_pars in parameters.py to run middle layer training only!')

########## Initialize orientation map and gabor filters ############

# Save out initial offset and reference orientation
ref_ori_saved = float(stimuli_pars.ref_ori)
offset_saved = float(stimuli_pars.offset)

# Save scripts into scripts folder and create figures and train_only folders
train_only_flag = False # Setting train_only_flag to True will run an additional training without pretraining
results_filename, final_folder_path = save_code(train_only_flag=train_only_flag)

# Run num_training number of pretraining + training
num_training = 5
starting_time_in_main= time.time()
initial_parameters = None
num_FailedRuns = 0
i=0

while i < num_training and num_FailedRuns < 20:
    numpy.random.seed(i+1)

    # Set pretraining flag to False
    pretrain_pars.is_on=True
    # Set offset and reference orientation to their initial values
    stimuli_pars.offset=offset_saved
    stimuli_pars.ref_ori=ref_ori_saved

    # Create file names
    results_filename = os.path.join(final_folder_path, f"results_{i}.csv")
    results_filename_train_only = os.path.join(final_folder_path, 'train_only', f"results_train_only{i}.csv")
    orimap_filename = os.path.join(final_folder_path, f"orimap_{i}.npy")

    # Initialize untrained parameters (calculate gabor filters, orientation map related variables)
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, orimap_filename)

    ##### PRETRAINING: GENERAL ORIENTAION DISCRIMINATION #####

    # Perturb readout_pars and trained_pars by percent % and collect them into two dictionaries for the two stages of the pretraining
    # Note that orimap is regenerated if conditions do not hold!
    trained_pars_stage1, trained_pars_stage2, untrained_pars = perturb_params(readout_pars, trained_pars, untrained_pars, percent=0.1, orimap_filename=orimap_filename)
    initial_parameters = create_initial_parameters_df(initial_parameters, trained_pars_stage2, untrained_pars.training_pars.eta)
    
    # Run pre-training
    training_output_df, pretraining_final_step = train_ori_discr_mid(
            trained_pars_stage1,
            trained_pars_stage2,
            untrained_pars,
            results_filename=results_filename,
            jit_on=True,
            offset_step = 0.1
        )
    
    # Handle the case when pretraining failed (possible reason can be the divergence of ssn diff equations)
    if training_output_df is None:
        print('######### Stopped run {} because of NaN values  - num failed runs = {} #########'.format(i, num_FailedRuns))
        num_FailedRuns = num_FailedRuns + 1
        continue
    
    ##### FINE DISCRIMINATION #####
    
    # Set pretraining flag to False
    untrained_pars.pretrain_pars.is_on = False
    # Load the last parameters from the pretraining
    trained_pars_stage1, trained_pars_stage2, offset_last = load_parameters_mid(results_filename, iloc_ind = pretraining_final_step, trained_pars_keys=trained_pars_stage2.keys())
    # Set the offset to the offset, where a threshold accuracy is achieved with the parameters from the last SGD step (loaded as offset_last)
    untrained_pars.stimuli_pars.offset = min(offset_last,10)
    # Run training
    training_output_df, _ = train_ori_discr_mid(
            trained_pars_stage1,
            trained_pars_stage2,
            untrained_pars,
            results_filename=results_filename,
            jit_on=True,
            offset_step=0.1
        )
    
    # Save initial_parameters to csv
    initial_parameters.to_csv(os.path.join(final_folder_path, 'initial_parameters.csv'), index=False)
    
    i = i + 1
    print('runtime of {} pretraining + training run(s)'.format(i), time.time()-starting_time_in_main)
    print('number of failed runs = ', num_FailedRuns)
