import jax
import jax.numpy as np
from jax import vmap
import optax
import time
import pandas as pd
import numpy
import os

from util import create_grating_pairs, sep_exponentiate, sigmoid, create_grating_pretraining
from SSN_classes import SSN_mid, SSN_sup
from model import evaluate_model_response


def train_ori_discr(
    readout_pars_dict,
    ssn_layer_pars_dict,
    constant_pars,
    results_filename=None,
    jit_on=True
):
    """
    Trains a two-layer SSN network model in two distinct stages.
    
    Stage 1: Trains the readout layer parameters until either the accuracy 
    threshold (first_stage_acc) is met or a specified number of epochs (training_pars.epochs) is reached.
    
    Stage 2: Trains the SSN layer parameters for a fixed number of epochs (training_pars.epochs).
    
    Parameters:
    - readout_parameters (dict): Parameters for the readout layer.
    - ssn_layer_parameters (dict): Parameters for the SSN layer.
    - constant_parameters (dict): Includes grid_pars, stimuli_pars, conn_pars_m, 
                                  conn_pars_s, filter_pars, ssn_ori_map, ssn_pars, 
                                  ssn_layer_pars, conv_pars, loss_pars, training_pars.
    - results_filename (str, optional): Filename for saving results.
    - jit_on (bool): If True, enables JIT compilation for performance improvement.
    """
    
    # Initialization of tracking variables for training and validation metrics
    train_losses_all = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    train_accs = [0]
    train_max_rates = [np.array([0.0, 0.0, 0.0, 0.0])]
    val_losses_all = []
    val_accs = []
    stages = [0]

    # Extract initial weights and biases for the readout and SSN layers
    w_sig_temp=readout_pars_dict['w_sig']
    if constant_pars.pretrain_pars.is_on:
        start = (constant_pars.readout_grid_size[0] - constant_pars.readout_grid_size[1]) // 2
        end = start + constant_pars.readout_grid_size[1]
        w_indices_to_save = numpy.array([i*constant_pars.readout_grid_size[0] + j for i in range(start, end) for j in range(start, end)])
    else:
        w_indices_to_save = numpy.array([i for i in range(constant_pars.readout_grid_size[1] ** 2)])
    w_sigs = [w_sig_temp[w_indices_to_save]]
    b_sigs = [readout_pars_dict['b_sig']]
    log_J_2x2_m = [ssn_layer_pars_dict['log_J_2x2_m'].ravel()]
    log_J_2x2_s = [ssn_layer_pars_dict['log_J_2x2_s'].ravel()]

    c_E = [ssn_layer_pars_dict['c_E']]
    c_I = [ssn_layer_pars_dict['c_I']]
    f_E = [ssn_layer_pars_dict['f_E']]
    f_I = [ssn_layer_pars_dict['f_I']]
    if 'kappa_pre' in ssn_layer_pars_dict:
        kappa_pre = [np.tanh(ssn_layer_pars_dict['kappa_pre'])]
        kappa_post = [np.tanh(ssn_layer_pars_dict['kappa_post'])]
    else:
        kappa_pre = np.tanh(constant_pars.ssn_layer_pars.kappa_pre)
        kappa_post = np.tanh(constant_pars.ssn_layer_pars.kappa_post)

    # Unpack input
    training_pars = constant_pars.training_pars
    stimuli_pars = constant_pars.stimuli_pars

    # Initialise optimizer and set first stage accuracy threshold
    optimizer = optax.adam(training_pars.eta)
    if constant_pars.pretrain_pars.is_on:
        Nepochs = training_pars.epochs[0]
        first_stage_final_epoch = Nepochs
        if constant_pars.pretrain_pars.Nstages == 1:
            readout_state = optimizer.init(readout_pars_dict)
        else:
            first_stage_acc = constant_pars.pretrain_pars.acc_th
            readout_state = optimizer.init(readout_pars_dict)
    else:
        Nepochs = training_pars.epochs[1]
        first_stage_final_epoch = Nepochs
        first_stage_acc = training_pars.first_stage_acc
        readout_state = optimizer.init(readout_pars_dict)

    print(
        "epochs: {} ¦ learning rate: {} ¦ sig_noise: {} ¦ ref ori: {} ¦ offset: {} ¦ batch size {}".format(
            Nepochs,
            training_pars.eta,
            training_pars.sig_noise,
            stimuli_pars.ref_ori,
            stimuli_pars.offset,
            training_pars.batch_size,
        )
    )

    if results_filename:
        print("Saving results to ", results_filename)
    else:
        print("#### NOT SAVING! ####")

    ######## Two-stage training: 1) parameters of sigmoid layer 2) parameters of SSN layers #############
    # Define epoch indices where losses an accuracy are validated
    val_epochs = np.arange(1, Nepochs + 1, training_pars.validation_freq)

    start_time = time.time()

    for stage in range(1,3):
        if stage ==2:
            # Reinitialise optimizer
            ssn_layer_state = optimizer.init(ssn_layer_pars_dict)        

        for epoch in range(1, Nepochs + 1):
            # Calculate loss and gradient on training data
            train_loss, train_loss_all, train_acc, _, _, train_max_rate, grad = loss_and_grad_ori_discr(stimuli_pars, training_pars,ssn_layer_pars_dict, readout_pars_dict, constant_pars, stage, jit_on)
            
            # Store SGD_step output
            train_losses_all.append(train_loss_all.ravel())
            train_accs.append(train_acc)
            train_max_rates.append(train_max_rate)
            stages.append(stage)

            # VALIDATION
            if epoch in val_epochs:
                
                #### Calculate loss for testing data ####
                val_loss, val_loss_all , val_acc, _, _, _, _ = loss_and_grad_ori_discr(stimuli_pars, training_pars,ssn_layer_pars_dict, readout_pars_dict, constant_pars, stage, jit_on)

                # Store SGD_step output from validation
                if epoch == val_epochs[0] and stage==1:
                    val_losses_all = val_loss_all
                else:
                    val_losses_all = np.hstack((val_losses_all, val_loss_all))

                val_accs.append(val_acc)
                
                epoch_time = time.time() - start_time
                print("Training loss: {:.3f} ¦ Val loss: {:.3f} ¦ Train accuracy: {:.3f} ¦ Val accuracy: {:.3f} ¦ Epoch: {} ¦ Runtime: {} ".format(
                    train_loss, val_loss, train_acc, val_acc, epoch, epoch_time
                ))

            # Updating parameters
            if stage == 1:
                # Early stop in first stage of training
                if (epoch > 20
                    and np.mean(np.asarray(train_accs[-20:])) > first_stage_acc
                )  or epoch > 500:
                    print(
                        "Early stop: {} accuracy achieved at epoch {}".format(
                            first_stage_acc, epoch
                        )
                    )
                    # Update readout parameters
                    first_stage_final_epoch = epoch
                    updates, readout_state = optimizer.update(grad, readout_state)
                    readout_pars_dict = optax.apply_updates(readout_pars_dict, updates)
                    w_sig_temp=readout_pars_dict['w_sig']
                    w_sigs.append(w_sig_temp[w_indices_to_save])
                    b_sigs.append(readout_pars_dict['b_sig'])
                    # Exit first training loop
                    break
                else:
                    # Update readout parameters
                    updates, readout_state = optimizer.update(grad, readout_state)
                    readout_pars_dict = optax.apply_updates(readout_pars_dict, updates)
                    w_sig_temp=readout_pars_dict['w_sig']
                    w_sigs.append(w_sig_temp[w_indices_to_save])
                    b_sigs.append(readout_pars_dict['b_sig'])

            else:                    
                # Update ssn layer parameters
                updates, ssn_layer_state = optimizer.update(grad, ssn_layer_state)
                ssn_layer_pars_dict = optax.apply_updates(ssn_layer_pars_dict, updates)
            
                log_J_2x2_m.append(ssn_layer_pars_dict['log_J_2x2_m'].ravel())
                log_J_2x2_s.append(ssn_layer_pars_dict['log_J_2x2_s'].ravel())
                c_E.append(ssn_layer_pars_dict['c_E'])
                c_I.append(ssn_layer_pars_dict['c_I'])
                f_E.append(ssn_layer_pars_dict['f_E'])
                f_I.append(ssn_layer_pars_dict['f_I'])
                if 'kappa_pre' in ssn_layer_pars_dict:
                    kappa_pre.append(ssn_layer_pars_dict['kappa_pre'])
                    kappa_post.append(ssn_layer_pars_dict['kappa_post'])

    ############# SAVING and RETURN OUTPUT #############

    train_max_rates = np.vstack(np.asarray(train_max_rates))
    
    # Save results - both training and validation
    #Define epoch indices for training and validation
    epochs = np.arange(0, first_stage_final_epoch + Nepochs + 1)
    val_epochs_stage1 = val_epochs[val_epochs < first_stage_final_epoch + 1]
    val_epochs_stage2 = np.arange(first_stage_final_epoch + 1, first_stage_final_epoch + Nepochs + 1, training_pars.validation_freq)
    val_epochs_all = np.concatenate((val_epochs_stage1,val_epochs_stage2))
    
    # Create DataFrame to save
    
    if 'kappa_pre' in ssn_layer_pars_dict:
        df = make_dataframe(stages, epochs,val_epochs_all, train_accs, train_losses_all,train_max_rates, val_accs, val_losses_all, b_sigs,w_sigs, log_J_2x2_m, log_J_2x2_s, c_E, c_I, f_E, f_I, kappa_pre, kappa_post)
    else:
        df = make_dataframe(stages, epochs,val_epochs_all, train_accs, train_losses_all,train_max_rates, val_accs, val_losses_all, b_sigs, w_sigs, log_J_2x2_m, log_J_2x2_s, c_E, c_I, f_E, f_I)

    # Save the DataFrame to a CSV file: append if file exists, write header only if file doesn't exist
    if results_filename:
        file_exists = os.path.isfile(results_filename)
        df.to_csv(results_filename, mode='a', header=not file_exists, index=False)

    return df


def loss_and_grad_ori_discr(stimuli_pars, training_pars, ssn_layer_pars_dict, readout_pars_dict, constant_pars, stage,jit_on):

    # Generate noise
    noise_ref = generate_noise(
        training_pars.sig_noise, training_pars.batch_size, readout_pars_dict["w_sig"].shape[0]
    )
    noise_target = generate_noise(
        training_pars.sig_noise, training_pars.batch_size, readout_pars_dict["w_sig"].shape[0]
    )

    if constant_pars.pretrain_pars.is_on:
        # Create stimulus for middle layer: train_data has ref, target and label
        train_data = create_grating_pretraining(stimuli_pars, constant_pars.pretrain_pars, training_pars.batch_size)
        if stage == 1:
            training_loss_val_and_grad = jax.value_and_grad(batch_loss_ori_discr, argnums=1, has_aux=True)
        else:
            training_loss_val_and_grad = jax.value_and_grad(batch_loss_ori_discr, argnums=0, has_aux=True)
        [loss, [all_losses, accuracy, sig_input, sig_output, max_rates]], grad = training_loss_val_and_grad(
            ssn_layer_pars_dict,
            readout_pars_dict,
            constant_pars,
            train_data,
            noise_ref,
            noise_target,
            jit_on
        )
    else:
        # Create stimulus for middle layer: train_data has ref, target and label
        train_data = create_grating_pairs(stimuli_pars, training_pars.batch_size)
        if stage == 1:
            training_loss_val_and_grad = jax.value_and_grad(batch_loss_ori_discr, argnums=1, has_aux=True)
        else:
            training_loss_val_and_grad = jax.value_and_grad(batch_loss_ori_discr, argnums=0, has_aux=True)
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


def batch_loss_ori_discr(ssn_layer_pars_dict, readout_pars_dict, constant_pars, train_data, noise_ref, noise_target, jit_on=True):
    
    '''
    Run orientation discrimination task on given batch of data. Returns losses averaged over the trials within the batch. Function over which the gradient is taken.
    '''
    #Parallelize orientation discrimination task
    vmap_ori_discrimination = vmap(loss_ori_discr, in_axes = (None, None, None, {'ref':0, 'target':0, 'label':0}, 0, 0) )
    jit_ori_discrimination = jax.jit(vmap_ori_discrimination, static_argnums = [2])


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
    if constant_pars.pretrain_pars.is_on:
        true_accuracy = np.corrcoef(sig_output, train_data['label'])[0,1]
    else:
        true_accuracy = np.sum(train_data['label'] == pred_label)/len(train_data['label'])  
    
    return loss, [all_losses, true_accuracy, sig_input, sig_output, max_rates]


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
    if 'kappa_pre' in ssn_layer_pars_dict:
        kappa_pre = np.tanh(ssn_layer_pars_dict['kappa_pre'])
        kappa_post = np.tanh(ssn_layer_pars_dict['kappa_post'])
    else:
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
    # Create middle and superficial SSN layers *** this is something that would be great to change - to call the ssn classes from inside the training
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
        N_grid=constant_pars.grid_pars.gridsize_Nx
        r_ref_2D=np.reshape(r_ref,(N_grid,N_grid))
        r_target_2D=np.reshape(r_target,(N_grid,N_grid))
        N_readout=constant_pars.readout_grid_size[1]
        start=(N_grid-N_readout)//2
        r_ref_box = jax.lax.dynamic_slice(r_ref_2D, (start, start), (N_readout,N_readout)).ravel()
        r_target_box = jax.lax.dynamic_slice(r_target_2D, (start, start), (N_readout,N_readout)).ravel()        
    
    #Add noise
    r_ref_box = r_ref_box + noise_ref*np.sqrt(jax.nn.softplus(r_ref_box))
    r_target_box = r_target_box + noise_target*np.sqrt(jax.nn.softplus(r_target_box))
    
    #Calculate readout loss
    if pretraining:
        sig_input = np.dot(w_sig, r_ref_box) + b_sig  
        sig_output = sig_input
        loss_readout = np.mean(np.abs(sig_output-train_data['label']))
        pred_label = None
        loss_avg_dx = loss_pars.lambda_dx*(avg_dx_ref_mid + avg_dx_target_mid + avg_dx_ref_sup + avg_dx_target_sup)/4
        loss_r_max =  loss_pars.lambda_r_max*(r_max_ref_mid + r_max_target_mid + r_max_ref_sup + r_max_target_sup)/4
    else:
        #Multiply (reference - target) by sigmoid layer weights, add bias and apply sigmoid funciton
        sig_input = np.dot(w_sig, (r_ref_box - r_target_box)) + b_sig     
        sig_output = sigmoid(sig_input)
        #calculate readout loss and the predicted label
        loss_readout = binary_loss(train_data['label'], sig_output)
        pred_label = np.round(sig_output) 
        loss_avg_dx = loss_pars.lambda_dx*(avg_dx_ref_mid + avg_dx_target_mid + avg_dx_ref_sup + avg_dx_target_sup )/4
        loss_r_max =  loss_pars.lambda_r_max*(r_max_ref_mid + r_max_target_mid + r_max_ref_sup + r_max_target_sup )/4
    loss_w = loss_pars.lambda_w*(np.linalg.norm(w_sig)**2)
    loss_b = loss_pars.lambda_b*(b_sig**2)
        
    #Combine all losses    
    loss = loss_readout + loss_w + loss_b +  loss_avg_dx + loss_r_max
    all_losses = np.vstack((loss_readout, loss_avg_dx, loss_r_max, loss_w, loss_b, loss))
    
    return loss, all_losses, pred_label, sig_input, sig_output,  [max_E_mid, max_I_mid, max_E_sup, max_I_sup]


#binary cross entropy
def binary_loss(n, x):
    return -(n * np.log(x) + (1 - n) * np.log(1 - x))


def generate_noise(sig_noise,  batch_size, length):
    '''
    Creates vectors of neural noise. Function creates N vectors, where N = batch_size, each vector of length = length. 
    '''
    return sig_noise*numpy.random.randn(batch_size, length)


def make_dataframe(stages, epochs,val_epochs, train_accs, train_losses_all,train_max_rates, val_accs, val_losses_all,b_sigs,w_sigs, log_J_2x2_m, log_J_2x2_s,c_E,c_I,f_E,f_I,kappa_pre=None,kappa_post=None):
       
    #Create DataFrame and fill it with variables
    df = pd.DataFrame({
        'stage': stages,
        'epoch': epochs,
        'acc': train_accs,
        'val_acc' : None,
    })
    loss_names = ['loss_binary', 'loss_avg_dx', 'loss_r_max', 'loss_w_sig', 'loss_b_sig', 'loss_all']
    for i in range(len(train_losses_all[0])):
        df[loss_names[i]]=np.stack(train_losses_all)[:,i]
    max_rates_names = ['maxr_E_mid', 'maxr_I_mid', 'maxr_E_sup', 'maxr_I_sup']
    for i in range(len(train_max_rates.T)):
        df[max_rates_names[i]]=train_max_rates[:,i]
    
    df.loc[val_epochs, 'val_acc'] = val_accs        
    val_loss_names = ['val_loss_binary', 'val_loss_avg_dx', 'val_loss_r_max', 'val_loss_w_sig', 'val_loss_b_sig', 'val_loss_all']
    for i in range(len(val_losses_all)):
        df[val_loss_names[i]] = None
        df.loc[val_epochs, val_loss_names[i]] = val_losses_all[i,:]

    #Add trained parameters
    epochs_stage1=np.arange(len(b_sigs))
    epochs_stage2=np.arange(len(b_sigs),len(b_sigs)+len(c_E)-1) #first values are default values saved at epoch 0
    for i in range(len(np.stack(w_sigs).T)):
        weight_name = f'w_sig_{i+1}'
        df[weight_name] = None
        df.loc[epochs_stage1, weight_name] = np.stack(w_sigs)[:,i]
        df.loc[epochs_stage2, weight_name] = np.stack(w_sigs)[-1,i]
    df['b_sig'] = None
    df.loc[epochs_stage1,'b_sig'] = b_sigs
    df.loc[epochs_stage2,'b_sig'] = b_sigs[-1]

    J_m_names = ['logJ_m_EE', 'logJ_m_EI', 'logJ_m_IE', 'logJ_m_II']
    J_s_names = ['logJ_s_EE', 'logJ_s_EI', 'logJ_s_IE', 'logJ_s_II']
    for i in range(len(np.stack(log_J_2x2_m).T)):
        df[J_m_names[i]] = None
        df.loc[epochs_stage1, J_m_names[i]] = np.stack(log_J_2x2_m)[0,i]
        df.loc[epochs_stage2, J_m_names[i]] = np.stack(log_J_2x2_m)[1:,i]
    for i in range(len(np.stack(log_J_2x2_m).T)):
        df[J_s_names[i]] = None
        df.loc[epochs_stage1, J_s_names[i]] = np.stack(log_J_2x2_s)[0,i]
        df.loc[epochs_stage2, J_s_names[i]] = np.stack(log_J_2x2_s)[1:,i]
    df.loc[epochs_stage1,'c_E']=c_E[0]
    df.loc[epochs_stage2,'c_E']=c_E[1:]
    df.loc[epochs_stage1,'c_I']=c_I[0]
    df.loc[epochs_stage2,'c_I']=c_I[1:]
    df.loc[epochs_stage1,'f_E']=f_E[0]
    df.loc[epochs_stage2,'f_E']=f_E[1:]
    df.loc[epochs_stage1,'f_I']=f_I[0]
    df.loc[epochs_stage2,'f_I']=f_I[1:]
    if kappa_pre is not None:
        kappa_names=['kappa_preE','kappa_preI','kappa_postE','kappa_postI']
        for i in range(len(np.stack(kappa_pre).T)):
            df[kappa_names[i]] = None
            df.loc[epochs_stage1, kappa_names[i]] = np.stack(kappa_pre)[0,i]
            df.loc[epochs_stage2, kappa_names[i]] = np.stack(kappa_pre)[1:,i]
        for i in range(len(np.stack(kappa_pre).T)):
            df[kappa_names[2+i]] = None
            df.loc[epochs_stage1, kappa_names[2+i]] = np.stack(kappa_post)[0,i]
            df.loc[epochs_stage2, kappa_names[2+i]] = np.stack(kappa_post)[1:,i]

    return df      
