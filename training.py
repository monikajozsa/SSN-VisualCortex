import os
import jax
import jax.numpy as np
import optax
import time
import csv
import pandas as pd

from util import create_grating_pairs
from training_supp import training_loss, generate_noise
#from visualization import plot_w_sig, plot_max_rates

def SGD_step(stimuli_pars, training_pars, ssn_layer_pars_dict, readout_pars_dict, constant_pars, stage, jit_on):
    if stage == 1:
        loss_and_grad_func = jax.value_and_grad(training_loss, argnums=1, has_aux=True)
    else:
        loss_and_grad_func = jax.value_and_grad(training_loss, argnums=0, has_aux=True)

    # Create stimulus for middle layer
    train_data = create_grating_pairs(stimuli_pars, training_pars.batch_size)

    # Generate noise
    noise_ref = generate_noise(
        training_pars.sig_noise, training_pars.batch_size, readout_pars_dict["w_sig"].shape[0]
    )
    noise_target = generate_noise(
        training_pars.sig_noise, training_pars.batch_size, readout_pars_dict["w_sig"].shape[0]
    )

    # Compute loss and gradient
    [
        loss, [all_losses, accuracy, sig_input, sig_output, max_rates],
    ], grad = loss_and_grad_func(
        ssn_layer_pars_dict,
        readout_pars_dict,
        constant_pars,
        train_data,
        noise_ref,
        noise_target,
        jit_on
    )
    return loss, all_losses, accuracy, sig_input, sig_output, max_rates, grad

def train_model(
    ssn_layer_pars_dict,
    readout_pars_dict,
    constant_pars,
    results_filename=None,
    jit_on=True
):
    """
    Training function for two layer model in two stages: 
    First stage trains readout_pars_dict until accuracy reaches training_pars.first_stage_acc
    or for training_pars.epochs number of epochs if accuracy is not reached. Second stage trains ssn_layer_pars_dict for training_pars.epochs number of epochs
    Note that constant_pars includes grid_pars, stimuli_pars, conn_pars_m, conn_pars_s, filter_pars, ssn_ori_map, ssn_pars, ssn_layer_pars, conv_pars, loss_pars, training_pars.
    """
    
    # Initialize variables
    train_losses_all = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    train_accs = [None]
    train_max_rates = [np.array([0.0, 0.0, 0.0, 0.0])]
    val_losses_all = []
    val_accs = []

    w_sigs = [readout_pars_dict['w_sig'][:7]]
    b_sigs = [readout_pars_dict['b_sig']]
    log_J_2x2_m = [ssn_layer_pars_dict['log_J_2x2_m'].ravel()]
    log_J_2x2_s = [ssn_layer_pars_dict['log_J_2x2_s'].ravel()]
    c_E = [ssn_layer_pars_dict['c_E']]
    c_I = [ssn_layer_pars_dict['c_I']]
    f_E = [ssn_layer_pars_dict['f_E']]
    f_I = [ssn_layer_pars_dict['f_I']]
    kappa_pre = [ssn_layer_pars_dict['kappa_pre']]
    kappa_post = [ssn_layer_pars_dict['kappa_post']]

    # Unpack input
    training_pars = constant_pars.training_pars
    stimuli_pars = constant_pars.stimuli_pars
    first_stage_final_epoch = training_pars.epochs

    print(
        "epochs: {} ¦ learning rate: {} ¦ sig_noise: {} ¦ ref ori: {} ¦ offset: {} ¦ batch size {}".format(
            training_pars.epochs,
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
    val_epochs = np.arange(1, training_pars.epochs + 1, training_pars.validation_freq)

    # Initialise optimizer
    optimizer = optax.adam(training_pars.eta)
    readout_state = optimizer.init(readout_pars_dict)

    start_time = time.time()

    for stage in range(1,3):
        if stage ==2:
            # Reinitialise optimizer
            ssn_layer_state = optimizer.init(ssn_layer_pars_dict)        

        for epoch in range(1, training_pars.epochs + 1):
            # Calculate loss and gradient on training data
            train_loss, train_loss_all, train_acc, _, _, train_max_rate, grad = SGD_step(stimuli_pars, training_pars,ssn_layer_pars_dict, readout_pars_dict, constant_pars, stage, jit_on)
            
            # Store SGD_step output
            train_losses_all.append(train_loss_all.ravel())
            train_accs.append(train_acc)
            train_max_rates.append(train_max_rate)

            # VALIDATION
            if epoch in val_epochs:
                
                #### Calculate loss for testing data ####
                val_loss, val_loss_all , val_acc, _, _, _, _ = SGD_step(stimuli_pars, training_pars,ssn_layer_pars_dict, readout_pars_dict, constant_pars, stage, jit_on)

                # Store SGD_step output from validation
                if epoch == val_epochs[0] and stage==1:
                    val_losses_all = val_loss_all
                else:
                    val_losses_all = np.hstack((val_losses_all, val_loss_all))

                val_accs.append(val_acc)
                
                epoch_time = time.time() - start_time
                print("Training loss: {:.3f} ¦ Val loss: {:.3f} ¦ Train accuracy: {:.3f} ¦ Val accuracy: {:.3f} ¦ Epoch: {} ¦ Runtime: {}".format(
                    train_loss, val_loss, train_acc, val_acc, epoch, epoch_time
                ))

            # Updating parameters
            if stage == 1:
                # Early stop in first stage of training
                if (epoch > 20
                    and np.mean(np.asarray(train_accs[-20:])) > training_pars.first_stage_acc
                ):
                    print(
                        "Early stop: {} accuracy achieved at epoch {}".format(
                            training_pars.first_stage_acc, epoch
                        )
                    )
                    first_stage_final_epoch = epoch

                    # Exit first training loop
                    break
                else:
                    # Update readout parameters
                    updates, readout_state = optimizer.update(grad, readout_state)
                    readout_pars_dict = optax.apply_updates(readout_pars_dict, updates)
                    w_sigs.append(readout_pars_dict['w_sig'][:7])
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
                kappa_pre.append(ssn_layer_pars_dict['kappa_pre'])
                kappa_post.append(ssn_layer_pars_dict['kappa_post'])

    ############# SAVING and RETURN OUTPUT #############

    train_max_rates = np.vstack(np.asarray(train_max_rates))
    
    # Save results - both training and validation
    if results_filename:
        #Define epoch indices for training and validation
        epochs_stage1 = np.arange(0,first_stage_final_epoch+1)
        epochs_stage2 = np.arange(first_stage_final_epoch+1, first_stage_final_epoch+training_pars.epochs + 1)
        epochs = np.concatenate((epochs_stage1,epochs_stage2))
        val_epochs_stage1 = val_epochs[val_epochs < first_stage_final_epoch + 1]
        val_epochs_stage2 = np.arange(first_stage_final_epoch+1, first_stage_final_epoch+training_pars.epochs + 1, training_pars.validation_freq)
        val_epochs_all = np.concatenate((val_epochs_stage1,val_epochs_stage2))
        
        #Create DataFrame and fill it with variables
        df = pd.DataFrame({
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
        
        df.loc[val_epochs_all-1, 'val_acc'] = val_accs        
        val_loss_names = ['val_loss_binary', 'val_loss_avg_dx', 'val_loss_r_max', 'val_loss_w_sig', 'val_loss_b_sig', 'val_loss_all']
        for i in range(len(val_losses_all)):
            df[val_loss_names[i]] = None
            df.loc[val_epochs_all-1, val_loss_names[i]] = val_losses_all[i,:]

        #Add trained parameters
        for i in range(len(np.stack(w_sigs).T)):
            weight_name = f'w_sig_{i+1}'
            df[weight_name] = None
            df.loc[epochs_stage1, weight_name] = np.stack(w_sigs)[:,i]
        df['b_sig'] = None
        df.loc[epochs_stage1,'b_sig'] = b_sigs

        J_m_names = ['J_m_EE', 'J_m_EI', 'J_m_IE', 'J_m_II']
        J_s_names = ['J_s_EE', 'J_s_EI', 'J_s_IE', 'J_s_II']
        for i in range(len(np.stack(log_J_2x2_m).T)):
            df[J_m_names[i]] = None
            df.loc[epochs_stage1, J_m_names[i]] = np.stack(log_J_2x2_m)[0,i]
            df.loc[epochs_stage2-1, J_m_names[i]] = np.stack(log_J_2x2_m)[1:,i]
        for i in range(len(np.stack(log_J_2x2_m).T)):
            df[J_s_names[i]] = None
            df.loc[epochs_stage1, J_s_names[i]] = np.stack(log_J_2x2_s)[0,i]
            df.loc[epochs_stage2-1, J_s_names[i]] = np.stack(log_J_2x2_s)[1:,i]
        df['c_E'] = None
        df.loc[epochs_stage1,'c_E']=c_E[0]
        df.loc[epochs_stage2-1,'c_E']=c_E[1:]
        df['c_I'] = None
        df.loc[epochs_stage1,'c_I']=c_I[0]
        df.loc[epochs_stage2-1,'c_I']=c_I[1:]
        df['f_E'] = None
        df.loc[epochs_stage1,'f_E']=f_E[0]
        df.loc[epochs_stage2-1,'f_E']=f_E[1:]
        df['f_I'] = None
        df.loc[epochs_stage1,'f_I']=f_I[0]
        df.loc[epochs_stage2-1,'f_I']=f_I[1:]
        kappa_names=['kappa_preE','kappa_preI','kappa_postE','kappa_postI']
        for i in range(len(np.stack(kappa_pre).T)):
            df[kappa_names[i]] = None
            df.loc[epochs_stage1, kappa_names[i]] = np.stack(kappa_pre)[0,i]
            df.loc[epochs_stage2-1, kappa_names[i]] = np.stack(kappa_pre)[1:,i]
        for i in range(len(np.stack(kappa_pre).T)):
            df[kappa_names[2+i]] = None
            df.loc[epochs_stage1, kappa_names[2+i]] = np.stack(kappa_post)[0,i]
            df.loc[epochs_stage2-1, kappa_names[2+i]] = np.stack(kappa_post)[1:,i]

        # Save the DataFrame to a CSV file
        df.to_csv(results_filename, index=False)       

    return df
