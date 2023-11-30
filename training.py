from training_supp import training_loss, generate_noise, save_trained_params
import jax
import jax.numpy as np
from util import take_log, create_grating_pairs
import optax
import time
import csv

 
def train_model(
    ssn_layer_pars_dict,
    readout_pars_dict,
    constant_pars,
    results_filename=None,
    jit_on=True
):
    """
    Training function for two layer model in two stages: 
    first stage trains readout_pars_dict until accuracy reaches training_pars.first_stage_acc
    or for training_pars.epochs number of epochs if accuracy is not reached, 
    second stage trains ssn_layer_pars_dict for training_pars.epochs number of epochs
    constant_pars: grid_pars, stimuli_pars, conn_pars_m, conn_pars_s, filter_pars, ssn_ori_map, ssn_pars, ssn_layer_pars, conv_pars, loss_pars, training_pars
    """
    training_pars = constant_pars.training_pars
    stimuli_pars = constant_pars.stimuli_pars
    # Initialize loss
    val_loss_per_epoch = []
    training_losses = []
    train_accs = []
    train_sig_input = []
    train_sig_output = []
    val_sig_input = []
    val_sig_output = []
    val_accs = []
    r_refs = [] 
    save_w_sigs = []
    first_stage_final_epoch = training_pars.epochs
    save_w_sigs.append(readout_pars_dict["w_sig"][:5])

    # Take logs of parameters
    ssn_layer_pars_dict["log_J_2x2_m"] = take_log(ssn_layer_pars_dict["J_2x2_m"])
    ssn_layer_pars_dict["log_J_2x2_s"] = take_log(ssn_layer_pars_dict["J_2x2_s"])

    epochs_to_save =  np.linspace(1, training_pars.epochs, training_pars.num_epochs_to_save).astype(int)
    batch_size = training_pars.batch_size

    # Initialise optimizer
    optimizer = optax.adam(training_pars.eta)
    readout_state = optimizer.init(readout_pars_dict)

    # Gradient function for gradient descent algorithm
    loss_and_grad_readout = jax.value_and_grad(training_loss, argnums=1, has_aux=True)
    loss_and_grad_ssn = jax.value_and_grad(training_loss, argnums=0, has_aux=True)

    print(
        "epochs: {} learning rate: {} sig_noise: {} ref ori: {} offset: {} batch size {}".format(
            training_pars.epochs,
            training_pars.eta,
            training_pars.sig_noise,
            stimuli_pars.ref_ori,
            stimuli_pars.offset,
            batch_size,
        )
    )

    if results_filename:
        print("Saving results to csv ", results_filename)
    else:
        print("#### NOT SAVING! ####")

    ######## FIRST STAGE OF TRAINING: SIGMOID LAYER PARAMETERS #############

    for epoch in range(1, training_pars.epochs + 1):
        start_time = time.time()

        # Create stimulus for middle layer
        train_data = create_grating_pairs(stimuli_pars, batch_size)

        # Generate noise
        noise_ref = generate_noise(
            training_pars.sig_noise, batch_size, readout_pars_dict["w_sig"].shape[0]
        )
        noise_target = generate_noise(
            training_pars.sig_noise, batch_size, readout_pars_dict["w_sig"].shape[0]
        )

        # Compute loss and gradient
        [
            epoch_loss,
            [epoch_all_losses, train_true_acc, train_delta_x, train_x, train_r_ref],
        ], grad = loss_and_grad_readout(
            ssn_layer_pars_dict,
            readout_pars_dict,
            constant_pars,
            train_data,
            noise_ref,
            noise_target,
            jit_on
        )

        if epoch == 1:
            all_losses = epoch_all_losses
        else:
            all_losses = np.hstack((all_losses, epoch_all_losses))

        training_losses.append(epoch_loss)
        train_accs.append(train_true_acc)
        train_sig_input.append(train_delta_x)
        train_sig_output.append(train_x)
        r_refs.append(train_r_ref)

        epoch_time = time.time() - start_time

        # Save the parameters given a number of epochs
        if epoch in epochs_to_save:
            # Evaluate model
            test_data = create_grating_pairs(
                stimuli_pars=stimuli_pars, batch_size=batch_size
            )

            # Generate noise
            noise_ref = generate_noise(
                training_pars.sig_noise, batch_size, readout_pars_dict["w_sig"].shape[0]
            )
            noise_target = generate_noise(
                training_pars.sig_noise, batch_size, readout_pars_dict["w_sig"].shape[0]
            )

            start_time = time.time()

            # Calculate loss for testing data
            [
                val_loss,
                [ _ , true_acc, val_delta_x, val_x, _],
            ], _ = loss_and_grad_readout(
                ssn_layer_pars_dict,
                readout_pars_dict,
                constant_pars,
                test_data,
                noise_ref,
                noise_target,
                jit_on
            )
            val_time = time.time() - start_time

            print(
                "Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, at epoch {}, (time {}, {}), ".format(
                    epoch_loss, val_loss, true_acc, epoch, epoch_time, val_time
                )
            )

            # Every 50 epochs print individual values of the loss
            if epoch % 50 == 0:
                print(
                    "Training accuracy: {}, all losses{}".format(
                        np.mean(np.asarray(train_accs[-20:])), epoch_all_losses
                    )
                )

            # Save validation loss
            val_loss_per_epoch.append([val_loss, int(epoch)])
            val_sig_input.append([val_delta_x, epoch])
            val_sig_output.append(val_x)
            val_accs.append(true_acc)

            # Save results
            if results_filename:
                save_params = save_trained_params(
                    ssn_layer_pars_dict, readout_pars_dict, true_acc, epoch
                )

                # Initialise results file
                if epoch == 1:
                    results_handle = open(results_filename, "w")
                    results_writer = csv.DictWriter(
                        results_handle, fieldnames=save_params.keys(), delimiter=","
                    )
                    results_writer.writeheader()

                results_writer.writerow(save_params)

        # Early stop in first stage of training
        if (
            epoch > 20
            and np.mean(np.asarray(train_accs[-20:])) > training_pars.first_stage_acc
        ):
            print(
                "Early stop: {} accuracy achieved at epoch {}".format(
                    training_pars.first_stage_acc, epoch
                )
            )
            first_stage_final_epoch = epoch

            # Save final parameters
            if results_filename:
                save_params = save_trained_params(
                    ssn_layer_pars_dict, readout_pars_dict, true_acc, epoch
                )
                results_writer.writerow(save_params)
            # Exit first training loop
            break

        # Update readout parameters
        updates, readout_state = optimizer.update(grad, readout_state)
        readout_pars_dict = optax.apply_updates(readout_pars_dict, updates)
        save_w_sigs.append(readout_pars_dict["w_sig"][:5])

    ############# SECOND STAGE OF TRAINING: Middle and superficial layer parameters #############

    print("Entering second stage at epoch {}".format(epoch))
    # Restart number of epochs

    # Reinitialize optimizer for second stage
    ssn_layer_state = optimizer.init(ssn_layer_pars_dict)

    for epoch in range(1, training_pars.epochs + 1):
        # Generate next batch of data
        train_data = create_grating_pairs(
            stimuli_pars=stimuli_pars, batch_size=batch_size
        )

        # Generate noise
        noise_ref = generate_noise(
            training_pars.sig_noise, batch_size, readout_pars_dict["w_sig"].shape[0]
        )
        noise_target = generate_noise(
            training_pars.sig_noise, batch_size, readout_pars_dict["w_sig"].shape[0]
        )

        # Run model and calculate gradient
        [
            epoch_loss,
            [epoch_all_losses, train_true_acc, train_delta_x, train_x, train_r_ref],
        ], grad = loss_and_grad_ssn(
            ssn_layer_pars_dict,
            readout_pars_dict,
            constant_pars,
            train_data,
            noise_ref,
            noise_target,
        )

        # Save training losses
        all_losses = np.hstack((all_losses, epoch_all_losses))
        training_losses.append(epoch_loss)
        train_accs.append(train_true_acc)
        train_sig_input.append(train_delta_x)
        train_sig_output.append(train_x)
        r_refs.append(train_r_ref)

        # Save the parameters given a number of epochs
        if epoch in epochs_to_save:
            # Evaluate model
            test_data = create_grating_pairs(
                stimuli_pars=stimuli_pars, batch_size=batch_size
            )
            # Generate noise
            noise_ref = generate_noise(
                training_pars.sig_noise, batch_size, readout_pars_dict["w_sig"].shape[0]
            )
            noise_target = generate_noise(
                training_pars.sig_noise, batch_size, readout_pars_dict["w_sig"].shape[0]
            )

            start_time = time.time()

            [
                val_loss,
                [ _ , true_acc, val_delta_x, val_x, _],
            ], _ = loss_and_grad_ssn(
                ssn_layer_pars_dict,
                readout_pars_dict,
                constant_pars,
                test_data,
                noise_ref,
                noise_target,
            )
            val_time = time.time() - start_time
            print(
                "Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, at epoch {}, (time {}, {})".format(
                    epoch_loss, val_loss, true_acc, epoch, epoch_time, val_time
                )
            )

            if epoch % 50 == 0 or epoch == 1:
                print(
                    "Training accuracy: {}, all losses{}".format(
                        train_true_acc, epoch_all_losses
                    )
                )

            val_loss_per_epoch.append([val_loss, epoch + first_stage_final_epoch])
            val_sig_input.append([val_delta_x, epoch + first_stage_final_epoch])
            val_sig_output.append(val_x)

            if results_filename:
                save_params = save_trained_params(
                    ssn_layer_pars_dict,
                    readout_pars_dict,
                    true_acc,
                    epoch=epoch + first_stage_final_epoch,
                )
                results_writer.writerow(save_params)

        # Update parameters
        updates, ssn_layer_state = optimizer.update(grad, ssn_layer_state)
        ssn_layer_pars_dict = optax.apply_updates(ssn_layer_pars_dict, updates)

    # Save transition epochs to plot losses and accuracy
    epochs_plot = first_stage_final_epoch

    # Plot maximum rates achieved during training
    r_refs = np.vstack(np.asarray(r_refs))
    save_w_sigs = np.asarray(np.vstack(save_w_sigs))

    return (
        [ssn_layer_pars_dict, readout_pars_dict],
        np.vstack([val_loss_per_epoch]),
        all_losses,
        train_accs,
        train_sig_input,
        train_sig_output,
        val_sig_input,
        val_sig_output,
        epochs_plot,
        save_w_sigs,
    )
