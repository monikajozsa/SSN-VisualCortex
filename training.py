from training_supp import *

def new_two_stage_training(
    ssn_layer_pars,
    sigm_pars,
    training_pars,
    constant_ssn_pars,
    stimuli_pars,
    results_filename=None,
    second_eta=None,
    test_size=None,
    results_dir=None,
    early_stop=0.7,
    extra_stop=20,
    ssn_ori_map=None,):
    """
    Training function for two layer model in two stages: once readout layer is trained until early_stop (first stage), extra epochs are ran without updating, and then SSN layer parameters are trained (second stage). Second stage is nested in first stage. Accuracy is calculated on testing set before training and after first stage.
    Inputs:
        individual parameters of the model
    Outputs:
    """

    # Define parameters from input data classes
    J_2x2_m = ssn_layer_pars.J_2x2_m
    J_2x2_s = ssn_layer_pars.J_2x2_s
    s_2x2_s = ssn_layer_pars.s_2x2_s
    sigma_oris_s = ssn_layer_pars.sigma_oris
    kappa_pre = ssn_layer_pars.kappa_pre
    kappa_post = ssn_layer_pars.kappa_post
    c_E = ssn_layer_pars.c_E
    c_I = ssn_layer_pars.c_I
    f_E = ssn_layer_pars.f_E
    f_I = ssn_layer_pars.f_I
    w_sig = sigm_pars.w_sig
    b_sig = sigm_pars.b_sig
    epochs = training_pars.epochs
    epochs_to_save =  np.insert((np.unique(np.linspace(1 , epochs, training_pars.num_epochs_to_save).astype(int))), 0 , 0)
    batch_size = training_pars.batch_size
    eta = training_pars.eta
    sig_noise = training_pars.sig_noise
    noise_type = training_pars.noise_type
    ref_ori = stimuli_pars.ref_ori
    offset = stimuli_pars.offset

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
    save_w_sigs.append(w_sig[:5])
    epoch = 0

    # Take logs of parameters
    logJ_2x2_s = take_log(J_2x2_s)
    logs_2x2 = np.log(s_2x2_s)
    logJ_2x2_m = take_log(J_2x2_m)
    logJ_2x2 = [logJ_2x2_m, logJ_2x2_s]
    sigma_oris = np.log(sigma_oris_s)

    # Define jax seed
    constant_ssn_pars["key"] = random.PRNGKey(numpy.random.randint(0, 10000))

    if ssn_ori_map == None:
        # Initialise networks
        print("Creating new orientation map")
        ssn_mid = SSN2DTopoV1_ONOFF_local(
            ssn_pars=constant_ssn_pars["ssn_pars"],
            grid_pars=constant_ssn_pars["grid_pars"],
            conn_pars=constant_ssn_pars["conn_pars_m"],
            filter_pars = constant_ssn_pars["filter_pars"],
            J_2x2 = J_2x2_m, 
            gE = constant_ssn_pars['gE'][0], 
            gI = constant_ssn_pars['gI'][0]
        )
        constant_ssn_pars["ssn_mid_ori_map"] = ssn_mid.ori_map
        constant_ssn_pars["ssn_sup_ori_map"] = ssn_mid.ori_map
    else:
        print("Loading orientation map")
        constant_ssn_pars["ssn_mid_ori_map"] = ssn_ori_map
        constant_ssn_pars["ssn_sup_ori_map"] = ssn_ori_map

    # Reassemble parameters into corresponding dictionaries
    constant_ssn_pars["logs_2x2"] = logs_2x2
    constant_ssn_pars["train_ori"] = ref_ori
    constant_ssn_pars["sigma_oris"] = sigma_oris
    constant_ssn_pars["f_E"] = f_E
    constant_ssn_pars["f_I"] = f_I
    constant_ssn_pars["c_E"] = c_E
    constant_ssn_pars["c_I"] = c_I
    readout_pars = dict(w_sig=w_sig, b_sig=b_sig)
    ssn_layer_pars_dict = dict(logJ_2x2=logJ_2x2, kappa_pre=kappa_pre, kappa_post=kappa_post)

    print(constant_ssn_pars["ssn_mid_ori_map"])

    test_size = batch_size if test_size is None else test_size

    # Initialise optimizer
    optimizer = optax.adam(eta)
    readout_state = optimizer.init(readout_pars)

    print(
        "Training model for {} epochs  with learning rate {}, sig_noise {} at offset {}, lam_w {}, batch size {}, noise_type {}".format(
            epochs,
            eta,
            sig_noise,
            offset,
            constant_ssn_pars["loss_pars"].lambda_w,
            batch_size,
            noise_type,
        )
    )
    print(
        "Loss parameters dx {}, w {} ".format(
            constant_ssn_pars["loss_pars"].lambda_dx,
            constant_ssn_pars["loss_pars"].lambda_w,
        )
    )

    epoch_c = epochs
    loop_epochs = epochs
    flag = True

    # Initialise csv file
    if results_filename:
        print("Saving results to csv ", results_filename)
    else:
        print("#### NOT SAVING! ####")

    loss_and_grad_readout = jax.value_and_grad(loss, argnums=1, has_aux=True)
    loss_and_grad_ssn = jax.value_and_grad(loss, argnums=0, has_aux=True)

    while epoch < loop_epochs + 1:
        start_time = time.time()
        epoch_loss = 0

        # Load next batch of data and convert
        train_data = create_data(stimuli_pars, number=batch_size)

        if epoch == epoch_c + extra_stop:
            debug_flag = True
        else:
            debug_flag = False

        # Generate noise
        constant_ssn_pars = generate_noise(
            constant_ssn_pars,
            sig_noise=sig_noise,
            batch_size=batch_size,
            length=w_sig.shape[0],
        )

        # Compute loss and gradient
        [
            epoch_loss,
            [epoch_all_losses, train_true_acc, train_delta_x, train_x, train_r_ref],
        ], grad = loss_and_grad_readout(
            ssn_layer_pars_dict, readout_pars, constant_ssn_pars, train_data, debug_flag
        )

        training_losses.append(epoch_loss)
        if epoch == 0:
            all_losses = epoch_all_losses
        else:
            all_losses = np.hstack((all_losses, epoch_all_losses))
        train_accs.append(train_true_acc)
        train_sig_input.append(train_delta_x)
        train_sig_output.append(train_x)
        r_refs.append(train_r_ref)

        epoch_time = time.time() - start_time

        # Save the parameters given a number of epochs
        if epoch in epochs_to_save:
            # Evaluate model
            test_data = create_data(stimuli_pars, number=test_size)

            start_time = time.time()
            # Compute loss and gradient
            constant_ssn_pars = generate_noise(
                constant_ssn_pars,
                sig_noise=sig_noise,
                batch_size=batch_size,
                length=w_sig.shape[0],
            )

            [
                val_loss,
                [val_all_losses, true_acc, val_delta_x, val_x, _],
            ], _ = loss_and_grad_readout(
                ssn_layer_pars_dict, readout_pars, constant_ssn_pars, test_data
            )
            val_time = time.time() - start_time

            print(
                "Training loss: {} ¦ Validation -- loss: {}, true accuracy: {}, at epoch {}, (time {}, {}), ".format(
                    epoch_loss, val_loss, true_acc, epoch, epoch_time, val_time
                )
            )

            if epoch % 50 == 0:
                print(
                    "Training accuracy: {}, all losses{}".format(
                        np.mean(np.asarray(train_accs[-20:])), epoch_all_losses
                    )
                )
            val_loss_per_epoch.append([val_loss, int(epoch)])
            val_sig_input.append([val_delta_x, epoch])
            val_sig_output.append(val_x)
            val_accs.append(true_acc)

            if results_filename:
                save_params = save_params_dict_two_stage(
                    ssn_layer_pars_dict, readout_pars, true_acc, epoch
                )

                # Initialise results file
                if epoch == 0:
                    results_handle = open(results_filename, "w")
                    results_writer = csv.DictWriter(
                        results_handle, fieldnames=save_params.keys(), delimiter=","
                    )
                    results_writer.writeheader()

                results_writer.writerow(save_params)

        # Early stop in first stage of training
        if epoch > 20 and flag and np.mean(np.asarray(train_accs[-20:])) > early_stop:
            epoch_c = epoch
            print(
                "Early stop: {} accuracy achieved at epoch {}".format(early_stop, epoch)
            )
            loop_epochs = epoch_c + extra_stop
            save_dict = dict(training_accuracy=train_true_acc)
            save_dict.update(readout_pars)

            flag = False

        # Only update parameters before criterion
        if epoch < epoch_c:
            updates, readout_state = optimizer.update(grad, readout_state)
            readout_pars = optax.apply_updates(readout_pars, updates)
            save_w_sigs.append(readout_pars["w_sig"][:5])

        # Start second stage of training after reaching criterion or after given number of epochs
        if (flag == False and epoch >= epoch_c + extra_stop) or (
            flag == True and epoch == loop_epochs
        ):
            # Final save before second stagenn
            if results_filename:
                save_params = save_params_dict_two_stage(
                    ssn_layer_pars_dict, readout_pars, true_acc, epoch
                )
                results_writer.writerow(save_params)

            final_epoch = epoch
            print("Entering second stage at epoch {}".format(epoch))

            #############START TRAINING NEW STAGE ##################################

            # Initialise second optimizer
            ssn_layer_state = optimizer.init(ssn_layer_pars_dict)
            epoch = 1
            second_eta = eta if second_eta is None else second_eta

            for epoch in range(epoch, epochs + 1):
                # Load next batch of data and convert
                train_data = create_data(stimuli_pars, number=batch_size)

                # Compute loss and gradient
                constant_ssn_pars = generate_noise(
                    constant_ssn_pars,
                    sig_noise=sig_noise,
                    batch_size=batch_size,
                    length=w_sig.shape[0],
                )
                [
                    epoch_loss,
                    [
                        epoch_all_losses,
                        train_true_acc,
                        train_delta_x,
                        train_x,
                        train_r_ref,
                    ],
                ], grad = loss_and_grad_ssn(
                    ssn_layer_pars_dict,
                    readout_pars,
                    constant_ssn_pars,
                    train_data,
                    debug_flag,
                )

                all_losses = np.hstack((all_losses, epoch_all_losses))
                training_losses.append(epoch_loss)
                train_accs.append(train_true_acc)
                train_sig_input.append(train_delta_x)
                train_sig_output.append(train_x)
                r_refs.append(train_r_ref)

                # Save the parameters given a number of epochs
                if epoch in epochs_to_save:
                    # Evaluate model
                    test_data = create_data(stimuli_pars, number=test_size)

                    start_time = time.time()
                    constant_ssn_pars = generate_noise(
                        constant_ssn_pars,
                        sig_noise=sig_noise,
                        batch_size=batch_size,
                        length=w_sig.shape[0],
                    )
                    [
                        val_loss,
                        [val_all_losses, true_acc, val_delta_x, val_x, _],
                    ], _ = loss_and_grad_ssn(
                        ssn_layer_pars_dict, readout_pars, constant_ssn_pars, test_data
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

                    val_loss_per_epoch.append([val_loss, epoch + final_epoch])
                    val_sig_input.append([val_delta_x, epoch + final_epoch])
                    val_sig_output.append(val_x)

                    if results_filename:
                        save_params = save_params_dict_two_stage(
                            ssn_layer_pars_dict,
                            readout_pars,
                            true_acc,
                            epoch=epoch + final_epoch,
                        )
                        results_writer.writerow(save_params)

                # Update parameters
                updates, ssn_layer_state = optimizer.update(grad, ssn_layer_state)
                ssn_layer_pars_dict = optax.apply_updates(ssn_layer_pars_dict, updates)

                # OPTION TWO - only train values for E post
                if np.shape(kappa_post) != (2,):
                    ssn_layer_pars_dict["kappa_pre"] = (
                        ssn_layer_pars_dict["kappa_pre"].at[1, :].set(kappa_pre[1, :])
                    )
                    ssn_layer_pars_dict["kappa_post"] = (
                        ssn_layer_pars_dict["kappa_post"].at[1, :].set(kappa_post[1, :])
                    )
                    ssn_layer_pars_dict["sigma_oris"] = (
                        ssn_layer_pars_dict["sigma_oris"].at[1, :].set(sigma_oris[1, :])
                    )

                if epoch == 1:
                    save_dict = dict(training_accuracy=train_true_acc)
                    save_dict.update(readout_pars)
                    # util.save_h5(os.path.join(results_dir+'dict_4'), save_dict)

            final_epoch_2 = epoch + final_epoch

            break
        ################################################################################

        epoch += 1

    save_w_sigs = np.asarray(np.vstack(save_w_sigs))
    visualization.plot_w_sig(
        save_w_sigs,
        epochs_to_save[: len(save_w_sigs)],
        epoch_c,
        save=os.path.join(results_dir + "_w_sig_evolution"),
    )

    if flag == False:
        epoch_c = [epoch_c, extra_stop, final_epoch_2]
    r_refs = np.vstack(np.asarray(r_refs))

    # Plot maximum rates achieved during training
    visualization.plot_max_rates(
        r_refs, epoch_c=epoch_c, save=os.path.join(results_dir + "_max_rates")
    )

    return (
        [ssn_layer_pars_dict, readout_pars],
        np.vstack([val_loss_per_epoch]),
        all_losses,
        train_accs,
        train_sig_input,
        train_sig_output,
        val_sig_input,
        val_sig_output,
        epoch_c,
        save_w_sigs,
    )

