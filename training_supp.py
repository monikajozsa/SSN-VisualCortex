import matplotlib.pyplot as plt
import jax
from jax import random
import jax.numpy as np
from jax import vmap
from torch.utils.data import DataLoader

from util import create_gratings, sep_exponentiate, constant_to_vec
from model import two_layer_model, evaluate_model_response


def create_data(stimuli_pars, n_trials=100):
    """
    Create data for given jitter and noise value for testing (not dataloader)
    """
    data = create_gratings(stimuli_pars=stimuli_pars, n_trials=n_trials)
    train_data = next(iter(DataLoader(data, batch_size=len(data), shuffle=False)))
    train_data["ref"] = train_data["ref"].numpy()
    train_data["target"] = train_data["target"].numpy()
    train_data["label"] = train_data["label"].numpy()

    return train_data


def generate_noise(
    constant_ssn_pars, sig_noise, batch_size, length, noise_type="poisson"
):
    constant_ssn_pars["key"], _ = random.split(constant_ssn_pars["key"])
    constant_ssn_pars["noise_ref"] = sig_noise * jax.random.normal(
        constant_ssn_pars["key"], shape=(batch_size, length)
    )
    constant_ssn_pars["key"], _ = random.split(constant_ssn_pars["key"])
    constant_ssn_pars["noise_target"] = sig_noise * jax.random.normal(
        constant_ssn_pars["key"], shape=(batch_size, length)
    )
    return constant_ssn_pars


def test_accuracy(
    ssn_layer_pars,
    readout_pars,
    constant_ssn_pars,
    stimuli_pars,
    offset,
    ref_ori,
    sig_noise,
    save=None,
    number_trials=5,
    batch_size=5,
):
    """
    Given network parameters, function generates random trials of data and calculates the accuracy per batch.
    Input:
        network parameters, number of trials and batch size of each trial
    Output:
        histogram of accuracies

    """

    all_accs = []

    for i in range(number_trials):
        testing_data = create_data(stimuli_pars, number=batch_size)

        constant_ssn_pars = generate_noise(
            constant_ssn_pars,
            sig_noise=sig_noise,
            batch_size=batch_size,
            length=readout_pars["w_sig"].shape[0],
        )

        _, _, pred_label, _, _, _ = model(
            ssn_layer_pars=ssn_layer_pars,
            readout_pars=readout_pars,
            constant_ssn_pars=constant_ssn_pars,
            data=testing_data,
            debug_flag=True,
        )

        true_accuracy = np.sum(testing_data["label"] == pred_label) / len(
            testing_data["label"]
        )
        all_accs.append(true_accuracy)

    plt.hist(all_accs)
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")

    if save:
        plt.savefig(save + ".png")

    plt.show()
    plt.close()


jitted_model = jax.jit(
    two_layer_model,
    static_argnums=[14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29],
)


# Vmap implementation of model function
vmap_model_jit = vmap(
    jitted_model,
    in_axes=(
        [None, None],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {"ref": 0, "target": 0, "label": 0},
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        0,
        0,
        None,
        None,
        None,
    ),
)

vmap_model = vmap(
    two_layer_model,
    in_axes=(
        [None, None],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {"ref": 0, "target": 0, "label": 0},
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        0,
        0,
        None,
        None,
        None,
    ),
)


def model(ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag=False):
    """
    Wrapper function for model.
    Inputs:
        parameters assembled  into dictionaries
    Output:
        output of model using unwrapped parameters
    """

    # Obtain variables from dictionaries
    logJ_2x2 = ssn_layer_pars["logJ_2x2"]
    c_E = constant_ssn_pars["c_E"]
    c_I = constant_ssn_pars["c_I"]
    f_E = constant_ssn_pars["f_E"]
    f_I = constant_ssn_pars["f_I"]

    log_sigma_oris = constant_ssn_pars["log_sigma_oris"]
    kappa_pre = ssn_layer_pars["kappa_pre"]
    kappa_post = ssn_layer_pars["kappa_post"]

    w_sig = readout_pars["w_sig"]
    b_sig = readout_pars["b_sig"]

    ssn_mid_ori_map = constant_ssn_pars["ssn_mid_ori_map"]
    logs_2x2 = constant_ssn_pars["logs_2x2"]
    ssn_pars = constant_ssn_pars["ssn_pars"]
    grid_pars = constant_ssn_pars["grid_pars"]
    conn_pars_m = constant_ssn_pars["conn_pars_m"]
    conn_pars_s = constant_ssn_pars["conn_pars_s"]
    gE_m = constant_ssn_pars["gE"][0]
    gE_s = constant_ssn_pars["gE"][1]
    gI_m = constant_ssn_pars["gI"][0]
    gI_s = constant_ssn_pars["gI"][1]
    filter_pars = constant_ssn_pars["filter_pars"]
    conv_pars = constant_ssn_pars["conv_pars"]
    loss_pars = constant_ssn_pars["loss_pars"]
    noise_ref = constant_ssn_pars["noise_ref"]
    noise_target = constant_ssn_pars["noise_target"]
    noise_type = constant_ssn_pars["noise_type"]
    train_ori = constant_ssn_pars["train_ori"]

    return vmap_model_jit(
        logJ_2x2,
        logs_2x2,
        c_E,
        c_I,
        f_E,
        f_I,
        w_sig,
        b_sig,
        log_sigma_oris,
        kappa_pre,
        kappa_post,
        ssn_mid_ori_map,
        ssn_mid_ori_map,
        data,
        ssn_pars,
        grid_pars,
        conn_pars_m,
        conn_pars_s,
        gE_m,
        gI_m,
        gE_s,
        gI_s,
        filter_pars,
        conv_pars,
        loss_pars,
        noise_ref,
        noise_target,
        noise_type,
        train_ori,
        debug_flag,
    )


def loss(ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag=False):
    """
    Function to take gradient with respect to. Output returned as two variables (jax grad takes gradient with respect to first output)
    Inputs:
        parameters assembled into dictionaries
    Ouputs:
        total loss to take gradient with respect to
    """

    total_loss, all_losses, pred_label, sig_input, x, max_rates = model(
        ssn_layer_pars=ssn_layer_pars,
        readout_pars=readout_pars,
        constant_ssn_pars=constant_ssn_pars,
        data=data,
        debug_flag=debug_flag,
    )
    loss = np.mean(total_loss)
    all_losses = np.mean(all_losses, axis=0)
    max_rates = [item.max() for item in max_rates]
    true_accuracy = np.sum(data["label"] == pred_label) / len(data["label"])

    return loss, [all_losses, true_accuracy, sig_input, x, max_rates]


def save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch):
    """
    Assemble trained parameters and epoch information into single dictionary for saving
    Inputs:
        dictionaries containing trained parameters
        other epoch parameters (accuracy, epoch number)
    Outputs:
        single dictionary concatenating all information to be saved
    """

    save_params = {}
    save_params = dict(epoch=epoch, val_accuracy=true_acc)

    J_2x2_m = sep_exponentiate(ssn_layer_pars["logJ_2x2"][0])
    Jm = dict(
        J_EE_m=J_2x2_m[0, 0],
        J_EI_m=J_2x2_m[0, 1],
        J_IE_m=J_2x2_m[1, 0],
        J_II_m=J_2x2_m[1, 1],
    )

    J_2x2_s = sep_exponentiate(ssn_layer_pars["logJ_2x2"][1])
    Js = dict(
        J_EE_s=J_2x2_s[0, 0],
        J_EI_s=J_2x2_s[0, 1],
        J_IE_s=J_2x2_s[1, 0],
        J_II_s=J_2x2_s[1, 1],
    )

    save_params.update(Jm)
    save_params.update(Js)

    if "c_E" in ssn_layer_pars.keys():
        save_params["c_E"] = ssn_layer_pars["c_E"]
        save_params["c_I"] = ssn_layer_pars["c_I"]

    if "sigma_oris" in ssn_layer_pars.keys():
        if len(ssn_layer_pars["sigma_oris"]) == 1:
            # save_params[key] = np.exp(ssn_layer_pars[key]) #
            print(
                "MJ error message: save_params[key] = np.exp(ssn_layer_pars[key]) would not work as key is undefined!"
            )
        elif np.shape(ssn_layer_pars["sigma_oris"]) == (2, 2):
            save_params["sigma_orisEE"] = np.exp(ssn_layer_pars["sigma_oris"][0, 0])
            save_params["sigma_orisEI"] = np.exp(ssn_layer_pars["sigma_oris"][0, 1])
        else:
            sigma_oris = dict(
                sigma_orisE=np.exp(ssn_layer_pars["sigma_oris"][0]),
                sigma_orisI=np.exp(ssn_layer_pars["sigma_oris"][1]),
            )
            save_params.update(sigma_oris)

    if "kappa_pre" in ssn_layer_pars.keys():
        if np.shape(ssn_layer_pars["kappa_pre"]) == (2, 2):
            save_params["kappa_preEE"] = np.tanh(ssn_layer_pars["kappa_pre"][0, 0])
            save_params["kappa_preEI"] = np.tanh(ssn_layer_pars["kappa_pre"][0, 1])
            save_params["kappa_postEE"] = np.tanh(ssn_layer_pars["kappa_post"][0, 0])
            save_params["kappa_postEI"] = np.tanh(ssn_layer_pars["kappa_post"][0, 1])

        else:
            save_params["kappa_preE"] = np.tanh(ssn_layer_pars["kappa_pre"][0])
            save_params["kappa_preI"] = np.tanh(ssn_layer_pars["kappa_pre"][1])
            save_params["kappa_postE"] = np.tanh(ssn_layer_pars["kappa_post"][0])
            save_params["kappa_postI"] = np.tanh(ssn_layer_pars["kappa_post"][1])

    if "f_E" in ssn_layer_pars.keys():
        save_params["f_E"] = np.exp(ssn_layer_pars["f_E"])
        save_params["f_I"] = np.exp(ssn_layer_pars["f_I"])

    # Add readout parameters
    save_params.update(readout_pars)

    return save_params


vmap_evaluate_response = vmap(
    evaluate_model_response, in_axes=(None, None, None, None, None, None, None, 0)
)


def vmap_eval2(
    opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars, conv_pars
):
    """
    For a given value of the weights, calculate the loss for all the stimuli.
    Output:
        losses: size(n_stimuli)
        Accuracy: scalar
    """

    eval_vmap = vmap(
        model,
        in_axes=(
            {
                "b_sig": None,
                "logJ_2x2": None,
                "logs_2x2": None,
                "w_sig": None,
                "c_E": None,
                "c_I": None,
            },
            None,
            None,
            {"PERIODIC": None, "p_local": [None, None], "sigma_oris": None},
            {"ref": 0, "target": 0, "label": 0},
            {
                "conv_factor": None,
                "degree_per_pixel": None,
                "edge_deg": None,
                "k": None,
                "sigma_g": None,
            },
            {"Tmax": None, "dt": None, "silent": None, "verbose": None, "xtol": None},
        ),
    )
    losses, pred_labels = eval_vmap(
        opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars, conv_pars
    )

    accuracy = np.sum(test_data["label"] == pred_labels) / len(test_data["label"])

    return losses, accuracy


def vmap_eval3(
    opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars, conv_pars
):
    """
    Iterates through all values of 'w' to give the losses at each stimuli and weight, and the accuracy at each weight
    Output:
        losses: size(n_weights, n_stimuli )
        accuracy: size( n_weights)
    """

    eval_vmap = vmap(
        vmap_eval2,
        in_axes=(
            {
                "b_sig": None,
                "logJ_2x2": None,
                "logs_2x2": None,
                "w_sig": 0,
                "c_E": None,
                "c_I": None,
            },
            None,
            None,
            {"PERIODIC": None, "p_local": [None, None], "sigma_oris": None},
            {"ref": None, "target": None, "label": None},
            {
                "conv_factor": None,
                "degree_per_pixel": None,
                "edge_deg": None,
                "k": None,
                "sigma_g": None,
            },
            {"Tmax": None, "dt": None, "silent": None, "verbose": None, "xtol": None},
        ),
    )
    losses, accuracies = eval_vmap(
        opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars, conv_pars
    )
    print(losses.shape)
    print(accuracies.shape)
    return losses, accuracies


def test_accuracies(
    opt_pars,
    ssn_pars,
    grid_pars,
    conn_pars,
    filter_pars,
    conv_pars,
    stimuli_pars,
    trials=5,
    p=0.9,
    printing=True,
):
    key = random.PRNGKey(7)
    N_neurons = 25
    accuracies = []
    key, _ = random.split(key)
    opt_pars["w_sig"] = random.normal(key, shape=(trials, N_neurons)) / np.sqrt(
        N_neurons
    )

    train_data = create_data(stimuli_pars)

    print(opt_pars["w_sig"].shape)
    val_loss, accuracies = vmap_eval3(
        opt_pars, ssn_pars, grid_pars, conn_pars, train_data, filter_pars, conv_pars
    )

    # calcualate how many accuracies are above 90
    higher_90 = np.sum(accuracies[accuracies > p]) / len(accuracies)

    if printing:
        print(
            "grating contrast = {}, jitter = {}, noise std={}, acc (% >90 ) = {}".format(
                stimuli_pars.grating_contrast,
                stimuli_pars.jitter_val,
                stimuli_pars.std,
                higher_90,
            )
        )
    return higher_90, accuracies
