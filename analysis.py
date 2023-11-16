import pandas as pd
import numpy

from jax import random, vmap
import jax.numpy as np

from training_supp import (
    constant_to_vec,
    create_data,
    obtain_fixed_point,
    model,
    generate_noise,
)

def param_ratios(results_file):
    results = pd.read_csv(results_file, header=0)

    if "J_EE" in results.columns:
        Js = results[["J_EE", "J_EI", "J_IE", "J_II"]]
        Js = Js.to_numpy()
        print("J ratios = ", np.array((Js[-1, :] / Js[0, :] - 1) * 100, dtype=int))

    if "s_EE" in results.columns:
        ss = results[["s_EE", "s_EI", "s_IE", "s_II"]]
        ss = ss.to_numpy()
        print("s ratios = ", np.array((ss[-1, :] / ss[0, :] - 1) * 100, dtype=int))

    if "c_E" in results.columns:
        cs = results[["c_E", "c_I"]]
        cs = cs.to_numpy()
        print("c ratios = ", np.array((cs[-1, :] / cs[0, :] - 1) * 100, dtype=int))

    if "sigma_orisE" in results.columns:
        sigma_oris = results[["sigma_orisE", "sigma_orisI"]]
        sigma_oris = sigma_oris.to_numpy()
        print(
            "sigma_oris ratios = ",
            np.array((sigma_oris[-1, :] / sigma_oris[0, :] - 1) * 100, dtype=int),
        )

    if "kappa_preE" in results.columns:
        kappa_pre = results[["kappa_preE", "kappa_preI"]]
        kappa_pre = kappa_pre.to_numpy()
        print(
            "kappa_pre ratios = ",
            np.array((kappa_pre[-1, :] / kappa_pre[0, :] - 1) * 100, dtype=int),
        )

    if "kappa_postE" in results.columns:
        kappa_post = results[["kappa_postE", "kappa_postI"]]
        kappa_post = kappa_post.to_numpy()
        print(
            "kappa_post ratios = ",
            np.array((kappa_post[-1, :] / kappa_post[0, :] - 1) * 100, dtype=int),
        )

    if "sigma_oris" in results.columns:
        sigma_oris = results[["sigma_oris"]]
        sigma_oris = sigma_oris.to_numpy()

        print(
            "sigma_oris ratios = ",
            np.array((sigma_oris[-1, :] / sigma_oris[0, :] - 1) * 100, dtype=int),
        )


def param_ratios_two_layer(results_file, epoch=None, percent_acc=0.85):
    results = pd.read_csv(results_file, header=0)

    if epoch == None:
        accuracies = list(results["val_accuracy"][:20].values)
        count = 9
        while np.asarray(accuracies).mean() < percent_acc:
            count += 1
            del accuracies[0]
            if count > len(results["val_accuracy"]):
                break
            else:
                accuracies.append(results["val_accuracy"][count])

        epoch = results["epoch"][count]
        epoch_index = results[results["epoch"] == epoch].index
        print(epoch, epoch_index)

    if epoch == -1:
        epoch_index = epoch
    else:
        epoch_index = results[results["epoch"] == epoch].index

    if "J_EE_m" in results.columns:
        Js = results[["J_EE_m", "J_EI_m", "J_IE_m", "J_II_m"]]
        Js = Js.to_numpy()
        print("J_m ratios = ", np.array((Js[epoch_index, :] / Js[0, :] - 1) * 100))

    if "J_EE_s" in results.columns:
        Js = results[["J_EE_s", "J_EI_s", "J_IE_s", "J_II_s"]]
        Js = Js.to_numpy()
        print("J_s ratios = ", np.array((Js[epoch_index, :] / Js[0, :] - 1) * 100))

    if "s_EE_m" in results.columns:
        ss = results[["s_EE_m", "s_EI_m", "s_IE_m", "s_II_m"]]
        ss = ss.to_numpy()
        print(
            "s_m ratios = ",
            np.array(
                (ss[epoch_index, :] / ss[0, :] - 1) * 100,
            ),
        )

    if "s_EE_s" in results.columns:
        ss = results[["s_EE_s", "s_EI_s", "s_IE_s", "s_II_s"]]
        ss = ss.to_numpy()
        print("s_s ratios = ", np.array((ss[epoch_index, :] / ss[0, :] - 1) * 100))

    if "c_E" in results.columns:
        cs = results[["c_E", "c_I"]]
        cs = cs.to_numpy()
        print("c ratios = ", np.array((cs[epoch_index, :] / cs[0, :] - 1) * 100))

    if "sigma_orisE" in results.columns:
        sigma_oris = results[["sigma_orisE", "sigma_orisI"]]
        sigma_oris = sigma_oris.to_numpy()
        print(
            "sigma_oris ratios = ",
            np.array((sigma_oris[epoch_index, :] / sigma_oris[0, :] - 1) * 100),
        )

    if "sigma_oris" in results.columns:
        sigma_oris = results[["sigma_oris"]]
        sigma_oris = sigma_oris.to_numpy()
        print(
            "sigma_oris ratios = ",
            np.array((sigma_oris[epoch_index, :] / sigma_oris[0, :] - 1) * 100),
        )

    if "f_E" in results.columns:
        fs = results[["f_E", "f_I"]]
        fs = fs.to_numpy()
        print("f ratios = ", np.array((fs[epoch_index, :] / fs[0, :] - 1) * 100))

    if "kappa_preE" in results.columns:
        kappas = results[["kappa_preE", "kappa_preI", "kappa_postE", "kappa_postI"]]
        kappas = kappas.to_numpy()
        print("kappas = ", kappas[epoch_index, :])


def assemble_pars(all_pars, matrix=True):
    """
    Take parameters from csv file and

    """
    pre_train = np.asarray(all_pars.iloc[0].tolist())
    post_train = np.asarray(all_pars.iloc[-1].tolist())

    if matrix == True:
        matrix_pars = lambda Jee, Jei, Jie, Jii: np.array([[Jee, Jei], [Jie, Jii]])

        pre_train = matrix_pars(*pre_train)
        post_train = matrix_pars(*post_train)

    return pre_train, post_train


def vmap_eval_hist(
    ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag=False
):
    losses, all_losses, pred_label, sig_in, sig_out, _ = model(
        ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag
    )
    # Find accuracy based on predicted labels

    true_accuracy = np.sum(data["label"] == pred_label) / len(data["label"])
    vmap_loss = np.mean(losses)
    all_losses = np.mean(all_losses, axis=0)
    sig_input = np.mean(sig_in)
    # std = np.std(sig_in)
    sig_output = np.mean(sig_out)

    return vmap_loss, true_accuracy, sig_input, sig_output


def vmap_eval3(ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag=False):
    """
    Iterates through all values of 'w' to give the losses at each stimuli and weight, and the accuracy at each weight
    Output:
        losses: size(n_weights, n_stimuli )
        accuracy: size( n_weights)
    """
    eval_vmap = vmap(
        vmap_eval_hist,
        in_axes=(
            {
                "c_E": None,
                "c_I": None,
                "f_E": None,
                "f_I": None,
                "logJ_2x2": [None, None],
                "kappa_pre": None,
                "kappa_post": None,
            },
            {"b_sig": None, "w_sig": 0},
            {
                "ssn_mid_ori_map": None,
                "ssn_sup_ori_map": None,
                "conn_pars_m": None,
                "conn_pars_s": None,
                "conv_pars": None,
                "filter_pars": None,
                "gE": [None, None],
                "gI": [None, None],
                "grid_pars": None,
                "loss_pars": None,
                "logs_2x2": None,
                "noise_type": None,
                "noise_ref": None,
                "noise_target": None,
                "ssn_pars": None,
                "key": None,
                "sigma_oris": None,
                "train_ori": None,
            },
            {"label": None, "ref": None, "target": None},
            None,
        ),
    )
    losses, true_acc, sig_input, sig_output = eval_vmap(
        ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag
    )

    return losses, true_acc, sig_input, sig_output


def test_accuracies(
    ssn_layer_pars,
    readout_pars,
    constant_ssn_pars,
    stimuli_pars,
    noise,
    offset,
    trials=5,
    p=0.9,
    printing=True,
):
    N_neurons = 25
    accuracies = []

    readout_pars["w_sig"] = numpy.random.normal(
        scale=0.25, size=(trials, N_neurons)
    ) / np.sqrt(N_neurons)

    train_data = create_data(stimuli_pars, number=trials)
    constant_ssn_pars = generate_noise(
        constant_ssn_pars,
        sig_noise=noise,
        batch_size=len(train_data["ref"]),
        length=readout_pars["w_sig"].shape[1],
    )

    val_loss, true_acc, sig_input, sig_output = vmap_eval3(
        ssn_layer_pars, readout_pars, constant_ssn_pars, train_data
    )

    # calcualate how many accuracies are above 90
    higher_90 = np.sum(true_acc[true_acc > p]) / len(true_acc)

    if printing:
        print(
            "grating contrast = {}, jitter = {}, noise std={}, acc (% >90 ) = {}".format(
                stimuli_pars["grating_contrast"],
                stimuli_pars["jitter_val"],
                stimuli_pars["std"],
                higher_90,
            )
        )
    print(true_acc.shape)

    return higher_90, true_acc, readout_pars["w_sig"], sig_input, sig_output


def initial_acc(
    ssn_layer_pars,
    readout_pars,
    constant_ssn_pars,
    stimuli_pars,
    ref_ori,
    offset,
    min_sig_noise,
    max_sig_noise,
    min_jitter=3,
    max_jitter=5,
    p=0.9,
    len_noise=11,
    len_jitters=3,
    save_fig=None,
    trials=100,
):
    """
    Find initial accuracy for varying jitter and noise levels.

    """

    print(constant_ssn_pars["noise_type"])
    # list_noise  =  np.logspace(start=np.log10(min_sig_noise), stop=np.log10(max_sig_noise), num=len_noise, endpoint=True, base=10.0, dtype=None, axis=0)
    list_noise = np.linspace(min_sig_noise, max_sig_noise, len_noise)
    list_jitters = np.linspace(min_jitter, max_jitter, len_jitters)

    low_acc = []
    all_accuracies = []
    percent_50 = []
    good_w_s = []
    all_sig_inputs = []
    all_sig_outputs = []
    constant_ssn_pars["train_ori"] = ref_ori

    for sig_noise in list_noise:
        for jitter in list_jitters:
            # stimuli_pars['std'] = noise
            stimuli_pars["jitter_val"] = jitter

            constant_ssn_pars["key"], _ = random.split(constant_ssn_pars["key"])
            higher_90, acc, w_s, sig_input, sig_output = test_accuracies(
                ssn_layer_pars,
                readout_pars,
                constant_ssn_pars,
                stimuli_pars,
                noise=sig_noise,
                offset=offset,
                p=p,
                trials=trials,
                printing=False,
            )
            print(acc.shape)
            # save low accuracies
            if higher_90 < 0.05:
                low_acc.append([jitter, sig_noise, higher_90])

            indices = list(filter(lambda x: acc[x] == 0.5, range(len(acc))))
            w_s = [w_s[idx] for idx in indices]
            good_w_s.append(w_s)
            all_sig_inputs.append(sig_input)
            all_sig_outputs.append(sig_output)

            all_accuracies.append([jitter, sig_noise, acc])

    return all_accuracies, low_acc, percent_50, good_w_s


def accuracies(all_acc, p=0.75):
    """
    Print accuracies and jitters that give have a probability p of having initial accuracy betwen 0.45-0.55
    """

    acc_to_save = []
    for x in range(len(all_acc)):
        acc = all_acc[x][2]
        if ((0.45 < acc) & (acc < 0.55)).sum() / len(acc) > p:
            print(all_acc[x][0], all_acc[x][1])
            acc_to_save.append([all_acc[x][0], all_acc[x][1]])

    return acc_to_save


def obtain_regular_indices(ssn, number=8, test_oris=None):
    """
    Function takes SSN network and outputs linearly separated list of orientation indices
    """

    array = ssn.ori_map[2:7, 2:7]
    array = array.ravel()

    if test_oris:
        pass
    else:
        test_oris = np.linspace(array.min(), array.max(), number)
    indices = []

    for test_ori in test_oris:
        idx = (np.abs(array - test_ori)).argmin()
        indices.append(idx)

    testing_angles = [array[idx] for idx in indices]
    print(testing_angles)

    return indices


def ori_tuning_curve_responses(
    ssn, conv_pars, stimuli_pars, index=None, offset=4, c_E=5, c_I=5
):
    all_responses = []
    ori_list = np.linspace(0, 180, 18 * 2 + 1)

    # Add preferred orientation
    if index:
        ori_list = np.unique(np.insert(ori_list, 0, ssn.ori_vec[index]).sort())

    # Obtain response for different orientations
    for ori in ori_list:
        stimulus_data = create_data(stimuli_pars, number=1)
        constant_vector = constant_to_vec(c_E, c_I, ssn)

        output_ref = np.matmul(ssn.gabor_filters, stimulus_data["ref"].squeeze())

        # Rectify output
        SSN_input_ref = np.maximum(0, output_ref) + constant_vector

        r_init = np.zeros(SSN_input_ref.shape[0])
        fp, _ = obtain_fixed_point(ssn, SSN_input_ref, conv_pars)

        if index == None:
            all_responses.append(fp)
        else:
            all_responses.append(fp[index])

    return np.vstack(all_responses), ori_list


def obtain_min_max_indices(ssn, fp):
    idx = (ssn.ori_vec > 45) * (ssn.ori_vec < 65)
    indices = np.where(idx)
    responses_45_65 = fp[indices]
    j_s = []
    max_min_indices = np.concatenate(
        [np.argsort(responses_45_65)[:3], np.argsort(responses_45_65)[-3:]]
    )

    for i in max_min_indices:
        j = indices[0][i]
        j_s.append(j)

    return j_s


def label_neuron(index):
    labels = ["E_ON", "I_ON", "E_OFF", "I_OFF"]
    return labels[int(np.floor(index / 81))]


def full_width_half_max(vector, d_theta):
    vector = vector - vector.min()
    half_height = vector.max() / 2
    points_above = len(vector[vector > half_height])

    distance = d_theta * points_above

    return distance


def sort_neurons(ei_indices, close_far_indices):
    empty_list = []
    for i in ei_indices:
        if i in close_far_indices:
            empty_list.append(i)

    return np.asarray([empty_list])


def close_far_indices(train_ori, ssn):
    close_indices = []
    far_indices = []

    upper_range = train_ori + 90 / 2
    print(upper_range)

    for i in range(len(ssn.ori_vec)):
        if 0 < ssn.ori_vec[i] <= upper_range:
            close_indices.append(i)
        else:
            far_indices.append(i)

    return np.asarray([close_indices]), np.asarray([far_indices])


def sort_close_far_EI(ssn, train_ori):
    close, far = close_far_indices(55, ssn)
    close = close.squeeze()
    far = far.squeeze()
    e_indices = np.where(ssn.tau_vec == ssn.tauE)[0]
    i_indices = np.where(ssn.tau_vec == ssn.tauI)[0]

    e_close = sort_neurons(e_indices, close)
    e_far = sort_neurons(e_indices, far)
    i_close = sort_neurons(i_indices, close)
    i_far = sort_neurons(i_indices, far)

    return e_close, e_far, i_close, i_far
