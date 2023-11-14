import matplotlib.pyplot as plt
import pandas as pd
import os
import jax.numpy as np

#from analysis import obtain_min_max_indices, label_neuron, ori_tuning_curve_responses

def plot_losses(training_losses, save_file=None):
    plt.plot(
        training_losses.T,
        label=["Binary cross entropy", "Avg_dx", "R_max", "w", "b", "Total"],
    )
    plt.legend()
    if save_file:
        plt.savefig(save_file)
    else:
        plt.show()


def plot_training_accs(training_accs, epoch_c=None, save=None):
    plt.plot(training_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy")

    if epoch_c == None:
        pass
    else:
        if np.isscalar(epoch_c):
            plt.axvline(x=epoch_c, c="r")
        else:
            plt.axvline(x=epoch_c[0], c="r")
            plt.axvline(x=epoch_c[0] + epoch_c[1], c="r")
            plt.axvline(x=epoch_c[2], c="r")

    if save:
        plt.savefig(save + ".png")
    plt.show()
    plt.close()


def plot_sigmoid_outputs(
    train_sig_input,
    val_sig_input,
    train_sig_output,
    val_sig_output,
    epoch_c=None,
    save=None,
):
    # Find maximum and minimum of
    max_train_sig_input = [item.max() for item in train_sig_input]
    mean_train_sig_input = [item.mean() for item in train_sig_input]
    min_train_sig_input = [item.min() for item in train_sig_input]

    max_val_sig_input = [item[0].max() for item in val_sig_input]
    mean_val_sig_input = [item[0].mean() for item in val_sig_input]
    min_val_sig_input = [item[0].min() for item in val_sig_input]

    epochs_to_plot = [item[1] for item in val_sig_input]

    max_train_sig_output = [item.max() for item in train_sig_output]
    mean_train_sig_output = [item.mean() for item in train_sig_output]
    min_train_sig_output = [item.min() for item in train_sig_output]

    max_val_sig_output = [item.max() for item in val_sig_output]
    mean_val_sig_output = [item.mean() for item in val_sig_output]
    min_val_sig_output = [item.min() for item in val_sig_output]

    # Create plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    # axes.vlines(x=epoch_c)

    axes[0, 0].plot(max_train_sig_input, label="Max")
    axes[0, 0].plot(mean_train_sig_input, label="Mean")
    axes[0, 0].plot(min_train_sig_input, label="Min")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].set_title("Input to sigmoid layer (training) ")
    # axes[0,0].vlines(x=epoch_c)

    axes[0, 1].plot(epochs_to_plot, max_val_sig_input, label="Max")
    axes[0, 1].plot(epochs_to_plot, mean_val_sig_input, label="Mean")
    axes[0, 1].plot(epochs_to_plot, min_val_sig_input, label="Min")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].set_title("Input to sigmoid layer (validation)")
    # axes[0,1].vlines(x=epoch_c)

    axes[1, 0].plot(max_train_sig_output, label="Max")
    axes[1, 0].plot(mean_train_sig_output, label="Mean")
    axes[1, 0].plot(min_train_sig_output, label="Min")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].set_title("Output of sigmoid layer (training)")
    # axes[1,0].vlines(x=epoch_c)

    axes[1, 1].plot(epochs_to_plot, max_val_sig_output, label="Max")
    axes[1, 1].plot(epochs_to_plot, mean_val_sig_output, label="Mean")
    axes[1, 1].plot(epochs_to_plot, min_val_sig_output, label="Min")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].set_title("Output to sigmoid layer (validation)")
    # axes[1,1].vlines(x=epoch_c)

    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    if epoch_c == None:
        pass
    else:
        if np.isscalar(epoch_c):
            plt.axvline(x=epoch_c, c="r")
        else:
            plt.axvline(x=epoch_c[0], c="r")
            plt.axvline(x=epoch_c[0] + epoch_c[1], c="r")

    if save:
        fig.savefig(save + ".png")
    


def plot_results(
    results_filename, bernoulli=True, epoch_c=None, save=None, norm_w=False
):
    """
    Read csv file with results and plot parameters against epochs. Option to plot norm of w if it is saved.
    """
    results = pd.read_csv(results_filename, header=0)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    if "J_EE" in results.columns:
        results.plot(x="epoch", y=["J_EE", "J_EI", "J_IE", "J_II"], ax=axes[0, 0])

    if "s_EE" in results.columns:
        results.plot(x="epoch", y=["s_EE", "s_EI", "s_IE", "s_II"], ax=axes[0, 1])

    if "c_E" in results.columns:
        results.plot(x="epoch", y=["c_E", "c_I"], ax=axes[1, 0])

    if "sigma_orisE" in results.columns:
        results.plot(x="epoch", y=["sigma_oriE", "sigma_oriI"], ax=axes[0, 1])

    if "sigma_oris" in results.columns:
        results.plot(x="epoch", y=["sigma_oris"], ax=axes[0, 1])

    if "norm_w" in results.columns and norm_w == True:
        results.plot(x="epoch", y=["norm_w"], ax=axes[1, 0])

    if bernoulli == True:
        results.plot(x="epoch", y=["val_accuracy", "ber_accuracy"], ax=axes[1, 1])
    else:
        results.plot(x="epoch", y=["val_accuracy"], ax=axes[1, 1])
        if epoch_c:
            plt.axvline(x=epoch_c, c="r")
    if save:
        fig.savefig(save + ".png")
    fig.show()
    plt.close()


def plot_results_two_layers(
    results_filename,
    bernoulli=False,
    save=None,
    epoch_c=None,
    norm_w=False,
    param_sum=False,
):
    results = pd.read_csv(results_filename, header=0)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

    if "J_EE_m" in results.columns:
        colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
        param_list_m = ["J_EE_m", "J_EI_m", "J_IE_m", "J_II_m"]
        param_list_s = ["J_EE_s", "J_EI_s", "J_IE_s", "J_II_s"]

        for i in range(4):
            results.plot(
                x="epoch", y=param_list_m[i], linestyle="-", ax=axes[0, 0], c=colors[i]
            )
            results.plot(
                x="epoch", y=param_list_s[i], linestyle="--", ax=axes[0, 0], c=colors[i]
            )

    if "s_EE_s" in results.columns:
        results.plot(
            x="epoch", y=["s_EE_s", "s_EI_s", "s_IE_s", "s_II_s"], ax=axes[0, 1]
        )

    if "c_E" in results.columns:
        results.plot(x="epoch", y=["c_E", "c_I"], ax=axes[1, 0])

    if "sigma_orisE" in results.columns:
        results.plot(
            x="epoch", y="sigma_orisE", linestyle="-", ax=axes[0, 1], c="tab:blue"
        )
        results.plot(
            x="epoch", y="sigma_orisI", linestyle="--", ax=axes[0, 1], c="tab:blue"
        )

    if "sigma_orisEE" in results.columns:
        results.plot(
            x="epoch", y="sigma_orisEE", linestyle="-", ax=axes[0, 1], c="tab:blue"
        )
        results.plot(
            x="epoch", y="sigma_orisEI", linestyle="--", ax=axes[0, 1], c="tab:blue"
        )

    # option 1 training
    if "kappa_preE" in results.columns:
        colors = ["tab:green", "tab:orange"]
        param_list_E = ["kappa_preE", "kappa_postE"]
        param_list_I = ["kappa_preI", "kappa_postI"]

        for i in range(2):
            results.plot(
                x="epoch", y=param_list_E[i], linestyle="-", ax=axes[2, 1], c=colors[i]
            )
            results.plot(
                x="epoch", y=param_list_I[i], linestyle="--", ax=axes[2, 1], c=colors[i]
            )

    # option 2 training
    if "kappa_preEE" in results.columns:
        colors = ["tab:green", "tab:orange"]
        param_list_E = ["kappa_preEE", "kappa_postEE"]
        param_list_I = ["kappa_preEI", "kappa_postEI"]

        for i in range(2):
            results.plot(
                x="epoch", y=param_list_E[i], linestyle="-", ax=axes[2, 1], c=colors[i]
            )
            results.plot(
                x="epoch", y=param_list_I[i], linestyle="--", ax=axes[2, 1], c=colors[i]
            )

    if "sigma_oris" in results.columns:
        results.plot(x="epoch", y=["sigma_oris"], ax=axes[0, 1])

    if "norm_w" in results.columns and norm_w == True:
        results.plot(x="epoch", y=["norm_w"], ax=axes[0, 1])

    if "f_E" in results.columns:
        results.plot(x="epoch", y=["f_E", "f_I"], ax=axes[1, 1])

    if bernoulli == True:
        results.plot(x="epoch", y=["val_accuracy", "ber_accuracy"], ax=axes[2, 0])
    else:
        results.plot(x="epoch", y=["val_accuracy"], ax=axes[2, 0])
        # If passed criterion, plot both lines
        if epoch_c == None:
            pass
        else:
            if np.isscalar(epoch_c):
                axes[2, 0].axvline(x=epoch_c, c="r")
            else:
                axes[2, 0].axvline(x=epoch_c[0], c="r")
                axes[2, 0].axvline(x=epoch_c[0] + epoch_c[1], c="r")
    if save:
        fig.savefig(save + ".png")
    fig.show()
    plt.close()

    # Create plots of sum of parameters
    if param_sum == True:
        fig_2, axes_2 = plt.subplots(nrows=1, ncols=3, figsize=(14, 3))

        axes_2[0].plot(results["J_IE_s"].to_numpy() + results["J_EE_s"])
        axes_2[0].set_title("Sum of J_EE_s + J_IE_s")

        axes_2[1].plot(results["J_IE_m"].to_numpy() + results["J_EE_m"])
        axes_2[1].set_title("Sum of J_EE_m + J_IE_m")

        axes_2[2].plot(results["f_E"].to_numpy() + results["f_I"])
        axes_2[2].set_title("Sum of f_E + f_I")

        if save:
            fig_2.savefig(save + "_param_sum.png")

        fig_2.show()
        plt.close()


def plot_losses(
    training_losses, validation_losses, epochs_to_save, epoch_c=None, save=None
):
    plt.plot(
        training_losses.T,
        label=["Binary cross entropy", "Avg_dx", "R_max", "w", "b", "Training total"],
    )
    plt.plot(epochs_to_save, validation_losses, label="Validation")
    plt.legend()
    plt.title("Training losses")
    if epoch_c:
        plt.axvline(x=epoch_c, c="r")
    if save:
        plt.savefig(save + ".png")
    plt.show()
    plt.close()


def plot_losses_two_stage(
    training_losses, val_loss_per_epoch, epoch_c=None, save=None, inset=None
):
    fig, axs1 = plt.subplots()
    axs1.plot(
        training_losses.T,
        label=["Binary cross entropy", "Avg_dx", "R_max", "w", "b", "Training total"],
    )
    axs1.plot(val_loss_per_epoch[:, 1], val_loss_per_epoch[:, 0], label="Validation")
    axs1.legend()
    axs1.set_title("Training losses")

    if inset:
        left, bottom, width, height = [0.2, 0.22, 0.35, 0.25]
        axs2 = fig.add_axes([left, bottom, width, height])

        axs2.plot(training_losses[0, :], label="Binary loss")
        axs2.legend()

    if epoch_c == None:
        pass
    else:
        if np.isscalar(epoch_c):
            axs1.axvline(x=epoch_c, c="r")
            if inset:
                axs2.axvline(x=epoch_c, c="r")
        else:
            axs1.axvline(x=epoch_c[0], c="r")
            axs1.axvline(x=epoch_c[0] + epoch_c[1], c="r")
            axs1.axvline(x=epoch_c[2], c="r")
            if inset:
                axs2.axvline(x=epoch_c[0], c="r")
                axs2.axvline(x=epoch_c[0] + epoch_c[1], c="r")
                axs1.axvline(x=epoch_c[2], c="r")
    if save:
        fig.savefig(save + ".png")


def plot_acc_vs_param(to_plot, lambdas, type_param=None, param=None):
    """
    Input:
        Matrix with shape (N+1, length of lambda) - each row corresponds to a different value of lambda, params at that value and
        the accuracy obtained
    Output:
        Plot of the desired param against the accuracy
    """

    plt.scatter(np.abs(to_plot[:, param]).T, to_plot[:, 0].T, c=lambdas)
    plt.colorbar()

    plt.ylabel("Accuracy")

    if type_param == "J":
        if param == 1:
            plt.xlabel("J_EE")
        if param == 2:
            plt.xlabel("J_EI")
        if param == 3:
            plt.xlabel("J_IE")
        if param == 4:
            plt.xlabel("J_II")

    if type_param == "s":
        if param == 1:
            plt.xlabel("s_EE")
        if param == 2:
            plt.xlabel("s_EI")
        if param == 3:
            plt.xlabel("s_IE")
        if param == 4:
            plt.xlabel("s_II")

    if type_param == "c":
        if param == 1:
            plt.xlabel("c_E")
        if param == 2:
            plt.xlabel("c_I")

    plt.show()


def plot_all_sig(all_sig_inputs, axis_title=None, save_fig=None):
    n_rows = int(np.sqrt(len(all_sig_inputs)))
    n_cols = int(np.ceil(len(all_sig_inputs) / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 20))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    count = 0

    # plot histograms
    for k in range(n_rows):
        for j in range(n_cols):
            axs[k, j].hist(all_sig_inputs[count])
            axs[k, j].set_xlabel(axis_title)
            axs[k, j].set_ylabel("Frequency")
            count += 1
            if count == len(all_sig_inputs):
                break

    if save_fig:
        fig.savefig(save_fig + "_" + axis_title + ".png")

    fig.show()
    plt.close()


def plot_histograms(all_accuracies, save_fig=None):
    n_rows = int(np.sqrt(len(all_accuracies)))
    n_cols = int(np.ceil(len(all_accuracies) / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 20))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    count = 0

    # plot histograms
    for k in range(n_rows):
        for j in range(n_cols):
            axs[k, j].hist(all_accuracies[count][2])
            axs[k, j].set_xlabel("Initial accuracy")
            axs[k, j].set_ylabel("Frequency")
            axs[k, j].set_title(
                "noise = "
                + str(np.round(all_accuracies[count][1], 2))
                + " jitter = "
                + str(np.round(all_accuracies[count][0], 2)),
                fontsize=10,
            )
            count += 1
            if count == len(all_accuracies):
                break

    if save_fig:
        fig.savefig(save_fig + ".png")

    fig.show()
    plt.close()


def plot_tuning_curves(
    pre_response_matrix,
    neuron_indices,
    radius_idx,
    ori_list,
    post_response_matrix=None,
    save=None,
):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(neuron_indices)))
    i = 0

    for idx in neuron_indices:
        plt.plot(
            ori_list, pre_response_matrix[radius_idx, idx, :], "--", color=colors[i]
        )

        if post_response_matrix.all():
            plt.plot(
                ori_list, post_response_matrix[radius_idx, idx, :], color=colors[i]
            )
        i += 1
    plt.xlabel("Orientation (degrees)")
    plt.ylabel("Response")

    if save:
        plt.savefig(save + ".png")
    plt.show()


def plot_vec2map(ssn, fp, save_fig=False):
    if ssn.Ne == 162:
        fp_E = ssn.select_type(fp, map_number=1).ravel()
        fp_I = ssn.select_type(fp, map_number=2).ravel()
        titles = ["E", "I"]
        all_responses = [fp_E, fp_I]

    if ssn.Ne > 162:
        fp_E_on = ssn.select_type(fp, map_number=1).ravel()
        fp_E_off = ssn.select_type(fp, map_number=3).ravel()
        fp_I_on = ssn.select_type(fp, map_number=2).ravel()
        fp_I_off = ssn.select_type(fp, map_number=4).ravel()
        titles = ["E_on", "I_on", "E_off", "I_off"]
        all_responses = [fp_E_on, fp_I_on, fp_E_off, fp_I_off]

    rows = int(len(titles) / 2)
    cols = int(len(titles) / rows)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    count = 0
    for row in range(0, rows):
        for col in range(0, cols):
            ax = axes[row, col]
            im = ax.imshow(
                all_responses[count].reshape(9, 9), vmin=fp.min(), vmax=fp.max()
            )
            ax.set_title(titles[count])
            ax.set_xlabel(
                "max "
                + str(all_responses[count].max())
                + " at index "
                + str(np.argmax(all_responses[count]))
            )
            count += 1

    fig.colorbar(im, ax=axes.ravel().tolist())

    if save_fig:
        fig.savefig(save_fig + ".png")

    plt.close()


def plot_mutiple_gabor_filters(ssn, fp, save_fig=None, indices=None):
    if indices == None:
        indices = obtain_min_max_indices(ssn=ssn, fp=fp)

    fig, axes = plt.subplots(2, 3, figsize=(8, 8))
    count = 0
    for row in range(0, 2):
        for col in range(0, 3):
            ax = axes[row, col]
            im = plot_individual_gabor(ax, fp, ssn, index=indices[count])
            count += 1
    if save_fig:
        fig.savefig(os.path.join(save_fig + ".png"))
    plt.show()
    plt.close()


def plot_individual_gabor(ax, fp, ssn, index):
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    labels = ["E_ON", "I_ON", "E_OFF", "I_OFF"]
    ax.imshow(ssn.gabor_filters[index].reshape(129, 129), cmap="Greys")
    ax.set_xlabel("Response " + str(fp[index]))
    ax.set_title("ori " + str(ssn.ori_vec[index]) + " " + str(label_neuron(index)))
    return ax


def plot_tuning_curves_ssn(
    ssn, index, conv_pars, stimuli_pars, offset=4, all_responses=None, save_fig=None
):
    print("Neuron preferred orientation: ", str(ssn.ori_vec[index]))

    if all_responses != None:
        pass
    else:
        all_responses, ori_list = ori_tuning_curve_responses(
            ssn=ssn,
            index=index,
            conv_pars=conv_pars,
            stimuli_pars=stimuli_pars,
            offset=offset,
        )

    plt.plot(ori_list, all_responses)
    plt.axvline(x=ssn.ori_vec[index], linestyle="dashed", c="r", label="Pref ori")
    plt.xlabel("Stimulus orientations")
    plt.ylabel("Response")
    plt.title("Neuron type " + str(label_neuron(index)))
    plt.legend()
    if save_fig:
        plt.savefig(save_fig + ".png")
    plt.show()
    plt.close()

    return all_responses


def plot_close_far(
    E_pre, E_post, I_pre, I_post, e_close, e_far, i_close, i_far, save=None, title=None
):
    # EE
    E_E_pre_close = [E_pre[e_close].mean(), E_pre[e_close].std()]
    E_E_post_close = [E_post[e_close].mean(), E_post[e_close].std()]
    E_E_pre_far = [E_pre[e_far].mean(), E_pre[e_far].std()]
    E_E_post_far = [E_post[e_far].mean(), E_post[e_far].std()]

    # IE
    I_E_pre_close = [E_pre[i_close].mean(), E_pre[i_close].std()]
    I_E_post_close = [E_post[i_close].mean(), E_post[i_close].std()]
    I_E_pre_far = [E_pre[i_far].mean(), E_pre[i_far].std()]
    I_E_post_far = [E_post[i_far].mean(), E_post[i_far].std()]

    # EI
    E_I_pre_close = [np.abs(I_pre[e_close].mean()), np.abs(I_pre[e_close].std())]
    E_I_post_close = [np.abs(I_post[e_close].mean()), np.abs(I_post[e_close].std())]
    E_I_pre_far = [np.abs(I_pre[e_far].mean()), np.abs(I_pre[e_far].std())]
    E_I_post_far = [np.abs(I_post[e_far].mean()), np.abs(I_post[e_far].std())]

    # II
    I_I_pre_close = [np.abs(I_pre[i_close].mean()), np.abs(I_pre[i_close].std())]
    I_I_post_close = [np.abs(I_post[i_close].mean()), np.abs(I_post[i_close].std())]
    I_I_pre_far = [np.abs(I_pre[i_far].mean()), np.abs(I_pre[i_far].std())]
    I_I_post_far = [np.abs(I_post[i_far].mean()), np.abs(I_post[i_far].std())]

    pre_close_mean = [
        E_E_pre_close[0],
        I_E_pre_close[0],
        E_I_pre_close[0],
        I_I_pre_close[0],
    ]
    post_close_mean = [
        E_E_post_close[0],
        I_E_post_close[0],
        E_I_post_close[0],
        I_I_post_close[0],
    ]

    pre_far_mean = [E_E_pre_far[0], I_E_pre_far[0], E_I_pre_far[0], I_I_pre_far[0]]
    post_far_mean = [E_E_post_far[0], I_E_post_far[0], E_I_post_far[0], I_I_post_far[0]]

    pre_close_error = [
        E_E_pre_close[1],
        I_E_pre_close[1],
        E_I_pre_close[1],
        I_I_pre_close[1],
    ]
    post_close_error = [
        E_E_post_close[1],
        I_E_post_close[1],
        E_I_post_close[1],
        I_I_post_close[1],
    ]

    pre_far_error = [E_E_pre_far[1], I_E_pre_far[1], E_I_pre_far[1], I_I_pre_far[1]]
    post_far_error = [
        E_E_post_far[1],
        I_E_post_far[1],
        E_I_post_far[1],
        I_I_post_far[1],
    ]

    X = np.arange(4)
    labels = ["EE", "IE", "EI", "II"]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(
        X + 0.00, pre_close_mean, color="c", width=0.15, hatch="/", label="pre_close"
    )
    ax.bar(X + 0.15, post_close_mean, color="c", width=0.15, label="post_close")
    ax.bar(X + 0.30, pre_far_mean, color="r", width=0.15, hatch="/", label="pre_far")
    ax.bar(X + 0.45, post_far_mean, color="r", width=0.15, label="post_far")
    if title:
        plt.title(title)
    plt.xticks(X + 0.225, labels)
    plt.ylabel("Average input")
    plt.legend()
    plt.axis("on")
    if save:
        plt.savefig(os.path.join(save, title + ".png"))
    fig.show()


def plot_r_ref(r_ref, epoch_c=None, save=None):
    plt.plot(r_ref)
    plt.xlabel("Epoch")
    plt.ylabel("noise")

    if epoch_c == None:
        pass
    else:
        if np.isscalar(epoch_c):
            plt.axvline(x=epoch_c, c="r")
        else:
            plt.axvline(x=epoch_c[0], c="r")
            plt.axvline(x=epoch_c[0] + epoch_c[1], c="r")
            plt.axvline(x=epoch_c[2], c="r")

    if save:
        plt.savefig(save + ".png")
    plt.show()
    plt.close()


def plot_max_rates(max_rates, epoch_c=None, save=None):
    plt.plot(max_rates, label=["E_mid", "I_mid", "E_sup", "I_sup"])
    plt.xlabel("Epoch")
    plt.ylabel("Maximum rates")
    plt.legend()

    if epoch_c == None:
        pass
    else:
        if np.isscalar(epoch_c):
            plt.axvline(x=epoch_c, c="r")
        else:
            plt.axvline(x=epoch_c[0], c="r")
            plt.axvline(x=epoch_c[0] + epoch_c[1], c="r")
            plt.axvline(x=epoch_c[2], c="r")

    if save:
        plt.savefig(save + ".png")
    plt.show()
    plt.close()


def plot_w_sig(w_sig, epochs_to_save, epoch_c=None, save=None):
    plt.plot(w_sig)
    plt.xlabel("Epoch")
    plt.ylabel("Values of w")
    if epoch_c:
        plt.axvline(x=epoch_c, c="r", label="criterion")
    if save:
        plt.savefig(save + ".png")
    plt.show()
    plt.close()

