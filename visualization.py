import matplotlib.pyplot as plt
import pandas as pd
import os
import jax.numpy as np
import numpy
import sys

from analysis import obtain_min_max_indices, label_neuron

def calculate_relative_change(df):
    # Find index where 'stage' is 1 for the first time
    ind0 = df.index[df['stage'] == 0][0]
    ind1 = df.index[df['stage'] == 1][0]

    relative_changes = numpy.zeros((12,2))
    # Calculate relative changes in Jm and Js
    J_m_EE = numpy.exp(df['logJ_m_EE'])
    J_m_IE = numpy.exp(df['logJ_m_IE'])
    J_m_EI = -1 * numpy.exp(df['logJ_m_EI'])
    J_m_II = -1 * numpy.exp(df['logJ_m_II'])
    J_s_EE = numpy.exp(df['logJ_s_EE'])
    J_s_IE = numpy.exp(df['logJ_s_IE'])
    J_s_EI = -1 * numpy.exp(df['logJ_s_EI'])
    J_s_II = -1 * numpy.exp(df['logJ_s_II'])
    relative_changes[0,0] =(J_m_EE.iloc[ind1] - J_m_EE.iloc[ind0]) / J_m_EE.iloc[ind0]
    relative_changes[0,1] =(J_m_EE.iloc[-1] - J_m_EE.iloc[ind1]) / J_m_EE.iloc[ind1]
    relative_changes[1,0] =(J_m_IE.iloc[ind1] - J_m_IE.iloc[ind0]) / J_m_IE.iloc[ind0]
    relative_changes[1,1] =(J_m_IE.iloc[-1] - J_m_IE.iloc[ind1]) / J_m_IE.iloc[ind1]
    relative_changes[2,0] =(J_m_EI.iloc[ind1] - J_m_EI.iloc[ind0]) / J_m_EI.iloc[ind0]
    relative_changes[2,1] =(J_m_EI.iloc[-1] - J_m_EI.iloc[ind1]) / J_m_EI.iloc[ind1]
    relative_changes[3,0] =(J_m_II.iloc[ind1] - J_m_II.iloc[ind0]) / J_m_II.iloc[ind0]
    relative_changes[3,1] =(J_m_II.iloc[-1] - J_m_II.iloc[ind1]) / J_m_II.iloc[ind1]

    relative_changes[4,0] =(J_s_EE.iloc[ind1] - J_s_EE.iloc[ind0]) / J_s_EE.iloc[ind0]
    relative_changes[4,1] =(J_s_EE.iloc[-1] - J_s_EE.iloc[ind1]) / J_s_EE.iloc[ind1]
    relative_changes[5,0] =(J_s_IE.iloc[ind1] - J_s_IE.iloc[ind0]) / J_s_IE.iloc[ind0]
    relative_changes[5,1] =(J_s_IE.iloc[-1] - J_s_IE.iloc[ind1]) / J_s_IE.iloc[ind1]
    relative_changes[6,0] =(J_s_EI.iloc[ind1] - J_s_EI.iloc[ind0]) / J_s_EI.iloc[ind0]
    relative_changes[6,1] =(J_s_EI.iloc[-1] - J_s_EI.iloc[ind1]) / J_s_EI.iloc[ind1]
    relative_changes[7,0] =(J_s_II.iloc[ind1] - J_s_II.iloc[ind0]) / J_s_II.iloc[ind0]
    relative_changes[7,1] =(J_s_II.iloc[-1] - J_s_II.iloc[ind1]) / J_s_II.iloc[ind1]

    # in c and f 
    relative_changes[8,0] = (df['c_E'].iloc[ind1] - df['c_E'].iloc[ind0]) / df['c_E'].iloc[ind0]
    relative_changes[8,1] = (df['c_E'].iloc[-1] - df['c_E'].iloc[ind1]) / df['c_E'].iloc[ind1]
    relative_changes[9,0] = (df['c_I'].iloc[ind1] - df['c_I'].iloc[ind0]) / df['c_I'].iloc[ind0]
    relative_changes[9,1] = (df['c_I'].iloc[-1] - df['c_I'].iloc[ind1]) / df['c_I'].iloc[ind1]
    relative_changes[10,0] = (df['f_E'].iloc[ind1] - df['f_E'].iloc[ind0]) / df['f_E'].iloc[ind0]
    relative_changes[10,1] = (df['f_E'].iloc[-1] - df['f_E'].iloc[ind1]) / df['f_E'].iloc[ind1]
    relative_changes[11,0] = (df['f_I'].iloc[ind1] - df['f_I'].iloc[ind0]) / df['f_I'].iloc[ind0]
    relative_changes[11,1] = (df['f_I'].iloc[-1] - df['f_I'].iloc[ind1]) / df['f_I'].iloc[ind1]
    return relative_changes


def barplots_from_csvs(directory, plot_filename = None):
    # List to store relative changes from each file
    relative_changes_pretrain = []
    relative_changes_train = []

    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            # Read CSV file
            df = pd.read_csv(filepath)
            # Calculate relative change
            relative_changes = calculate_relative_change(df)
            relative_changes_pretrain.append(relative_changes[:,0])
            relative_changes_train.append(relative_changes[:,1])
    
    # Plotting bar plots
    # Define groups of parameters and plot each parameter group
    groups = [
        ['J_m_EE', 'J_m_EI', 'J_m_IE', 'J_m_II'],
        ['J_s_EE', 'J_s_EI', 'J_s_IE', 'J_s_II'],
        ['c_E', 'c_I'], ['f_E', 'f_I']
    ]
    num_groups = len(groups)
    fig, axs = plt.subplots(2, num_groups, figsize=(5*num_groups, 10))  # Create subplots for each group
    
    relative_changes_train = numpy.array(relative_changes_train)
    relative_changes_pretrain = numpy.array(relative_changes_pretrain)
    group_indices = [0,4,8,10,12]
    titles_pretrain=['Jm changes in pretraining', 'Js changes in pretraining', 'c changes in pretraining', 'f changes in pretraining']
    titles_train=['Jm changes in training', 'Js changes in training', 'c changes in training', 'f changes in training']
    for i, group in enumerate(groups):
        group_data = relative_changes_pretrain[:, group_indices[i]:group_indices[i+1]]  # Extract data for the current group
        axs[0,i].boxplot(group_data, labels=group)
        axs[0,i].set_title(titles_pretrain[i])
    for i, group in enumerate(groups):
        group_data = relative_changes_train[:, group_indices[i]:group_indices[i+1]]  # Extract data for the current group
        axs[1,i].boxplot(group_data, labels=group)
        axs[1,i].set_title(titles_train[i])
        
    plt.tight_layout()
    
    if plot_filename is not None:
        full_path = os.path.join(directory, plot_filename + ".png")
        fig.savefig(full_path)

    plt.close()


def plot_results_from_csv(
    results_filename,
    fig_filename=None):
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(results_filename, header=0)
    N=len(df[df.columns[0]])
    # Create a subplot with 2 rows and 4 columns
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(35, 10))

    # Plot accuracy and losses
    for column in df.columns:
        if 'acc' in column and 'val_' not in column:
            axes[0, 0].plot(range(N), df[column], label=column, alpha=0.6)
        if 'val_acc' in column:
            axes[0, 0].scatter(range(N), df[column], label=column, marker='o', s=50, c='green')
    axes[0, 0].legend(loc='lower right')
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_ylim(0, 1) 

    for column in df.columns:
        if 'loss_' in column and 'val_loss' not in column:
            axes[1, 0].plot(range(N), df[column], label=column, alpha=0.6)
        if 'val_loss' in column:
            axes[1, 0].scatter(range(N), df[column], marker='o', s=50)
    axes[1, 0].legend(["readout loss", "avg_dx", "r_max", "w_sig", "b_sig", "total"])
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].set_title('Loss')

    #Plot changes in sigmoid weights and bias of the sigmoid layer
    axes[1,1].plot(range(N), df['b_sig'], label=column, linestyle='--', linewidth = 2)
    axes[1,1].legend("b sig")

    num_pretraining_steps= sum(df['stage'] == df['stage'][0])
    for column in df.columns:
        if 'offset' in column:
            axes[0,4].plot(range(num_pretraining_steps), np.ones(num_pretraining_steps)*5, label='offsets at bl acc', alpha=0.2)
            axes[0,4].scatter(range(num_pretraining_steps), df[column][0:num_pretraining_steps], label='offsets at bl acc', marker='o', s=50)
            axes[0,4].grid(color='gray', linestyle='-', linewidth=0.5)
            axes[0,4].set_title('Offset with accuracy 0.749')
            axes[1,4].plot(range(N-num_pretraining_steps,N), df[column][N-num_pretraining_steps:N], label='offset')
            axes[1,4].grid(color='gray', linestyle='-', linewidth=0.5)
            axes[1,4].set_ylim(0, 20)
            axes[1,4].set_title('Offset during staircase training')
            
    i=0
    for column in df.columns:
        if 'w_sig_' in column and i<10:
            axes[1,1].plot(range(N), df[column], label=column)
            i = i+1
    axes[1,1].set_title('Readout bias and weights')

    #Plot changes in J_m and J_s
    for column in df.columns:
        if 'J_m_' in column:
            if 'EE' in column:
                axes[0, 2].plot(range(N), numpy.exp(df[column]), label=column, c='tab:red')
            if 'IE' in column:
                axes[0, 2].plot(range(N), numpy.exp(df[column]), label=column, linestyle='--', c='tab:red')
    for column in df.columns:
        if 'J_m_' in column:
            if 'II' in column:
                axes[0, 2].plot(range(N), -numpy.exp(df[column]), label=column, c='tab:blue')
            if 'EI' in column:
                axes[0, 2].plot(range(N), -numpy.exp(df[column]), label=column, linestyle='--', c='tab:blue')
    axes[0,2].legend(loc="upper left")
    axes[0,2].set_title('J in middle layer')
    
    for column in df.columns:
        if 'J_s_' in column:
            if 'EE' in column:
                axes[1, 2].plot(range(N), numpy.exp(df[column]), label=column, c='tab:red')
            if 'IE' in column:
                axes[1, 2].plot(range(N), numpy.exp(df[column]), label=column, linestyle='--', c='tab:red')
    for column in df.columns:
        if 'J_s_' in column:
            if 'II' in column:
                axes[1, 2].plot(range(N), -numpy.exp(df[column]), label=column, c='tab:blue')
            if 'EI' in column:
                axes[1, 2].plot(range(N), -numpy.exp(df[column]), label=column, linestyle='--', c='tab:blue')
    axes[1,2].legend(loc="upper left")
    axes[1,2].set_title('J in superficial layer')

    # Plot maximum rates
    i=0
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    for column in df.columns:
        if 'maxr' in column:
            axes[0, 1].plot(range(N), df[column], label=column, c=colors[i])
            i=i+1
    axes[0,1].legend(loc="upper left")
    axes[0,1].set_title('Maximum rates')

    #Plot changes in baseline inhibition and excitation and feedforward weights (second stage of the training)
    axes[0,3].plot(range(N), df['c_E'], label='c_E')
    axes[0,3].plot(range(N), df['c_I'], label='c_I')
    axes[0,3].legend(["c_E","c_I"])
    axes[0,3].set_title('Baseline inh and exc inputs')

    #Plot feedforward weights from middle to superficial layer (second stage of the training)
    axes[1,3].plot(range(N), numpy.exp(df['f_E']), label='f_E')
    axes[1,3].plot(range(N), numpy.exp(df['f_I']), label='f_I')
    axes[1,3].legend(["f_E","f_I"])
    axes[1,3].set_title('Weights between mid and sup layers')

    for i in range(1, len(df['stage'])):
        if df['stage'][i] != df['stage'][i-1]:
            axes[0,0].axvline(x=i, color='red', linestyle='--')
            axes[0,1].axvline(x=i, color='red', linestyle='--')
            axes[0,2].axvline(x=i, color='red', linestyle='--')
            axes[0,3].axvline(x=i, color='red', linestyle='--')
            axes[1,0].axvline(x=i, color='red', linestyle='--')
            axes[1,1].axvline(x=i, color='red', linestyle='--')
            axes[1,2].axvline(x=i, color='red', linestyle='--')
            axes[1,3].axvline(x=i, color='red', linestyle='--')

    if fig_filename:
        fig.savefig(fig_filename + ".png")
    #fig.show()
    plt.close()


def plot_results_from_csvs(folder_path, num_runs=3, num_rnd_cells=5):
    # Add folder_path to path
    if folder_path not in sys.path:
        sys.path.append(folder_path)

    # Plot loss, accuracy and trained parameters
    for j in range(num_runs):
        results_filename = os.path.join(folder_path,f'results_{j}.csv')
        results_fig_filename = os.path.join(folder_path,f'resultsfig_{j}')
    
        plot_results_from_csv(results_filename,results_fig_filename)
    
    # Plot tuning curves before pretraining, after pretraining and after training  
    num_mid_cells = 648
    num_sup_cells = 164
    num_runs_plotted = min(5,num_runs)    
    
    for j in range(num_runs_plotted):

        tc_pre_pretrain = os.path.join(folder_path,f'tc_prepre_{j}.csv')
        tc_post_pretrain =os.path.join(folder_path,f'tc_prepost_{j}.csv')
        tc_post_train =os.path.join(folder_path,f'tc_post_{j}.csv')
        df_tc_pre_pretrain = pd.read_csv(tc_pre_pretrain, header=0)
        df_tc_post_pretrain = pd.read_csv(tc_post_pretrain, header=0)
        df_tc_post_train = pd.read_csv(tc_post_train, header=0)

        # Select num_rnd_cells randomly selected cells to plot from both middle and superficial layer cells
        if j==0:
            fig, axes = plt.subplots(nrows=num_rnd_cells, ncols=2*num_runs_plotted, figsize=(10*num_runs_plotted, 25))
            mid_columns = df_tc_pre_pretrain.columns[0:num_mid_cells]
            sup_columns = df_tc_pre_pretrain.columns[num_mid_cells:num_mid_cells+num_sup_cells]
            selected_mid_columns = numpy.random.choice(mid_columns, size=num_rnd_cells, replace=False)
            selected_sup_columns = numpy.random.choice(sup_columns, size=num_rnd_cells, replace=False)
            N = len(df_tc_pre_pretrain[selected_mid_columns[0]])
        
        # Plot tuning curves
        for i in range(num_rnd_cells):    
            axes[i,2*j].plot(range(N), df_tc_pre_pretrain[selected_mid_columns[i]], label='pre-pretraining')
            axes[i,2*j].plot(range(N), df_tc_post_pretrain[selected_mid_columns[i]], label='post-pretraining')
            axes[i,2*j].plot(range(N), df_tc_post_train[selected_mid_columns[i]], label='post-training')
            axes[i,2*j+1].plot(range(N), df_tc_pre_pretrain[selected_sup_columns[i]], label='pre-pretraining')
            axes[i,2*j+1].plot(range(N), df_tc_post_pretrain[selected_sup_columns[i]], label='post-pretraining')
            axes[i,2*j+1].plot(range(N), df_tc_post_train[selected_sup_columns[i]], label='post-training')
            axes[i,2*j].legend(loc='upper left')
            axes[i,2*j+1].legend(loc='upper left')

        axes[0,2*j].set_title(f'Run{j+1} - Middle layer')
        axes[0,2*j+1].set_title(f'Run{j+1} - Superficial layer')
       
    # Save plot
    fig.savefig(os.path.join(folder_path,'tc_fig.png'))
    fig.show()
    plt.close()


def plot_losses_two_stage(
    training_losses, val_loss_per_epoch, epochs_plot=None, save=None, inset=None
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

    if epochs_plot == None:
        pass
    else:
        if np.isscalar(epochs_plot):
            axs1.axvline(x=epochs_plot, c="r")
            if inset:
                axs2.axvline(x=epochs_plot, c="r")
        else:
            axs1.axvline(x=epochs_plot[0], c="r")
            axs1.axvline(x=epochs_plot[0] + epochs_plot[1], c="r")
            axs1.axvline(x=epochs_plot[2], c="r")
            if inset:
                axs2.axvline(x=epochs_plot[0], c="r")
                axs2.axvline(x=epochs_plot[0] + epochs_plot[1], c="r")
                axs1.axvline(x=epochs_plot[2], c="r")
    if save:
        fig.savefig(save + ".png")
    plt.close()


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


def plot_r_ref(r_ref, epochs_plot=None, save=None):
    plt.plot(r_ref)
    plt.xlabel("Epoch")
    plt.ylabel("noise")

    if epochs_plot == None:
        pass
    else:
        if np.isscalar(epochs_plot):
            plt.axvline(x=epochs_plot, c="r")
        else:
            plt.axvline(x=epochs_plot[0], c="r")
            plt.axvline(x=epochs_plot[0] + epochs_plot[1], c="r")
            plt.axvline(x=epochs_plot[2], c="r")

    if save:
        plt.savefig(save + ".png")
    plt.show()
    plt.close()


def plot_max_rates(max_rates, epochs_plot=None, save=None):
    plt.plot(max_rates, label=["E_mid", "I_mid", "E_sup", "I_sup"])
    plt.xlabel("Epoch")
    plt.ylabel("Maximum rates")
    plt.legend()

    if epochs_plot == None:
        pass
    else:
        if np.isscalar(epochs_plot):
            plt.axvline(x=epochs_plot, c="r")
        else:
            plt.axvline(x=epochs_plot[0], c="r")
            plt.axvline(x=epochs_plot[0] + epochs_plot[1], c="r")
            plt.axvline(x=epochs_plot[2], c="r")

    if save:
        plt.savefig(save + ".png")
    plt.close()


def plot_w_sig(w_sig, epochs_plot=None, save=None):
    plt.plot(w_sig)
    plt.xlabel("Epoch")
    plt.ylabel("Values of w")
    if epochs_plot:
        plt.axvline(x=epochs_plot, c="r", label="criterion")
    if save:
        plt.savefig(save + ".png")
    plt.close()
