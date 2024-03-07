import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
import jax.numpy as np
import numpy
import sys


def calculate_relative_change(df):
    # Calculate relative changes in Jm and Js
    J_m_EE = numpy.exp(df['logJ_m_EE'])
    J_m_IE = numpy.exp(df['logJ_m_IE'])
    J_m_EI = -1 * numpy.exp(df['logJ_m_EI'])
    J_m_II = -1 * numpy.exp(df['logJ_m_II'])
    J_s_EE = numpy.exp(df['logJ_s_EE'])
    J_s_IE = numpy.exp(df['logJ_s_IE'])
    J_s_EI = -1 * numpy.exp(df['logJ_s_EI'])
    J_s_II = -1 * numpy.exp(df['logJ_s_II'])
    f_E = numpy.exp(df['f_E'])
    f_I = -1 * numpy.exp(df['f_I'])

    relative_changes = numpy.zeros((18,2))

    # Calculate relative changes in parameters and other metrics before and after training
    train_start_ind = df.index[df['stage'] == 1][0]
    relative_changes[0,1] =(J_m_EE.iloc[-1] - J_m_EE.iloc[train_start_ind]) / J_m_EE.iloc[train_start_ind]
    relative_changes[1,1] =(J_m_IE.iloc[-1] - J_m_IE.iloc[train_start_ind]) / J_m_IE.iloc[train_start_ind]
    relative_changes[2,1] =(J_m_EI.iloc[-1] - J_m_EI.iloc[train_start_ind]) / J_m_EI.iloc[train_start_ind]
    relative_changes[3,1] =(J_m_II.iloc[-1] - J_m_II.iloc[train_start_ind]) / J_m_II.iloc[train_start_ind]
    relative_changes[4,1] =(J_s_EE.iloc[-1] - J_s_EE.iloc[train_start_ind]) / J_s_EE.iloc[train_start_ind]
    relative_changes[5,1] =(J_s_IE.iloc[-1] - J_s_IE.iloc[train_start_ind]) / J_s_IE.iloc[train_start_ind]
    relative_changes[6,1] =(J_s_EI.iloc[-1] - J_s_EI.iloc[train_start_ind]) / J_s_EI.iloc[train_start_ind]
    relative_changes[7,1] =(J_s_II.iloc[-1] - J_s_II.iloc[train_start_ind]) / J_s_II.iloc[train_start_ind]
    relative_changes[8,1] = (df['c_E'].iloc[-1] - df['c_E'].iloc[train_start_ind]) / df['c_E'].iloc[train_start_ind]
    relative_changes[9,1] = (df['c_I'].iloc[-1] - df['c_I'].iloc[train_start_ind]) / df['c_I'].iloc[train_start_ind]
    relative_changes[10,1] = (f_E.iloc[-1] - f_E.iloc[train_start_ind]) / f_E.iloc[train_start_ind]
    relative_changes[11,1] = (f_I.iloc[-1] - f_I.iloc[train_start_ind]) / f_I.iloc[train_start_ind]

    relative_changes[12,1] = (df['acc'].iloc[-1] - df['acc'].iloc[train_start_ind]) / df['acc'].iloc[train_start_ind] # accuracy
    for column in df.columns:
        if 'offset' in column:
            relative_changes[13,1] = (df[column].iloc[-1] - df[column].iloc[train_start_ind]) / df[column].iloc[train_start_ind] # offset threshold
    relative_changes[14,1] = (df['maxr_E_mid'].iloc[-1] - df['maxr_E_mid'].iloc[train_start_ind]) / df['maxr_E_mid'].iloc[train_start_ind] # r_mid
    relative_changes[15,1] = (df['maxr_I_mid'].iloc[-1] - df['maxr_I_mid'].iloc[train_start_ind]) / df['maxr_I_mid'].iloc[train_start_ind] # r_mid
    relative_changes[16,1] = (df['maxr_E_sup'].iloc[-1] - df['maxr_E_sup'].iloc[train_start_ind]) / df['maxr_E_sup'].iloc[train_start_ind] # r_mid
    relative_changes[17,1] = (df['maxr_I_sup'].iloc[-1] - df['maxr_I_sup'].iloc[train_start_ind]) / df['maxr_I_sup'].iloc[train_start_ind] # r_sup

    # Calculate relative changes in parameters and other metrics before and after pretraining
    pretraining_on = float(np.sum(df['stage'].ravel() == 0))>0
    if pretraining_on:
        pretrain_start_ind = df.index[df['stage'] == 0][0]
        relative_changes[0,0] =(J_m_EE.iloc[train_start_ind] - J_m_EE.iloc[pretrain_start_ind]) / J_m_EE.iloc[pretrain_start_ind]
        relative_changes[1,0] =(J_m_IE.iloc[train_start_ind] - J_m_IE.iloc[pretrain_start_ind]) / J_m_IE.iloc[pretrain_start_ind]
        relative_changes[2,0] =(J_m_EI.iloc[train_start_ind] - J_m_EI.iloc[pretrain_start_ind]) / J_m_EI.iloc[pretrain_start_ind]
        relative_changes[3,0] =(J_m_II.iloc[train_start_ind] - J_m_II.iloc[pretrain_start_ind]) / J_m_II.iloc[pretrain_start_ind]
        relative_changes[4,0] =(J_s_EE.iloc[train_start_ind] - J_s_EE.iloc[pretrain_start_ind]) / J_s_EE.iloc[pretrain_start_ind]
        relative_changes[5,0] =(J_s_IE.iloc[train_start_ind] - J_s_IE.iloc[pretrain_start_ind]) / J_s_IE.iloc[pretrain_start_ind]
        relative_changes[6,0] =(J_s_EI.iloc[train_start_ind] - J_s_EI.iloc[pretrain_start_ind]) / J_s_EI.iloc[pretrain_start_ind]
        relative_changes[7,0] =(J_s_II.iloc[train_start_ind] - J_s_II.iloc[pretrain_start_ind]) / J_s_II.iloc[pretrain_start_ind]
        relative_changes[8,0] = (df['c_E'].iloc[train_start_ind] - df['c_E'].iloc[pretrain_start_ind]) / df['c_E'].iloc[pretrain_start_ind]
        relative_changes[9,0] = (df['c_I'].iloc[train_start_ind] - df['c_I'].iloc[pretrain_start_ind]) / df['c_I'].iloc[pretrain_start_ind]
        relative_changes[10,0] = (df['f_E'].iloc[train_start_ind] - df['f_E'].iloc[pretrain_start_ind]) / df['f_E'].iloc[pretrain_start_ind]
        relative_changes[11,0] = (df['f_I'].iloc[train_start_ind] - df['f_I'].iloc[pretrain_start_ind]) / df['f_I'].iloc[pretrain_start_ind]
        
        relative_changes[12,0] = (df['acc'].iloc[-1] - df['acc'].iloc[train_start_ind]) / df['acc'].iloc[train_start_ind] # accuracy
        for column in df.columns:
            if 'offset' in column:
                relative_changes[13,1] = (df[column].iloc[train_start_ind] - df[column].iloc[pretrain_start_ind]) / df[column].iloc[pretrain_start_ind] # offset threshold
        relative_changes[14,0] = (df['maxr_E_mid'].iloc[train_start_ind] - df['maxr_E_mid'].iloc[pretrain_start_ind]) / df['maxr_E_mid'].iloc[pretrain_start_ind] # r_mid
        relative_changes[15,0] = (df['maxr_I_mid'].iloc[train_start_ind] - df['maxr_I_mid'].iloc[pretrain_start_ind]) / df['maxr_I_mid'].iloc[pretrain_start_ind] # r_mid
        relative_changes[16,0] = (df['maxr_E_sup'].iloc[train_start_ind] - df['maxr_E_sup'].iloc[pretrain_start_ind]) / df['maxr_E_sup'].iloc[pretrain_start_ind] # r_mid
        relative_changes[17,0] = (df['maxr_I_sup'].iloc[train_start_ind] - df['maxr_I_sup'].iloc[pretrain_start_ind]) / df['maxr_I_sup'].iloc[pretrain_start_ind] # r_sup
    
    return relative_changes


def barplots_from_csvs(directory, plot_filename = None):
    # List to store relative changes from each file
    relative_changes_pretrain = []
    relative_changes_train = []

    # Iterate through each file in the directory
    numFiles = 0
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and filename.startswith('result'):
            numFiles = numFiles + 1
            filepath = os.path.join(directory, filename)
            # Read CSV file
            df = pd.read_csv(filepath)
            # Calculate relative change
            relative_changes = 100 * calculate_relative_change(df)
            relative_changes_pretrain.append(relative_changes[:,0])
            relative_changes_train.append(relative_changes[:,1])
    
    # Plotting bar plots
    # Define groups of parameters and plot each parameter group
    groups = [
        ['J_m_EE', 'J_m_IE', 'J_m_EI', 'J_m_II'],
        ['J_s_EE', 'J_s_IE', 'J_s_EI', 'J_s_II'],
        ['c_E', 'c_I'], 
        ['f_E', 'f_I']
    ]
    num_groups = len(groups)
    fig, axs = plt.subplots(2, num_groups, figsize=(5*num_groups, 10))  # Create subplots for each group
    
    relative_changes_train = numpy.array(relative_changes_train)
    relative_changes_pretrain = numpy.array(relative_changes_pretrain)
    group_start_ind = [0,4,8,10,12] # putting together Jm, Js, c, f
    titles_pretrain= ['Jm changes in pretraining, {} runs'.format(numFiles),'Js changes in pretraining, {} runs'.format(numFiles),'c changes in pretraining, {} runs'.format(numFiles), 'f changes in pretraining, {} runs'.format(numFiles)]
    titles_train=['Jm changes in training, {} runs'.format(numFiles), 'Js changes in training, {} runs'.format(numFiles), 'c changes in training, {} runs'.format(numFiles), 'f changes in training, {} runs'.format(numFiles)]
    J_box_colors = ['tab:red','tab:red','tab:blue','tab:blue']
    c_f_box_colors = ['tab:red','tab:blue']
    if np.sum(np.abs(relative_changes[:,0])) >0:
        for i, group in enumerate(groups):
            group_data = relative_changes_pretrain[:, group_start_ind[i]:group_start_ind[i+1]]  # Extract data for the current group
            bp = axs[0,i].boxplot(group_data, labels=group, patch_artist=True)
            if i<2:
                for box, color in zip(bp['boxes'], J_box_colors):
                    box.set_facecolor(color)
            else:
                for box, color in zip(bp['boxes'], c_f_box_colors):
                    box.set_facecolor(color)
            axs[0,i].axhline(y=0, color='black', linestyle='--')
            axs[0,i].set_title(titles_pretrain[i])
            axs[0,i].set_ylabel('rel change in %')
    for i, group in enumerate(groups):
        group_data = relative_changes_train[:, group_start_ind[i]:group_start_ind[i+1]]  # Extract data for the current group
        bp = axs[1,i].boxplot(group_data, labels=group, patch_artist=True)
        if i<2:
            for box, color in zip(bp['boxes'], J_box_colors):
                box.set_facecolor(color)
        else:
            for box, color in zip(bp['boxes'], c_f_box_colors):
                box.set_facecolor(color)
        axs[1,i].axhline(y=0, color='black', linestyle='--')
        axs[1,i].set_title(titles_train[i])
        axs[1,i].set_ylabel('rel change in %')
        
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
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(35, 45))

    # Plot accuracy and losses
    for column in df.columns:
        if 'acc' in column and 'val_' not in column:
            axes[0,0].plot(range(N), df[column], label=column, alpha=0.6, c='tab:green')
        if 'val_acc' in column:
            axes[0,0].scatter(range(N), df[column], label=column, marker='o', s=50, c='green')
    axes[0,0].legend(loc='lower right')
    axes[0,0].set_title('Accuracy', fontsize=20)
    axes[0,0].axhline(y=0.5, color='black', linestyle='--')
    axes[0,0].set_ylim(0, 1)
    axes[0,0].set_xlabel('SGD steps', fontsize=20)

    for column in df.columns:
        if 'loss_' in column and 'val_loss' not in column:
            axes[1,0].plot(range(N), df[column], label=column, alpha=0.6)
        if 'val_loss' in column:
            axes[1,0].scatter(range(N), df[column], marker='o', s=50)
    axes[1,0].legend(loc='upper right')
    axes[1,0].set_title('Loss', fontsize=20)
    axes[1,0].set_xlabel('SGD steps', fontsize=20)

    # BARPLOTS about relative changes
    categories_params = ['Jm_EE', 'Jm_IE', 'Jm_EI', 'Jm_II', 'Js_EE', 'Js_IE', 'Js_EI', 'Js_II']
    categories_metrics = [ 'c_E', 'c_I', 'f_E', 'f_I', 'acc', 'offset', 'rm_E', 'rm_I', 'rs_E','rs_I']
    rel_changes = calculate_relative_change(df)
    for i_train_pretrain in range(2):
        values_params = 100 * rel_changes[0:8,i_train_pretrain]
        values_metrics = 100 * rel_changes[8:18,i_train_pretrain]

        # Choosing colors for each bar
        colors_params = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:blue']
        colors_metrics = [ 'tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'tab:green', 'tab:brown', 'tab:red', 'tab:blue', 'tab:red', 'tab:blue']

        # Creating the bar plot
        row_i = 2+i_train_pretrain
        bars_params = axes[row_i,1].bar(categories_params, values_params, color=colors_params)
        bars_metrics = axes[row_i,2].bar(categories_metrics, values_metrics, color=colors_metrics)

        # Annotating each bar with its value for bars_params
        for bar in bars_params:
            yval = bar.get_height()
            # Adjust text position for positive values to be inside the bar
            if abs(yval) > 2:
                if yval > 0:
                    text_position = yval - 0.1*max(abs(values_params))
                else:
                    text_position = yval + 0.05*max(abs(values_params))
            else:
                text_position = yval
            axes[row_i,1].text(bar.get_x() + bar.get_width() / 2, text_position, f'{yval:.2f}', ha='center', va='bottom', fontsize=20)
        for bar in bars_metrics:
            yval = bar.get_height()
            if abs(yval) > 2:
                if yval > 0:
                    text_position = yval - 0.2*max(abs(values_params))
                else:
                    text_position = yval + 0.05*max(abs(values_params))
            else:
                text_position = yval
            axes[row_i,2].text(bar.get_x() + bar.get_width() / 2, text_position, f'{yval:.2f}', ha='center', va='bottom', fontsize=20)

        # Adding labels and titles
        axes[row_i,1].set_ylabel('relative change %', fontsize=20)
        axes[row_i,2].set_ylabel('relative change %', fontsize=20)
        axes[row_i,1].set_title('Rel changes in J before and after training', fontsize=20)
        axes[row_i,2].set_title('Other rel changes before and after training', fontsize=20)
        axes[row_i,2].axvline(x=3.5, color='black', linestyle='--')
    
    ################
    num_pretraining_steps= sum(df['stage'] == 0)
    for column in df.columns:
        if 'offset' in column:
            axes[2,0].plot(range(num_pretraining_steps), np.ones(num_pretraining_steps)*5, label='stopping threshold', alpha=0.6, c='tab:brown')
            axes[2,0].scatter(range(num_pretraining_steps), df[column][0:num_pretraining_steps], label='offsets at bl acc', marker='o', s=50, c='tab:brown')
            axes[2,0].grid(color='gray', linestyle='-', linewidth=0.5)
            axes[2,0].set_title('Offset with accuracy 0.749', fontsize=20)
            axes[2,0].set_ylabel('degrees', fontsize=20)
            axes[2,0].set_xlabel('SGD steps', fontsize=20)
            axes[2,0].set_ylim(0, 50)
            if num_pretraining_steps==0:#if there was no pretraining
                axes[3,0].plot(range(N), df[column], label='offset', c='tab:brown')
                axes[3,0].grid(color='gray', linestyle='-', linewidth=0.5)
                axes[3,0].set_ylim(0, max(df[column])+1)
                axes[3,0].set_title('Offset during staircase training', fontsize=20)
            else:
                axes[3,0].plot(range(num_pretraining_steps,N), df[column][num_pretraining_steps:N], label='offset', c='tab:brown')
                axes[3,0].grid(color='gray', linestyle='-', linewidth=0.5)
                axes[3,0].set_ylim(0, max(df[column][num_pretraining_steps:N])+1)
                axes[3,0].set_title('Offset during staircase training', fontsize=20)
            axes[3,0].set_ylabel('degrees', fontsize=20)
            axes[3,0].set_xlabel('SGD steps', fontsize=20)            
    
    #Plot changes in sigmoid weights and bias of the sigmoid layer
    axes[1,1].plot(range(N), df['b_sig'], label='b_sig', linestyle='--', linewidth = 3)
    axes[1,1].set_xlabel('SGD steps', fontsize=20)
    i=0
    for column in df.columns:
        if 'w_sig_' in column and i<10:
            axes[1,1].plot(range(N), df[column], label=column)
            i = i+1
    axes[1,1].set_title('Readout bias and weights', fontsize=20)
    axes[1,1].legend(loc='lower right')

    # Plot changes in J_m and J_s
    axes[0,2].plot(range(N), numpy.exp(df['logJ_m_EE']), label='J_m_EE', linestyle='--', c='tab:red',linewidth=3)
    axes[0,2].plot(range(N), numpy.exp(df['logJ_m_IE']), label='J_m_IE', linestyle='--', c='tab:orange',linewidth=3)
    axes[0,2].plot(range(N), -numpy.exp(df['logJ_m_II']), label='J_m_II', linestyle='--', c='tab:blue',linewidth=3)
    axes[0,2].plot(range(N), -numpy.exp(df['logJ_m_EI']), label='J_m_EI', linestyle='--', c='tab:green',linewidth=3)
    
    axes[0,2].plot(range(N), numpy.exp(df['logJ_s_EE']), label='J_s_EE', c='tab:red',linewidth=3)
    axes[0,2].plot(range(N), numpy.exp(df['logJ_s_IE']), label='J_s_IE', c='tab:orange',linewidth=3)
    axes[0,2].plot(range(N), -numpy.exp(df['logJ_s_II']), label='J_s_II', c='tab:blue',linewidth=3)
    axes[0,2].plot(range(N), -numpy.exp(df['logJ_s_EI']), label='J_s_EI', c='tab:green',linewidth=3)
    axes[0,2].legend(loc="upper left", fontsize=20)
    axes[0,2].set_title('J in middle and superficial layers', fontsize=20)
    axes[0,2].set_xlabel('SGD steps', fontsize=20)

    # Plot maximum rates
    colors = ["tab:blue", "tab:red"]
    axes[0,1].plot(range(N), df['maxr_E_mid'], label='maxr_E_mid', c=colors[1], linestyle=':')
    axes[0,1].plot(range(N), df['maxr_I_mid'], label='maxr_I_mid', c=colors[0], linestyle=':')
    axes[0,1].plot(range(N), df['maxr_E_sup'], label='maxr_E_sup', c=colors[1])
    axes[0,1].plot(range(N), df['maxr_I_sup'], label='maxr_I_sup', c=colors[0])
    axes[0,1].legend(loc="upper left", fontsize=20)
    axes[0,1].set_title('Maximum rates', fontsize=20)
    axes[0,1].set_xlabel('SGD steps', fontsize=20)

    #Plot changes in baseline inhibition and excitation and feedforward weights (second stage of the training)
    axes[1,2].plot(range(N), df['c_E'], label='c_E',c='tab:red',linewidth=3)
    axes[1,2].plot(range(N), df['c_I'], label='c_I',c='tab:blue',linewidth=3)

    #Plot feedforward weights from middle to superficial layer (second stage of the training)
    axes[1,2].plot(range(N), numpy.exp(df['f_E']), label='f_E', linestyle='--',c='tab:red',linewidth=3)
    axes[1,2].plot(range(N), numpy.exp(df['f_I']), label='f_I', linestyle='--',c='tab:blue',linewidth=3)
    axes[1,2].set_title('c: constant inputs, f: weights between mid and sup layers', fontsize=20)

    for i in range(1, len(df['stage'])):
        if df['stage'][i] != df['stage'][i-1]:
            axes[0,0].axvline(x=i, color='black', linestyle='--')
            axes[0,1].axvline(x=i, color='black', linestyle='--')
            axes[0,2].axvline(x=i, color='black', linestyle='--')
            axes[1,0].axvline(x=i, color='black', linestyle='--')
            axes[1,1].axvline(x=i, color='black', linestyle='--')
            axes[1,2].axvline(x=i, color='black', linestyle='--')
    for i in range(4):
        for j in range(3):
            axes[i,j].tick_params(axis='both', which='major', labelsize=18)  # Set tick font size
            axes[i,j].legend(fontsize=20)

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
    tc_post_pretrain =os.path.join(folder_path,f'tc_postpre_0.csv')
    pretrain_ison = os.path.exists(tc_post_pretrain)
    for j in range(num_runs_plotted):

        tc_pre_pretrain = os.path.join(folder_path,f'tc_prepre_{j}.csv')
        tc_post_pretrain =os.path.join(folder_path,f'tc_postpre_{j}.csv')
        tc_post_train =os.path.join(folder_path,f'tc_post_{j}.csv')
        df_tc_pre_pretrain = pd.read_csv(tc_pre_pretrain, header=0)
        if pretrain_ison:
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
            axes[i,2*j].plot(range(N), df_tc_pre_pretrain[selected_mid_columns[i]], label='pre-pretraining',linewidth=2)
            axes[i,2*j+1].plot(range(N), df_tc_pre_pretrain[selected_sup_columns[i]], label='pre-pretraining',linewidth=2)
            if pretrain_ison:
                axes[i,2*j].plot(range(N), df_tc_post_pretrain[selected_mid_columns[i]], label='post-pretraining',linewidth=2)
                axes[i,2*j+1].plot(range(N), df_tc_post_pretrain[selected_sup_columns[i]], label='post-pretraining',linewidth=2)
            axes[i,2*j].plot(range(N), df_tc_post_train[selected_mid_columns[i]], label='post-training',linewidth=2)
            axes[i,2*j+1].plot(range(N), df_tc_post_train[selected_sup_columns[i]], label='post-training',linewidth=2)
            axes[i,2*j].legend(loc='upper left', fontsize=20)
            axes[i,2*j+1].legend(loc='upper left', fontsize=20)

        axes[0,2*j].set_title(f'Run{j+1} - Middle layer', fontsize=20)
        axes[0,2*j+1].set_title(f'Run{j+1} - Superficial layer', fontsize=20)
       
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


def plot_pre_post_scatter(x_axis, y_axis, orientations, indices_to_plot, title, save_dir = None):
    
    '''
    Create scatter plot for pre and post training responses. Colour represents preferred orientation according to Schoups et al bins
    '''
    
    #Create legend
    patches = []
    cmap = plt.get_cmap('rainbow')
    colors = numpy.flip(cmap(numpy.linspace(0,1, 8)), axis = 0)
    bins = ['0-4', '4-12', '12-20', '20-28', '28-36', '36-44', '44-50', '+50']
    for j in range(0,len(colors)):
        patches.append(mpatches.Patch(color=colors[j], label=bins[j]))
    
    #Iterate through required neurons
    for idx in indices_to_plot:
        #Select bin and colour
        if np.abs(orientations[idx]) <4:
            colour = colors[0]
            label = bins[0]
        elif np.abs(orientations[idx]) >50:
            colour = colors[-1]
            label = bins[-1]
        else:
            colour = colors[int(1+np.floor((np.abs(orientations[idx]) -4)/8) )]
            label = bins[int(1+np.floor((np.abs(orientations[idx]) -4)/8) )]
        plt.scatter(x = x_axis[idx], y =y_axis[idx], color =colour, label = label )
    
    #Plot x = y line
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='gold')
    plt.xlabel('Pre training')
    plt.ylabel('Post training')
    plt.title(title)
    plt.legend(handles = patches, loc = 'upper right', bbox_to_anchor=(1.3, 1.0), title = 'Pref ori - train ori')
    if save_dir:
        plt.savefig(os.path.join(save_dir, str(title)+'.png'), bbox_inches='tight')
    plt.show()