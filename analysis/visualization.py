import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import jax.numpy as np
import numpy
import seaborn as sns
import statsmodels.api as sm
import scipy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from analysis.analysis_functions import rel_change_for_run, rel_change_for_runs, MVPA_param_offset_correlations, data_from_run, exclude_runs, param_offset_correlations
from util import filter_for_run_and_stage, check_header

plt.rcParams['xtick.labelsize'] = 12 # Set the size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12 # Set the size for y-axis tick labels

########### Plotting functions ##############
def boxplots_from_csvs(folder, save_folder, num_time_inds = 3, num_training = 1):
    def scatter_data_with_lines(ax, data):
        for i in range(2):
            group_data = data[i, :]
            # let the position be i 
            x_positions = [i for _ in range(len(group_data))]
            ax.scatter(x_positions, group_data, color='black', alpha=0.7)  # Scatter plot of individual data points
        
        # Draw lines connecting the points between the two groups
        for j in range(len(data[0])):
            ax.plot([0, 1], [data[0, j], data[1, j]], color='black', alpha=0.2)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Group 1', 'Group 2'])

    # Iterate through each file in the directory
    for i in range(num_training):
        df_i, stage_time_inds = data_from_run(folder, run_index=i, num_indices=num_time_inds)
        train_end_ind = stage_time_inds[-1]
        if num_time_inds>2:
            pretrain_start_ind = stage_time_inds[0]
            train_start_ind = stage_time_inds[1]
        else:
            train_start_ind = stage_time_inds[0]
        if i==0:
            vals_pre = df_i.iloc[[train_start_ind]]
            vals_post = df_i.iloc[[train_end_ind]]
            if num_time_inds>2:
                vals_prepre = df_i.iloc[[pretrain_start_ind]]
        else:
            vals_pre=pd.concat([vals_pre,df_i.iloc[[train_start_ind]]], ignore_index=True)
            vals_post=pd.concat([vals_post, df_i.iloc[[train_end_ind]]], ignore_index=True)
            if num_time_inds>2:
                vals_prepre=pd.concat([vals_prepre, df_i.iloc[[pretrain_start_ind]]], ignore_index=True)
    
    means_pre = vals_pre.mean()
    means_post = vals_post.mean()    

    ################# Plotting bar plots of offset before and after given time indices #################
    # Create figure and axis
    fig, ax = plt.subplots()
    # Colors for bars
    colors = ['blue', 'green']

    # Bar plot for mean values
    offset_means = [means_pre['staircase_offset'], means_post['staircase_offset']]
    bars = ax.bar(['Pre', 'Post'], offset_means, color=colors, alpha=0.7)
    # Annotate each bar with its value
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=20)

    # Plot individual data points and connect them
    scatter_data_with_lines(ax, numpy.array([vals_pre['staircase_offset'], vals_post['staircase_offset']]))

    # Save plot
    full_path = save_folder + '/offset_pre_post.png'
    fig.savefig(full_path)
    plt.close()

    ################# Plotting bar plots of J parameters before and after given time indices #################
    # Create figure and axis
    fig, ax = plt.subplots(2,4, figsize=(20, 10))
    # Colors for bars
    colors=['red' ,'tab:red','blue', 'tab:blue' ,'red' ,'tab:red', 'blue', 'tab:blue']
    keys_J = ['J_EE_m', 'J_IE_m', 'J_EI_m', 'J_II_m', 'J_EE_s', 'J_IE_s', 'J_EI_s', 'J_II_s']
    J_means_pre = [means_pre['J_EE_m'], means_pre['J_IE_m'], -means_pre['J_EI_m'], -means_pre['J_II_m'], means_pre['J_EE_s'], means_pre['J_IE_s'], -means_pre['J_EI_s'], -means_pre['J_II_s']]
    J_means_post = [means_post['J_EE_m'], means_post['J_IE_m'], -means_post['J_EI_m'], -means_post['J_II_m'], means_post['J_EE_s'], means_post['J_IE_s'], -means_post['J_EI_s'], -means_post['J_II_s']]
    ax_flat = ax.flatten()

    for i in range(8):
        bars = ax_flat[i].bar(['Pre', 'Post'], [J_means_pre[i], J_means_post[i]], color=colors[i], alpha=0.7)
        ax_flat[i].set_title(keys_J[i])
        for bar in bars:
            yval = bar.get_height()
            ax_flat[i].text(bar.get_x() + bar.get_width() / 2, 0.9*yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=20)
        scatter_data_with_lines(ax_flat[i], numpy.abs(numpy.array([vals_pre[keys_J[i]], vals_post[keys_J[i]]])))

    # Save plot
    full_path = save_folder + '/J_pre_post.png'
    fig.savefig(full_path)
    plt.close()

    ################# Plotting bar plots of loss parameters before and after  #################
    # Create figure and axis
    fig, ax = plt.subplots(2,4, figsize=(20, 10))
    # Colors for bars
    colors=['red' ,'blue', 'tab:red', 'tab:blue' ,'red' , 'blue','tab:red', 'tab:blue']
    keys_r = ['maxr_E_mid', 'maxr_I_mid', 'maxr_E_sup', 'maxr_I_sup', 'meanr_E_mid', 'meanr_I_mid', 'meanr_E_sup', 'meanr_I_sup']
    r_means_pre = [means_pre[keys_r[i]] for i in range(8)]
    r_means_post = [means_post[keys_r[i]] for i in range(8)]
    ax_flat = ax.flatten()

    for i in range(8):
        bars = ax_flat[i].bar(['Pre', 'Post'], [r_means_pre[i], r_means_post[i]], color=colors[i], alpha=0.7)
        ax_flat[i].set_title(keys_r[i])
        for bar in bars:
            yval = bar.get_height()
            ax_flat[i].text(bar.get_x() + bar.get_width() / 2, 0.9*yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=20)
        scatter_data_with_lines(ax_flat[i], numpy.abs(numpy.array([vals_pre[keys_r[i]], vals_post[keys_r[i]]])))

    # Save plot
    full_path = save_folder + '/r_pre_post.png'
    fig.savefig(full_path)
    plt.close()

    ################# Boxplots for relative parameter changes #################

    rel_changes_train, rel_changes_pretrain = rel_change_for_runs(folder, num_indices=3, num_runs=num_training)

    # Define groups of parameters and plot each parameter group
    group_labels = [
        [r'$\Delta J^{\text{mid}}_{E \rightarrow E}$', r'$\Delta J^{\text{mid}}_{E \rightarrow I}$', r'$\Delta J^{\text{mid}}_{I \rightarrow E}$', r'$\Delta J^{\text{mid}}_{I \rightarrow I}$'],
        [r'$\Delta J^{\text{sup}}_{E \rightarrow E}$', r'$\Delta J^{\text{sup}}_{E \rightarrow I}$', r'$\Delta J^{\text{sup}}_{I \rightarrow E}$', r'$\Delta J^{\text{sup}}_{I \rightarrow I}$'],
        [r'$\Delta c^{\text{mid}}_{\rightarrow E}$', r'$\Delta c^{\text{mid}}_{\rightarrow I}$', r'$\Delta c^{\text{sup}}_{\rightarrow E}$', r'$\Delta c^{\text{sup}}_{\rightarrow I}$'],
        [r'$\Delta f^{\text{mid}\rightarrow \text{sup}}_{E \rightarrow E}$', r'$\Delta f^{\text{mid}\rightarrow \text{sup}}_{E \rightarrow I}$'],
        [r'$\Delta \kappa^{\text{pre}}_{E \rightarrow  E}$',r'$\Delta \kappa^{\text{pre}}_{E \rightarrow  I}$',r'$\Delta \kappa^{\text{post}}_{E \rightarrow  E}$',r'$\Delta \kappa^{\text{post}}_{E \rightarrow  I}$']
    ]
    keys_group = [keys_J[:4], keys_J[4:], ['cE_m', 'cI_m','cE_s', 'cI_s'], ['f_E', 'f_I'], ['kappa_EE_pre','kappa_IE_pre','kappa_EE_post','kappa_IE_post']]
    num_groups = len(group_labels)    

    fig, axs = plt.subplots(num_time_inds-1, num_groups, figsize=(5*num_groups, 5*(num_time_inds-1)))  # Create subplots for each group
    axes_flat = axs.flatten()
    
    J_box_colors = ['tab:red','tab:red','tab:blue','tab:blue']
    c_box_colors = ['orange', 'orange', 'orange', 'orange']
    f_box_colors = ['orange', 'orange']
    kappa_box_colors = ['tab:green','tab:green','tab:green','tab:green']
    box_colors = [J_box_colors,J_box_colors,c_box_colors, f_box_colors, kappa_box_colors]
    for j in range(num_time_inds-1):
        for i, label in enumerate(group_labels):
            group_data = numpy.zeros((num_training, len(keys_group[i])))
            for var_ind in range(len(keys_group[i])):
                if j==0:
                    if keys_group[i][var_ind] in rel_changes_pretrain.keys():
                        group_data[:,var_ind]=rel_changes_pretrain[keys_group[i][var_ind]].T
                    else:
                        group_data[:,var_ind]=numpy.zeros(num_training)
                else:
                    group_data[:,var_ind]=rel_changes_train[keys_group[i][var_ind]].T
            bp = axes_flat[j*num_groups+i].boxplot(group_data,labels=label,patch_artist=True)
            for median in bp['medians']:
                median.set(color='black', linewidth=2.5)
            for box, color in zip(bp['boxes'], box_colors[i]):
                box.set_facecolor(color)
            axes_flat[j*num_groups+i].axhline(y=0, color='black', linestyle='--')
            axes_format(axes_flat[j*num_groups+i], fs_ticks=20, ax_width=2, tick_width=5, tick_length=10, xtick_flag=False)
            if i==num_groups-1:
                axes_flat[j*num_groups+i].set_ylabel('Change from 0 init', fontsize=20)
            else:
                axes_flat[j*num_groups+i].set_ylabel('Relative change (%)', fontsize=20)
    
    plt.tight_layout()
    
    full_path = save_folder + '/boxplot_relative_changes' + ".png"
    fig.savefig(full_path)

    plt.close()


def axes_format(axs, fs_ticks=20, ax_width=2, tick_width=5, tick_length=10, xtick_flag=True, ytick_flag=True):
    # Adjust axes line width (spines thickness)
    for spine in axs.spines.values():
        spine.set_linewidth(ax_width)

    # Reduce the number of ticks to 2
    if xtick_flag:
        xtick_loc = axs.get_xticks()
        if len(xtick_loc)>5:
            axs.set_xticks(xtick_loc[numpy.array([1,-2])])
        else:
            axs.set_xticks(xtick_loc[numpy.array([0,-1])])
    if ytick_flag:
        ytick_loc = axs.get_yticks()
        if len(ytick_loc)>5:
            axs.set_yticks(ytick_loc[numpy.array([1,-2])])
        else:
            axs.set_yticks(ytick_loc[numpy.array([0,-1])])
    axs.tick_params(axis='both', which='major', labelsize=fs_ticks, width=tick_width, length=tick_length)


def plot_results_from_csv(folder,run_index = 0, fig_filename=None):
    
    def annotate_bar(ax, bars, values):
        """Annotate each bar with its value"""
        for bar in bars:
            yval = bar.get_height()
            # Adjust text position to be inside the bar
            if abs(yval) > 2:
                if yval > 0:
                    text_position = yval - 0.1*max(abs(numpy.array(values)))
                else:
                    text_position = yval + 0.05*max(abs(numpy.array(values)))
            else:
                text_position = yval
            ax.text(bar.get_x() + bar.get_width() / 2, text_position, f'{yval:.2f}', ha='center', va='bottom', fontsize=20)

    # Create a subplot with 4 rows and 3 columns
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(60, 45))
    df, time_inds  = data_from_run(folder, run_index=run_index, num_indices=3)
    N = time_inds[-1]+1

    ################ Plot changes in accuracy and losses over time ################
    for column in df.columns:
        if 'acc' in column and 'val_' not in column:
            axes[0,0].plot(range(N), df[column], label=column, alpha=0.6, c='tab:green')
        if 'val_acc' in column:
            axes[0,0].scatter(range(N), df[column], label=column, marker='o', s=50, c='green')
    axes[0,0].legend(loc='lower right', fontsize=20)
    axes[0,0].set_title('Accuracy', fontsize=20)
    axes[0,0].axhline(y=0.5, color='black', linestyle='--')
    axes[0,0].set_ylim(0, 1)
    axes[0,0].set_xlabel('SGD steps', fontsize=20)

    for column in df.columns:
        if 'loss_' in column and 'val_loss' not in column:
            axes[0,1].plot(range(N), df[column], label=column, alpha=0.6)
        if 'val_loss' in column:
            axes[0,1].scatter(range(N), df[column], label='val_loss', marker='o', s=50)
    axes[0,1].legend(loc='upper right', fontsize=20)
    axes[0,1].set_title('Loss', fontsize=20)
    axes[0,1].set_xlabel('SGD steps', fontsize=20)
   
    ################ Barplots ################
    # Get the relative changes of the parameters
    rel_changes_train, _, _ = rel_change_for_run(folder, run_index, 3)
    keys_J = [key for key in rel_changes_train.keys() if key.startswith('J_')]
    values_J = [rel_changes_train[key] for key in keys_J] 
    keys_metrics =  [key for key in rel_changes_train.keys() if '_offset' in key or key.startswith('acc')]
    values_metrics = [rel_changes_train[key] for key in keys_metrics] 
    keys_meanr = [key for key in rel_changes_train.keys() if key.startswith('meanr_')]
    values_meanr = [rel_changes_train[key] for key in keys_meanr] 
    keys_fc = [key for key in rel_changes_train.keys() if key.startswith('c_') or key.startswith('f_')]
    values_fc = [rel_changes_train[key]for key in keys_fc]

    # exclude the run from further analysis if for both staircase and psychometric offsets more than 8 values out of the last 10 were above 10 degrees
    exclude_run = sum(df[keys_metrics[1]][-11:-1] > 9) > 8 and sum(df[keys_metrics[2]][-11:-1] > 9) > 8

    # Choosing colors for each bar
    colors_J = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:blue']
    colors_cf = ['tab:orange', 'tab:green','tab:red', 'tab:blue', 'tab:red', 'tab:blue']
    colors_metrics = [ 'tab:green', 'tab:orange', 'tab:brown']
    colors_r = ['tab:red', 'tab:blue', 'tab:red', 'tab:blue']

    # Creating the bar plot
    bars_metrics = axes[0,3].bar(keys_metrics, values_metrics, color=colors_metrics)
    bars_r = axes[1,3].bar(keys_meanr, values_meanr, color=colors_r)
    bars_params = axes[2,2].bar(keys_J, values_J, color=colors_J)    
    bars_cf = axes[2,3].bar(keys_fc, values_fc, color=colors_cf)    

    # Annotating each bar with its value for bars_params
    annotate_bar(axes[2,2], bars_params, values_J)
    annotate_bar(axes[0,3], bars_metrics, values_metrics)
    annotate_bar(axes[1,3], bars_r, values_meanr)
    annotate_bar(axes[2,3], bars_cf, values_fc)
    # Set the title and labels for the bar plots
    axes[2,2].set_ylabel('relative change %', fontsize=20)
    axes[0,3].set_ylabel('relative change %', fontsize=20)
    axes[1,3].set_ylabel('relative change %', fontsize=20)
    axes[2,3].set_ylabel('relative change %', fontsize=20)
    axes[2,2].set_title('Rel. changes in J before and after training', fontsize=20)
    axes[0,3].set_title('Rel. changes in metrics before and after training', fontsize=20)
    axes[1,3].set_title('Rel. changes in mean rates before and after training', fontsize=20)
    axes[2,3].set_title('Other rel changes before and after training', fontsize=20)

    ################ Plot changes in offset thresholds over time ################
    num_pretraining_steps= sum(df['stage'] == 0)
    axes[0,2].plot(range(num_pretraining_steps), np.ones(num_pretraining_steps)*6, alpha=0.6, c='black', linestyle='--')
    axes[0,2].scatter(range(N), df[keys_metrics[1]], label=keys_metrics[1], marker='o', s=70, c='tab:orange')
    axes[0,2].scatter(range(N), df[keys_metrics[2]], label=keys_metrics[2], marker='o', s=50, c='tab:brown')
    axes[0,2].grid(color='gray', linestyle='-', linewidth=0.5)
    axes[0,2].set_title('Offset', fontsize=20)
    axes[0,2].set_ylabel('degrees', fontsize=20)
    axes[0,2].set_xlabel('SGD steps', fontsize=20)
    axes[0,2].set_ylim(0, min(25,max(df[keys_metrics[2]])+1)) # keys_metrics[2] should be the staircase offset
    axes[0,2].legend(loc='upper right', fontsize=20)
       
    ################ Plot changes in sigmoid weights and bias over time ################
    axes[1,2].plot(range(N), df['b_sig'], label='b_sig', linestyle='--', linewidth = 3)
    axes[1,2].set_xlabel('SGD steps', fontsize=20)
    i=0
    for i in range(10):
        column = f'w_sig_{29+i}' # if all 81 w_sigs are saved
        if column in df.keys():
            axes[1,2].plot(range(N), df[column], label=column)
        else:
            column = f'w_sig_{i}' # if only the middle 25 w_sigs are saved
            if column in df.keys():                
                axes[1,2].plot(range(N), df[column], label=column)
    axes[1,2].set_title('Readout bias and weights', fontsize=20)
    axes[1,2].legend(loc='upper right', fontsize=20)

    ################ Plot changes in J_m and J_s over time ################
    axes[2,0].plot(range(N), df['J_EE_m'], label='J_EE_m', linestyle='--', c='tab:red',linewidth=3)
    axes[2,0].plot(range(N), df['J_IE_m'], label='J_IE_m', linestyle='--', c='tab:orange',linewidth=3)
    axes[2,0].plot(range(N), df['J_II_m'], label='J_II_m', linestyle='--', c='tab:blue',linewidth=3)
    axes[2,0].plot(range(N), df['J_EI_m'], label='J_EI_m', linestyle='--', c='tab:green',linewidth=3)
    
    axes[2,0].plot(range(N), df['J_EE_s'], label='J_EE_s', c='tab:red',linewidth=3)
    axes[2,0].plot(range(N), df['J_IE_s'], label='J_IE_s', c='tab:orange',linewidth=3)
    axes[2,0].plot(range(N), df['J_II_s'], label='J_II_s', c='tab:blue',linewidth=3)
    axes[2,0].plot(range(N), df['J_EI_s'], label='J_EI_s', c='tab:green',linewidth=3)
    axes[2,0].legend(loc="upper right", fontsize=20)
    axes[2,0].set_title('J in middle and superficial layers', fontsize=20)
    axes[2,0].set_xlabel('SGD steps', fontsize=20)

    ################ Plot changes in maximum rates over time ################
    colors = ["tab:blue", "tab:red"]
    axes[1,0].plot(range(N), df['maxr_E_mid'], label='maxr_E_mid', c=colors[1], linestyle=':')
    axes[1,0].plot(range(N), df['maxr_I_mid'], label='maxr_I_mid', c=colors[0], linestyle=':')
    axes[1,0].plot(range(N), df['maxr_E_sup'], label='maxr_E_sup', c=colors[1])
    axes[1,0].plot(range(N), df['maxr_I_sup'], label='maxr_I_sup', c=colors[0])
    axes[1,0].legend(loc="upper right", fontsize=20)
    axes[1,0].set_title('Maximum rates', fontsize=20)
    axes[1,0].set_xlabel('SGD steps', fontsize=20)

    ################ Plot changes in mean rates over time ################
    colors = ["tab:blue", "tab:red"]
    if 'meanr_E_mid' in df.columns:
        axes[1,1].plot(range(N), df['meanr_E_mid'], label='meanr_E_mid', c=colors[1], linestyle=':')
        axes[1,1].plot(range(N), df['meanr_I_mid'], label='meanr_I_mid', c=colors[0], linestyle=':')
        axes[1,1].plot(range(N), df['meanr_E_sup'], label='meanr_E_sup', c=colors[1])
        axes[1,1].plot(range(N), df['meanr_I_sup'], label='meanr_I_sup', c=colors[0])
        axes[1,1].legend(loc="upper right", fontsize=20)
        axes[1,1].set_title('Mean rates', fontsize=20)
        axes[1,1].set_xlabel('SGD steps', fontsize=20)

    ################ Plot changes in c and f ################
    axes[2,1].plot(range(N), df['cE_m'], label='cE_m',c='tab:orange',linewidth=3)
    axes[2,1].plot(range(N), df['cI_m'], label='cI_m',c='tab:green',linewidth=3)
    axes[2,1].plot(range(N), df['cE_s'], label='cE_s',c='tab:red',linewidth=3)
    axes[2,1].plot(range(N), df['cI_s'], label='cI_s',c='tab:blue',linewidth=3)
    axes[2,1].plot(range(N), df['f_E'], label='f_E', linestyle='--',c='tab:red',linewidth=3)
    axes[2,1].plot(range(N), df['f_I'], label='f_I', linestyle='--',c='tab:blue',linewidth=3)
    axes[2,1].set_title('c: constant inputs, f: weights between mid and sup layers', fontsize=20)
    axes[2,1].legend(loc="upper right", fontsize=20)

    # Add vertical lines to indicate the different stages
    for i in range(1, len(df['stage'])):
        if df['stage'][i] != df['stage'][i-1]:
            axes[0,0].axvline(x=i, color='black', linestyle='--')
            axes[0,1].axvline(x=i, color='black', linestyle='--')
            axes[0,2].axvline(x=i, color='black', linestyle='--')
            axes[1,0].axvline(x=i, color='black', linestyle='--')
            axes[1,1].axvline(x=i, color='black', linestyle='--')
            axes[1,2].axvline(x=i, color='black', linestyle='--')
            axes[2,0].axvline(x=i, color='black', linestyle='--')
            axes[2,1].axvline(x=i, color='black', linestyle='--')
    for i in range(3):
        for j in range(4):
            axes[i,j].tick_params(axis='both', which='major', labelsize=18)  # Set tick font size

    if fig_filename:
        fig.savefig(fig_filename + ".png")
    plt.close()

    return exclude_run


def plot_results_from_csvs(folder_path, num_runs=3, folder_to_save=None, starting_run=0):
    # Add folder_path to path
    if folder_path not in sys.path:
        sys.path.append(folder_path)

    # Plot loss, accuracy and trained parameters
    excluded_run_inds = None
    for j in range(starting_run, num_runs):
        if folder_to_save is not None:
            results_fig_filename = os.path.join(folder_to_save,f'resultsfig_{j}')
        else:
            results_fig_filename = os.path.join(folder_path,f'resultsfig_{j}')
        exclude_run = plot_results_from_csv(folder_path,j,results_fig_filename)
        if exclude_run:
            excluded_run_inds.append(j)
    return excluded_run_inds

################### TUNING CURVES ###################
def plot_tuning_curves_all_cells(results_dir, run_index, folder_to_save, num_rnd_cells=81):
    """Plot example tuning curves for middle and superficial layer cells at different stages of training"""
    if num_rnd_cells == 81:
        tc_cells = numpy.arange(81)
    else:
        tc_cells_unsorted = numpy.random.choice(81, num_rnd_cells, replace=False)
        tc_cells = numpy.sort(tc_cells_unsorted)

    # Load tuning curves
    train_tc_filename = os.path.join(results_dir, 'tuning_curves.csv')
    train_tuning_curves = numpy.array(pd.read_csv(train_tc_filename, header=None))
    pretrain_tc_filename = os.path.join(os.path.dirname(results_dir), 'pretraining_tuning_curves.csv')
    pretrain_tuning_curves = numpy.array(pd.read_csv(pretrain_tc_filename, header=0))

    # Combine pretraining and training tuning curves
    tuning_curves = numpy.vstack((pretrain_tuning_curves, train_tuning_curves))

    # Select tuning curves for the current run
    mesh_i = tuning_curves[:,0]==run_index
    tuning_curve_i = tuning_curves[mesh_i,1:]
    
    # Select tuning curves for each stage of training
    mesh_stage_0 = tuning_curve_i[:,0]==0
    tc_0 = tuning_curve_i[mesh_stage_0,1:]
    mesh_stage_1 = tuning_curve_i[:,0]==1
    tc_1 = tuning_curve_i[mesh_stage_1,1:]
    mesh_stage_2 = tuning_curve_i[:,0]==2
    tc_2 = tuning_curve_i[mesh_stage_2,1:]

    # Create figure of 2 x 5 subplot arranged as E-I for the two rows and mid-phase0, mid-phase1, mid-phase2, mid-phase3, and sup for the columns
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(5*5, 5*2))
    num_oris = tc_1.shape[0]
    # Plot tuning curves
    for cell_ind in range(len(tc_cells)):
        for i in range(2):
            # Middle layer cells
            for j in range(4):
                axes[i,j].plot(numpy.arange(num_oris)*180/num_oris, tc_0[:,j*162+i*81+cell_ind], label='pretraining',linewidth=2, color='black', alpha=0.2)
                axes[i,j].plot(numpy.arange(num_oris)*180/num_oris, tc_1[:,j*162+i*81+cell_ind], label='post-pretraining',linewidth=2, color='orange', alpha=0.4)
                axes[i,j].plot(numpy.arange(num_oris)*180/num_oris, tc_2[:,j*162+i*81+cell_ind], label='post-training',linewidth=2, color='green', alpha=0.6)
                # emphasize 55 on the x-axis
                axes[i,j].axvline(x=55, color='black', linestyle='--', alpha=0.5)
            
            # Superficial layer cells        
            axes[i,4].plot(numpy.arange(num_oris)*180/num_oris, tc_0[:,648+i*81+cell_ind], label='pretraining',linewidth=2, color='black', alpha=0.2)
            axes[i,4].plot(numpy.arange(num_oris)*180/num_oris, tc_1[:,648+i*81+cell_ind], label='post-pretraining',linewidth=2, color='orange', alpha=0.4)
            axes[i,4].plot(numpy.arange(num_oris)*180/num_oris, tc_2[:,648+i*81+cell_ind], label='post-training',linewidth=2, color='green', alpha=0.6)
            # emphasize 55 on the x-axis
            axes[i,4].axvline(x=55, color='black', linestyle='--')
    # axes[0,0].legend(loc='upper left', fontsize=20)
    # Set main title
    fig.suptitle('Top: E, Bottom: I, Left 4: mid, Right 1: sup, Black: prepre, Orange: pre, Green:post', fontsize=20)

    # Save plot
    if folder_to_save is not None:
        fig.savefig(os.path.join(folder_to_save,f'tc_fig_all_cells_run{run_index}.png'))
    else:
        fig.savefig(os.path.join(results_dir,f'tc_fig_all_cells_run{run_index}.png'))
    plt.close()

def plot_slope_config_groups(results_dir, config_groups, folder_to_save):
    slope_data = pd.read_csv(os.path.join(results_dir, 'tc_slopediff_train.csv'))
    fig, axes = plt.subplots(nrows=2, ncols=len(config_groups), figsize=(15*len(config_groups), 10*2))
    layers = ['mid', 'sup']
    color_shades_of_blue = ['slategray', 'blue', 'teal', 'deepskyblue', 'darkblue']
    color_shades_of_red = ['rosybrown', 'red', 'salmon', 'darkred', 'orangered']
    for i, config_group in enumerate(config_groups):
        for j in range(len(config_group)):
            mesh_config = slope_data['configuration']==config_group[j]
            slope_data_config = slope_data[mesh_config]
            for ori in ['57', '123']:
                if ori == '57':
                    linestyle = '-'
                else:
                    linestyle = '--'
                for type in ['E', 'I']:
                    if type == 'E':
                        color = color_shades_of_blue[j]
                    else:
                        color = color_shades_of_red[j]
                    for layer_ind in range(2):
                        layer = layers[layer_ind]
                        x_data = slope_data_config[f'slope_{ori}_diff_{type}_{layer}_x']
                        y_data = slope_data_config[f'slope_{ori}_diff_{type}_{layer}_y']
                        label = f'{config_group[j]} ({type}, ori {ori})'
                        axes[layer_ind, i].plot(x_data, y_data, label=label, linewidth=2, color=color, linestyle=linestyle)
                        # Add legend outside the axes
                        axes[layer_ind, i].legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12, borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95]) # Adjust rect to make space for legends
    fig.suptitle('Top: Mid, Bottom: Sup, Blue: I, Red: E, Solid: 57, Dashed: 123', fontsize=25)
    if folder_to_save is not None:
        fig.savefig(os.path.join(folder_to_save,'slope_fig_config_groups.png'))
    else:
        fig.savefig(os.path.join(results_dir,'slope_fig_config_groups.png'))


def plot_tuning_curves(results_dir, tc_cells, num_runs, folder_to_save, seed=0, tc_cell_labels=None):
    """Plot example tuning curves for middle and superficial layer cells at different stages of training"""

    train_tc_filename = os.path.join(results_dir, 'tuning_curves.csv')
    # Specify header based on what type of data there is in the first row of the csv file
    train_tc_header = check_header(train_tc_filename)
    train_tuning_curves = numpy.array(pd.read_csv(train_tc_filename, header=train_tc_header))
    pretrain_tc_filename = os.path.join(os.path.dirname(results_dir), 'pretraining_tuning_curves.csv')
    pretrain_tuning_curves = numpy.array(pd.read_csv(pretrain_tc_filename, header=0))

    # Combine pretraining and training tuning curves
    tuning_curves = numpy.vstack((pretrain_tuning_curves, train_tuning_curves))

    numpy.random.seed(seed)
    num_mid_cells = 648
    num_sup_cells = 164
    num_runs_plotted = min(5,num_runs)
    for i in range(num_runs_plotted):
        # Select tuning curves for the current run
        mesh_i = tuning_curves[:,0]==i
        tuning_curve_i = tuning_curves[mesh_i,1:]
        
        # Select tuning curves for each stage of training
        mesh_stage_0 = tuning_curve_i[:,0]==0
        tc_0 = tuning_curve_i[mesh_stage_0,1:]
        mesh_stage_1 = tuning_curve_i[:,0]==1
        tc_1 = tuning_curve_i[mesh_stage_1,1:]
        mesh_stage_2 = tuning_curve_i[:,0]==2
        tc_2 = tuning_curve_i[mesh_stage_2,1:]

        # Select num_rnd_cells randomly selected cells to plot from both middle and superficial layer cells
        if i==0:
            if isinstance(tc_cells,int):
                num_cells_plotted=tc_cells
                mid_cells = numpy.random.randint(0, num_mid_cells, size=int(num_cells_plotted/2), replace=False)
                sup_cells = numpy.random.randint(num_mid_cells, num_mid_cells+num_sup_cells, size=int(num_cells_plotted/2), replace=False)
                tc_cells = numpy.concatenate((mid_cells, sup_cells))
                tc_cell_labels = ['mid_'+str(cell) for cell in mid_cells] + ['sup_'+str(cell) for cell in sup_cells]
            else:
                num_cells_plotted=len(tc_cells)
                if tc_cell_labels is None:
                    tc_cell_labels = ['mid_'+str(cell) if cell<num_mid_cells else 'sup_'+str(cell-num_mid_cells) for cell in tc_cells]
            fig, axes = plt.subplots(nrows=num_runs_plotted, ncols=num_cells_plotted, figsize=(5*num_cells_plotted, 5*num_runs_plotted))
        axes_flatten = axes.flatten()
        num_oris = tc_1.shape[0]
        # Plot tuning curves
        for cell_ind in range(num_cells_plotted):
            axes_flatten[i*num_cells_plotted+cell_ind].plot(range(num_oris), tc_0[:,tc_cells[cell_ind]], label='pretraining',linewidth=2)
            axes_flatten[i*num_cells_plotted+cell_ind].plot(range(num_oris), tc_1[:,tc_cells[cell_ind]], label='post-pretraining',linewidth=3)
            axes_flatten[i*num_cells_plotted+cell_ind].plot(range(num_oris), tc_2[:,tc_cells[cell_ind]], label='post-training',linewidth=4)
            
            axes_flatten[i*num_cells_plotted+cell_ind].set_title(tc_cell_labels[cell_ind], fontsize=20)
    axes_flatten[i*num_cells_plotted+cell_ind].legend(loc='upper left', fontsize=20)

    # Save plot
    if folder_to_save is not None:
        fig.savefig(os.path.join(folder_to_save,'tc_fig.png'))
    else:
        fig.savefig(os.path.join(results_dir,'tc_fig.png'))
    plt.close()


def plot_pre_post_scatter(ax, x_axis, y_axis, orientations, indices_to_plot, num_training, title, colors=None, linecolor='black'):
    """Scatter plot of pre vs post training values for a given set of indices"""
    if colors is None:
        ax.scatter(x_axis[:,indices_to_plot], y_axis[:,indices_to_plot], s=20, alpha=0.5)
    else:
        for run_ind in range(num_training):
            bin_indices = numpy.digitize(numpy.abs(orientations[run_ind,:]), [4, 12, 20, 28, 36, 44, 50, 180])
        
            # Iterate over bins rather than individual points
        
            for bin_idx, color in enumerate(colors, start=1):  # Adjust as needed
                # Find indices within this bin
                in_bin = numpy.where(bin_indices == bin_idx)[0]
                # Find intersection with indices_to_plot
                plot_indices = numpy.intersect1d(in_bin, indices_to_plot)
                
                if len(plot_indices) > 0:
                    ax.scatter(x_axis[run_ind,plot_indices], y_axis[run_ind,plot_indices], color=color, s=20, alpha=0.5)
            
    # Plot x = y line
    xpoints = ypoints = ax.get_xlim()
    ax.plot(xpoints, ypoints, color=linecolor, linewidth=2)
    ax.set_xlabel('Pre training')
    ax.set_ylabel('Post training')
    ax.set_title(title)


def plot_tc_features(results_dir, stages=[0,1,2]):
    """Plot tuning curve features for E and I cells at different stages of training"""
    def sliding_mannwhitney(x1, y1, x2, y2, window_size, sliding_unit):
        from scipy.stats import mannwhitneyu
        """ Perform Mann-Whitney U test on y1 and y2 in sliding windows along x1 and x2. """
        # Determine the range of x-values for sliding windows
        min_x = max(np.min(x1), np.min(x2))  # Start at the max of the minimum x-values
        max_x = min(np.max(x1), np.max(x2))  # End at the min of the maximum x-values
        
        # Initialize lists to store results
        x_window_center = []
        p_val_vec = []
        
        # Slide the window across the x-axis
        current_start = min_x
        while current_start + window_size <= max_x:
            current_end = current_start + window_size
            center = (current_start + current_end) / 2  # Center of the window
            
            # Find indices where x1 and x2 fall within the current window
            indices_x1 = (x1 >= current_start) & (x1 < current_end)
            indices_x2 = (x2 >= current_start) & (x2 < current_end)
            
            # Extract the corresponding y1 and y2 values in this window
            y1_in_window = y1[indices_x1]
            y2_in_window = y2[indices_x2]
            
            # Perform Mann-Whitney U test if there are enough data points in both windows
            if len(y1_in_window) > 0 and len(y2_in_window) > 0:
                u_stat, p_value = mannwhitneyu(y1_in_window, y2_in_window)
            else:
                p_value = np.nan  # If there's not enough data, return NaN
            
            # Store results
            x_window_center.append(center)
            p_val_vec.append(p_value)
            
            # Move the window by sliding_unit
            current_start += sliding_unit
        
        # Convert lists to arrays and return
        return np.array(x_window_center), np.array(p_val_vec)

    def shift_x_data(x_data, indices, shift_value=90):
        """ Shift circular x_data by shift_value and center it around the new 0 (around shift_value) """
        x_data_shifted = x_data[:, indices].flatten() - shift_value
        x_data_shifted = numpy.where(x_data_shifted > 90, x_data_shifted - 180, x_data_shifted)
        x_data_shifted = numpy.where(x_data_shifted < -90, x_data_shifted + 180, x_data_shifted)
        return x_data_shifted
    
    def scatter_feature(ax, y, x, indices_E, indices_I, shift_val=55, fs_ticks=40, feature=None, values_for_colors=None):
        """Scatter plot of feature for E and I cells. If shift_val is not None, then x should be the preferred orientation that will be centered around shift_val."""
        if shift_val is not None:
            x_I = shift_x_data(x, indices_I, shift_value=shift_val)
            x_E = shift_x_data(x, indices_E, shift_value=shift_val)
        else:
            x_I = x[:,indices_I].flatten()
            x_E = x[:,indices_E].flatten()
        
        if values_for_colors is not None:
            # Define a colormap and the colors for E and I cells based on values_for_colors
            cmap = plt.get_cmap('rainbow')
            norm = plt.Normalize(vmin=numpy.min(values_for_colors), vmax=numpy.max(values_for_colors))
            
            # Get colors for E and I cells based on values_for_colors
            color_E = cmap(norm(values_for_colors[:,indices_E].flatten()))
            color_I = cmap(norm(values_for_colors[:,indices_I].flatten()))
        else:
            color_E = 'red'
            color_I = 'blue'
        ax.scatter(x_E, y[:,indices_E].flatten(), s=50, alpha=0.3, color=color_E)
        ax.scatter(x_I, y[:,indices_I].flatten(), s=50, alpha=0.3, color=color_I)
        if shift_val is None:
            xpoints = ypoints = ax.get_xlim()
            ax.plot(xpoints, ypoints, color='black', linewidth=2)
            ax.set_xlabel('Pre training', fontsize=fs_ticks)
            ax.set_ylabel('Post training', fontsize=fs_ticks)
            axes_format(ax, fs_ticks)
            
        if feature is not None:
            ax.set_title(feature, fontsize=fs_ticks)

    def lowess_feature(ax, y, x, indices_E, indices_I, shift_val=55, frac=0.15, shades=''):
        """Plot lowess smoothed feature for E and I cells. Curves are fitted separately for E and I cells and for x<0 and x>0."""
        # fit curves separately for the smoothed_x_E<0 and smoothed_x_E>0
        if shift_val is not None:
            x_I = shift_x_data(x, indices_I, shift_value=shift_val)
            x_E = shift_x_data(x, indices_E, shift_value=shift_val)
        else:
            x_I = x[:,indices_I].flatten()
            x_E = x[:,indices_E].flatten()

        y_E = y[:,indices_E].flatten()
        x_E_pos = x_E[x_E>0]
        lowess_E_pos = sm.nonparametric.lowess(y_E[x_E>0], x_E_pos, frac=frac)
        x_E_neg = x_E[x_E<0]
        lowess_E_neg = sm.nonparametric.lowess(y_E[x_E<0], x_E_neg, frac=frac)
        lowess_E = numpy.concatenate((lowess_E_neg, lowess_E_pos), axis=0)
        #lowess_E = sm.nonparametric.lowess(y_E, x_E, frac=frac) # if we don't want to separate the curve into x<0 and x>0
        
        y_I = y[:,indices_I].flatten()
        x_I_pos = x_I[x_I>0]
        lowess_I_pos = sm.nonparametric.lowess(y_I[x_I>0], x_I_pos, frac=frac)
        x_I_neg = x_I[x_I<0]
        lowess_I_neg = sm.nonparametric.lowess(y_I[x_I<0], x_I_neg, frac=frac)
        lowess_I = numpy.concatenate((lowess_I_neg, lowess_I_pos), axis=0)
        #lowess_I = sm.nonparametric.lowess(y_I, x_I, frac=frac) # if we don't want to separate the curve into x<0 and x>0
        
        ax.plot(lowess_E[:, 0], numpy.abs(lowess_E[:, 1]), color=shades+'red', linewidth=10, alpha=0.8)
        ax.plot(lowess_I[:, 0], numpy.abs(lowess_I[:, 1]), color=shades+'blue', linewidth=10, alpha=0.8)

    def mesh_for_feature(data, feature, mesh_cells):
        mesh_feature = data['feature']==feature
        feature_data = data[mesh_feature]
        feature_data = feature_data.loc[:,mesh_cells]
        return feature_data.to_numpy()
              
    # Load tuning curves
    train_tc_features_filename = os.path.join(results_dir, 'tuning_curve_features.csv')
    train_tc_header = check_header(train_tc_features_filename)
    train_tc_features = pd.read_csv(train_tc_features_filename, header=train_tc_header)
    pretrain_tc_features_filename = os.path.join(os.path.dirname(results_dir), 'pretraining_tuning_curve_features.csv')
    pretrain_tc_features = pd.read_csv(pretrain_tc_features_filename, header=0)

    ############## Plots about changes before vs after training and pretraining (per layer and per centered or all) ##############
             
    # Define indices for each group of cells
    #E_sup = numpy.linspace(0, 80, 81).astype(int) + 648 
    #I_sup = numpy.linspace(81, 161, 81).astype(int) + 648
    E_sup_centre = numpy.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)+648
    I_sup_centre = (E_sup_centre+81).astype(int)
    
    #mid_array = numpy.linspace(0, 647, 648).round().reshape(4, 2, 81).astype(int)
    #E_mid = mid_array[:,0,:].ravel().astype(int)
    #I_mid = mid_array[:,1,:].ravel().astype(int)
    E_mid_centre = numpy.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)
    I_mid_centre = (E_mid_centre+81).astype(int)
    indices = [E_sup_centre, I_sup_centre, E_mid_centre, I_mid_centre]
    
    # Create legends for the plot
    patches = []
    cmap = plt.get_cmap('rainbow')
    colors = numpy.flip(cmap(numpy.linspace(0,1, 8)), axis = 0)
    bins = ['0-4', '4-12', '12-20', '20-28', '28-36', '36-44', '44-50', '+50']
    for layer in range(0,len(colors)):
        patches.append(mpatches.Patch(color=colors[layer], label=bins[layer]))

    ###############################################
    ######### Schoups-style scatter plots #########
    ###############################################
    colors = numpy.flip(cmap(numpy.linspace(0,1, 8)), axis = 0)
    fs_text = 20
    fs_ticks = 20
    
    # Scatter slope, where x-axis is orientation and y-axis is the change in slope before and after training
    stage_labels = ['pretrain', 'train']
    
    for training_stage in stages[0:-1]:
        fig, axs = plt.subplots(2, 5, figsize=(50, 20))
        
        mesh_stage_pre = pretrain_tc_features['stage']==training_stage
        data_pre = pretrain_tc_features[mesh_stage_pre]
        mesh_cells = data_pre.columns.str.startswith('G')
        if training_stage == 1:
            mesh_stage_post = train_tc_features['stage']==training_stage+1
            data_post = train_tc_features[mesh_stage_post]
        else:
            mesh_stage_post = pretrain_tc_features['stage']==training_stage+1
            data_post = pretrain_tc_features[mesh_stage_post]    
        
        min_pre = mesh_for_feature(data_pre, 'min', mesh_cells)
        min_post = mesh_for_feature(data_post, 'min', mesh_cells)
        max_pre = mesh_for_feature(data_pre, 'max', mesh_cells) # (max-min)/max
        max_post = mesh_for_feature(data_post, 'max', mesh_cells)
        mean_pre = mesh_for_feature(data_pre, 'mean', mesh_cells)
        mean_post = mesh_for_feature(data_post, 'mean', mesh_cells)
        slope_hm_pre = mesh_for_feature(data_pre, 'slope_hm', mesh_cells)
        slope_hm_post = mesh_for_feature(data_post, 'slope_hm', mesh_cells)
        fwhm_pre = mesh_for_feature(data_pre, 'fwhm', mesh_cells)      
        fwhm_post = mesh_for_feature(data_post, 'fwhm', mesh_cells)
        slope_55_pre = mesh_for_feature(data_pre, 'slope_55', mesh_cells)
        slope_55_post = mesh_for_feature(data_post, 'slope_55', mesh_cells)
        slope_125_pre = mesh_for_feature(data_pre, 'slope_125', mesh_cells)
        slope_125_post = mesh_for_feature(data_post, 'slope_125', mesh_cells)
        slopediff_55 = numpy.abs(slope_55_post) - numpy.abs(slope_55_pre)
        slopediff_125 = numpy.abs(slope_125_post) - numpy.abs(slope_125_pre)

        mesh_pref_ori = data_post['feature']=='pref_ori'
        pref_ori = data_post[mesh_pref_ori]
        pref_ori = pref_ori.loc[:,mesh_cells].to_numpy()
        #coloring_feature = pref_ori
        #coloring_feature = numpy.reshape(numpy.repeat(numpy.arange(pref_ori.shape[0]), pref_ori.shape[1]), (pref_ori.shape[0], pref_ori.shape[1]))
        coloring_feature = None
        for layer in range(2):            
            ##### Plot features before vs after training per layer and cell type #####
            ax_row_ind = 0 if layer == 1 else 1  # For clearer readability
            scatter_feature(axs[ax_row_ind,0], fwhm_post, fwhm_pre, indices[2*layer], indices[2*layer+1], shift_val=None, feature='fwhm', values_for_colors=coloring_feature)
            scatter_feature(axs[ax_row_ind,1], min_post, min_pre, indices[2*layer], indices[2*layer+1], shift_val=None, feature='min', values_for_colors=coloring_feature)
            scatter_feature(axs[ax_row_ind,2], max_post, max_pre, indices[2*layer], indices[2*layer+1], shift_val=None, feature='max', values_for_colors=coloring_feature)
            scatter_feature(axs[ax_row_ind,3], mean_post, mean_pre, indices[2*layer], indices[2*layer+1], shift_val=None, feature='mean', values_for_colors=coloring_feature)
            scatter_feature(axs[ax_row_ind,4], slope_hm_post, slope_hm_pre, indices[2*layer], indices[2*layer+1], shift_val=None, feature='slope_hm', values_for_colors=coloring_feature)

        plt.tight_layout()

        # Save plot
        if training_stage == 0:
            file_path = os.path.join(os.path.dirname(results_dir),f'tc_features_{stage_labels[training_stage]}.png')
        else:
            file_path = os.path.join(results_dir,'figures', f'tc_features_{stage_labels[training_stage]}.png')
        plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=np.min(pref_ori), vmax=np.max(pref_ori)), cmap='rainbow'),ax=axs[-1,-1],orientation='horizontal',label='Preferred orientation')
        fig.savefig(file_path)
        plt.close()

        # 3 x 2 scatter plot of data[slopediff_55_0 and 1], data[slopediff_55_0 and 1] and data[slopediff_diff]
        fig, axs = plt.subplots(2, 4, figsize=(40, 20))
        coloring_feature = numpy.reshape(numpy.repeat(numpy.arange(pref_ori.shape[0]), pref_ori.shape[1]), (pref_ori.shape[0], pref_ori.shape[1]))
        # Middle layer
        for layer in range(2):
            scatter_feature(axs[layer,0], slopediff_55, pref_ori, indices[2*layer], indices[2*layer+1], shift_val=55, feature='slopediff_55')
            scatter_feature(axs[layer,1], slopediff_55, pref_ori, indices[2*layer], indices[2*layer+1], shift_val=55, feature='slopediff_55')
            lowess_feature(axs[layer,0], slopediff_55, pref_ori, indices[2*layer], indices[2*layer+1], shift_val=55, shades='dark')
            lowess_feature(axs[layer,1], slopediff_125, pref_ori, indices[2*layer], indices[2*layer+1], shift_val=125, shades='tab:')
            lowess_feature(axs[layer,2], slopediff_55, pref_ori, indices[2*layer], indices[2*layer+1], shift_val=55, shades='dark')
            lowess_feature(axs[layer,2], slopediff_125, pref_ori, indices[2*layer], indices[2*layer+1], shift_val=125, shades='tab:')
            pref_ori_55_E = shift_x_data(pref_ori, indices[2*layer], shift_value=55)
            pref_ori_125_E = shift_x_data(pref_ori, indices[2*layer], shift_value=125)
            pref_ori_55_I = shift_x_data(pref_ori, indices[2*layer+1], shift_value=55)
            pref_ori_125_I = shift_x_data(pref_ori, indices[2*layer+1], shift_value=125)
            slopediff_55_E = slopediff_55[:,indices[2*layer]].flatten()
            slopediff_55_I = slopediff_55[:,indices[2*layer+1]].flatten()
            slopediff_125_E = slopediff_125[:,indices[2*layer]].flatten()
            slopediff_125_I = slopediff_125[:,indices[2*layer+1]].flatten()
            x_mw_55_E, y_mw_55_E = sliding_mannwhitney(pref_ori_55_E, slopediff_55_E, pref_ori_125_E, slopediff_125_E, window_size=10, sliding_unit=0.2)
            x_mw_55_I, y_mw_55_I = sliding_mannwhitney(pref_ori_55_I, slopediff_55_I, pref_ori_125_I, slopediff_125_I, window_size=10, sliding_unit=0.2)
            axs[layer,3].plot(x_mw_55_E, y_mw_55_E, color='red', linewidth=2)
            axs[layer,3].plot(x_mw_55_I, y_mw_55_I, color='blue', linewidth=2)
            # Add dashed black line at 0.05
            axs[layer,3].axhline(y=0.05, color='black', linestyle='--', linewidth=2)
            
            # Set titles
            axs[layer,0].set_title(r'$\Delta$' + 'slope(55)', fontsize=fs_text)
            axs[layer,1].set_title(r'$\Delta$' + 'slope(125)', fontsize=fs_text)
            axs[layer,2].set_title(r'$\Delta$' + 'slope(55), '+ r'$\Delta$' + 'slope(125)', fontsize=fs_text)
        
        # Format and save plot
        for ax in axs.flatten():
            axes_format(ax, fs_ticks)
        
        plt.tight_layout(w_pad=10, h_pad=7)
        if training_stage == 0:
            file_path = os.path.join(os.path.dirname(results_dir),f'tc_slope_{stage_labels[training_stage]}.png')
        else:
            file_path = os.path.join(results_dir,'figures', f'tc_slope_{stage_labels[training_stage]}.png')
        fig.savefig(file_path)
        plt.close()

################### CORRELATION ANALYSIS ###################

def plot_param_offset_correlations(folder):
    """Plot the correlations between the offset parameters and the psychometric, staircase, and loss parameters"""
    # Helper functions
    def get_color(corr_and_p):
        """Determine the color based on correlation value and significance."""
        if corr_and_p[1] < 0.05:  # Significant correlation
            if corr_and_p[0] > 0:
                return 'darkgreen', 'lightgreen'  # Significantly positive
            else:
                return 'darkred', 'lightcoral'    # Significantly negative
        else:
            return 'gray', 'lightgray'  # Insignificant correlation
        
    def regplots(param_key, rel_changes_train, corr_and_p_1, corr_and_p_2, corr_and_p_3, axes1, axes2, axes3, i, j):
        line_color1, scatter_color1 = get_color(corr_and_p_1)
        line_color2, scatter_color2 = get_color(corr_and_p_2)
        line_color3, scatter_color3 = get_color(corr_and_p_3)
        sns.regplot(x=param_key, y='psychometric_offset', data=rel_changes_train, ax=axes1[i,j], ci=95, color='red', 
            line_kws={'color':line_color1}, scatter_kws={'alpha':0.3, 'color':scatter_color1})
        sns.regplot(x=param_key, y='staircase_offset', data=rel_changes_train, ax=axes2[i,j], ci=95, color='blue', 
            line_kws={'color':line_color2}, scatter_kws={'alpha':0.3, 'color':scatter_color2})
        sns.regplot(x=param_key, y='loss_binary_cross_entr', data=rel_changes_train, ax=axes3[i,j], ci=95, color='green', 
            line_kws={'color':line_color3}, scatter_kws={'alpha':0.3, 'color':scatter_color3})
        # add label to x-axis
        axes1[i,j].set_xlabel(param_key, fontsize=20)
        axes2[i,j].set_xlabel(param_key, fontsize=20)
        axes3[i,j].set_xlabel(param_key, fontsize=20)
        # display corr in the right bottom of the figure
        axes1[i,j].text(0.05, 0.05, f'r= {corr_and_p_1[0]:.2f}', transform=axes1[i,j].transAxes, fontsize=20)
        axes2[i,j].text(0.05, 0.05, f'r= {corr_and_p_2[0]:.2f}', transform=axes2[i,j].transAxes, fontsize=20)
        axes3[i,j].text(0.05, 0.05, f'r= {corr_and_p_3[0]:.2f}', transform=axes3[i,j].transAxes, fontsize=20)

    def save_fig(fig, folder, filename, title=None):
        if title is not None:
            fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        fig.savefig(folder + filename)
        plt.close(fig)

    corr_psychometric_offset_param, corr_staircase_offset_param, corr_loss_param, rel_changes_train = param_offset_correlations(folder)
    # Make three plots of the correlations between the offset parameters and the psychometric, staircase, and loss parameters
    fig1, axes1 = plt.subplots(nrows=3, ncols=9, figsize=(9*5, 3*5))
    fig2, axes2 = plt.subplots(nrows=3, ncols=9, figsize=(9*5, 3*5))
    fig3, axes3 = plt.subplots(nrows=3, ncols=9, figsize=(9*5, 3*5))
    
    j1, j2, j3 = 0, 0, 0
    for param_key, corr_and_p_1 in corr_psychometric_offset_param.items():
        # rows of the plot collect types of parameters: 1) mid, 2) sup, 3) f and kappa, 4) J combined
        corr_and_p_2 = corr_staircase_offset_param[param_key]
        corr_and_p_3 = corr_loss_param[param_key]
        if param_key.endswith('_m'):
            regplots(param_key, rel_changes_train, corr_and_p_1, corr_and_p_2, corr_and_p_3, axes1, axes2, axes3, 0, j1)
            j1 += 1
        elif param_key.endswith('_s'):
            regplots(param_key, rel_changes_train, corr_and_p_1, corr_and_p_2, corr_and_p_3, axes1, axes2, axes3, 1, j2)
            j2 += 1
        else:
            regplots(param_key, rel_changes_train, corr_and_p_1, corr_and_p_2, corr_and_p_3, axes1, axes2, axes3, 2, j3)
            j3 += 1
    # Adjust layout and save + close the plot
    save_fig(fig1, folder, '/figures/Offset_corr_psychometric.png', title='Paramaters vs psychometric_offset')
    save_fig(fig2, folder, '/figures/Offset_corr_staircase.png', title='Paramaters vs staircase_offset')
    save_fig(fig3, folder, '/figures/Offset_corr_loss_binary.png', title='Paramaters vs BCE loss')


def plot_correlations(folder, num_training, num_time_inds=3):
    offset_pars_corr, offset_staircase_pars_corr, MVPA_corrs, data = MVPA_param_offset_correlations(folder, num_training, num_time_inds, mesh_for_valid_offset=False)

    ########## Plot the correlation between offset_th_rel_change and the combination of the J_m_E_rel_change, J_m_I_rel_change, J_s_E_rel_change, and J_s_I_rel_change ##########
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    axes_flat = axes.flatten()

    # x-axis labels for the subplots on J_m_E, J_m_I, J_s_E, and J_s_I
    x_keys_J = ['J_m_E_rel_change', 'J_m_I_rel_change', 'J_s_E_rel_change', 'J_s_I_rel_change']
    x_labels_J = [
    r'$\Delta (J^\text{mid}_{E \rightarrow E} + J^\text{mid}_{E \rightarrow I})$',
    r'$\Delta (J^\text{mid}_{I \rightarrow I} + J^\text{mid}_{I \rightarrow E})$',
    r'$\Delta (J^\text{sup}_{E \rightarrow E} + J^\text{sup}_{E \rightarrow I})$',
    r'$\Delta (J^\text{sup}_{I \rightarrow I} + J^\text{sup}_{I \rightarrow E})$']
    E_indices = [0,2]

    # Plot the correlation between staircase_offset_rel_change and the combination of the J_m_E_rel_change, J_m_I_rel_change, J_s_E_rel_change, and J_s_I_rel_change
    for i in range(4):
        # Create lmplot for each pair of variables
        if i in E_indices:
            sns.regplot(x=x_keys_J[i], y='staircase_offset_rel_change', data=data, ax=axes_flat[i], ci=95, color='red', 
                line_kws={'color':'darkred'}, scatter_kws={'alpha':0.3, 'color':'red'})
        else:
            sns.regplot(x=x_keys_J[i], y='staircase_offset_rel_change', data=data, ax=axes_flat[i], ci=95, color='blue', 
                line_kws={'color':'darkblue'}, scatter_kws={'alpha':0.3, 'color':'blue'})
        # Calculate the Pearson correlation coefficient and the p-value
        corr = offset_pars_corr[i]['corr']
        p_value = offset_pars_corr[i]['p_value']
        print('Correlation between offset_th_diff and', x_keys_J[i], 'is', corr, 'with p-value', p_value)
        
        # display corr and p-value in the right bottom of the figure
        axes_flat[i].text(0.05, 0.05, f'r= {corr:.2f}', transform=axes_flat[i].transAxes, fontsize=20)
        axes_format(axes_flat[i], fs_ticks=20)
        # add xlabel and ylabel
        axes_flat[i].set_xlabel(x_labels_J[i], fontsize=20, labelpad=20)
    # Adjust layout and save + close the plot
    plt.tight_layout()
    plt.savefig(folder + "/figures/Offset_corr_J_IE.png")
    plt.close()
    plt.clf()

    ########## Plot the correlation between offset_staircase_diff and the combination of the J_m_E_diff, J_m_I_diff, J_s_E_diff, and J_s_I_diff ##########
    data['offset_staircase_improve'] = -1*data['staircase_offset_rel_change']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    axes_flat = axes.flatten()

    for i in range(4):
        # Create lmplot for each pair of variables
        if i in E_indices:
            sns.regplot(x=x_keys_J[i], y='offset_staircase_improve', data=data, ax=axes_flat[i], ci=95, color='red', 
                line_kws={'color':'darkred'}, scatter_kws={'alpha':0.3, 'color':'red'})
        else:
            sns.regplot(x=x_keys_J[i], y='offset_staircase_improve', data=data, ax=axes_flat[i], ci=95, color='blue', 
                line_kws={'color':'darkblue'}, scatter_kws={'alpha':0.3, 'color':'blue'})
        axes_flat[i].set(ylabel=None)
        # Calculate the Pearson correlation coefficient and the p-value
        corr = offset_staircase_pars_corr[i]['corr']
        p_value = offset_staircase_pars_corr[i]['p_value']
        print('Correlation between offset_staircase_improve and', x_keys_J[i], 'is', corr, 'with p-value', p_value)
        
        # display corr and p-value in the right bottom of the figure
        axes_flat[i].text(0.05, 0.05, f'r= {corr:.2f}', transform=axes_flat[i].transAxes, fontsize=20)
        axes_format(axes_flat[i], fs_ticks=20)
        # add xlabel and ylabel
        axes_flat[i].set_xlabel(x_labels_J[i], fontsize=20, labelpad=20)
    # Add shared y-label
    fig.text(-0.05, 0.5, 'offset threshold improvement (%)', va='center', rotation='vertical', fontsize=20)

    # Adjust layout and save + close the plot
    plt.tight_layout()
    plt.savefig(folder + "/figures/Offset_staircase_corr_J_IE.png", bbox_inches='tight')
    plt.close()
    plt.clf()

    ########## Plot the correlation between offset_th_diff and the combination of the f_E_diff, f_I_diff, cE_m_diff, cI_m_diff, cE_s_diff, cI_s_diff ##########
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axes_flat = axes.flatten()

    # x-axis labels
    x_labels_fc = ['f_E_diff','f_I_diff', 'cE_m_diff', 'cI_m_diff', 'cE_s_diff', 'cI_s_diff']
    E_indices = [0,2]
    colors = ['brown', 'purple', 'orange', 'green', 'tab:orange', 'tab:green']
    linecolors = ['#8B4513', '#800080', '#FF8C00',  '#006400', '#FF8C00',  '#006400']  # Approximate dark versions of purple, green, orange, and brown
    axes_indices = [0,0,1,1,1,1]
    for i in range(len(x_labels_fc)):
        # Create lmplot for each pair of variables
        # Set colors to purple, green, orange and brown for the different indices
        sns.regplot(x=x_labels_fc[i], y='staircase_offset_rel_change', data=data, ax=axes_flat[axes_indices[i]], ci=95, color=colors[i],
            line_kws={'color':linecolors[i]}, scatter_kws={'alpha':0.3, 'color':colors[i]})
        # Calculate the Pearson correlation coefficient and the p-value
        print('Correlation between staircase_offset_rel_change and', x_labels_fc[i])

    # Adjust layout and save + close the plot
    plt.tight_layout()
    plt.savefig(folder + "/figures/Offset_corr_f_c.png")
    plt.close()
    plt.clf()

    # Plot MVPA_pars_corr for each ori_ind on the same plot but different subplots with scatter and different colors for each parameter
    x= numpy.arange(1,13)
    x_labels = ['offset threshold','J_m_E', 'J_m_I', 'J_s_E', 'J_s_I', 'm_f_E', 'm_f_I', 'm_c_E', 'm_c_I', 's_f_E', 's_f_I', 's_c_E', 's_c_I']
    plt.scatter(x, MVPA_corrs[0]['corr'])
    plt.scatter(x, MVPA_corrs[1]['corr'])
    plt.scatter(x, MVPA_corrs[2]['corr'])
    plt.xticks(x, x_labels)
    plt.legend(['55', '125', '0'])
    plt.title('Correlation of MVPA scores with parameter differences')
    # Save and close the plot
    plt.savefig(folder + '/MVPA_pars_corr.png')
    plt.close()


def plot_corr_triangle(data,folder_to_save='',filename='corr_triangle.png'):
    """Plot a triangle with correlation plots in the middle of the edges of the triangle. Data is supposed to be a dictionary with keys corresponding to MVPA results and relative parameter changes and offset changes."""
    # Get keys and values
    keys = data.keys()
    labels = ['rel. change in ' + keys[0], 'rel. change in '+keys[1], 'rel. change in ' + keys[2]]

    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Define positions of labels
    left_bottom_text = [0,0]
    right_bottom_text = [2,0]
    top_text = [2*np.cos(np.pi/3), 2*np.sin(np.pi/3)]

    # Define positions of correlation plots in the middle of the edges of the triangle
    width = 0.7
    height = 0.7
    left_top_regplot = [(left_bottom_text[0] + top_text[0]) / 2 - width/2, (left_bottom_text[1] + top_text[1]) / 2- height/2, width, height]
    right_top_regplot = [(right_bottom_text[0] + top_text[0]) / 2 - width/2, (right_bottom_text[1] + top_text[1]) / 2- height/2, width, height]
    bottom_regplot = [(left_bottom_text[0] + right_bottom_text[0]) / 2 - width/2, (left_bottom_text[1] + right_bottom_text[1]) / 2- height/2, width, height]
    
    # Add node labels
    buffer_x = 0.05
    buffer_y = 0.1
    fig.text(top_text[0],top_text[1], labels[0], ha='center', fontsize=35)
    fig.text(left_bottom_text[0]-buffer_x, left_bottom_text[1]-buffer_y, labels[1], ha='center', fontsize=35)
    fig.text(right_bottom_text[0]+buffer_x, right_bottom_text[1]-buffer_y, labels[2], ha='center', fontsize=35)

    # Add lines connecting the nodes and the correlation plots (note that imput is x_data, y_data and not point1, point2)
    line1_vec = [top_text[0]-left_bottom_text[0], top_text[1]-left_bottom_text[1]]
    line2_vec = [top_text[0]-right_bottom_text[0], top_text[1]-right_bottom_text[1]]
    line3_vec = [right_bottom_text[0]-left_bottom_text[0], right_bottom_text[1]-left_bottom_text[1]]
    fig.add_artist(plt.Line2D([left_bottom_text[0], left_bottom_text[0]+line1_vec[0]/4], [left_bottom_text[1], left_bottom_text[1]+line1_vec[1]/4], lw=3, color='black'))
    fig.add_artist(plt.Line2D([left_bottom_text[0]+3*line1_vec[0]/4, left_bottom_text[0]+line1_vec[0]], [left_bottom_text[1]+3*line1_vec[1]/4, left_bottom_text[1]+line1_vec[1]], lw=3, color='black'))
    fig.add_artist(plt.Line2D([right_bottom_text[0], right_bottom_text[0]+line2_vec[0]/4], [right_bottom_text[1], right_bottom_text[1]+line2_vec[1]/4], lw=3, color='black'))
    fig.add_artist(plt.Line2D([right_bottom_text[0]+3*line2_vec[0]/4, right_bottom_text[0]+line2_vec[0]], [right_bottom_text[1]+3*line2_vec[1]/4, right_bottom_text[1]+line2_vec[1]], lw=3, color='black'))
    fig.add_artist(plt.Line2D([left_bottom_text[0], left_bottom_text[0]+line3_vec[0]/4], [left_bottom_text[1], left_bottom_text[1]+line3_vec[1]/4], lw=3, color='black'))
    fig.add_artist(plt.Line2D([left_bottom_text[0]+3*line3_vec[0]/4, left_bottom_text[0]+line3_vec[0]], [left_bottom_text[1]+3*line3_vec[1]/4, left_bottom_text[1]+line3_vec[1]], lw=3, color='black'))

    # Add subplots
    ax_left_top = fig.add_axes(left_top_regplot)
    ax_right_top = fig.add_axes(right_top_regplot)
    ax_bottom = fig.add_axes(bottom_regplot)

    # Plot the first correlation (MVPA vs dJm_ratio)
    sns.regplot(ax=ax_left_top, x=keys[0], y=keys[1], data=data, scatter_kws={'s':10}, line_kws={'color':'orange'})
    axes_format(ax_left_top, fs_ticks=30)
    ax_left_top.set_xlabel('')
    ax_left_top.set_ylabel('')
    corr, p_val = scipy.stats.pearsonr(data[keys[0]], data[keys[1]])
    ax_left_top.text(0.05, 0.05, f'r= {corr:.2f}, p= {p_val:.2f}', transform=ax_left_top.transAxes, fontsize=30)

    # Plot the third correlation (dJm_ratio vs d_offset)
    sns.regplot(ax=ax_bottom, x=keys[2], y=keys[1], data=data, scatter_kws={'s':10}, line_kws={'color':'orange'})
    axes_format(ax_bottom, fs_ticks=30)
    ax_bottom.set_xlabel('')
    ax_bottom.set_ylabel('')
    corr, p_val = scipy.stats.pearsonr(data[keys[2]], data[keys[1]])
    ax_bottom.text(0.05, 0.05, f'r= {corr:.2f}, p= {p_val:.2f}', transform=ax_bottom.transAxes, fontsize=30)
    
    # Plot the second correlation (MVPA vs d_offset)
    sns.regplot(ax=ax_right_top, x=keys[0], y=keys[2], data=data, scatter_kws={'s':10}, line_kws={'color':'orange'})
    axes_format(ax_right_top, fs_ticks=30)
    ax_right_top.set_xlabel('')
    ax_right_top.set_ylabel('')
    corr, p_val = scipy.stats.pearsonr(data[keys[0]], data[keys[2]])
    ax_right_top.text(0.05, 0.05, f'r= {corr:.2f}, p= {p_val:.2f}', transform=ax_right_top.transAxes, fontsize=30)

    # Remove unused axes
    ax.axis('off')

    # Save the figure
    file_path = os.path.join(folder_to_save, filename)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()


######### PLOT RESULTS ON PARAMETERS and TUNING CURVES ############
def plot_results_on_parameters(final_folder_path, num_training, plot_per_run = True, plot_boxplots = True):
    """ Plot the results from the results csv files and tuning curves csv files"""
    folder_to_save = os.path.join(final_folder_path, 'figures')

    ######### PLOT RESULTS ############
    if plot_per_run:
        # Create a per_run folder within the figures folder
        per_run_folder = os.path.join(folder_to_save, 'per_run')
        if not os.path.exists(per_run_folder):
            os.makedirs(per_run_folder)
        excluded_run_inds = plot_results_from_csvs(final_folder_path, num_training, folder_to_save=per_run_folder)
        # save excluded_run_inds into a csv in final_folder_path
        excluded_run_inds_df = pd.DataFrame(excluded_run_inds)
        excluded_run_inds_df.to_csv(os.path.join(final_folder_path, 'excluded_runs.csv'))
        if excluded_run_inds is not None:
            exclude_runs(final_folder_path, excluded_run_inds)
            num_training=num_training-len(excluded_run_inds)
    if plot_boxplots:
        boxplots_from_csvs(final_folder_path, folder_to_save, num_time_inds = 3, num_training=num_training)


################## MVPA related plots ##################

def plot_corr_triangles(final_folder_path, folder_to_save):
    """ Plot the correlation triangles for the MVPA results """

    data_rel_changes, _ = rel_change_for_runs(final_folder_path, num_indices = 3)
    MVPA_scores = pd.read_csv(final_folder_path +'/MVPA_scores.csv').to_numpy() # MVPA_scores - num_trainings x layer x SGD_ind x ori_ind (sup layer = 0)
    data_sup_55 = pd.DataFrame({
        'MVPA': (MVPA_scores[:,0,-1,0]- MVPA_scores[:,0,-2,0])/MVPA_scores[:,0,-2,0],
        'JsI/JsE': data_rel_changes['EI_ratio_J_s'],
        'offset_th': data_rel_changes['staircase_offset']
    })
    plot_corr_triangle(data_sup_55, folder_to_save, 'corr_triangle_sup_55')
    data_sup_125 = pd.DataFrame({
        'MVPA': (MVPA_scores[:,0,-1,1]- MVPA_scores[:,0,-2,1])/MVPA_scores[:,0,-2,1],
        'JsI/JsE': data_rel_changes['EI_ratio_J_s'],
        'offset_th': data_rel_changes['staircase_offset']
    })
    plot_corr_triangle(data_sup_125, folder_to_save, 'corr_triangle_sup_125')
    data_mid_55 = pd.DataFrame({
        'MVPA': (MVPA_scores[:,1,-1,0]- MVPA_scores[:,1,-2,0])/MVPA_scores[:,1,-2,0],
        'JmI/JmE': data_rel_changes['EI_ratio_J_m'],
        'offset_th': data_rel_changes['staircase_offset']
    })
    plot_corr_triangle(data_mid_55, folder_to_save, 'corr_triangle_mid_55')
    data_mid_125 = pd.DataFrame({
        'MVPA': (MVPA_scores[:,1,-1,1]- MVPA_scores[:,1,-2,1])/MVPA_scores[:,1,-2,1],
        'JmI/JmE': data_rel_changes['EI_ratio_J_m'],
        'offset_th': data_rel_changes['staircase_offset']
    })
    plot_corr_triangle(data_mid_125, folder_to_save, 'corr_triangle_mid_125')
    
    
def plot_MVPA_or_Mahal_scores(final_folder_path, num_runs, scores, file_name):
    """ Plot the MVPA scores or Mahalanobis distances for the two layers and two orientations """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # iterate over the two layers
    layer_label = ['sup', 'mid']
    for layer in range(2):
        # create a list of the 4 conditions
        data = [scores[:,layer,1, 0], scores[:,layer,2, 0], scores[:,layer,1, 1], scores[:,layer,2, 1]]
        # draw the boxplot
        ax[layer].boxplot(data, positions=[1, 2, 3, 4])
        ax[layer].set_xticklabels(['55 pre', '55 post', '125 pre', '125 post'])
        ax[layer].set_title(f'Layer {layer_label[layer]}')
        ylabel_text = 'MVPA score' if 'MVPA' in file_name else 'Mahalanobis distance'
        ax[layer].set_ylabel(ylabel_text)
        # draw lines to connect the pre and post training for each sample
        for i in range(num_runs):
            #gray lines
            ax[layer].plot([1, 2], [data[0][i], data[1][i]], color='gray', alpha=0.5, linewidth=0.5)
            ax[layer].plot([3, 4], [data[2][i], data[3][i]], color='gray', alpha=0.5, linewidth=0.5)
    plt.savefig(os.path.join(final_folder_path ,'MVPA', file_name+'.png'))
    plt.close()