import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as np
import numpy
import seaborn as sns
import statsmodels.api as sm
import scipy
import os
import sys
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from analysis.analysis_functions import rel_change_for_run, rel_change_for_runs, MVPA_param_offset_correlations, data_from_run, param_offset_correlations, pre_post_for_runs
from util import check_header, csv_to_numpy

plt.rcParams['xtick.labelsize'] = 12 # Set the size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12 # Set the size for y-axis tick labels

######### HELPER FUNCTIONS ############
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


def annotate_bar(ax, bars, values, ylabel=None, title=None):
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
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=20)
    if title is not None:
        ax.set_title(title, fontsize=20)

######### PLOT RESULTS ON PARAMETERS ############

def plot_results_from_csv(folder, run_index = 0, fig_filename=''):
    """ Plot the results on the parameters from a signle result csv file """
    def plot_params_over_time(ax, df, keys, colors, linestyles, title, SGD_steps):
        for i, key in enumerate(keys):
            ax.plot(range(SGD_steps), df[key], label=key, color=colors[i], linestyle=linestyles[i])
        ax.set_title(title, fontsize=20)
        ax.legend(loc='upper right', fontsize=20)
        ax.set_xlabel('SGD steps', fontsize=20)
        
    def plot_accuracy(ax, df, SGD_steps):
        """Plot accuracy changes over time."""
        for column in df.columns:
            if 'acc' in column and 'val_' not in column:
                ax.plot(range(SGD_steps), df[column], label=column, alpha=0.6, c='tab:green')
            elif 'val_acc' in column:
                ax.scatter(range(SGD_steps), df[column], label=column, marker='o', s=50, c='green')
        ax.legend(loc='lower right', fontsize=20)
        ax.set_title('Accuracy', fontsize=20)
        ax.axhline(y=0.5, color='black', linestyle='--')
        ax.set_ylim(0, 1)
        ax.set_xlabel('SGD steps', fontsize=20)

    def plot_loss(ax, df, SGD_steps):
        for column in df.columns:
            if 'loss_' in column and 'val_loss' not in column:
                ax.plot(range(SGD_steps), df[column], label=column, alpha=0.6)
            if 'val_loss' in column:
                ax.scatter(range(SGD_steps), df[column], label='val_loss', marker='o', s=50)
        ax.legend(loc='upper right', fontsize=20)
        ax.set_title('Loss', fontsize=20)
        ax.set_xlabel('SGD steps', fontsize=20)

    def plot_offset(ax, df, SGD_steps, keys_metrics):
        num_pretraining_steps= sum(df['stage'] == 0)
        ax.plot(range(num_pretraining_steps), np.ones(num_pretraining_steps)*6, alpha=0.6, c='black', linestyle='--')
        if len(keys_metrics)>1:
            ax.scatter(range(SGD_steps), df[keys_metrics[0]], label=keys_metrics[0], marker='o', s=70, c='tab:orange')
            ax.scatter(range(SGD_steps), df[keys_metrics[1]], label=keys_metrics[1], marker='o', s=50, c='tab:brown')
            ax.set_ylim(0, min(25,max(df[keys_metrics[1]])+1)) # keys_metrics[1] is the staircase offset
        else: # when the run returned NA and so training was not saved
            ax.scatter(range(SGD_steps), df[keys_metrics[0]], label=keys_metrics[0], marker='o', s=50, c='tab:green')
        ax.grid(color='gray', linestyle='-', linewidth=0.5)
        ax.set_title('Offset', fontsize=20)
        ax.set_ylabel('degrees', fontsize=20)
        ax.set_xlabel('SGD steps', fontsize=20)        
        ax.legend(loc='upper right', fontsize=20)

    def plot_readout_weights(ax, df, SGD_steps):
        ax.plot(range(SGD_steps), df['b_sig'], label='b_sig', linestyle='--', linewidth = 3)
        ax.set_xlabel('SGD steps', fontsize=20)
        i=0
        for i in range(10):
            column = f'w_sig_{29+i}' # if all 81 w_sigs are saved
            if column in df.keys():
                ax.plot(range(SGD_steps), df[column], label=column)
            else:
                column = f'w_sig_{i}' # if only the middle 25 w_sigs are saved
                if column in df.keys():                
                    ax.plot(range(SGD_steps), df[column], label=column)
        ax.set_title('Readout bias and weights', fontsize=20)
        ax.legend(loc='upper right', fontsize=20)
    
    # check if filename fig_filename + ".png" exists, if it does, return
    if os.path.exists(fig_filename + ".png"):
        return
    else:    
        df, time_inds, no_train_data  = data_from_run(folder, run_index=run_index, num_indices=3)
        # if df is empty, then there is no training data from that run (possibly because of divergence of the model)
        if df.empty:
            return
        SGD_steps = time_inds[-1]+1
        
        # Define colors and linestyles for the plots
        colors_J=['tab:red' ,'tab:orange','tab:green', 'tab:blue', 'tab:red' ,'tab:orange','tab:green', 'tab:blue']
        linestyles_J = ['--', '--', '--', '--', '-', '-', '-', '-']
        keys_J_raw = ['J_EE_m', 'J_IE_m', 'J_EI_m', 'J_II_m', 'J_EE_s', 'J_IE_s', 'J_EI_s', 'J_II_s']
        keys_maxr = ['maxr_E_mid', 'maxr_I_mid', 'maxr_E_sup', 'maxr_I_sup']
        keys_meanr = ['meanr_E_mid', 'meanr_I_mid', 'meanr_E_sup', 'meanr_I_sup']
        linestyles_r = [':', ':', '-', '-']
        colors_r = ["tab:red", "tab:blue", "tab:red", "tab:blue"]
        keys_cf = ['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I']
        colors_cf = ['tab:orange', 'tab:green','tab:red', 'tab:blue', 'tab:red', 'tab:blue']
        linestyles_cf = ['-', '-', '-', '-', '--', '--']
        colors_metrics = [ 'tab:orange', 'tab:brown','tab:green']
        
        # Define keys for the metrics - might be different than keys_metrics_rel_change in case there is no training data
        keys_offsets = [key for key in df.keys() if '_offset' in key]
        keys_acc = [key for key in df.keys() if key.startswith('acc')]
        keys_metrics = keys_offsets + keys_acc
        if not no_train_data:
            # Define values for the bar plots
            rel_changes_train, _, _ = rel_change_for_run(folder, run_index, 3)
            values_J = [rel_changes_train[key] for key in keys_J_raw]
            keys_offsets_rel_change = [key for key in rel_changes_train.keys() if '_offset' in key]
            if len(keys_offsets_rel_change)>0:
                keys_metrics_rel_change = keys_offsets_rel_change + keys_acc
            else:
                keys_metrics_rel_change = keys_acc
            values_metrics = [rel_changes_train[key] for key in keys_metrics_rel_change]
            values_meanr = [rel_changes_train[key] for key in keys_meanr] 
            values_fc = [rel_changes_train[key]for key in keys_cf]        

        # Create the figure
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(60, 45))

        # Plot each section
        plot_accuracy(axes[0,0], df, SGD_steps)
        plot_loss(axes[0,1], df, SGD_steps)
        plot_offset(axes[0,2], df, SGD_steps, keys_metrics)
        plot_params_over_time(axes[1,0], df, keys_maxr, colors_r, linestyles_r, title='Maximum rates', SGD_steps=SGD_steps)
        plot_params_over_time(axes[1,1], df, keys_meanr, colors_r, linestyles_r, title='Mean rates', SGD_steps=SGD_steps)
        plot_readout_weights(axes[1,2], df, SGD_steps)
        plot_params_over_time(axes[2,0], df, keys_J_raw, colors_J, linestyles_J, title='J in middle and superficial layers', SGD_steps=SGD_steps)
        plot_params_over_time(axes[2,1], df, keys_cf, colors_cf, linestyles_cf, title='c: constant inputs, f: weights between mid and sup layers', SGD_steps=SGD_steps)
        if not no_train_data:
            bars_params = axes[2,2].bar(keys_J_raw, values_J, color=colors_J)   
            bars_metrics = axes[0,3].bar(keys_metrics_rel_change, values_metrics, color=colors_metrics)
            bars_r = axes[1,3].bar(keys_meanr, values_meanr, color=colors_r)         
            bars_cf = axes[2,3].bar(keys_cf, values_fc, color=colors_cf)
            
            # Annotating each bar with its value for bars_params
            annotate_bar(axes[2,2], bars_params, values_J, ylabel='relative change %', title= 'Rel. changes in J before and after training')
            annotate_bar(axes[0,3], bars_metrics, values_metrics, ylabel='relative change %', title= 'Rel. changes in metrics before and after training')
            annotate_bar(axes[1,3], bars_r, values_meanr, ylabel='relative change %', title= 'Rel. changes in mean rates before and after training')
            annotate_bar(axes[2,3], bars_cf, values_fc, ylabel='relative change %', title= 'Other rel changes before and after training')
        
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

        fig.savefig(fig_filename + ".png")
        plt.close()


def plot_results_from_csvs(folder_path, num_runs=3, starting_run=0):
    """ Plot the per-run results from the results csv on the parameters """
    # Add folder_path to path
    if folder_path not in sys.path:
        sys.path.append(folder_path)

    # Create a per_run folder within the figures folder
    folder_to_save = os.path.join(folder_path,  'figures', 'per_run')
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)    

    # Plot loss, accuracy and trained parameters and save the figures
    for j in range(starting_run, num_runs):
        results_fig_filename = os.path.join(folder_to_save,f'resultsfig_{j}')
        plot_results_from_csv(folder_path, j, results_fig_filename)


def barplots_from_csvs(folder, save_folder=None, excluded_runs = []):
    def scatter_data_with_lines(ax, data):
        for i in range(2):
            group_data = data[i, :]
            # let the position be i 
            x_positions = [i for _ in range(len(group_data))]
            ax.scatter(x_positions, group_data, color='black', alpha=0.7)  # Scatter plot of individual data points
        
        # Draw lines connecting the points between the two groups
        for j in range(len(data[0])):
            ax.plot([0, 1], [data[0, j], data[1, j]], color='black', alpha=0.2)
        
    def barplot_params(colors, keys, titles, means_pre, means_post, vals_pre, vals_post, figname, ylims=None):
        if len(keys) == 1:
            fig, ax = plt.subplots()
        elif len(keys) == 2:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax_flat = ax
        else:
            fig, ax = plt.subplots(2, len(keys)//2, figsize=(len(keys)//2 * 5, 10))
            ax_flat = ax.flatten()
        # Colors for bars
        param_means_pre = [means_pre[keys[i]] for i in range(len(keys))]
        param_means_post = [means_post[keys[i]] for i in range(len(keys))]
        x_positions = [0, 1]
        x_labels = ['Pre', 'Post']
        for i in range(len(keys)):
            if len(keys) == 1:
                ax_i = ax
            else:
                ax_i = ax_flat[i]
            bars = ax_i.bar(x_positions, [param_means_pre[i], param_means_post[i]], color=colors[i], alpha=0.7)
            ax_i.set_title(titles[i], fontsize=20)
            ax_i.set_xticks(x_positions)
            ax_i.set_xticklabels(x_labels, fontsize=20)
            annotate_bar(ax_i, bars, [param_means_pre[i], param_means_post[i]])
            scatter_data_with_lines(ax_i, numpy.abs(numpy.array([vals_pre[keys[i]], vals_post[keys[i]]])))
            if ylims is not None:
                ax_i.set_ylim(ylims[i])
        plt.tight_layout()
        fig.savefig(figname)
        plt.close()
    
    if save_folder is None:
        save_folder = os.path.join(folder, 'figures')

    # Get relevant data for plots: values for pre and post training and relative changes
    pretrain_filepath = os.path.join(os.path.dirname(folder), 'pretraining_results.csv') 
    pretrain_df = pd.read_csv(pretrain_filepath)
    num_training = pretrain_df['run_index'].nunique()
    _, vals_pre, vals_post = pre_post_for_runs(folder, num_training, num_time_inds=2, excluded_runs=excluded_runs)

    # Extend vals_pre, vals_post with 'J_EI_m_abs', 'J_II_m_abs', 'J_EI_s_abs', 'J_II_s_abs' to include the absolute values
    vals_pre['J_EI_m_abs'] = numpy.abs(vals_pre['J_EI_m'])
    vals_pre['J_II_m_abs'] = numpy.abs(vals_pre['J_II_m'])
    vals_pre['J_EI_s_abs'] = numpy.abs(vals_pre['J_EI_s'])
    vals_pre['J_II_s_abs'] = numpy.abs(vals_pre['J_II_s'])
    vals_post['J_EI_m_abs'] = numpy.abs(vals_post['J_EI_m'])
    vals_post['J_II_m_abs'] = numpy.abs(vals_post['J_II_m'])
    vals_post['J_EI_s_abs'] = numpy.abs(vals_post['J_EI_s'])
    vals_post['J_II_s_abs'] = numpy.abs(vals_post['J_II_s'])
    means_pre = vals_pre.mean()
    means_post = vals_post.mean()

    # Plotting bar plots of offset before and after training or pretraining 
    keys_offset = ['staircase_offset', 'psychometric_offset']
    titles_offset = ['Staircase offset threshold', 'Psychometric offset threshold']
    colors = ['green', 'forestgreen']
    barplot_params(colors, keys_offset, titles_offset, means_pre, means_post, vals_pre, vals_post, save_folder + '/offset_pre_post.png', ylims=[[0, 20], [0, 20]])

    # Plotting bar plots of J parameters before and after training or pretraining 
    colors=['red' ,'tab:red','blue', 'tab:blue' ,'red' ,'tab:red', 'blue', 'tab:blue']
    keys_J = ['J_EE_m', 'J_IE_m', 'J_EI_m_abs', 'J_II_m_abs', 'J_EE_s', 'J_IE_s', 'J_EI_s_abs', 'J_II_s_abs']
    titles_J = [ r'$J^{\text{mid}}_{E \rightarrow E}$', r'$J^{\text{mid}}_{E \rightarrow I}$', r'$J^{\text{mid}}_{I \rightarrow E}$', r'$J^{\text{mid}}_{I \rightarrow I}$', r'$J^{\text{sup}}_{E \rightarrow E}$', r'$J^{\text{sup}}_{E \rightarrow I}$', r'$J^{\text{sup}}_{I \rightarrow E}$', r'$J^{\text{sup}}_{I \rightarrow I}$']
    barplot_params(colors, keys_J, titles_J, means_pre, means_post, vals_pre, vals_post, save_folder + '/J_pre_post.png')

    # Plotting bar plots of r parameters before and after training or pretraining
    colors=['red' ,'blue', 'tab:red', 'tab:blue' ,'red' , 'blue','tab:red', 'tab:blue']
    keys_r = ['maxr_E_mid', 'maxr_I_mid', 'maxr_E_sup', 'maxr_I_sup', 'meanr_E_mid', 'meanr_I_mid', 'meanr_E_sup', 'meanr_I_sup']
    titles_r = [r'$r^{\text{mid}}_{\text{max},E}$', r'$r^{\text{mid}}_{\text{max},I}$', r'$r^{\text{sup}}_{\text{max},E}$', r'$r^{\text{sup}}_{\text{max},I}$', r'$r^{\text{mid}}_{\text{mean},E}$', r'$r^{\text{mid}}_{\text{mean},I}$', r'$r^{\text{sup}}_{\text{mean},E}$', r'$r^{\text{sup}}_{\text{mean},I}$']
    barplot_params(colors, keys_r, titles_r, means_pre, means_post, vals_pre, vals_post, save_folder + '/r_pre_post.png')

    # Plotting bar plots of c, f and kappa parameters before and after  
    colors=['orange', 'orange', 'green', 'green', 'green', 'green', 'green', 'green', 'green', 'green']
    keys_c_f_kappa = ['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'kappa_EE_pre', 'kappa_IE_pre', 'kappa_EE_post', 'kappa_IE_post', 'f_I']
    titles_c_f_kappa = [r'$c^{\text{mid}}_{\rightarrow E}$', r'$c^{\text{mid}}_{\rightarrow I}$', r'$c^{\text{sup}}_{\rightarrow E}$', r'$c^{\text{sup}}_{\rightarrow I}$', r'$f^{\text{mid}\rightarrow \text{sup}}_{E \rightarrow E}$', r'$\kappa^{\text{pre}}_{E \rightarrow  E}$', r'$\kappa^{\text{pre}}_{I \rightarrow  E}$', r'$\kappa^{\text{post}}_{E \rightarrow  E}$', r'$\kappa^{\text{post}}_{I \rightarrow  E}$', r'$f^{\text{mid}\rightarrow \text{sup}}_{E \rightarrow I}$']
    barplot_params(colors, keys_c_f_kappa, titles_c_f_kappa, means_pre, means_post, vals_pre, vals_post, save_folder + '/c_f_kappa_pre_post.png')
    

def boxplots_from_csvs(folder, save_folder = None, num_time_inds = 3, excluded_runs = []):
    """ Create boxplots and barplots for the relative changes of the parameters before and after training """
    def boxplot_params(keys_group, rel_changes, group_labels, box_colors, full_path, set_ylim=False):
        num_groups=len(keys_group)
        num_training = len(rel_changes[keys_group[0][0]])
        fig, axs = plt.subplots(1, num_groups, figsize=(6*num_groups, 5))  # Create subplots
        
        for i, label in enumerate(group_labels):
            group_data = numpy.zeros((num_training, len(keys_group[i])))
            for var_ind in range(len(keys_group[i])):
                if keys_group[i][var_ind] in rel_changes.keys():
                    group_data[:, var_ind] = rel_changes[keys_group[i][var_ind]].T
                else:
                    group_data[:, var_ind] = numpy.zeros(num_training)
            
            bp = axs[i].boxplot(group_data, labels=label, patch_artist=True)
            for median in bp['medians']:
                median.set(color='black', linewidth=2.5)
            for box, color in zip(bp['boxes'], box_colors[i]):
                box.set_facecolor(color)
            axs[i].axhline(y=0, color='black', linestyle='--')
            axes_format(axs[i], fs_ticks=20, ax_width=2, tick_width=5, tick_length=10, xtick_flag=False)
            if i == num_groups - 1:
                axs[i].set_ylabel('Absolute change', fontsize=20)
            else:
                axs[i].set_ylabel('Relative change (%)', fontsize=20)
            if set_ylim:
                if i < len(group_labels) - 1:
                    axs[i].set_ylim(-60, 100)
                else:
                    axs[i].set_ylim(-2.5, 2.5)
                # set the y-ticks to be at -60, -30, 0, 30, 60 for the first 4 plots and -2.5, -1.25, 0, 1.25, 2.5 for the last plot
                yticks = numpy.linspace(-60, 60, 5) if i < len(group_labels) - 1 else numpy.linspace(-2.5, 2.5, 5)
                axs[i].set_yticks(yticks)
                
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.5)
        fig.savefig(full_path)
        plt.close()

    if save_folder is None:
        save_folder = os.path.join(folder, 'figures')

    # Get relevant data for plots: values for pre and post training and relative changes
    pretrain_filepath = os.path.join(os.path.dirname(folder), 'pretraining_results.csv') 
    pretrain_df = pd.read_csv(pretrain_filepath)
    num_training = pretrain_df['run_index'].nunique()
    rel_changes_train, rel_changes_pretrain = rel_change_for_runs(folder, num_time_inds=num_time_inds, num_runs=num_training, excluded_runs=excluded_runs)

    # Define groups of parameters and plot each parameter group
    keys_group = [['J_EE_m', 'J_IE_m', 'J_EI_m', 'J_II_m'], ['J_EE_s', 'J_IE_s', 'J_EI_s', 'J_II_s'], ['cE_m', 'cI_m','cE_s', 'cI_s'], ['f_E', 'f_I'], ['kappa_EE_pre','kappa_IE_pre','kappa_EE_post','kappa_IE_post']]
    group_labels = [
        [r'$\Delta J^{\text{mid}}_{E \rightarrow E}$', r'$\Delta J^{\text{mid}}_{E \rightarrow I}$', r'$\Delta J^{\text{mid}}_{I \rightarrow E}$', r'$\Delta J^{\text{mid}}_{I \rightarrow I}$'],
        [r'$\Delta J^{\text{sup}}_{E \rightarrow E}$', r'$\Delta J^{\text{sup}}_{E \rightarrow I}$', r'$\Delta J^{\text{sup}}_{I \rightarrow E}$', r'$\Delta J^{\text{sup}}_{I \rightarrow I}$'],
        [r'$\Delta c^{\text{mid}}_{\rightarrow E}$', r'$\Delta c^{\text{mid}}_{\rightarrow I}$', r'$\Delta c^{\text{sup}}_{\rightarrow E}$', r'$\Delta c^{\text{sup}}_{\rightarrow I}$'],
        [r'$\Delta f^{\text{mid}\rightarrow \text{sup}}_{E \rightarrow E}$', r'$\Delta f^{\text{mid}\rightarrow \text{sup}}_{E \rightarrow I}$'],
        [r'$\Delta \kappa^{\text{pre}}_{E \rightarrow  E}$',r'$\Delta \kappa^{\text{pre}}_{E \rightarrow  I}$',r'$\Delta \kappa^{\text{post}}_{E \rightarrow  E}$',r'$\Delta \kappa^{\text{post}}_{E \rightarrow  I}$']
    ]
    J_box_colors = ['tab:red','tab:red','tab:blue','tab:blue']
    c_box_colors = ['orange', 'orange', 'orange', 'orange']
    f_box_colors = ['orange', 'orange']
    kappa_box_colors = ['tab:green','tab:green','tab:green','tab:green']    
    box_colors = [J_box_colors,J_box_colors,c_box_colors, f_box_colors, kappa_box_colors]
    
    # Create boxplots and save the figure
    pretrain_fig_folder = os.path.join(os.path.dirname(folder),'pretraining_figures')
    if not os.path.exists(pretrain_fig_folder):
        os.makedirs(pretrain_fig_folder)
    if num_time_inds == 3:
        boxplot_params(keys_group, rel_changes_pretrain, group_labels, box_colors, os.path.join(pretrain_fig_folder,'boxplot_relative_changes.png'))
    boxplot_params(keys_group, rel_changes_train, group_labels, box_colors, os.path.join(save_folder,'boxplot_relative_changes.png'))
    boxplot_params(keys_group, rel_changes_train, group_labels, box_colors, os.path.join(save_folder,'boxplot_relative_changes_ylim.png'), set_ylim=True)
    plt.close()


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


def plot_tuning_curves(results_dir, tc_cells, num_runs, folder_to_save=None, seed=0, tc_cell_labels=None, excluded_runs=[]):
    """Plot example tuning curves for middle and superficial layer cells at different stages of training"""
    if folder_to_save is None:
        folder_to_save = os.path.join(results_dir, 'figures')

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
    valid_runs = numpy.setdiff1d(numpy.arange(num_runs), excluded_runs)
    for i in range(num_runs_plotted):
        # Select tuning curves for the current run
        mesh_i = tuning_curves[:,0]==valid_runs[i]
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


def plot_tc_features(results_dir, stages=[0,1,2], color_by=None, add_cross=False, only_slope_plot=False, only_center_cells=False, excluded_runs=[]):
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

    def shift_x_data(x_data, indices, shift_val=90, L_ori=180):
        """ Shift circular x_data by shift_value and center it around the new 0 (around shift_value) """
        x_data_shifted = x_data[:, indices].flatten() - shift_val
        x_data_shifted = numpy.where(x_data_shifted > L_ori/2, x_data_shifted - L_ori, x_data_shifted)
        x_data_shifted = numpy.where(x_data_shifted < -L_ori/2, x_data_shifted + L_ori, x_data_shifted)
        return x_data_shifted
    
    def scatter_feature(ax, y, x, indices_E, indices_I, shift_val=55, fs_ticks=40, feature=None, values_for_colors=None, title=None, axes_inds=[0,0], add_cross=False):
        """Scatter plot of feature for E and I cells. If shift_val is not None, then x should be the preferred orientation that will be centered around shift_val."""
        if shift_val is not None:
            x_I = shift_x_data(x, indices_I, shift_val=shift_val)
            x_E = shift_x_data(x, indices_E, shift_val=shift_val)
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
            color_E = 'tab:orange'
            color_I = 'tab:cyan'
        ax.scatter(x_E, y[:,indices_E].flatten(), s=50, alpha=0.3, color=color_E)
        ax.scatter(x_I, y[:,indices_I].flatten(), s=50, alpha=0.3, color=color_I)
        if shift_val is None:
            xpoints = ypoints = ax.get_xlim()
            ax.plot(xpoints, ypoints, color='black', linewidth=2)
            if axes_inds[0]==1:
                ax.set_xlabel('Pre training', fontsize=fs_ticks)
            if axes_inds[1] == 0 and axes_inds[0] == 0:
                ax.set_ylabel('MIDDLE LAYER \n\n Post training', fontsize=fs_ticks)
            if axes_inds[1] == 0 and axes_inds[0] == 1:
                ax.set_ylabel('SUPERFICIAL LAYER \n\n Post training', fontsize=fs_ticks)
            axes_format(ax, fs_ticks)
        else:
            if axes_inds[1] == 0 and axes_inds[0] == 0:
                ax.set_ylabel('MIDDLE LAYER \n', fontsize=fs_ticks)
            if axes_inds[1] == 0 and axes_inds[0] == 1:
                ax.set_ylabel('SUPERFICIAL LAYER \n', fontsize=fs_ticks)
            
        if title is None:
            title = feature
        if axes_inds[0]==0:
            ax.set_title(title, fontsize=fs_ticks)

        if add_cross: # add a cross at the mean of the data and two error bars for the standard deviations on x and y axes
            mean_E_y = numpy.mean(y[:,indices_E].flatten())
            mean_I_y = numpy.mean(y[:,indices_I].flatten())
            mean_E_x = numpy.mean(x_E)
            mean_I_x = numpy.mean(x_I)
            std_E_y = numpy.std(y[:,indices_E].flatten())
            std_I_y = numpy.std(y[:,indices_I].flatten())
            std_E_x = numpy.std(x_E)
            std_I_x = numpy.std(x_I)
            ax.errorbar(mean_E_x, mean_E_y, xerr=std_E_x, yerr=std_E_y, fmt='o', color='tab:red', markersize=15, elinewidth=8, capsize=5, capthick=2)
            #print(f'{mean_E_x:.1f}, {mean_E_y:.1f}, {std_E_x:.1f}, {std_E_y:.1f}')
            ax.errorbar(mean_I_x, mean_I_y, xerr=std_I_x, yerr=std_I_y, fmt='o', color='blue', markersize=15, elinewidth=8, capsize=5, capthick=2)
            #print(f'{mean_I_x:.1f}, {mean_I_y:.1f}, {std_I_x:.1f}, {std_I_y:.1f}')

    def lowess_feature(ax, y, x, indices_E, indices_I, shift_val=55, frac=0.15, shades=''):
        """Plot lowess smoothed feature for E and I cells. Curves are fitted separately for E and I cells and for x<0 and x>0."""
        # fit curves separately for the smoothed_x_E<0 and smoothed_x_E>0
        if shift_val is not None:
            x_I = shift_x_data(x, indices_I, shift_val=shift_val)
            x_E = shift_x_data(x, indices_E, shift_val=shift_val)
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
        
        ax.plot(lowess_E[:, 0], lowess_E[:, 1], color=shades+'red', linewidth=10, alpha=0.8)
        ax.plot(lowess_I[:, 0], lowess_I[:, 1], color=shades+'blue', linewidth=10, alpha=0.8)

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
    
    # Filter out excluded runs
    valid_runs = numpy.setdiff1d(numpy.arange(pretrain_tc_features['run_index'].nunique()), excluded_runs)
    pretrain_tc_features = pretrain_tc_features[pretrain_tc_features['run_index'].isin(valid_runs)]
    train_tc_features = train_tc_features[train_tc_features['run_index'].isin(valid_runs)]

    ############## Plots about changes before vs after training and pretraining (per layer and per centered or all) ##############
      
    # Define indices for each group of cells
    if only_center_cells:
        E_sup_centre = numpy.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)+648
        I_sup_centre = (E_sup_centre+81).astype(int)
        E_mid_centre_ph0 = numpy.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)
        E_mid_centre_ph1, E_mid_centre_ph2, E_mid_centre_ph3 = E_mid_centre_ph0+162, E_mid_centre_ph0+2*162, E_mid_centre_ph0+3*162
        E_mid_centre = numpy.concatenate((E_mid_centre_ph0, E_mid_centre_ph1, E_mid_centre_ph2, E_mid_centre_ph3))
        I_mid_centre = E_mid_centre + 81
        indices = [E_mid_centre, I_mid_centre, E_sup_centre, I_sup_centre]
        E_mid_centre_phase_0_pi= numpy.concatenate((E_mid_centre_ph0,E_mid_centre_ph2))
        indices_mid_phase_0_pi = [E_mid_centre_phase_0_pi, E_mid_centre_phase_0_pi + 81, E_sup_centre, I_sup_centre]
    else:
        E_sup = numpy.linspace(0, 80, 81).astype(int) + 648 
        I_sup = numpy.linspace(81, 161, 81).astype(int) + 648
        mid_array = numpy.linspace(0, 647, 648).round().reshape(4, 2, 81).astype(int)
        E_mid = mid_array[:,0,:].ravel().astype(int)
        I_mid = mid_array[:,1,:].ravel().astype(int)
        indices = [E_mid, I_mid, E_sup, I_sup]
        E_mid_phase_0_pi = numpy.concatenate((mid_array[0,0,:], mid_array[2,0,:]))
        indices_mid_phase_0_pi = [E_mid_phase_0_pi, E_mid_phase_0_pi + 81, E_sup, I_sup]
    
    ###############################################
    ######### Schoups-style scatter plots #########
    ###############################################
    cmap = plt.get_cmap('rainbow')
    fs_text = 20
    fs_ticks = 20
    
    # Scatter slope, where x-axis is orientation and y-axis is the change in slope before and after training
    stage_labels = ['pretrain', 'train']
    # Get the preferred orientation from orimap.csv
    orimap_file_path = os.path.join(os.path.dirname(results_dir),'orimap.csv')
    df_orimap = pd.read_csv(orimap_file_path)
    np_orimap=df_orimap.to_numpy()
    np_orimap = np_orimap[:,1:]
    # Repeat the orimap 10 times over the second axis
    pref_ori_all = np.tile(np_orimap, 10)
    pref_ori = pref_ori_all[valid_runs, :]

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
        if not only_slope_plot:
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

        # Get the preferred orientation from data_post and compare it with pref_ori_all
        #mesh_pref_ori = data_post['feature']=='pref_ori'
        #pref_ori_v1 = data_post.loc[mesh_pref_ori]
        #pref_ori_v1 = pref_ori_v1.loc[:,mesh_cells].to_numpy()
        #print(numpy.allclose(pref_ori, pref_ori_v1))
        
        if color_by == 'pref_ori':
            colors_scatter_feature = pref_ori
        elif color_by == 'phase':
            color_cell_group = numpy.ones(pref_ori.shape[1]//5)
            color_all_cells = numpy.concatenate((color_cell_group, 20*color_cell_group, 30*color_cell_group, 40*color_cell_group, 50*color_cell_group))
            colors_scatter_feature = numpy.reshape(numpy.tile(color_all_cells, pref_ori.shape[0]), (pref_ori.shape[0], pref_ori.shape[1]))
        elif color_by == 'run_index':
            colors_scatter_feature = numpy.reshape(numpy.repeat(numpy.arange(pref_ori.shape[0]), pref_ori.shape[1]), (pref_ori.shape[0], pref_ori.shape[1])) # coloring by run
        else:
            colors_scatter_feature = None
        if not only_slope_plot:
            for layer in range(2):            
                ##### Plot features before vs after training per layer and cell type #####
                scatter_feature(axs[layer,0], fwhm_post, fwhm_pre, indices[2*layer], indices[2*layer+1], shift_val=None, feature='fwhm', values_for_colors=colors_scatter_feature, title='Tuning width', axes_inds = [layer,0], add_cross=add_cross)
                scatter_feature(axs[layer,1], min_post, min_pre, indices[2*layer], indices[2*layer+1], shift_val=None, feature='min', values_for_colors=colors_scatter_feature, title='Baseline rate', axes_inds = [layer,1], add_cross=add_cross)
                scatter_feature(axs[layer,2], max_post, max_pre, indices[2*layer], indices[2*layer+1], shift_val=None, feature='max', values_for_colors=colors_scatter_feature, title='Peak firing rate', axes_inds = [layer,2], add_cross=add_cross)
                scatter_feature(axs[layer,3], mean_post, mean_pre, indices[2*layer], indices[2*layer+1], shift_val=None, feature='mean', values_for_colors=colors_scatter_feature, title='Mean firing rate', axes_inds = [layer,3], add_cross=add_cross)
                scatter_feature(axs[layer,4], slope_hm_post, slope_hm_pre, indices[2*layer], indices[2*layer+1], shift_val=None, feature='slope_hm', values_for_colors=colors_scatter_feature, title='Slope at half maximum', axes_inds = [layer,4], add_cross=add_cross)

            plt.tight_layout()

            # Add colorbar
            if colors_scatter_feature is not None:
                if (colors_scatter_feature == pref_ori).all():
                    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=np.min(pref_ori), vmax=np.max(pref_ori)), cmap='rainbow'),ax=axs,orientation='vertical')
                    cbar.set_label('Preferred orientation', fontsize=40)
                    cbar.ax.tick_params(labelsize=20) 
            else:
                # Discrete colors and labels
                display_legend = True
                discrete_colors = ['red', 'blue', 'green', 'orange']  # Example discrete colors
                if colors_scatter_feature is not None:
                    unique_values = np.unique(colors_scatter_feature)
                    discrete_colors = cmap(np.linspace(0, 1, len(unique_values)))
                    if len(discrete_colors)==5:
                        discrete_labels = ['Phase 0', 'Phase Pi/2', 'Phase 3Pi/2', 'Phase Pi', 'No phase']  # Corresponding labels
                    else:
                        display_legend = False
                else:
                    discrete_colors = ['tab:orange', 'tab:cyan']
                    discrete_labels = ['Excitatory', 'Inhibitory']
                if display_legend:
                    # Create custom legend for discrete set of colors
                    handles = [mpl.patches.Patch(color=color, label=label) for color, label in zip(discrete_colors, discrete_labels)]
                    axs[0, 0].legend(handles=handles, loc='upper left', fontsize=20)
            
            # Save plot
            if training_stage == 0:
                if not os.path.exists(os.path.join(os.path.dirname(results_dir),'pretraining_figures')):
                    os.makedirs(os.path.join(os.path.dirname(results_dir),'pretraining_figures'))
                file_path = os.path.join(os.path.dirname(results_dir),'pretraining_figures', f'tc_features_{stage_labels[training_stage]}_color_by_{color_by}.png')
            else:
                file_path = os.path.join(results_dir,'figures', f'tc_features_{stage_labels[training_stage]}_color_by_{color_by}.png')
            fig.savefig(file_path)
            plt.close()

        # Scatter plot of data[slopediff_55_0 and 1], data[slopediff_55_0 and 1] with lowess smoothed lines and Mann-Whitney U test
        fig, axs = plt.subplots(2, 4, figsize=(40, 20))
        # Loop through layers
        for layer in range(2):
            scatter_feature(axs[layer,0], slopediff_55, pref_ori, indices_mid_phase_0_pi[2*layer], indices_mid_phase_0_pi[2*layer+1], shift_val=55, feature='slopediff_55', values_for_colors=colors_scatter_feature, axes_inds=[layer,0])
            scatter_feature(axs[layer,1], slopediff_125, pref_ori, indices_mid_phase_0_pi[2*layer], indices_mid_phase_0_pi[2*layer+1], shift_val=125, feature='slopediff_125', values_for_colors=colors_scatter_feature, axes_inds=[layer,1])
            lowess_feature(axs[layer,0], slopediff_55, pref_ori, indices_mid_phase_0_pi[2*layer], indices_mid_phase_0_pi[2*layer+1], shift_val=55, shades='dark')
            lowess_feature(axs[layer,1], slopediff_125, pref_ori, indices_mid_phase_0_pi[2*layer], indices_mid_phase_0_pi[2*layer+1], shift_val=125, shades='tab:')
            lowess_feature(axs[layer,2], slopediff_55, pref_ori, indices_mid_phase_0_pi[2*layer], indices_mid_phase_0_pi[2*layer+1], shift_val=55, shades='dark')
            lowess_feature(axs[layer,2], slopediff_125, pref_ori, indices_mid_phase_0_pi[2*layer], indices_mid_phase_0_pi[2*layer+1], shift_val=125, shades='tab:')
            pref_ori_55_E = shift_x_data(pref_ori, indices_mid_phase_0_pi[2*layer], shift_val=55)
            pref_ori_125_E = shift_x_data(pref_ori, indices_mid_phase_0_pi[2*layer], shift_val=125)
            pref_ori_55_I = shift_x_data(pref_ori, indices_mid_phase_0_pi[2*layer+1], shift_val=55)
            pref_ori_125_I = shift_x_data(pref_ori, indices_mid_phase_0_pi[2*layer+1], shift_val=125)
            slopediff_55_E = slopediff_55[:,indices_mid_phase_0_pi[2*layer]].flatten()
            slopediff_55_I = slopediff_55[:,indices_mid_phase_0_pi[2*layer+1]].flatten()
            slopediff_125_E = slopediff_125[:,indices_mid_phase_0_pi[2*layer]].flatten()
            slopediff_125_I = slopediff_125[:,indices_mid_phase_0_pi[2*layer+1]].flatten()
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
            file_path = os.path.join(os.path.dirname(results_dir), 'pretraining_figures', f'tc_slope_{stage_labels[training_stage]}_color_by_{color_by}.png')
        else:
            file_path = os.path.join(results_dir,'figures', f'tc_slope_{stage_labels[training_stage]}_color_by_{color_by}.png')
        fig.savefig(file_path)
        plt.close()

################### CORRELATION ANALYSIS ###################
def match_keys_to_labels(key_list):
    # Create an empty dictionary to store the matching
    matched_labels = {}
    
    # Mapping rules from the keys to LaTeX notation
    for key in key_list:
        if 'J_EE_m' in key:
            matched_labels[key] = r'$\Delta J^{\text{mid}}_{E \rightarrow E}$'
        elif 'J_EI_m' in key:
            matched_labels[key] = r'$\Delta J^{\text{mid}}_{I \rightarrow E}$'
        elif 'J_IE_m' in key:
            matched_labels[key] = r'$\Delta J^{\text{mid}}_{E \rightarrow I}$'
        elif 'J_II_m' in key:
            matched_labels[key] = r'$\Delta J^{\text{mid}}_{I \rightarrow I}$'
        elif 'J_I_m' in key:
            matched_labels[key] = r'$\Delta J^{\text{mid}}_{I \rightarrow }$'
        elif 'J_E_m' in key:
            matched_labels[key] = r'$\Delta J^{\text{mid}}_{E \rightarrow }$'
        elif 'EI_ratio_J_m' in key or 'JmI/JmE' in key:
            matched_labels[key] = r'$\Delta J^{\text{mid}}_{E \rightarrow }/ J^{\text{mid}}_{I \rightarrow }$'
        elif 'J_EE_s' in key:
            matched_labels[key] = r'$\Delta J^{\text{sup}}_{E \rightarrow E}$'
        elif 'J_EI_s' in key:
            matched_labels[key] = r'$\Delta J^{\text{sup}}_{I \rightarrow E}$'
        elif 'J_IE_s' in key:
            matched_labels[key] = r'$\Delta J^{\text{sup}}_{E \rightarrow I}$'
        elif 'J_II_s' in key:
            matched_labels[key] = r'$\Delta J^{\text{sup}}_{I \rightarrow I}$'
        elif 'J_I_s' in key:
            matched_labels[key] = r'$\Delta J^{\text{sup}}_{I \rightarrow }$'
        elif 'J_E_s' in key:
            matched_labels[key] = r'$\Delta J^{\text{sup}}_{E \rightarrow }$'
        elif 'EI_ratio_J_s' in key or 'JsI/JsE' in key:
            matched_labels[key] = r'$\Delta J^{\text{sup}}_{E \rightarrow }/ J^{\text{sup}}_{I \rightarrow }$'
        elif 'EI_ratio_J_ms' in key:
            matched_labels[key] = r'$\Delta J_{E \rightarrow }/ J_{I \rightarrow }$'
        elif 'cE_m' in key:
            matched_labels[key] = r'$\Delta c^{\text{mid}}_{\rightarrow E}$'
        elif 'cI_m' in key:
            matched_labels[key] = r'$\Delta c^{\text{mid}}_{\rightarrow I}$'
        elif 'cE_s' in key:
            matched_labels[key] = r'$\Delta c^{\text{sup}}_{\rightarrow E}$'
        elif 'cI_s' in key:
            matched_labels[key] = r'$\Delta c^{\text{sup}}_{\rightarrow I}$'
        elif 'f_E' in key:
            matched_labels[key] = r'$\Delta f^{\text{mid}\rightarrow \text{sup}}_{E \rightarrow E}$'
        elif 'f_I' in key:
            matched_labels[key] = r'$\Delta f^{\text{mid}\rightarrow \text{sup}}_{E \rightarrow I}$'
        elif 'kappa_EE_pre' in key:
            matched_labels[key] = r'$\Delta \kappa^{\text{pre}}_{E \rightarrow  E}$'
        elif 'kappa_IE_pre' in key:
            matched_labels[key] = r'$\Delta \kappa^{\text{pre}}_{E \rightarrow  I}$'
        elif 'kappa_EE_post' in key:
            matched_labels[key] = r'$\Delta \kappa^{\text{post}}_{E \rightarrow  E}$'
        elif 'kappa_IE_post' in key:
            matched_labels[key] = r'$\Delta \kappa^{\text{post}}_{E \rightarrow  I}$'
        elif 'offset_th' in key: # Psychometric offset threshold for correlation triangles
            matched_labels[key] = r'$\Delta \theta_{\text{offset}}$'
        else:
            matched_labels[key] = key  # Default to the key itself if no match is found
    
    return matched_labels

def plot_param_offset_correlations(folder, excluded_runs=[]):
    """ Plot the correlations between the psychometric offset threshold and the model parameters. """
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
        
    def regplots(param_key, y_key, rel_changes_train, corr_and_p_1, axes_flat, j, x_labels=None, y_labels=None):
        if x_labels is None:
            x_labels = param_key
        line_color1, scatter_color = get_color(corr_and_p_1)
        sns.regplot(x=param_key, y=y_key, data=rel_changes_train, ax=axes_flat[j], ci=95, color='red', 
            line_kws={'color':line_color1}, scatter_kws={'alpha':0.3, 'color':scatter_color})
        # add labels to x-axis and y-axis
        axes_flat[j].set_xlabel(x_labels, fontsize=20)
        if y_labels is not None:
            axes_flat[j].set_ylabel(y_labels, fontsize=20)
        # display corr in the right bottom of the figure
        axes_flat[j].text(0.05, 0.05, f'r= {corr_and_p_1[0]:.2f}', transform=axes_flat[j].transAxes, fontsize=20)

    def save_fig(fig, folder, filename, title=None):
        if title is not None:
            fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        fig.savefig(str(folder) + filename)
        plt.close(fig)

    corr_psychometric_offset_param, corr_staircase_offset_param, corr_loss_param, rel_changes_train = param_offset_correlations(folder, excluded_runs=excluded_runs)
    # Make correlations plots between the psychometric offset and model parameters
    keys_a = ['J_EE_m', 'J_IE_m', 'J_EI_m', 'J_II_m', 'J_EE_s', 'J_IE_s', 'J_EI_s', 'J_II_s']
    keys_b = ['J_E_m', 'J_I_m', 'EI_ratio_J_m', 'J_E_s', 'J_I_s', 'EI_ratio_J_s']
    keys_c = ['f_E', 'f_I', 'cE_m', 'cI_m', 'cE_s', 'cI_s']
    keys_d = ['kappa_EE_pre', 'kappa_IE_pre', 'kappa_EE_post', 'kappa_IE_post']
    fig1a, axes1a = plt.subplots(nrows=2, ncols=4, figsize=(4*5, 2*5)) # raw J params
    fig1b, axes1b = plt.subplots(nrows=2, ncols=3, figsize=(3*5, 2*5)) # combined J params
    fig1c, axes1c = plt.subplots(nrows=3, ncols=2, figsize=(2*5, 3*5)) # f and c params
    fig1d, axes1d = plt.subplots(nrows=2, ncols=2, figsize=(2*5, 2*5)) # kappa params
    # Process each group of parameters
    parameter_groups = [keys_a, keys_b, keys_c, keys_d]
    x_labels = match_keys_to_labels(corr_psychometric_offset_param.keys())
    for i, keys_group in enumerate(parameter_groups):
        if i == 0:
            axes_flat = axes1a.flatten()
        elif i == 1:
            axes_flat = axes1b.flatten()
        elif i == 2:
            axes_flat = axes1c.flatten()
        else:
            axes_flat = axes1d.flatten()
        j = 0
        for param_key_ind in range(len(keys_group)):
            param_key = keys_group[param_key_ind]
            if param_key_ind ==0 and i < 2:
                y_label ='MIDDLE LAYER \n'
            elif param_key_ind == len(keys_group)//2 and i < 2:
                y_label = 'SUPERFICIAL LAYER \n'
            else:
                y_label = None
            # rows of the plot collect types of parameters: 1) mid, 2) sup, 3) f and kappa, 4) J combined
            corr_and_p_1 = corr_psychometric_offset_param[param_key]
            regplots(param_key, 'psychometric_offset', rel_changes_train, corr_and_p_1, axes_flat, j, x_labels[param_key], y_label)
            j += 1
        # Adjust layout and save + close the plot
        save_fig(fig1a, folder, f'/figures/corr_psychometric_Jraw.png', title='Raw J parameters vs psychometric threshold')
        save_fig(fig1b, folder, f'/figures/corr_psychometric_Jcombined.png', title='Combined J parameters vs psychometric threshold')
        save_fig(fig1c, folder, f'/figures/corr_psychometric_f_c.png', title='f and c parameters vs psychometric threshold')
        save_fig(fig1d, folder, f'/figures/corr_psychometric_kappa.png', title='kappa parameters vs psychometric threshold')


def plot_correlations(folder, num_training, num_time_inds=3):
    offset_staircase_pars_corr, offset_psychometric_pars_corr, MVPA_corrs, data = MVPA_param_offset_correlations(folder, num_training, num_time_inds, mesh_for_valid_offset=False)

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

    # Plot the correlation between psychometric_offset_rel_change and the combination of the J_m_E_rel_change, J_m_I_rel_change, J_s_E_rel_change, and J_s_I_rel_change
    for i in range(4):
        # Create lmplot for each pair of variables
        if i in E_indices:
            sns.regplot(x=x_keys_J[i], y='psychometric_offset_rel_change', data=data, ax=axes_flat[i], ci=95, color='red', 
                line_kws={'color':'darkred'}, scatter_kws={'alpha':0.3, 'color':'red'})
        else:
            sns.regplot(x=x_keys_J[i], y='psychometric_offset_rel_change', data=data, ax=axes_flat[i], ci=95, color='blue', 
                line_kws={'color':'darkblue'}, scatter_kws={'alpha':0.3, 'color':'blue'})
        # Calculate the Pearson correlation coefficient and the p-value
        corr = offset_psychometric_pars_corr[i]['corr']
        p_value = offset_psychometric_pars_corr[i]['p_value']
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

    ########## Plot the correlation between offset_psychometric_diff and the combination of the J_m_E_diff, J_m_I_diff, J_s_E_diff, and J_s_I_diff ##########
    data['offset_psychometric_improve'] = -1*data['psychometric_offset_rel_change']
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    axes_flat = axes.flatten()

    for i in range(4):
        # Create lmplot for each pair of variables
        if i in E_indices:
            sns.regplot(x=x_keys_J[i], y='offset_psychometric_improve', data=data, ax=axes_flat[i], ci=95, color='red', 
                line_kws={'color':'darkred'}, scatter_kws={'alpha':0.3, 'color':'red'})
        else:
            sns.regplot(x=x_keys_J[i], y='offset_psychometric_improve', data=data, ax=axes_flat[i], ci=95, color='blue', 
                line_kws={'color':'darkblue'}, scatter_kws={'alpha':0.3, 'color':'blue'})
        axes_flat[i].set(ylabel=None)
        # Calculate the Pearson correlation coefficient and the p-value
        corr = offset_psychometric_pars_corr[i]['corr']
        p_value = offset_psychometric_pars_corr[i]['p_value']
        print('Correlation between offset_psychometric_improve and', x_keys_J[i], 'is', corr, 'with p-value', p_value)
        
        # display corr and p-value in the right bottom of the figure
        axes_flat[i].text(0.05, 0.05, f'r= {corr:.2f}', transform=axes_flat[i].transAxes, fontsize=20)
        axes_format(axes_flat[i], fs_ticks=20)
        # add xlabel and ylabel
        axes_flat[i].set_xlabel(x_labels_J[i], fontsize=20, labelpad=20)
    # Add shared y-label
    fig.text(-0.05, 0.5, 'psychometric threshold improvement (%)', va='center', rotation='vertical', fontsize=20)

    # Adjust layout and save + close the plot
    plt.tight_layout()
    plt.savefig(folder + "/figures/Offset_psychometric_corr_J_IE.png", bbox_inches='tight')
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
        sns.regplot(x=x_labels_fc[i], y='psychometric_offset_rel_change', data=data, ax=axes_flat[axes_indices[i]], ci=95, color=colors[i],
            line_kws={'color':linecolors[i]}, scatter_kws={'alpha':0.3, 'color':colors[i]})
        # Calculate the Pearson correlation coefficient and the p-value
        print('Correlation between psychometric_offset_rel_change and', x_labels_fc[i])

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


################## MVPA related plots ##################

def plot_corr_triangle(data, keys):
    """ In the given axis (ax), plot a correlation triangle between the data columns specified in keys """
    # Select data with keys
    data_keys = data[keys]
    data_keys = data_keys.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Keys and labels
    key_labels = match_keys_to_labels(keys)
    labels = [
    rf'$\Delta$ MVPA (%)' + '\n L: y, R: x', 
    f'{key_labels[keys[1]]} (%)' + '\n x-axis', 
    f'{key_labels[keys[2]]} (%)' + '\n y-axis'
    ]

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
    buffer_y = 0.15
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
    # Exclude runs (columns in data) with NaN or inf values
    data = data.replace([numpy.inf, -numpy.inf], numpy.nan).dropna()
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
    
    return fig

    
def plot_corr_triangles(folder_path, excluded_runs=[]):
    """ Plot the four correlation triangles into a single 2x2 figure """
    
    folder_to_save = os.path.join(folder_path, 'figures')
    
    # Load the MVPA scores
    MVPA_scores = csv_to_numpy(folder_path + '/MVPA_scores.csv')

    # Load the data and exclude the runs that are in the excluded_runs list
    data_rel_changes, _ = rel_change_for_runs(folder_path, num_time_inds=3)
    valid_runs = numpy.setdiff1d(data_rel_changes['run_index'], excluded_runs)
    valid_runs_inds = numpy.where(numpy.isin(data_rel_changes['run_index'], valid_runs))[0]
    data_rel_changes_valid_runs = {key: value[valid_runs_inds.tolist()] for key, value in data_rel_changes.items()}

    # Prepare the four data sets (MVPA dimensions are num_trainings x layer x SGD_ind x ori_ind)
    data_for_corr_triangles = pd.DataFrame({
        'MVPA_mid_55': 100*(MVPA_scores[:, 0, -1, 0] - MVPA_scores[:, 0, -2, 0]) / MVPA_scores[:, 0, -2, 0],
        'MVPA_mid_125': 100*(MVPA_scores[:, 0, -1, 1] - MVPA_scores[:, 0, -2, 1]) / MVPA_scores[:, 0, -2, 1],
        'MVPA_sup_55': 100*(MVPA_scores[:, 1, -1, 0] - MVPA_scores[:, 1, -2, 0]) / MVPA_scores[:, 1, -2, 0],
        'MVPA_sup_125': 100*(MVPA_scores[:, 1, -1, 1] - MVPA_scores[:, 1, -2, 1]) / MVPA_scores[:, 1, -2, 1],
        'JmI/JmE': data_rel_changes_valid_runs['EI_ratio_J_m'],
        'JsI/JsE': data_rel_changes_valid_runs['EI_ratio_J_s'],
        'offset_th': data_rel_changes_valid_runs['psychometric_offset']
    })
    
    # Plot each triangle into its respective subplot
    fig1 = plot_corr_triangle(data_for_corr_triangles, keys=['MVPA_mid_55', 'JmI/JmE', 'offset_th'])
    fig2 = plot_corr_triangle(data_for_corr_triangles, keys=['MVPA_mid_125', 'JmI/JmE', 'offset_th'])
    fig3 = plot_corr_triangle(data_for_corr_triangles, keys=['MVPA_sup_55', 'JsI/JsE', 'offset_th'])
    fig4 = plot_corr_triangle(data_for_corr_triangles, keys=['MVPA_sup_125', 'JsI/JsE', 'offset_th'])
    
    fig1.savefig(f"{folder_to_save}/corr_triangle_mid_55.png", bbox_inches='tight')
    fig2.savefig(f"{folder_to_save}/corr_triangle_mid_125.png", bbox_inches='tight')
    fig3.savefig(f"{folder_to_save}/corr_triangle_sup_55.png", bbox_inches='tight')
    fig4.savefig(f"{folder_to_save}/corr_triangle_sup_125.png", bbox_inches='tight')
    
    # Load the saved images
    img1 = Image.open(f"{folder_to_save}/corr_triangle_mid_55.png")
    img2 = Image.open(f"{folder_to_save}/corr_triangle_mid_125.png")
    img3 = Image.open(f"{folder_to_save}/corr_triangle_sup_55.png")
    img4 = Image.open(f"{folder_to_save}/corr_triangle_sup_125.png")
    
    # Create a new figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Turn off axes for the image plots
    for ax in axes.flat:
        ax.axis('off')
    
    # Plot the images in the respective subplots
    axes[0, 0].imshow(img1)
    axes[0, 1].imshow(img2)
    axes[1, 0].imshow(img3)
    axes[1, 1].imshow(img4)

    # Set titles for the subplots arranged to the left
    axes[0, 0].set_title('mid 55', fontsize=22, loc='left')
    axes[0, 1].set_title('mid 125', fontsize=22, loc='right')
    axes[1, 0].set_title('sup 55', fontsize=22, loc='left')
    axes[1, 1].set_title('sup 125', fontsize=22, loc='right')
    
    # Save the combined figure
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(os.path.join(folder_to_save, 'combined_corr_triangles.png'))
    plt.close()

    # close figures
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    img1.close()
    img2.close()
    img3.close()
    img4.close()


def plot_MVPA_or_Mahal_scores(folder_path, file_name):
    """ Plot the MVPA scores or Mahalanobis distances for the two layers and two orientations. 
    scores dimensions are runs x layers x SGD_inds x ori_inds """
    
    #Load the scores
    if file_name.endswith('MVPA_scores'):
        scores = csv_to_numpy(folder_path + '/MVPA_scores.csv')
    else:
        scores = csv_to_numpy(folder_path + '/Mahal_scores.csv') 
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # iterate over the two layers
    layer_label = ['mid', 'sup']
    num_runs = scores.shape[0]
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
    plt.savefig(os.path.join(folder_path ,'figures', file_name+'.png'))
    plt.close()


def plot_MVPA_or_Mahal_scores_match_Kes_fig(folder_path, file_name):
    ''' Plot the MVPA scores or Mahalanobis distances for the two layers and two orientations in the format of Ke's 2024 paper.'''

    # Load the scores
    if file_name.endswith('MVPA_scores'):
        scores = csv_to_numpy(folder_path + '/MVPA_scores.csv')
    else:
        scores = csv_to_numpy(folder_path + '/Mahal_scores.csv') 

    # Barplot of the JmsI/JmsE before and after training
    color_pretest = '#F3929A'
    color_posttest = '#70BFD9'
    colors_bar = [color_pretest, color_posttest]
    darker_colors = ['#91575C', '#385F6C']
    
    # Plotting the bars
    num_runs = scores.shape[0]
    mean_scores = numpy.mean(scores*100, axis=0)
    std_scores = numpy.std(scores*100, axis=0)/numpy.sqrt(num_runs)
    sub_titles = ['Trained orientation', 'Untrained orientation']
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for ori in range(2):
        bar_width = 0.4  # Width of the bars
        
        # Plot the bar chart with error bars, applying colors individually
        bars = []
        for layer in range(2):
            layer_ind_flip = 1 if layer == 0 else 0 # Flip the layer index for the bar plot because they have superficial and middle reversed in the paper
            for std_ind in range(2):
                bars.append(ax[ori].bar(layer*2.5+std_ind, mean_scores[layer_ind_flip, std_ind+1, ori], bar_width, yerr=std_scores[layer_ind_flip, std_ind+1, ori], capsize=5, 
                                color=colors_bar[std_ind], ecolor=colors_bar[std_ind], error_kw=dict(ecolor=darker_colors[std_ind], alpha=0.9, lw=2, capsize=5, capthick=2)))
        ax[ori].set_xticks([0.5, 3])
        ax[ori].set_xticklabels(['Superficial', 'Middle'], fontsize=20)
        ax[ori].set_ylabel('MVPA accuracy (%)', fontsize=20)
        ax[ori].tick_params(axis='y', which='both', labelsize=18)
        ax[ori].yaxis.set_tick_params(width=2, length=8)  # Customize tick size
        plt.tight_layout()
        ax[ori].set_xlim(-0.5, 4)
        ax[ori].set_ylim(50, 90)
        ax[ori].set_title(sub_titles[ori], fontsize=20)

    plt.savefig(os.path.join(folder_path, 'figures', "MVPA_match_paper_fig.png"))
    plt.close()