import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
import jax.numpy as np
import numpy
import sys

from model import evaluate_model_response
from util import sep_exponentiate
from SSN_classes import SSN_mid, SSN_sup
from util_gabor import BW_image_jit
#from compare_tuning_curves_standalone import two_layer_model, constant_to_vec

########### Plotting functions ##############
def boxplots_from_csvs(directory, save_directory, plot_filename = None):
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
        full_path = save_directory + '/' + plot_filename + ".png"
        fig.savefig(full_path)

    plt.close()


def plot_tuning_curves(folder_path,tc_cells,num_runs,folder_to_save,train_only_str='', seed=0):
    numpy.random.seed(seed)
    num_mid_cells = 648
    num_sup_cells = 164
    num_runs_plotted = min(5,num_runs)
    tc_post_pretrain = folder_path + '/tc_postpre_0.csv'
    pretrain_ison = os.path.exists(tc_post_pretrain)
    for j in range(num_runs_plotted):
        if pretrain_ison:
            tc_pre_pretrain = os.path.join(folder_path,f'tc_' + train_only_str + f'prepre_{j}.csv')
            df_tc_pre_pretrain = pd.read_csv(tc_pre_pretrain, header=0)
            tc_post_pretrain =os.path.join(folder_path,f'tc_' + train_only_str + f'postpre_{j}.csv')
            df_tc_post_pretrain = pd.read_csv(tc_post_pretrain, header=0)
        else:
            tc_pre_train = folder_path + f'/tc_train_only_pre_{j}.csv'
            df_tc_pre_pretrain = pd.read_csv(tc_pre_train, header=0)
        tc_post_train =os.path.join(folder_path,f'tc_' + train_only_str + f'post_{j}.csv')
        df_tc_post_train = pd.read_csv(tc_post_train, header=0)

        # Select num_rnd_cells randomly selected cells to plot from both middle and superficial layer cells
        if j==0:
            if isinstance(tc_cells,int):
                num_rnd_cells=tc_cells
                selected_mid_col_inds = numpy.random.randint(0, num_mid_cells, size=int(num_rnd_cells/2), replace=False)
                selected_sup_col_inds = numpy.random.randint(0, num_sup_cells, size=num_rnd_cells-int(num_rnd_cells/2), replace=False)
            else:
                num_rnd_cells=len(tc_cells)
                selected_mid_col_inds = numpy.array(tc_cells[:int(len(tc_cells)/2)])-1
                selected_sup_col_inds = numpy.array(tc_cells[int(len(tc_cells)/2):])-1
            fig, axes = plt.subplots(nrows=num_runs_plotted, ncols=num_rnd_cells, figsize=(5*num_rnd_cells, 5*num_runs_plotted))
        num_oris = df_tc_pre_pretrain.shape[0]
        # Plot tuning curves
        for i in range(int(num_rnd_cells/2)):
            if num_runs_plotted==1:
                ax1=axes[i]
                ax2=axes[int(num_rnd_cells/2)+i]
            else:
                ax1=axes[j,i]
                ax2=axes[j,int(num_rnd_cells/2)+i]
            ax1.plot(range(num_oris), df_tc_pre_pretrain.iloc[:,selected_mid_col_inds[i]], label='pre-pretraining',linewidth=2)
            ax2.plot(range(num_oris), df_tc_pre_pretrain.iloc[:,selected_sup_col_inds[i]], label='pre-pretraining',linewidth=2)
            ax1.plot(range(num_oris), df_tc_post_train.iloc[:,selected_mid_col_inds[i]], label='post-training',linewidth=2)
            ax2.plot(range(num_oris), df_tc_post_train.iloc[:,selected_sup_col_inds[i]], label='post-training',linewidth=2)
            if pretrain_ison:
                ax1.plot(range(num_oris), df_tc_post_pretrain.iloc[:,selected_mid_col_inds[i]], label='post-pretraining',linewidth=2)
                ax2.plot(range(num_oris), df_tc_post_pretrain.iloc[:,selected_sup_col_inds[i]], label='post-pretraining',linewidth=2)
            
            ax1.set_title(f'Middle layer cell {i}', fontsize=20)
            ax2.set_title(f'Superficial layer, cell {i}', fontsize=20)
    ax2.legend(loc='upper left', fontsize=20)
    
    # Save plot
    if folder_to_save is not None:
        fig.savefig(os.path.join(folder_to_save,'tc_' + train_only_str + 'fig.png'))
    else:
        fig.savefig(os.path.join(folder_path,'tc_' + train_only_str + 'fig.png'))
    plt.close()


def plot_pre_post_scatter(ax, x_axis, y_axis, orientations, indices_to_plot, N_runs, title, colors):
    '''
    
    '''
    
    for run_ind in range(N_runs):
        bin_indices = numpy.digitize(numpy.abs(orientations[run_ind,:]), [4, 12, 20, 28, 36, 44, 50, 180])
    
        # Iterate over bins rather than individual points
        for bin_idx, color in enumerate(colors, start=1):  # Adjust as needed
            # Find indices within this bin
            in_bin = numpy.where(bin_indices == bin_idx)[0]
            # Find intersection with indices_to_plot
            plot_indices = numpy.intersect1d(in_bin, indices_to_plot)
            
            if len(plot_indices) > 0:
                ax.scatter(x_axis[run_ind,plot_indices], y_axis[run_ind,plot_indices], color=color, s=20, alpha=0.7)
    
    # Plot x = y line
    xpoints = ypoints = ax.get_xlim()
    ax.plot(xpoints, ypoints, linestyle='--', color='gold', linewidth=1)
    ax.set_xlabel('Pre training')
    ax.set_ylabel('Post training')
    ax.set_title(title)
  

def plot_tc_features(results_dir, N_runs, ori_list, train_only_str=''):

    # Initialize dictionaries to store the data arrays
    if train_only_str=='':
        data = {
            'norm_slope_prepre': [],
            'norm_slope_postpre': [],
            'norm_slope_post': [],
            'fwhm_prepre': [],
            'fwhm_postpre': [],
            'fwhm_post': [],
            'orientations_prepre': [],
            'orientations_postpre': [],
            'orientations_post': [],
        }
    else:
            data = {
            'norm_slope_train_only_pre': [],
            'norm_slope_train_only_post': [],
            'fwhm_train_only_pre': [],
            'fwhm_train_only_post': [],
            'orientations_train_only_pre': [],
            'orientations_train_only_post': []
        }

    for i in range(N_runs):
        # File names associated with each data type
        if train_only_str=='':
            file_names = {
                'prepre': results_dir + f'/tc_prepre_{i}.csv',
                'postpre':  results_dir + f'/tc_postpre_{i}.csv',
                'post': results_dir + f'/tc_post_{i}.csv'
            }
        else:
            file_names = {
                'train_only_pre': results_dir + f'/tc_train_only_pre_{i}.csv',
                'train_only_post': results_dir + f'/tc_train_only_post_{i}.csv'
            }

        # Loop through each file name to process and store data
        for key, file_name in file_names.items():
            # Load data from file
            slope, fwhm, orientations = tc_features(file_name, ori_list=ori_list, expand_dims=True)
            
            # If first iteration, initialize; else, concatenate
            if  i==0:
                data[f'norm_slope_{key}'] = slope
                data[f'fwhm_{key}'] = fwhm
                data[f'orientations_{key}'] = orientations
            else:
                data[f'norm_slope_{key}'] = numpy.concatenate((data[f'norm_slope_{key}'], slope), axis=0)
                data[f'fwhm_{key}'] = numpy.concatenate((data[f'fwhm_{key}'], fwhm), axis=0)
                data[f'orientations_{key}'] = numpy.concatenate((data[f'orientations_{key}'], orientations), axis=0)


    # Plots about changes before vs after training and pretraining and training only (per layer and per centered or all)
    E_sup = 648+numpy.linspace(0, 80, 81).astype(int) 
    I_sup = 648+numpy.linspace(81, 161, 81).astype(int) 
    E_mid = numpy.linspace(0, 647, 648).round().reshape(8, 81, -1)[0:9:2].ravel().astype(int) 
    I_mid = numpy.linspace(0, 647, 648).round().reshape(8, 81, -1)[1:9:2].ravel().astype(int) 
    labels = ['E_sup','I_sup','E_mid','I_mid']
    indices = [E_sup, I_sup, E_mid, I_mid]

    #E_sup_centre = 648+numpy.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)
    #I_sup_centre = (E_sup_centre+81).astype(int)
    #E_mid_centre = numpy.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)
    #I_mid_centre = (E_mid_centre+81).astype(int)
    
    # Create legend
    patches = []
    cmap = plt.get_cmap('rainbow')
    colors = numpy.flip(cmap(numpy.linspace(0,1, 8)), axis = 0)
    bins = ['0-4', '4-12', '12-20', '20-28', '28-36', '36-44', '44-50', '+50']
    for j in range(0,len(colors)):
        patches.append(mpatches.Patch(color=colors[j], label=bins[j]))

    # Plot slope
    if train_only_str=='':
        fig, axs = plt.subplots(4, 4, figsize=(15, 20)) 
        for j in range(len(indices)):
            title = 'Slope pretraining ' + labels[j]
            plot_pre_post_scatter(axs[j,0], data['norm_slope_prepre'] , data['norm_slope_postpre'] ,  data['orientations_prepre'],  indices[j],N_runs, title = title,colors=colors)

            title = 'Slope training, ' + labels[j]
            plot_pre_post_scatter(axs[j,1], data['norm_slope_postpre'] , data['norm_slope_post'] ,  data['orientations_postpre'], indices[j],N_runs, title = title,colors=colors)

            title = 'Fwhm pretraining ' + labels[j]
            plot_pre_post_scatter(axs[j,2],  data['fwhm_prepre'] ,  data['fwhm_postpre'] ,  data['orientations_prepre'], indices[j], N_runs, title = title,colors=colors)

            title = 'Fwhm training, ' + labels[j] 
            plot_pre_post_scatter(axs[j,3], data['fwhm_postpre'] , data['fwhm_post'] ,data['orientations_postpre'], indices[j], N_runs,title = title,colors=colors)
        axs[j,3].legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1), title='Pref ori - train ori')
    else:
        fig, axs = plt.subplots(4, 2, figsize=(10, 20)) 
        for j in range(len(indices)):    
            title = 'Slope training_only ' + labels[j]
            plot_pre_post_scatter(axs[j,0],  data['norm_slope_train_only_pre'] , data['norm_slope_train_only_post'] , data['orientations_train_only_pre'], indices[j], N_runs, title = title,colors=colors)

            title = 'Fwhm training_only ' + labels[j] 
            plot_pre_post_scatter(axs[j,1],  data['fwhm_train_only_pre'] , data['fwhm_train_only_post'] ,data['orientations_train_only_pre'], indices[j], N_runs, title = title,colors=colors)
        axs[j,1].legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1), title='Pref ori - train ori')
    
    plt.tight_layout()
    fig.savefig(os.path.dirname(results_dir) + "/figures/tc_features" + train_only_str +".png")


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
    rel_changes = calculate_relative_change(df) # 0 is pretraining and 1 is training in the second dimensions
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
        if i_train_pretrain==0:
            axes[row_i,1].set_title('Rel changes in J before and after pretraining', fontsize=20)
            axes[row_i,2].set_title('Other rel changes before and after pretraining', fontsize=20)
        else:
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
    axes[0,2].plot(range(N), df['J_m_EE'], label='J_m_EE', linestyle='--', c='tab:red',linewidth=3)
    axes[0,2].plot(range(N), df['J_m_IE'], label='J_m_IE', linestyle='--', c='tab:orange',linewidth=3)
    axes[0,2].plot(range(N), df['J_m_II'], label='J_m_II', linestyle='--', c='tab:blue',linewidth=3)
    axes[0,2].plot(range(N), df['J_m_EI'], label='J_m_EI', linestyle='--', c='tab:green',linewidth=3)
    
    axes[0,2].plot(range(N), df['J_s_EE'], label='J_s_EE', c='tab:red',linewidth=3)
    axes[0,2].plot(range(N), df['J_s_IE'], label='J_s_IE', c='tab:orange',linewidth=3)
    axes[0,2].plot(range(N), df['J_s_II'], label='J_s_II', c='tab:blue',linewidth=3)
    axes[0,2].plot(range(N), df['J_s_EI'], label='J_s_EI', c='tab:green',linewidth=3)
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
    axes[1,2].plot(range(N), df['f_E'], label='f_E', linestyle='--',c='tab:red',linewidth=3)
    axes[1,2].plot(range(N), df['f_I'], label='f_I', linestyle='--',c='tab:blue',linewidth=3)
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
    plt.close()


def plot_results_from_csvs(folder_path, num_runs=3, num_rnd_cells=5, folder_to_save=None):
    # Add folder_path to path
    if folder_path not in sys.path:
        sys.path.append(folder_path)

    # Plot loss, accuracy and trained parameters
    for j in range(num_runs):
        results_filename = os.path.join(folder_path,f'results_{j}.csv')
        if folder_to_save is not None:
            results_fig_filename = os.path.join(folder_to_save,f'resultsfig_{j}.png')
        else:
            results_fig_filename = os.path.join(folder_path,f'resultsfig_{j}.png')
        if not os.path.exists(results_filename):
            results_filename = os.path.join(folder_path,f'results_train_only{j}.csv')
            if folder_to_save is not None:
                results_fig_filename = os.path.join(folder_to_save,f'resultsfig_train_only{j}.png')
            else:
                results_fig_filename = os.path.join(folder_path,f'resultsfig_train_only{j}.png')
        if not os.path.exists(results_fig_filename):
            plot_results_from_csv(results_filename,results_fig_filename)

############## Analysis functions ##########
            
            
def calculate_relative_change(df):
    # Calculate relative changes in Jm and Js
    J_m_EE = df['J_m_EE']
    J_m_IE = df['J_m_IE']
    J_m_EI = [np.abs(df['J_m_EI'][i]) for i in range(len(df['J_m_EI']))]
    J_m_II = [np.abs(df['J_m_II'][i]) for i in range(len(df['J_m_II']))]
    J_s_EE = df['J_s_EE']
    J_s_IE = df['J_s_IE']
    J_s_EI = [np.abs(df['J_s_EI'][i]) for i in range(len(df['J_s_EI']))]
    J_s_II = [np.abs(df['J_s_II'][i]) for i in range(len(df['J_s_II']))]
    c_E = df['c_E']
    c_I = df['c_I']
    f_E = df['f_E']
    f_I = [np.abs(df['f_I'][i]) for i in range(len(df['f_I']))]
    relative_changes = numpy.zeros((18,2))

    # Calculate relative changes in parameters and other metrics before and after training
    train_start_ind = df.index[df['stage'] == 1][0]
    train_end_ind = len(J_m_EE)-1
    relative_changes[0,1] =(J_m_EE[train_end_ind] - J_m_EE[train_start_ind]) / J_m_EE[train_start_ind]
    relative_changes[1,1] =(J_m_IE[train_end_ind] - J_m_IE[train_start_ind]) / J_m_IE[train_start_ind]
    relative_changes[2,1] =(J_m_EI[train_end_ind] - J_m_EI[train_start_ind]) / J_m_EI[train_start_ind]
    relative_changes[3,1] =(J_m_II[train_end_ind] - J_m_II[train_start_ind]) / J_m_II[train_start_ind]
    relative_changes[4,1] =(J_s_EE[train_end_ind] - J_s_EE[train_start_ind]) / J_s_EE[train_start_ind]
    relative_changes[5,1] =(J_s_IE[train_end_ind] - J_s_IE[train_start_ind]) / J_s_IE[train_start_ind]
    relative_changes[6,1] =(J_s_EI[train_end_ind] - J_s_EI[train_start_ind]) / J_s_EI[train_start_ind]
    relative_changes[7,1] =(J_s_II[train_end_ind] - J_s_II[train_start_ind]) / J_s_II[train_start_ind]
    relative_changes[8,1] = (c_E[train_end_ind] - c_E[train_start_ind]) / c_E[train_start_ind]
    relative_changes[9,1] = (c_I[train_end_ind] - c_I[train_start_ind]) / c_I[train_start_ind]
    relative_changes[10,1] = (f_E[train_end_ind] - f_E[train_start_ind]) / f_E[train_start_ind]
    relative_changes[11,1] = (f_I[train_end_ind] - f_I[train_start_ind]) / f_I[train_start_ind]

    relative_changes[12,1] = (df['acc'].iloc[train_end_ind] - df['acc'].iloc[train_start_ind]) / df['acc'].iloc[train_start_ind] # accuracy
    for column in df.columns:
        if 'offset' in column:
            relative_changes[13,1] = (df[column].iloc[train_end_ind] - df[column].iloc[train_start_ind]) / df[column].iloc[train_start_ind] # offset threshold
    relative_changes[14,1] = (df['maxr_E_mid'].iloc[train_end_ind] - df['maxr_E_mid'].iloc[train_start_ind]) / df['maxr_E_mid'].iloc[train_start_ind] # r_mid
    relative_changes[15,1] = (df['maxr_I_mid'].iloc[train_end_ind] - df['maxr_I_mid'].iloc[train_start_ind]) / df['maxr_I_mid'].iloc[train_start_ind] # r_mid
    relative_changes[16,1] = (df['maxr_E_sup'].iloc[train_end_ind] - df['maxr_E_sup'].iloc[train_start_ind]) / df['maxr_E_sup'].iloc[train_start_ind] # r_mid
    relative_changes[17,1] = (df['maxr_I_sup'].iloc[train_end_ind] - df['maxr_I_sup'].iloc[train_start_ind]) / df['maxr_I_sup'].iloc[train_start_ind] # r_sup

    # Calculate relative changes in parameters and other metrics before and after pretraining
    pretraining_on = float(np.sum(df['stage'].ravel() == 0))>0
    if pretraining_on:
        pretrain_start_ind = df.index[df['stage'] == 0][0]
        relative_changes[0,0] =(J_m_EE[train_start_ind] - J_m_EE[pretrain_start_ind]) / J_m_EE[pretrain_start_ind]
        relative_changes[1,0] =(J_m_IE[train_start_ind] - J_m_IE[pretrain_start_ind]) / J_m_IE[pretrain_start_ind]
        relative_changes[2,0] =(J_m_EI[train_start_ind] - J_m_EI[pretrain_start_ind]) / J_m_EI[pretrain_start_ind]
        relative_changes[3,0] =(J_m_II[train_start_ind] - J_m_II[pretrain_start_ind]) / J_m_II[pretrain_start_ind]
        relative_changes[4,0] =(J_s_EE[train_start_ind] - J_s_EE[pretrain_start_ind]) / J_s_EE[pretrain_start_ind]
        relative_changes[5,0] =(J_s_IE[train_start_ind] - J_s_IE[pretrain_start_ind]) / J_s_IE[pretrain_start_ind]
        relative_changes[6,0] =(J_s_EI[train_start_ind] - J_s_EI[pretrain_start_ind]) / J_s_EI[pretrain_start_ind]
        relative_changes[7,0] =(J_s_II[train_start_ind] - J_s_II[pretrain_start_ind]) / J_s_II[pretrain_start_ind]
        relative_changes[8,0] = (c_E[train_start_ind] - c_E[pretrain_start_ind]) / c_E[pretrain_start_ind]
        relative_changes[9,0] = (c_I[train_start_ind] - c_I[pretrain_start_ind]) / c_I[pretrain_start_ind]
        relative_changes[10,0] = (f_E[train_start_ind] - f_E[pretrain_start_ind]) / f_E[pretrain_start_ind]
        relative_changes[11,0] = (f_I[train_start_ind] - f_I[pretrain_start_ind]) / f_I[pretrain_start_ind]
        
        relative_changes[12,0] = (df['acc'].iloc[train_end_ind] - df['acc'].iloc[train_start_ind]) / df['acc'].iloc[train_start_ind] # accuracy
        for column in df.columns:
            if 'offset' in column:
                relative_changes[13,1] = (df[column].iloc[train_start_ind] - df[column].iloc[pretrain_start_ind]) / df[column].iloc[pretrain_start_ind] # offset threshold
        relative_changes[14,0] = (df['maxr_E_mid'].iloc[train_start_ind] - df['maxr_E_mid'].iloc[pretrain_start_ind]) / df['maxr_E_mid'].iloc[pretrain_start_ind] # r_mid
        relative_changes[15,0] = (df['maxr_I_mid'].iloc[train_start_ind] - df['maxr_I_mid'].iloc[pretrain_start_ind]) / df['maxr_I_mid'].iloc[pretrain_start_ind] # r_mid
        relative_changes[16,0] = (df['maxr_E_sup'].iloc[train_start_ind] - df['maxr_E_sup'].iloc[pretrain_start_ind]) / df['maxr_E_sup'].iloc[pretrain_start_ind] # r_mid
        relative_changes[17,0] = (df['maxr_I_sup'].iloc[train_start_ind] - df['maxr_I_sup'].iloc[pretrain_start_ind]) / df['maxr_I_sup'].iloc[pretrain_start_ind] # r_sup
    
    return relative_changes


def tuning_curve(untrained_pars, trained_pars, tuning_curves_filename=None, ori_vec=np.arange(0,180,6)):
    '''
    Calculate responses of middle and superficial layers to different orientations.
    '''
    ref_ori_saved = float(untrained_pars.stimuli_pars.ref_ori)
    for key in list(trained_pars.keys()):  # Use list to make a copy of keys to avoid RuntimeError
        # Check if key starts with 'log'
        if key.startswith('log'):
            # Create a new key by removing 'log' prefix
            new_key = key[4:]
            # Exponentiate the values and assign to the new key
            if numpy.isscalar(trained_pars[key]):
                if key.startswith('log_J') and key.endswith('I'):
                    trained_pars[new_key] = -numpy.exp(trained_pars[key])
                else:
                    trained_pars[new_key] = numpy.exp(trained_pars[key])
            else:
                trained_pars[new_key] = sep_exponentiate(trained_pars[key])
    
    ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=trained_pars['J_2x2_m'])
    
    N_ori = len(ori_vec)
    new_rows = []
    x = untrained_pars.BW_image_jax_inp[5]
    y = untrained_pars.BW_image_jax_inp[6]
    alpha_channel = untrained_pars.BW_image_jax_inp[7]
    mask = untrained_pars.BW_image_jax_inp[8]
    background = untrained_pars.BW_image_jax_inp[9]
    
    train_data = BW_image_jit(untrained_pars.BW_image_jax_inp[0:5], x, y, alpha_channel, mask, background, ori_vec, np.zeros(N_ori))
    for i in range(N_ori):
        ssn_sup=SSN_sup(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=trained_pars['J_2x2_s'], p_local=untrained_pars.ssn_layer_pars.p_local_s, oris=untrained_pars.oris, s_2x2=untrained_pars.ssn_layer_pars.s_2x2_s, sigma_oris = untrained_pars.ssn_layer_pars.sigma_oris, ori_dist = untrained_pars.ori_dist, train_ori = untrained_pars.stimuli_pars.ref_ori)
        _, _, [_,_], [_,_], [_,_,_,_], [r_mid_i, r_sup_i] = evaluate_model_response(ssn_mid, ssn_sup, train_data[i,:], untrained_pars.conv_pars, trained_pars['c_E'], trained_pars['c_I'], trained_pars['f_E'], trained_pars['f_I'], untrained_pars.gabor_filters)
        # testing tuning curve differece ***
        #constant_vector_mid = constant_to_vec(c_E = trained_pars['c_E'], c_I = trained_pars['c_I'], ssn= ssn_mid)
        #constant_vector_sup = constant_to_vec(c_E = trained_pars['c_E'], c_I = trained_pars['c_I'], ssn = ssn_sup, sup=True)
        #_, _, _, _, [fp_mid, fp_sup] = two_layer_model(ssn_mid, ssn_sup, train_data[i,:], untrained_pars.conv_pars, constant_vector_mid, constant_vector_sup,trained_pars['f_E'], trained_pars['f_I'])
        if i==0:
            responses_mid = numpy.zeros((N_ori,len(r_mid_i)))
            responses_sup = numpy.zeros((N_ori,len(r_sup_i)))
        responses_mid[i,:] = r_mid_i
        responses_sup[i,:] = r_sup_i
    
        # Save responses into csv file
        if tuning_curves_filename is not None:
 
            # Concatenate the new data as additional rows
            new_row = numpy.concatenate((r_mid_i, r_sup_i), axis=0)
            new_rows.append(new_row)

    if tuning_curves_filename is not None:
        new_rows_df = pd.DataFrame(new_rows)
        if os.path.exists(tuning_curves_filename):
            # Read existing data and concatenate new data
            existing_df = pd.read_csv(tuning_curves_filename)
            df = pd.concat([existing_df, new_rows_df], axis=0)
        else:
            # If CSV does not exist, use new data as the DataFrame
            df = new_rows_df

        # Write the DataFrame to CSV file
        df.to_csv(tuning_curves_filename, index=False)

    untrained_pars.stimuli_pars.ref_ori = ref_ori_saved

    return responses_sup, responses_mid


def tc_slope(tuning_curve, x_axis, x1, x2, normalised=False):
    """
    Calculates slope of normalized tuning_curve between points x1 and x2. tuning_curve is given at x_axis points.
    """
    #Remove baseline if normalising
    if normalised == True:
        tuning_curve = (tuning_curve - tuning_curve.min())/tuning_curve.max()
    
    #Find indices corresponding to desired x values
    idx_1 = (np.abs(x_axis - x1)).argmin()
    idx_2 = (np.abs(x_axis - x2)).argmin()
    
    grad =(np.abs(tuning_curve[idx_2] - tuning_curve[idx_1]))/(x2-x1)
    
    return grad


def full_width_half_max(vector, d_theta):
    
    #Remove baseline
    vector = vector-vector.min()
    half_height = vector.max()/2
    points_above = len(vector[vector>half_height])

    distance = d_theta * points_above
    
    return distance


def tc_features(file_name, ori_list=numpy.arange(0,180,6), expand_dims=False):
    
    # Tuning curve of given cell indices
    tuning_curve = numpy.array(pd.read_csv(file_name))
    num_cells = tuning_curve.shape[1]
    
    # Find preferred orientation and center it at 55
    pref_ori = ori_list[np.argmax(tuning_curve, axis = 0)]
    norm_pref_ori = pref_ori -55

    # Full width half height
    full_width_half_max_vec = numpy.zeros(num_cells) 
    d_theta = ori_list[1]-ori_list[0]
    for i in range(0, num_cells):
        full_width_half_max_vec[i] = full_width_half_max(tuning_curve[:,i], d_theta = d_theta)

    # Norm slope
    avg_slope_vec =numpy.zeros(num_cells) 
    for i in range(num_cells):
        avg_slope_vec[i] = tc_slope(tuning_curve[:, i], x_axis = ori_list, x1 = 52, x2 = 58, normalised =True)
    if expand_dims:
        avg_slope_vec = numpy.expand_dims(avg_slope_vec, axis=0)
        full_width_half_max_vec = numpy.expand_dims(full_width_half_max_vec, axis=0)
        norm_pref_ori = numpy.expand_dims(norm_pref_ori, axis=0)

    return avg_slope_vec, full_width_half_max_vec, norm_pref_ori

