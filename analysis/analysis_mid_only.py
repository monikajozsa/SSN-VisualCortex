##### This script 

import os
import pandas as pd
import jax.numpy as np
import statsmodels.api as sm
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from util import sep_exponentiate
from training.SSN_classes import SSN_mid
from training.model import vmap_evaluate_model_response_mid
from training.util_gabor import BW_image_jit
from parameters import conv_pars
from analysis.analysis_functions import SGD_step_indices, tc_features
from analysis.visualization import plot_pre_post_scatter

def load_parameters_mid_only(folder_path,file_ind, readout_grid_size=5, iloc_ind=-1, trained_pars_keys=['log_J_2x2_m', 'log_J_2x2_s', 'c_E', 'c_I', 'log_f_E', 'log_f_I']):

    # Get the last row of the given csv file
    df = pd.read_csv(os.path.join(folder_path,f"results_{file_ind}.csv"))
    selected_row = df.iloc[int(iloc_ind)]

    # Extract stage 1 parameters from df
    w_sig_keys = [f'w_sig_{i}' for i in range(1, readout_grid_size*readout_grid_size+1)] 
    w_sig_values = selected_row[w_sig_keys].values
    pars_stage1 = dict(w_sig=w_sig_values, b_sig=selected_row['b_sig'])

    # Extract stage 2 parameters from df
    log_J_m_keys = ['log_J_m_EE','log_J_m_EI','log_J_m_IE','log_J_m_II'] 
    log_J_m_values = selected_row[log_J_m_keys].values.reshape(2, 2)
    # get Js values from initial parameters file
    init_pars=pd.read_csv(os.path.join(folder_path,'initial_parameters.csv'))
    selected_row_init_pars = init_pars.iloc[int(file_ind)]
    
    # Create a dictionary with the trained parameters
    pars_stage2 = dict(
        log_J_2x2_m = log_J_m_values
    )
    if 'c_E' in trained_pars_keys:
        if 'c_E' in selected_row:
            pars_stage2['c_E'] = selected_row['c_E']
            pars_stage2['c_I'] = selected_row['c_I']
        else:
            pars_stage2['c_E'] = selected_row_init_pars['c_E']
            pars_stage2['c_I'] = selected_row_init_pars['c_I']
        
    offsets  = df['offset'].dropna().reset_index(drop=True)
    offset_last = offsets[len(offsets)-1]

    return pars_stage1, pars_stage2, offset_last

def tuning_curve_mid_only(untrained_pars, trained_pars, tuning_curves_filename=None, ori_vec=np.arange(0,180,6)):
    '''
    Calculate responses of middle and superficial layers to different orientations.
    '''
    ref_ori_saved = float(untrained_pars.stimuli_pars.ref_ori)
    if 'log_J_2x2_m' in trained_pars:
        J_2x2_m = sep_exponentiate(trained_pars['log_J_2x2_m'])
    if 'J_2x2_m' in trained_pars:
        J_2x2_m = trained_pars['J_2x2_m']
    if 'c_E' in trained_pars:
        c_E = trained_pars['c_E']
        c_I = trained_pars['c_I']
    else:
        c_E = untrained_pars.ssn_pars.c_E
        c_I = untrained_pars.ssn_pars.c_I        
    
    ssn_mid=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_m)
    
    num_ori = len(ori_vec)
    new_rows = []
    x = untrained_pars.BW_image_jax_inp[5]
    y = untrained_pars.BW_image_jax_inp[6]
    alpha_channel = untrained_pars.BW_image_jax_inp[7]
    mask = untrained_pars.BW_image_jax_inp[8]
    background = untrained_pars.BW_image_jax_inp[9]
    
    train_data = BW_image_jit(untrained_pars.BW_image_jax_inp[0:5], x, y, alpha_channel, mask, background, ori_vec, np.zeros(num_ori))
    _, _, _, _, _, responses_mid = vmap_evaluate_model_response_mid(ssn_mid, train_data, conv_pars, c_E, c_I, untrained_pars.gabor_filters)
    
    # Save responses into csv file
    new_rows=responses_mid
    write_csv_flag = True
    if tuning_curves_filename is not None:
        new_rows_df = pd.DataFrame(new_rows)
        if os.path.exists(tuning_curves_filename):
            # Read existing data and concatenate new data
            existing_df = pd.read_csv(tuning_curves_filename)
            # check if the number of rows in the csv file is less than num_ori
            if len(existing_df) >= num_ori:
                write_csv_flag = False
            else:
                df = pd.concat([existing_df, new_rows_df], axis=0)
        else:
            # If CSV does not exist, use new data as the DataFrame
            df = new_rows_df

        # Write the DataFrame to CSV file
        if write_csv_flag:
            df.to_csv(tuning_curves_filename, index=False)

    untrained_pars.stimuli_pars.ref_ori = ref_ori_saved

    return responses_mid

def plot_results_from_csvs_mid_only(folder_path, num_runs=3, num_rnd_cells=5, folder_to_save=None, starting_run=0):
    # Add folder_path to path
    if folder_path not in sys.path:
        sys.path.append(folder_path)

    # Plot loss, accuracy and trained parameters
    for j in range(starting_run,num_runs):
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
            plot_results_from_csv_mid_only(results_filename,results_fig_filename)

def plot_results_from_csv_mid_only(
    results_filename,
    fig_filename=None):
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(results_filename, header=0)
    N=len(df[df.columns[0]])
    # Create a subplot with 4 rows and 3 columns
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
    categories_params = ['Jm_EE', 'Jm_IE', 'Jm_EI', 'Jm_II']
    categories_metrics = [ 'c_E', 'c_I', 'acc', 'offset', 'rm_E', 'rm_I']
    rel_par_changes,_ = rel_changes_mid_only(df) # 0 is pretraining and 1 is training in the second dimensions
    for i_train_pretrain in range(2):
        values_params = 100 * rel_par_changes[0:4,i_train_pretrain]
        values_metrics = 100 * rel_par_changes[4:10,i_train_pretrain]

        # Choosing colors for each bar
        colors_params = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
        colors_metrics = [ 'tab:red', 'tab:blue', 'tab:green', 'tab:brown', 'tab:red', 'tab:blue']

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
    axes[0,2].legend(loc="upper left", fontsize=20)
    axes[0,2].set_title('J in middle layer', fontsize=20)
    axes[0,2].set_xlabel('SGD steps', fontsize=20)

    # Plot maximum rates
    colors = ["tab:blue", "tab:red"]
    axes[0,1].plot(range(N), df['maxr_E_mid'], label='maxr_E_mid', c=colors[1], linestyle=':')
    axes[0,1].plot(range(N), df['maxr_I_mid'], label='maxr_I_mid', c=colors[0], linestyle=':')
    axes[0,1].legend(loc="upper left", fontsize=20)
    axes[0,1].set_title('Maximum rates', fontsize=20)
    axes[0,1].set_xlabel('SGD steps', fontsize=20)

    #Plot changes in baseline inhibition and excitation and feedforward weights (second stage of the training)
    axes[1,2].plot(range(N), df['c_E'], label='c_E',c='tab:red',linewidth=3)
    axes[1,2].plot(range(N), df['c_I'], label='c_I',c='tab:blue',linewidth=3)
    axes[1,2].set_title('c: baseline inputs', fontsize=20)

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

def rel_changes_mid_only(df, num_indices=3):
    # Calculate relative changes in Jm and Js
    J_m_EE = df['J_m_EE']
    J_m_IE = df['J_m_IE']
    J_m_EI = [np.abs(df['J_m_EI'][i]) for i in range(len(df['J_m_EI']))]
    J_m_II = [np.abs(df['J_m_II'][i]) for i in range(len(df['J_m_II']))]
    c_E = df['c_E']
    c_I = df['c_I']
    acc = df['acc']
    offset = df['offset']
    maxr_E_mid = df['maxr_E_mid']
    maxr_I_mid = df['maxr_I_mid']
    relative_changes = numpy.zeros((10,num_indices-1))

    ############### Calculate relative changes in parameters and other metrics before and after training ###############
    # Define time indices for pretraining and training
    time_inds = SGD_step_indices(df, num_indices)

    # Calculate relative changes for pretraining and training (additional time points may be included)
    for j in range(2):
        if j==0:
            # changes during pretraining
            start_ind = time_inds[0]
            relative_changes[0,0] =(J_m_EE[time_inds[1]] - J_m_EE[start_ind]) / J_m_EE[start_ind] # J_m_EE
            relative_changes[1,0] =(J_m_IE[time_inds[1]] - J_m_IE[start_ind]) / J_m_IE[start_ind] # J_m_IE
            relative_changes[2,0] =(J_m_EI[time_inds[1]] - J_m_EI[start_ind]) / J_m_EI[start_ind] # J_m_EI
            relative_changes[3,0] =(J_m_II[time_inds[1]] - J_m_II[start_ind]) / J_m_II[start_ind] # J_m_II
            relative_changes[4,0] = (c_E[time_inds[1]] - c_E[start_ind]) / c_E[start_ind] # c_E
            relative_changes[5,0] = (c_I[time_inds[1]] - c_I[start_ind]) / c_I[start_ind] # c_I
            relative_changes[6,0] = (acc[time_inds[1]] - acc[start_ind]) / acc[start_ind] # accuracy
            relative_changes[7,0] = (offset[time_inds[1]] - offset[start_ind]) / offset[start_ind] # offset and offset threshold
            relative_changes[8,0] = (maxr_E_mid[time_inds[1]] - maxr_E_mid[start_ind]) /maxr_E_mid[start_ind] # r_E_mid
            relative_changes[9,0] = (maxr_I_mid[time_inds[1]] -maxr_I_mid[start_ind]) / maxr_I_mid[start_ind] # r_I_mid
        else: 
            # changes during training
            start_ind = time_inds[1]
            for i in range(num_indices-2):
                relative_changes[0,i+j] =(J_m_EE[time_inds[i+2]] - J_m_EE[start_ind]) / J_m_EE[start_ind] # J_m_EE
                relative_changes[1,i+j] =(J_m_IE[time_inds[i+2]] - J_m_IE[start_ind]) / J_m_IE[start_ind] # J_m_IE
                relative_changes[2,i+j] =(J_m_EI[time_inds[i+2]] - J_m_EI[start_ind]) / J_m_EI[start_ind] # J_m_EI
                relative_changes[3,i+j] =(J_m_II[time_inds[i+2]] - J_m_II[start_ind]) / J_m_II[start_ind] # J_m_II
                relative_changes[4,i+j] = (c_E[time_inds[i+2]] - c_E[start_ind]) / c_E[start_ind] # c_E
                relative_changes[5,i+j] = (c_I[time_inds[i+2]] - c_I[start_ind]) / c_I[start_ind] # c_I
                relative_changes[6,i+j] = (acc[time_inds[i+2]] - acc[start_ind]) / acc[start_ind] # accuracy
                relative_changes[7,i+j] = (offset[time_inds[i+2]] - offset[start_ind]) / offset[start_ind]
                relative_changes[8,i+j] = (maxr_E_mid[time_inds[i+2]] - maxr_E_mid[start_ind]) / maxr_E_mid[start_ind] # r_E_mid
                relative_changes[9,i+j] = (maxr_I_mid[time_inds[i+2]] - maxr_I_mid[start_ind]) / maxr_I_mid[start_ind] # r_I_mid

    return relative_changes, time_inds

def plot_tc_features_mid_only(results_dir, num_training, ori_list, train_only_str='', pre_post_scatter_flag=False):

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

    for i in range(num_training):
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
            
            # Save features: if first iteration, initialize; else, concatenate
            if  i==0:
                data[f'norm_slope_{key}'] = slope
                data[f'fwhm_{key}'] = fwhm
                data[f'orientations_{key}'] = orientations
            else:
                data[f'norm_slope_{key}'] = numpy.concatenate((data[f'norm_slope_{key}'], slope), axis=0)
                data[f'fwhm_{key}'] = numpy.concatenate((data[f'fwhm_{key}'], fwhm), axis=0)
                data[f'orientations_{key}'] = numpy.concatenate((data[f'orientations_{key}'], orientations), axis=0)


    ############## Plots about changes before vs after training and pretraining and training only (per layer and per centered or all) ##############
                
    # Define indices for each group of cells) 
    E_mid = numpy.linspace(0, 647, 648).round().reshape(8, 81, -1)[0:9:2].ravel().astype(int) 
    I_mid = numpy.linspace(0, 647, 648).round().reshape(8, 81, -1)[1:9:2].ravel().astype(int) 
    indices = [ E_mid, I_mid]

    # Create labels for the plot
    fs_text = 40
    ############# Plot fwhm before vs after training for E_sup and E_mid #############
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    j=0
    # add a little jitter to x and y to avoid overlapping points
    x = numpy.random.normal(0, 0.3, data['fwhm_prepre'].shape) + data['fwhm_prepre']
    y = numpy.random.normal(0, 0.3, data['fwhm_post'].shape) + data['fwhm_post']

    plot_pre_post_scatter(axs[abs((2-j))//2,1], x , y ,data['orientations_postpre'], indices[j], num_training, '', colors=None)
    
    axs[0,1].set_title('Full width \n at half maximum (deg.)', fontsize=fs_text)
    axs[0,1].set_xlabel('')
    axs[1,1].set_xlabel('Pre FWHM', fontsize=fs_text, labelpad=20)
    axs[1,1].set_ylabel('Post FWHM', fontsize=fs_text)
    axs[0,1].set_ylabel('Post FWHM', fontsize=fs_text)
    
    ############# Plot orientation vs slope #############
    # Add slope difference before and after training to the data dictionary
    data['slope_diff'] = data['norm_slope_post'] - data['norm_slope_prepre']
    
    # Scatter slope, where x-axis is orientation and y-axis is the change in slope before and after training
    j=0
    # Define x and y values for the scatter plot
    x= data['orientations_prepre'][:,indices[j]].flatten()
    #shift x to have 0 in its center (with circular orientation) and 180 at the end and apply the same shift to the slope_diff
    x = numpy.where(x>90, x-180, x)
    y= data['slope_diff'][:,indices[j]].flatten()
    lowess = sm.nonparametric.lowess(y, x, frac=0.15)  # Example with frac=0.2 for more local averaging
    axs[abs((2-j)) // 2,0].scatter(x, y, s=15, alpha=0.7)
    axs[abs((2-j)) // 2,0].plot(lowess[:, 0], lowess[:, 1], color='black', linewidth=3)
        
    axs[0,0].set_title('Tuning curve slope \n at trained orientation', fontsize=fs_text)
    axs[0,0].set_xlabel('')
    axs[1,0].set_xlabel('pref. ori - trained ori', fontsize=fs_text, labelpad=20)
    axs[1,0].set_ylabel(r'$\Delta$ slope', fontsize=fs_text)
    axs[0,0].set_ylabel(r'$\Delta$ slope', fontsize=fs_text)
    plt.tight_layout(w_pad=10, h_pad=7)
    fig.savefig(results_dir + "/figures/tc_features" + train_only_str +".png", bbox_inches='tight')
    plt.close()

def boxplots_from_csvs_mid_only(folder, save_folder, plot_filename = None, num_time_inds = 3, num_training = None):
    # List to store relative changes from each file
    relative_changes_at_time_inds = []
    
    # Iterate through each file in the directory
    numFiles = 0
    for filename in os.listdir(folder):
        if num_training is not None and numFiles > (num_training-1):
            break
        if filename.endswith('.csv') and filename.startswith('result'):
            numFiles = numFiles + 1
            filepath = os.path.join(folder, filename)
            # Read CSV file
            df = pd.read_csv(filepath)
            # Calculate relative change
            relative_changes, time_inds = rel_changes_mid_only(df, num_time_inds)
            start_time_ind = 1 if num_time_inds > 2 else 0
            relative_changes = relative_changes*100
            relative_changes_at_time_inds.append(relative_changes)
            if numFiles==1:
                # offset at time_inds[0] and at time_inds[i] handled as new row for each i
                offset_pre_post_temp = [[df['offset'][time_inds[start_time_ind]] ,df['offset'][time_inds[i] ]] for i in range(start_time_ind+1,num_time_inds)]
                if not numpy.isnan(numpy.array(offset_pre_post_temp)).any():
                    offset_pre_post = numpy.array(offset_pre_post_temp)
                else:
                    numFiles = numFiles - 1
            else:
                offset_pre_and_post_temp = [[df['offset'][time_inds[start_time_ind]] ,df['offset'][time_inds[i] ]] for i in range(start_time_ind+1,num_time_inds)]
                # skip if there is a nan value
                if not numpy.isnan(offset_pre_and_post_temp).any():
                    offset_pre_post = numpy.vstack((offset_pre_post,offset_pre_and_post_temp))
                else:
                    numFiles = numFiles - 1
    
    # Plotting bar plots of offset before and after given time indices
    offset_pre_post = offset_pre_post.T
    means = np.mean(offset_pre_post, axis=1)

    # Create figure and axis
    fig, ax = plt.subplots()
    # Colors for bars
    colors = ['blue', 'green']

    # Bar plot for mean values
    bars = ax.bar(['Pre', 'Post'], means, color=colors, alpha=0.7)
    # Annotate each bar with its value
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=20)

    # Plot individual data points and connect them
    for i in range(2):
        group_data = offset_pre_post[i,:]
        x_positions = [i for j in range(len(group_data))]
        ax.scatter(x_positions, group_data, color='black', alpha=0.7)  # Scatter plot of individual data points
        # Draw lines from bar to points
    for j in range(len(group_data)):
        ax.plot([0, 1], [offset_pre_post[0,j], offset_pre_post[1,j]], color='black', alpha=0.2)
    # Save plot
    if plot_filename is not None:
        full_path = save_folder + '/offset_pre_post.png'
        fig.savefig(full_path)

    ################# Boxplots for relative parameter changes #################
    # Define groups of parameters and plot each parameter group
    labels = [
        [r'$\Delta J^{\text{mid}}_{E \rightarrow E}$', r'$\Delta J^{\text{mid}}_{E \rightarrow I}$', r'$\Delta J^{\text{mid}}_{I \rightarrow E}$', r'$\Delta J^{\text{mid}}_{I \rightarrow I}$'],
        [r'$\Delta c_E$', r'$\Delta c_I$']
    ]
    num_groups = len(labels)

    fig, axs = plt.subplots(num_time_inds-1, num_groups, figsize=(5*num_groups, 5*(num_time_inds-1)))  # Create subplots for each group
    axes_flat = axs.flatten()
    
    relative_changes_at_time_inds = numpy.array(relative_changes_at_time_inds)
    group_start_ind = [0,4,6] # putting together Jm, and c
    J_box_colors = ['tab:red','tab:red','tab:blue','tab:blue']
    c_box_colors = ['#8B4513', '#800080']
    if np.sum(np.abs(relative_changes[:,0])) >0:
        for j in range(num_time_inds-1):
            for i, label in enumerate(labels):
                group_data = relative_changes_at_time_inds[:, group_start_ind[i]:group_start_ind[i+1], j]  # Extract data for the current group
                bp = axes_flat[j*num_groups+i].boxplot(group_data,labels=label,patch_artist=True)
                if i<2:
                    for box, color in zip(bp['boxes'], J_box_colors):
                        box.set_facecolor(color)
                else:
                    for box, color in zip(bp['boxes'], c_box_colors):
                        box.set_facecolor(color)
                axes_flat[j*num_groups+i].axhline(y=0, color='black', linestyle='--')
        
    plt.tight_layout()
    
    if plot_filename is not None:
        full_path = save_folder + '/' + plot_filename + ".png"
        fig.savefig(full_path)

    plt.close()


############### MAIN CODE ################
import pandas as pd
import numpy
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
numpy.random.seed(0)
'''
from training.util_gabor import init_untrained_pars
from util import load_parameters
from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    readout_pars,
    ssn_pars,
    conv_pars,
    training_pars,
    loss_pars,
    pretrain_pars # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
)

# Checking that pretrain_pars.is_on is on
if not pretrain_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to True in parameters.py to run training with pretraining!')


########## Calculate and save tuning curves ############
tc_ori_list = numpy.arange(0,180,2)
num_training = 10
final_folder_path = os.path.join('results','May21_v1')
start_time_in_main= time.time()
for i in range(num_training):
    # Define file names
    results_filename = os.path.join(final_folder_path, f"results_{i}.csv")
    tc_prepre_filename = os.path.join(final_folder_path, f"tc_prepre_{i}.csv")
    tc_postpre_filename = os.path.join(final_folder_path, f"tc_postpre_{i}.csv")
    tc_post_filename = os.path.join(final_folder_path, f"tc_post_{i}.csv")
    orimap_filename = os.path.join(final_folder_path, f"orimap_{i}.npy")
    df = pd.read_csv(results_filename)
    SGD_step_inds = SGD_step_indices(df, 3)

    # Load parameters and calculate (and save) tuning curves
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, orimap_filename)
    trained_pars_stage1, trained_pars_stage2, offset_last = load_parameters_mid_only(final_folder_path,i, iloc_ind = SGD_step_inds[0])
    tc_prepre= tuning_curve_mid_only(untrained_pars, trained_pars_stage2, tc_prepre_filename, ori_vec=tc_ori_list)   
    trained_pars_stage1, trained_pars_stage2, offset_last = load_parameters_mid_only(final_folder_path,i, iloc_ind = SGD_step_inds[1], trained_pars_keys=trained_pars_stage2.keys())
    tc_postpre = tuning_curve_mid_only(untrained_pars, trained_pars_stage2, tc_postpre_filename, ori_vec=tc_ori_list)
    _, trained_pars_stage2, _ = load_parameters_mid_only(final_folder_path,i, iloc_ind = SGD_step_inds[2], trained_pars_keys=trained_pars_stage2.keys())
    tc_post = tuning_curve_mid_only(untrained_pars, trained_pars_stage2, tc_post_filename, ori_vec=tc_ori_list)
    print(f'Finished calculating tuning curves for training {i} in {time.time()-start_time_in_main} seconds')

'''
######### PLOT RESULTS ############
import matplotlib.pyplot as plt

start_time=time.time()
final_folder_path=os.path.join('results','May21_v1')
num_training = 10
tc_ori_list = numpy.arange(0,180,2)
tc_cells=[10,20,30,40,50,60,70,80]

## Pretraining + training
folder_to_save = os.path.join(final_folder_path, 'figures')
boxplot_file_name = 'boxplot_pretraining'
mahal_file_name = 'Mahal_dist'
num_SGD_inds = 3
sigma_filter = 2

plot_results_from_csvs_mid_only(final_folder_path, num_training, folder_to_save=folder_to_save)#, starting_run=10)
#boxplots_from_csvs_mid_only(final_folder_path, folder_to_save, boxplot_file_name, num_time_inds = num_SGD_inds, num_training=num_training)
plot_tc_features_mid_only(final_folder_path, num_training, tc_ori_list)