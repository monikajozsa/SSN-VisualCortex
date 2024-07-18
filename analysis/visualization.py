import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
import jax.numpy as np
import numpy
import sys
import seaborn as sns
import statsmodels.api as sm
import scipy

from analysis_functions import rel_changes, tc_features, MVPA_param_offset_correlations
from util import filter_for_run

plt.rcParams['xtick.labelsize'] = 12 # Set the size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12 # Set the size for y-axis tick labels

########### Plotting functions ##############
def boxplots_from_csvs(folder, save_folder, plot_filename = None, num_time_inds = 3, num_training = 1):
    # List to store relative changes from each file
    relative_changes_at_time_inds = []
    
    # Iterate through each file in the directory
    numFiles = 0
    for i in range(num_training):
        filepath = os.path.join(folder, 'results.csv')
        # Read CSV file
        df = pd.read_csv(filepath)
        df_i = filter_for_run(df,i)
        # Calculate relative change
        relative_changes, time_inds = rel_changes(df_i, num_time_inds)
        start_time_ind = 1 if num_time_inds > 2 else 0
        relative_changes = relative_changes*100
        relative_changes_at_time_inds.append(relative_changes)
    
        offset_pre_post_temp = [[df_i['offset'][time_inds[start_time_ind]] ,df_i['offset'][time_inds[i] ]] for i in range(start_time_ind+1,num_time_inds)]
        if not numpy.isnan(numpy.array(offset_pre_post_temp)).any():
            if numFiles==0:
                offset_pre_post = numpy.array(offset_pre_post_temp)
            else:
                offset_pre_post = numpy.vstack((offset_pre_post,offset_pre_post_temp))
            numFiles = numFiles + 1
        else:
            numFiles = numFiles - 1
        
    ################# Plotting bar plots of offset before and after given time indices #################
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
        # let the position be i 
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
        [r'$\Delta J^{\text{sup}}_{E \rightarrow E}$', r'$\Delta J^{\text{sup}}_{E \rightarrow I}$', r'$\Delta J^{\text{sup}}_{I \rightarrow E}$', r'$\Delta J^{\text{sup}}_{I \rightarrow I}$'],
        [r'$\Delta c_E$', r'$\Delta c_I$', r'$\Delta f_E$', r'$\Delta f_I$']
    ]
    num_groups = len(labels)

    fig, axs = plt.subplots(num_time_inds-1, num_groups, figsize=(5*num_groups, 5*(num_time_inds-1)))  # Create subplots for each group
    axes_flat = axs.flatten()
    
    relative_changes_at_time_inds = numpy.array(relative_changes_at_time_inds)
    group_start_ind = [0,4,8,12] # putting together Jm, Js, c, f
    J_box_colors = ['tab:red','tab:red','tab:blue','tab:blue']
    c_f_box_colors = ['#8B4513', '#800080', '#FF8C00',  '#006400']
    if np.sum(np.abs(relative_changes[:,0])) >0:
        for j in range(num_time_inds-1):
            for i, label in enumerate(labels):
                group_data = relative_changes_at_time_inds[:, group_start_ind[i]:group_start_ind[i+1], j]  # Extract data for the current group
                bp = axes_flat[j*num_groups+i].boxplot(group_data,labels=label,patch_artist=True)
                if i<2:
                    for box, color in zip(bp['boxes'], J_box_colors):
                        box.set_facecolor(color)
                else:
                    for box, color in zip(bp['boxes'], c_f_box_colors):
                        box.set_facecolor(color)
                axes_flat[j*num_groups+i].axhline(y=0, color='black', linestyle='--')
                axes_format(axes_flat[j*num_groups+i], fs_ticks=20, ax_width=2, tick_width=5, tick_length=10, xtick_flag=False)
        
    plt.tight_layout()
    
    if plot_filename is not None:
        full_path = save_folder + '/' + plot_filename + ".png"
        fig.savefig(full_path)

    plt.close()


def plot_tuning_curves(results_dir,tc_cells,num_runs,folder_to_save, seed=0):

    tc_filename = results_dir + f'/tuning_curves.csv'
    tuning_curves = numpy.array(pd.read_csv(tc_filename))

    numpy.random.seed(seed)
    num_mid_cells = 648
    num_sup_cells = 164
    num_runs_plotted = min(5,num_runs)
    tc_post_pretrain = results_dir + '/tc_postpre_0.csv'
    pretrain_ison = os.path.exists(tc_post_pretrain)
    for i in range(num_runs_plotted):
        mesh_i = tuning_curves[:,0]==i
        tuning_curve_i = tuning_curves[mesh_i,1:]
        if pretrain_ison:
            mesh_stage_0 = tuning_curve_i[:,0]==0
            tc_0 = tuning_curve_i[mesh_stage_0,1:]

        mesh_stage_1 = tuning_curve_i[:,0]==1
        tc_1 = tuning_curve_i[mesh_stage_1,1:]
        mesh_stage_2 = tuning_curve_i[:,0]==2
        tc_2 = tuning_curve_i[mesh_stage_2,1:]

        # Select num_rnd_cells randomly selected cells to plot from both middle and superficial layer cells
        if i==0:
            if isinstance(tc_cells,int):
                num_rnd_cells=tc_cells
                selected_mid_col_inds = numpy.random.randint(0, num_mid_cells, size=int(num_rnd_cells/2), replace=False)
                selected_sup_col_inds = numpy.random.randint(0, num_sup_cells, size=num_rnd_cells-int(num_rnd_cells/2), replace=False)
            else:
                num_rnd_cells=len(tc_cells)
                selected_mid_col_inds = numpy.array(tc_cells[:int(len(tc_cells)/2)])-1
                selected_sup_col_inds = numpy.array(tc_cells[int(len(tc_cells)/2):])-1
            fig, axes = plt.subplots(nrows=num_runs_plotted, ncols=num_rnd_cells, figsize=(5*num_rnd_cells, 5*num_runs_plotted))
        num_oris = tc_1.shape[0]
        # Plot tuning curves
        for cell_ind in range(int(num_rnd_cells/2)):
            if num_runs_plotted==1:
                ax1=axes[cell_ind]
                ax2=axes[int(num_rnd_cells/2)+cell_ind]
            else:
                ax1=axes[i,cell_ind]
                ax2=axes[i,int(num_rnd_cells/2)+cell_ind]
            if pretrain_ison:
                ax1.plot(range(num_oris), tc_0[:,selected_mid_col_inds[cell_ind]], label='initial',linewidth=2)
                ax2.plot(range(num_oris), tc_0[:,selected_sup_col_inds[cell_ind]], label='initial',linewidth=2)
            ax1.plot(range(num_oris), tc_1[:,selected_mid_col_inds[cell_ind]], label='post-pretraining',linewidth=2)
            ax2.plot(range(num_oris), tc_1[:,selected_sup_col_inds[cell_ind]], label='post-pretraining',linewidth=2)
            ax1.plot(range(num_oris), tc_2[:,selected_mid_col_inds[cell_ind]], label='post-training',linewidth=2)
            ax2.plot(range(num_oris), tc_2[:,selected_sup_col_inds[cell_ind]], label='post-training',linewidth=2)
            
            ax1.set_title(f'Middle layer cell {cell_ind}', fontsize=20)
            ax2.set_title(f'Superficial layer, cell {cell_ind}', fontsize=20)
    ax2.legend(loc='upper left', fontsize=20)
    
    # Save plot
    if folder_to_save is not None:
        fig.savefig(os.path.join(folder_to_save,'tc_fig.png'))
    else:
        fig.savefig(os.path.join(results_dir,'tc_fig.png'))
    plt.close()


def plot_pre_post_scatter(ax, x_axis, y_axis, orientations, indices_to_plot, num_training, title, colors=None, linecolor='black'):
    '''
    Scatter plot of pre vs post training values for a given set of indices
    '''
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


def plot_tc_features(results_dir, num_training, ori_list):
    def shift_x_data(x_data, indices, shift_value=90):
        # Shift x_data by shift_value and center it around 0 (that is, around shift_value)
        x_data_shifted = x_data[:, indices].flatten() - shift_value
        x_data_shifted = numpy.where(x_data_shifted > 90, x_data_shifted - 180, x_data_shifted)
        x_data_shifted = numpy.where(x_data_shifted < -90, x_data_shifted + 180, x_data_shifted)
        return x_data_shifted
    # Initialize dictionaries to store the data arrays
    data = {
    'fwhm_0': [],
    'fwhm_2': [],
    'preforis_0': [],
    'preforis_2': []}
            
    # Load data from file
    tc_filename = results_dir + f'/tuning_curves.csv'
    tuning_curves = pd.read_csv(tc_filename)
    # Loop through each training and stage within training (pre pretraining, post pretrainig and post training)
    for i in range(num_training):
        # Filter tuning curves for the current run
        tuning_curves_i = filter_for_run(tuning_curves,i)
        tuning_curves_i['training_stage'] = pd.to_numeric(tuning_curves_i['training_stage'], errors='coerce')
        for training_stage in range(3):      
            # Filter tuning curves for the current training stage      
            mesh_stage = tuning_curves_i['training_stage']==training_stage
            tuning_curve = tuning_curves_i[mesh_stage]
            tuning_curve = tuning_curve.drop(columns=['training_stage'])
            tuning_curve = tuning_curve.to_numpy()

            # Calculate features for the current tuning curve: slope of normalized tuning_curve
            slope, fwhm, orientations = tc_features(tuning_curve, ori_list=ori_list, expand_dims=True, ori_to_center_slope=[55, 125])
            # Save features: if first iteration, initialize; else, concatenate
            if  i==0:
                data[f'slope_55_{training_stage}'] = slope[:,:,0]
                data[f'slope_125_{training_stage}'] = slope[:,:,1]
                data[f'fwhm_{training_stage}'] = fwhm
                data[f'preforis_{training_stage}'] = orientations
            else:
                data[f'slope_55_{training_stage}'] = numpy.concatenate((data[f'slope_55_{training_stage}'], slope[:,:,0]), axis=0)
                data[f'slope_125_{training_stage}'] = numpy.concatenate((data[f'slope_125_{training_stage}'], slope[:,:,1]), axis=0)
                data[f'fwhm_{training_stage}'] = numpy.concatenate((data[f'fwhm_{training_stage}'], fwhm), axis=0)
                data[f'preforis_{training_stage}'] = numpy.concatenate((data[f'preforis_{training_stage}'], orientations), axis=0)


    ############## Plots about changes before vs after training and pretraining and training only (per layer and per centered or all) ##############
             
    # Define indices for each group of cells
    E_sup = 648+numpy.linspace(0, 80, 81).astype(int) 
    I_sup = 648+numpy.linspace(81, 161, 81).astype(int) 
    E_mid_array = numpy.linspace(0, 647, 648).round().reshape(4, 2, 81).astype(int)
    E_mid = E_mid_array[:,0,:].ravel().astype(int)
    I_mid_array = numpy.linspace(0, 647, 648).round().reshape(4, 2, 81).astype(int)
    I_mid = I_mid_array[:,1,:].ravel().astype(int)
    indices = [E_sup, I_sup, E_mid, I_mid]
    #E_sup_centre = 648+numpy.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)
    #I_sup_centre = (E_sup_centre+81).astype(int)
    #E_mid_centre = numpy.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)
    #I_mid_centre = (E_mid_centre+81).astype(int)
    
    # Create legends for the plot
    patches = []
    cmap = plt.get_cmap('rainbow')
    colors = numpy.flip(cmap(numpy.linspace(0,1, 8)), axis = 0)
    bins = ['0-4', '4-12', '12-20', '20-28', '28-36', '36-44', '44-50', '+50']
    for j in range(0,len(colors)):
        patches.append(mpatches.Patch(color=colors[j], label=bins[j]))

    #############################################################################
    ######### Schoups-style scatter plots - coloring based on cell type #########
    #############################################################################
    phase_colors_E = [ 'red', 'yellow', 'darkred', 'orange']
    phase_colors_I = [ 'blue', 'green', 'darkblue', 'darkgreen']
    colors = numpy.flip(cmap(numpy.linspace(0,1, 8)), axis = 0)
    fs_text = 40
    fs_ticks = 30
    
    # Scatter slope, where x-axis is orientation and y-axis is the change in slope before and after training
    stage_labels = ['pretrain', 'train']
    for training_stage in range(2):
        fig, axs = plt.subplots(2, 2, figsize=(25, 25))
        for j in [0,2]:            
            ##### Plot fwhm before vs after training for E_sup and E_mid #####
            # add a little jitter to x and y to avoid overlapping points
            x = numpy.random.normal(0, 0.3, data[f'fwhm_{training_stage}'].shape) + data[f'fwhm_{training_stage}']
            y = numpy.random.normal(0, 0.3, data[f'fwhm_{training_stage+1}'].shape) + data[f'fwhm_{training_stage+1}']
            ax = axs[abs((2-j))//2,1]
            if j==2:
                for phase_ind in range(4):
                    indices_phase_E = I_mid_array[phase_ind,0,:]
                    indices_phase_I = I_mid_array[phase_ind,1,:]
                    ax.scatter(x[:,indices_phase_E], y[:,indices_phase_E], s=(50-10*phase_ind), alpha=0.5, color=phase_colors_E[phase_ind])
                    ax.scatter(x[:,indices_phase_I], y[:,indices_phase_I], s=(50-10*phase_ind), alpha=0.5, color=phase_colors_I[phase_ind])
            else:
                ax.scatter(x[:,I_sup], y[:,I_sup], s=30, alpha=0.5, color='blue')
                ax.scatter(x[:,E_sup], y[:,E_sup], s=30, alpha=0.5, color='red')
            xpoints = ypoints = ax.get_xlim()
            ax.plot(xpoints, ypoints, color='black', linewidth=2)
            ax.set_xlabel('Pre training')
            ax.set_ylabel('Post training')
            
            # Format axes
            axes_format(axs[abs((2-j))//2,1], fs_ticks)
            
            ##### Plot orientation vs slope #####
            data[f'slopediff_55_{training_stage}'] = data[f'slope_55_{training_stage+1}'] - data[f'slope_55_{training_stage}']
            data[f'slopediff_125_{training_stage}'] = data[f'slope_125_{training_stage+1}'] - data[f'slope_125_{training_stage}']
            data[f'slopediff_diff_{training_stage}'] = data[f'slopediff_55_{training_stage}'] - data[f'slopediff_125_{training_stage}']
            if j==2:
                # Middle layer scatter plots with added colors to the different cell categories
                for phase_ind in range(4):
                    indices_phase_E = E_mid_array[phase_ind,0,:]
                    indices_phase_I = I_mid_array[phase_ind,1,:]
                    y_E= data[f'slopediff_diff_{training_stage}'][:,indices_phase_E].flatten()
                    y_I= data[f'slopediff_diff_{training_stage}'][:,indices_phase_I].flatten()
                    x_I_90 = shift_x_data(data[f'preforis_{training_stage}'], indices_phase_I, shift_value=90)
                    x_E_90 = shift_x_data(data[f'preforis_{training_stage}'], indices_phase_E, shift_value=90)
                    axs[abs((2-j)) // 2,0].scatter(x_E_90, y_E, s=(50-10*phase_ind), alpha=0.5, color=phase_colors_E[phase_ind])
                    axs[abs((2-j)) // 2,0].scatter(x_I_90, y_I, s=(50-10*phase_ind), alpha=0.5, color=phase_colors_I[phase_ind])
            else:
                # Superficial layer scatter plots
                y_E= data[f'slopediff_diff_{training_stage}'][:,E_sup].flatten()
                y_I= data[f'slopediff_diff_{training_stage}'][:,I_sup].flatten()
                x_E_90= shift_x_data(data[f'preforis_{training_stage}'], E_sup, shift_value=90)
                x_I_90= shift_x_data(data[f'preforis_{training_stage}'], I_sup, shift_value=90)
                axs[abs((2-j)) // 2,0].scatter(x_E_90, y_E, s=30, alpha=0.7, color='red')
                axs[abs((2-j)) // 2,0].scatter(x_I_90, y_I, s=30, alpha=0.7, color='blue')
            # Line plots for both layers: define x and y values and shift x to have 0 in its center
            y_E= data[f'slopediff_diff_{training_stage}'][:,indices[j]].flatten()
            y_I= data[f'slopediff_diff_{training_stage}'][:,indices[j+1]].flatten()
            x_E= shift_x_data(data[f'preforis_{training_stage}'], indices[j], shift_value=90)
            x_I= shift_x_data(data[f'preforis_{training_stage}'], indices[j+1], shift_value=90)
            lowess_E = sm.nonparametric.lowess(y_E, x_E, frac=0.15)  # Example with frac=0.2 for more local averaging
            lowess_I = sm.nonparametric.lowess(y_I, x_I, frac=0.15)
            axs[abs((2-j)) // 2,0].plot(lowess_E[:, 0], lowess_E[:, 1], color='darkred', linewidth=6)
            axs[abs((2-j)) // 2,0].plot(lowess_I[:, 0], lowess_I[:, 1], color='darkblue', linewidth=6)
            axes_format(axs[abs((2-j)) // 2,0], fs_ticks)
        
        axs[0,1].set_title('Full width \n at half maximum (deg.)', fontsize=fs_text)
        axs[0,1].set_xlabel('')
        axs[1,1].set_xlabel('Pre FWHM', fontsize=fs_text, labelpad=20)
        axs[1,1].set_ylabel('Post FWHM', fontsize=fs_text)
        axs[0,1].set_ylabel('Post FWHM', fontsize=fs_text)

        axs[0,0].set_title('Tuning curve slope:\n'+ r'$\Delta$' + 'slope(55) - '+ r'$\Delta$' + 'slope(125)', fontsize=fs_text)
        axs[0,0].set_xlabel('')
        axs[1,0].set_xlabel('pref. ori - trained ori', fontsize=fs_text, labelpad=20)
        axs[1,0].set_ylabel(r'$\Delta$ slope(55)- \Delta$ slope(125)', fontsize=fs_text)
        axs[0,0].set_ylabel(r'$\Delta$ slope(55)- \Delta$ slope(125)', fontsize=fs_text)
        plt.tight_layout(w_pad=10, h_pad=7)
        fig.savefig(results_dir + f"/figures/tc_features_{stage_labels[training_stage]}.png", bbox_inches='tight')
        plt.close()

        # 3 x 2 scatter plot of data[slopediff_55_0 and 1], data[slopediff_55_0 and 1] and data[slopediff_diff]
        fig, axs = plt.subplots(2, 3, figsize=(25, 25))
        # Middle layer
        for k in [0,2]:
            # k=0 superficial layer, k=2 middle layer
            # Middle layer scatter plots
            if k==2:
                for phase_ind in range(4):
                    indices_phase_E = E_mid_array[phase_ind,0,:]
                    indices_phase_I = E_mid_array[phase_ind,1,:]
                    x_E_55 = shift_x_data(data[f'preforis_{training_stage}'], indices_phase_E, shift_value=55)
                    x_I_55 = shift_x_data(data[f'preforis_{training_stage}'], indices_phase_I, shift_value=55)
                    x_E_125 = shift_x_data(data[f'preforis_{training_stage}'], indices_phase_E, shift_value=125)
                    x_I_125 = shift_x_data(data[f'preforis_{training_stage}'], indices_phase_I, shift_value=125)
                    x_E_90 = shift_x_data(data[f'preforis_{training_stage}'], indices_phase_E, shift_value=90)
                    x_I_90 = shift_x_data(data[f'preforis_{training_stage}'], indices_phase_I, shift_value=90)
                    axs[0,0].scatter(x_E_55, data[f'slopediff_{55}_{training_stage}'][:,indices_phase_E].flatten(), s=(50-10*phase_ind), alpha=0.5, color=phase_colors_E[phase_ind])
                    axs[0,0].scatter(x_I_55, data[f'slopediff_{55}_{training_stage}'][:,indices_phase_I].flatten(), s=(50-10*phase_ind), alpha=0.5, color=phase_colors_I[phase_ind])
                    axs[0,1].scatter(x_E_125, data[f'slopediff_{125}_{training_stage}'][:,indices_phase_E].flatten(), s=(50-10*phase_ind), alpha=0.5, color=phase_colors_E[phase_ind])
                    axs[0,1].scatter(x_I_125, data[f'slopediff_{125}_{training_stage}'][:,indices_phase_I].flatten(), s=(50-10*phase_ind), alpha=0.5, color=phase_colors_I[phase_ind])
                    axs[0,2].scatter(x_E_90, data[f'slopediff_diff_{training_stage}'][:,indices_phase_E].flatten(), s=(50-10*phase_ind), alpha=0.5, color=phase_colors_E[phase_ind])
                    axs[0,2].scatter(x_I_90, data[f'slopediff_diff_{training_stage}'][:,indices_phase_I].flatten(), s=(50-10*phase_ind), alpha=0.5, color=phase_colors_I[phase_ind])
                # Line plots for middle layer: merge phases
                x_E_55 = shift_x_data(data[f'preforis_{training_stage}'], E_mid, shift_value=55)
                x_I_55 = shift_x_data(data[f'preforis_{training_stage}'], I_mid, shift_value=55)
                x_E_125 = shift_x_data(data[f'preforis_{training_stage}'], E_mid, shift_value=125)
                x_I_125 = shift_x_data(data[f'preforis_{training_stage}'], I_mid, shift_value=125)
                x_E_90 = shift_x_data(data[f'preforis_{training_stage}'], E_mid, shift_value=90)
                x_I_90 = shift_x_data(data[f'preforis_{training_stage}'], I_mid, shift_value=90)
                lowess_E_55 = sm.nonparametric.lowess(data[f'slopediff_{55}_{training_stage}'][:,indices[k]].flatten(), x_E_55, frac=0.15)  # Example with frac=0.2 for more local averaging
                lowess_I_55 = sm.nonparametric.lowess(data[f'slopediff_{55}_{training_stage}'][:,indices[k+1]].flatten(), x_I_55, frac=0.15)
                lowess_E_125 = sm.nonparametric.lowess(data[f'slopediff_{125}_{training_stage}'][:,indices[k]].flatten(), x_E_125, frac=0.15)
                lowess_I_125 = sm.nonparametric.lowess(data[f'slopediff_{125}_{training_stage}'][:,indices[k+1]].flatten(), x_I_125, frac=0.15)
                lowess_E_diff = sm.nonparametric.lowess(data[f'slopediff_diff_{training_stage}'][:,indices[k]].flatten(), x_E_90, frac=0.15)
                lowess_I_diff = sm.nonparametric.lowess(data[f'slopediff_diff_{training_stage}'][:,indices[k+1]].flatten(), x_I_90, frac=0.15)
            # Superficial layer scatter plots
            axs_ind_1 = abs((2-k))//2           
            if k==0:
                x_E_55 = shift_x_data(data[f'preforis_{training_stage}'], E_sup, shift_value=55)
                x_I_55 = shift_x_data(data[f'preforis_{training_stage}'], I_sup, shift_value=55)
                x_E_125 = shift_x_data(data[f'preforis_{training_stage}'], E_sup, shift_value=125)
                x_I_125 = shift_x_data(data[f'preforis_{training_stage}'], I_sup, shift_value=125)
                x_E_90 = shift_x_data(data[f'preforis_{training_stage}'], E_sup, shift_value=90)
                x_I_90 = shift_x_data(data[f'preforis_{training_stage}'], I_sup, shift_value=90)
                axs[axs_ind_1,0].scatter(x_E_55, data[f'slopediff_{55}_{training_stage}'][:,indices[k]].flatten(), s=30, alpha=0.7, color='red')
                axs[axs_ind_1,0].scatter(x_I_55, data[f'slopediff_{55}_{training_stage}'][:,indices[k+1]].flatten(), s=30, alpha=0.7, color='blue')
                axs[axs_ind_1,1].scatter(x_E_125, data[f'slopediff_{125}_{training_stage}'][:,indices[k]].flatten(), s=30, alpha=0.7, color='red')
                axs[axs_ind_1,1].scatter(x_I_125, data[f'slopediff_{125}_{training_stage}'][:,indices[k+1]].flatten(), s=30, alpha=0.7, color='blue')
                axs[axs_ind_1,2].scatter(x_E_90, data[f'slopediff_diff_{training_stage}'][:,indices[k]].flatten(), s=30, alpha=0.7, color='red')
                axs[axs_ind_1,2].scatter(x_I_90, data[f'slopediff_diff_{training_stage}'][:,indices[k+1]].flatten(), s=30, alpha=0.7, color='blue')
                
                # Line plots for superficial layer: define x and y values and shift x to have 0 in its center
                lowess_E_55 = sm.nonparametric.lowess(data[f'slopediff_{55}_{training_stage}'][:,indices[k]].flatten(), x_E_55, frac=0.15)  # Example with frac=0.2 for more local averaging
                lowess_I_55 = sm.nonparametric.lowess(data[f'slopediff_{55}_{training_stage}'][:,indices[k+1]].flatten(), x_I_55, frac=0.15)
                lowess_E_125 = sm.nonparametric.lowess(data[f'slopediff_{125}_{training_stage}'][:,indices[k]].flatten(), x_E_125, frac=0.15)
                lowess_I_125 = sm.nonparametric.lowess(data[f'slopediff_{125}_{training_stage}'][:,indices[k+1]].flatten(), x_I_125, frac=0.15)
                lowess_E_diff = sm.nonparametric.lowess(data[f'slopediff_diff_{training_stage}'][:,indices[k]].flatten(), x_E_90, frac=0.15)
                lowess_I_diff = sm.nonparametric.lowess(data[f'slopediff_diff_{training_stage}'][:,indices[k+1]].flatten(), x_I_90, frac=0.15)
            axs[axs_ind_1,0].plot(lowess_E_55[:, 0], lowess_E_55[:, 1], color='red', linewidth=4)
            axs[axs_ind_1,0].plot(lowess_I_55[:, 0], lowess_I_55[:, 1], color='blue', linewidth=4)
            axs[axs_ind_1,1].plot(lowess_E_125[:, 0], lowess_E_125[:, 1], color='red', linewidth=4)
            axs[axs_ind_1,1].plot(lowess_I_125[:, 0], lowess_I_125[:, 1], color='blue', linewidth=4)
            axs[axs_ind_1,2].plot(lowess_E_diff[:, 0], lowess_E_diff[:, 1], color='red', linewidth=4)
            axs[axs_ind_1,2].plot(lowess_I_diff[:, 0], lowess_I_diff[:, 1], color='blue', linewidth=4)
            # Set titles
            axs[axs_ind_1,0].set_title(r'$\Delta$' + 'slope(55)', fontsize=fs_text)
            axs[axs_ind_1,1].set_title(r'$\Delta$' + 'slope(125)', fontsize=fs_text)
            axs[axs_ind_1,2].set_title(r'$\Delta$' + 'slope(55) - '+ r'$\Delta$' + 'slope(125)', fontsize=fs_text)
        # Format and save plot
        for ax in axs.flatten():
            axes_format(ax, fs_ticks)
        plt.tight_layout(w_pad=10, h_pad=7)
        fig.savefig(results_dir + f"/figures/tc_slope_{stage_labels[training_stage]}.png", bbox_inches='tight')
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


def plot_results_from_csv(df,fig_filename=None):
    
    N=len(df[df.columns[0]])
    # Create a subplot with 4 rows and 3 columns
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(45, 35))

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
            axes[0,1].plot(range(N), df[column], label=column, alpha=0.6)
        if 'val_loss' in column:
            axes[0,1].scatter(range(N), df[column], label='val_loss', marker='o', s=50)
    axes[0,1].legend(loc='upper right')
    axes[0,1].set_title('Loss', fontsize=20)
    axes[0,1].set_xlabel('SGD steps', fontsize=20)

    # BARPLOTS about relative changes
    categories_J = ['Jm_EE', 'Jm_IE', 'Jm_EI', 'Jm_II', 'Js_EE', 'Js_IE', 'Js_EI', 'Js_II']
    categories_metrics = [ 'acc', 'offset']
    categories_r = [ 'rm_E', 'rm_I', 'rs_E','rs_I']
    categories_cf = ['c_E', 'c_I', 'f_E', 'f_I']
    rel_par_changes,_ = rel_changes(df) # 0 is pretraining and 1 is training in the second dimensions
    pretrain_train_ind=1
    values_J = 100 * rel_par_changes[0:8,pretrain_train_ind]
    values_cf = 100 * rel_par_changes[8:12,pretrain_train_ind]
    values_metrics = 100 * rel_par_changes[12:14,pretrain_train_ind]
    values_r = 100 * rel_par_changes[14:19,pretrain_train_ind]

    # Choosing colors for each bar
    colors_J = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:blue']
    colors_cf = ['tab:red', 'tab:blue', 'tab:red', 'tab:blue']
    colors_metrics = [ 'tab:green', 'tab:brown']
    colors_r = ['tab:red', 'tab:blue', 'tab:red', 'tab:blue']

    # Creating the bar plot
    bars_metrics = axes[0,3].bar(categories_metrics, values_metrics, color=colors_metrics)
    bars_r = axes[1,3].bar(categories_r, values_r, color=colors_r)
    bars_params = axes[2,2].bar(categories_J, values_J, color=colors_J)    
    bars_cf = axes[2,3].bar(categories_cf, values_cf, color=colors_cf)    

    # Annotating each bar with its value for bars_params
    for bar in bars_params:
        yval = bar.get_height()
        # Adjust text position for positive values to be inside the bar
        if abs(yval) > 2:
            if yval > 0:
                text_position = yval - 0.1*max(abs(values_J))
            else:
                text_position = yval + 0.05*max(abs(values_J))
        else:
            text_position = yval
        axes[2,2].text(bar.get_x() + bar.get_width() / 2, text_position, f'{yval:.2f}', ha='center', va='bottom', fontsize=20)
    for bar in bars_cf:
        yval = bar.get_height()
        if abs(yval) > 2:
            if yval > 0:
                text_position = yval - 0.1*max(abs(values_cf))
            else:
                text_position = yval + 0.05*max(abs(values_cf))
        else:
            text_position = yval
        axes[2,3].text(bar.get_x() + bar.get_width() / 2, text_position, f'{yval:.2f}', ha='center', va='bottom', fontsize=20)
    for bar in bars_r:
        yval = bar.get_height()
        if abs(yval) > 2:
            if yval > 0:
                text_position = yval - 0.1*max(abs(values_r))
            else:
                text_position = yval + 0.05*max(abs(values_r))
        else:
            text_position = yval
        axes[1,3].text(bar.get_x() + bar.get_width() / 2, text_position, f'{yval:.2f}', ha='center', va='bottom', fontsize=20)
    for bar in bars_metrics:
        yval = bar.get_height()
        if abs(yval) > 2:
            if yval > 0:
                text_position = yval - 0.2*max(abs(values_J))
            else:
                text_position = yval + 0.05*max(abs(values_J))
        else:
            text_position = yval
        axes[0,3].text(bar.get_x() + bar.get_width() / 2, text_position, f'{yval:.2f}', ha='center', va='bottom', fontsize=20)

        # Adding labels and titles
        axes[0,3].set_ylabel('relative change %', fontsize=20)
        axes[1,3].set_ylabel('relative change %', fontsize=20)
        axes[2,2].set_ylabel('relative change %', fontsize=20)
        axes[2,3].set_ylabel('relative change %', fontsize=20)
        if pretrain_train_ind==0:
            axes[2,2].set_title('Rel changes in J before and after pretraining', fontsize=20)
            axes[2,3].set_title('Other rel changes before and after pretraining', fontsize=20)
        else:
            axes[2,2].set_title('Rel changes in J before and after training', fontsize=20)
            axes[2,3].set_title('Other rel changes before and after training', fontsize=20)
    
    ################
    num_pretraining_steps= sum(df['stage'] == 0)
    for column in df.columns:
        if 'offset' in column:
            axes[0,2].plot(range(num_pretraining_steps), np.ones(num_pretraining_steps)*6, label='stopping threshold', alpha=0.6, c='tab:brown')
            axes[0,2].scatter(range(N), df[column], label='offset', marker='o', s=50, c='tab:brown')
            axes[0,2].grid(color='gray', linestyle='-', linewidth=0.5)
            axes[0,2].set_title('Offset', fontsize=20)
            axes[0,2].set_ylabel('degrees', fontsize=20)
            axes[0,2].set_xlabel('SGD steps', fontsize=20)
            axes[0,2].set_ylim(0, max(df[column])+1)
    
    # Plot changes in sigmoid weights and bias of the sigmoid layer
    axes[1,2].plot(range(N), df['b_sig'], label='b_sig', linestyle='--', linewidth = 3)
    axes[1,2].set_xlabel('SGD steps', fontsize=20)
    i=0
    for column in df.columns:
        if 'w_sig_' in column and i<10:
            axes[1,2].plot(range(N), df[column], label=column)
            i = i+1
    axes[1,2].set_title('Readout bias and weights', fontsize=20)
    axes[1,2].legend(loc='lower right')

    # Plot changes in J_m and J_s
    axes[2,0].plot(range(N), df['J_m_EE'], label='J_m_EE', linestyle='--', c='tab:red',linewidth=3)
    axes[2,0].plot(range(N), df['J_m_IE'], label='J_m_IE', linestyle='--', c='tab:orange',linewidth=3)
    axes[2,0].plot(range(N), df['J_m_II'], label='J_m_II', linestyle='--', c='tab:blue',linewidth=3)
    axes[2,0].plot(range(N), df['J_m_EI'], label='J_m_EI', linestyle='--', c='tab:green',linewidth=3)
    
    axes[2,0].plot(range(N), df['J_s_EE'], label='J_s_EE', c='tab:red',linewidth=3)
    axes[2,0].plot(range(N), df['J_s_IE'], label='J_s_IE', c='tab:orange',linewidth=3)
    axes[2,0].plot(range(N), df['J_s_II'], label='J_s_II', c='tab:blue',linewidth=3)
    axes[2,0].plot(range(N), df['J_s_EI'], label='J_s_EI', c='tab:green',linewidth=3)
    axes[2,0].legend(loc="upper left", fontsize=20)
    axes[2,0].set_title('J in middle and superficial layers', fontsize=20)
    axes[2,0].set_xlabel('SGD steps', fontsize=20)

    # Plot maximum rates
    colors = ["tab:blue", "tab:red"]
    axes[1,0].plot(range(N), df['maxr_E_mid'], label='maxr_E_mid', c=colors[1], linestyle=':')
    axes[1,0].plot(range(N), df['maxr_I_mid'], label='maxr_I_mid', c=colors[0], linestyle=':')
    axes[1,0].plot(range(N), df['maxr_E_sup'], label='maxr_E_sup', c=colors[1])
    axes[1,0].plot(range(N), df['maxr_I_sup'], label='maxr_I_sup', c=colors[0])
    axes[1,0].legend(loc="upper left", fontsize=20)
    axes[1,0].set_title('Maximum rates', fontsize=20)
    axes[1,0].set_xlabel('SGD steps', fontsize=20)

    # Plot mean rates
    colors = ["tab:blue", "tab:red"]
    if 'meanr_E_mid' in df.columns:
        axes[1,1].plot(range(N), df['meanr_E_mid'], label='meanr_E_mid', c=colors[1], linestyle=':')
        axes[1,1].plot(range(N), df['meanr_I_mid'], label='meanr_I_mid', c=colors[0], linestyle=':')
        axes[1,1].plot(range(N), df['meanr_E_sup'], label='meanr_E_sup', c=colors[1])
        axes[1,1].plot(range(N), df['meanr_I_sup'], label='meanr_I_sup', c=colors[0])
        axes[1,1].legend(loc="upper left", fontsize=20)
        axes[1,1].set_title('Mean rates', fontsize=20)
        axes[1,1].set_xlabel('SGD steps', fontsize=20)

    #Plot changes in baseline inhibition and excitation and feedforward weights (second stage of the training)
    axes[2,1].plot(range(N), df['c_E'], label='c_E',c='tab:red',linewidth=3)
    axes[2,1].plot(range(N), df['c_I'], label='c_I',c='tab:blue',linewidth=3)

    #Plot feedforward weights from middle to superficial layer (second stage of the training)
    axes[2,1].plot(range(N), df['f_E'], label='f_E', linestyle='--',c='tab:red',linewidth=3)
    axes[2,1].plot(range(N), df['f_I'], label='f_I', linestyle='--',c='tab:blue',linewidth=3)
    axes[2,1].set_title('c: constant inputs, f: weights between mid and sup layers', fontsize=20)

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
            axes[i,j].legend(fontsize=20)

    if fig_filename:
        fig.savefig(fig_filename + ".png")
    plt.close()


def plot_results_from_csvs(folder_path, num_runs=3, num_rnd_cells=5, folder_to_save=None, starting_run=0):
    # Add folder_path to path
    if folder_path not in sys.path:
        sys.path.append(folder_path)

    # Plot loss, accuracy and trained parameters
    results_filename = os.path.join(folder_path,f'results.csv')
    df = pd.read_csv(results_filename)
    for j in range(starting_run,num_runs):
        if folder_to_save is not None:
            results_fig_filename = os.path.join(folder_to_save,f'resultsfig_{j}')
        else:
            results_fig_filename = os.path.join(folder_path,f'resultsfig_{j}')
        if not os.path.exists(results_fig_filename):
            df_j = filter_for_run(df,j)
            plot_results_from_csv(df_j,results_fig_filename)

################### CORRELATION ANALYSIS ###################

def plot_correlations(folder, num_training, num_time_inds=3):
    offset_pars_corr, offset_staircase_pars_corr, MVPA_corrs, data = MVPA_param_offset_correlations(folder, num_training, num_time_inds, mesh_for_valid_offset=False)

    ########## Plot the correlation between offset_th_diff and the combination of the J_m_E_diff, J_m_I_diff, J_s_E_diff, and J_s_I_diff ##########
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    axes_flat = axes.flatten()

    # x-axis labels for the subplots on J_m_E, J_m_I, J_s_E, and J_s_I
    x_keys_J = ['J_m_E_diff', 'J_m_I_diff', 'J_s_E_diff', 'J_s_I_diff']
    x_labels_J = [
    r'$\Delta (J^\text{mid}_{E \rightarrow E} + J^\text{mid}_{E \rightarrow I})$',
    r'$\Delta (J^\text{mid}_{I \rightarrow I} + J^\text{mid}_{I \rightarrow E})$',
    r'$\Delta (J^\text{sup}_{E \rightarrow E} + J^\text{sup}_{E \rightarrow I})$',
    r'$\Delta (J^\text{sup}_{I \rightarrow I} + J^\text{sup}_{I \rightarrow E})$']
    E_indices = [0,2]

    # Plot the correlation between offset_th_diff and the combination of the J_m_E_diff, J_m_I_diff, J_s_E_diff, and J_s_I_diff
    for i in range(4):
        # Create lmplot for each pair of variables
        if i in E_indices:
            sns.regplot(x=x_keys_J[i], y='offset_th_diff', data=data, ax=axes_flat[i], ci=95, color='red', 
                line_kws={'color':'darkred'}, scatter_kws={'alpha':0.3, 'color':'red'})
        else:
            sns.regplot(x=x_keys_J[i], y='offset_th_diff', data=data, ax=axes_flat[i], ci=95, color='blue', 
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
    data['offset_staircase_improve'] = -1*data['offset_staircase_diff']
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

    ########## Plot the correlation between offset_th_diff and the combination of the f_E_diff, f_I_diff, c_E_diff, and c_I_diff ##########
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axes_flat = axes.flatten()

    # x-axis labels
    x_labels_fc = ['f_E_diff','f_I_diff', 'c_E_diff', 'c_I_diff']
    E_indices = [0,2]
    colors = ['brown', 'purple', 'orange', 'green']
    linecolors = ['#8B4513', '#800080', '#FF8C00',  '#006400']  # Approximate dark versions of purple, green, orange, and brown
    axes_indices = [0,0,1,1]
    for i in range(len(x_labels_fc)):
        # Create lmplot for each pair of variables
        # set colors to purple, green, orange and brown for the different indices
        sns.regplot(x=x_labels_fc[i], y='offset_th_diff', data=data, ax=axes_flat[axes_indices[i]], ci=95, color=colors[i],
            line_kws={'color':linecolors[i]}, scatter_kws={'alpha':0.3, 'color':colors[i]})
        # Calculate the Pearson correlation coefficient and the p-value
        corr = offset_pars_corr[4+i]['corr']
        p_value = offset_pars_corr[4+i]['p_value']
        print('Correlation between offset_th_diff and', x_labels_fc[i], 'is', corr, 'with p-value', p_value)
        # display corr and p-value at the left bottom of the figure
        # axes_flat[i % 2].text(0.05, 0.05, f'Corr: {corr:.2f}, p-val: {p_value:.2f}', transform=axes_flat[i % 2].transAxes, fontsize=10)    
        # Close the lmplot's figure to prevent overlapping
        axes_flat[axes_indices[i]].set_title( f'Corr: {corr:.2f}, p-val: {p_value:.2f}')

    # Adjust layout and save + close the plot
    plt.tight_layout()
    plt.savefig(folder + "/figures/Offset_corr_f_c.png")
    plt.close()
    plt.clf()

    # plot MVPA_pars_corr for each ori_ind on the same plot but different subplots with scatter and different colors for each parameter
    x= numpy.arange(1,13)
    x_labels = ['offset','J_m_E', 'J_m_I', 'J_s_E', 'J_s_I', 'm_f_E', 'm_f_I', 'm_c_E', 'm_c_I', 's_f_E', 's_f_I', 's_c_E', 's_c_I']
    plt.scatter(x, MVPA_corrs[0]['corr'])
    plt.scatter(x, MVPA_corrs[1]['corr'])
    plt.scatter(x, MVPA_corrs[2]['corr'])
    plt.xticks(x, x_labels)
    plt.legend(['55', '125', '0'])
    plt.title('Correlation of MVPA scores with parameter differences')
    #save the plot
    plt.savefig(folder + '/MVPA_pars_corr.png')
    plt.close()


def plot_corr_triangle(data,folder_to_save='',filename='corr_triangle.png'):
    '''Plot a triangle with correlation plots in the middle of the edges of the triangle. Data is supposed to be a dictionary with keys corresponding to MVPA results and relative parameter changes and offset changes.'''
    # Get keys and values
    keys = data.keys()
    labels = ['rel change in ' + keys[0], 'rel change in '+keys[1], 'rel change in ' + keys[2]]

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