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

from analysis import rel_changes, tc_features, MVPA_param_offset_correlations

plt.rcParams['xtick.labelsize'] = 12 # Set the size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12 # Set the size for y-axis tick labels

########### Plotting functions ##############
def boxplots_from_csvs(folder, save_folder, plot_filename = None, num_time_inds = 3, num_training = None):
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
            relative_changes, time_inds = rel_changes(df, num_time_inds)
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
        x_positions = numpy.random.normal(i, 0.04, size=len(group_data))
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
    #titles= ['Jm changes, {} runs'.format(numFiles),'Js changes, {} runs'.format(numFiles),'c changes, {} runs'.format(numFiles), 'f changes, {} runs'.format(numFiles)]
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
                #axes_flat[j*num_groups+i].set_title(titles[i])
                #axes_flat[j*num_groups+i].set_ylabel('rel change in %')
        
    plt.tight_layout()
    
    if plot_filename is not None:
        full_path = save_folder + '/' + plot_filename + ".png"
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
            ax1.plot(range(num_oris), df_tc_pre_pretrain.iloc[:,selected_mid_col_inds[i]], label='initial',linewidth=2)
            ax2.plot(range(num_oris), df_tc_pre_pretrain.iloc[:,selected_sup_col_inds[i]], label='initial',linewidth=2)
            if pretrain_ison:
                ax1.plot(range(num_oris), df_tc_post_pretrain.iloc[:,selected_mid_col_inds[i]], label='post-pretraining',linewidth=2)
                ax2.plot(range(num_oris), df_tc_post_pretrain.iloc[:,selected_sup_col_inds[i]], label='post-pretraining',linewidth=2)
            ax1.plot(range(num_oris), df_tc_post_train.iloc[:,selected_mid_col_inds[i]], label='post-training',linewidth=2)
            ax2.plot(range(num_oris), df_tc_post_train.iloc[:,selected_sup_col_inds[i]], label='post-training',linewidth=2)
            
            ax1.set_title(f'Middle layer cell {i}', fontsize=20)
            ax2.set_title(f'Superficial layer, cell {i}', fontsize=20)
    ax2.legend(loc='upper left', fontsize=20)
    
    # Save plot
    if folder_to_save is not None:
        fig.savefig(os.path.join(folder_to_save,'tc_' + train_only_str + 'fig.png'))
    else:
        fig.savefig(os.path.join(folder_path,'tc_' + train_only_str + 'fig.png'))
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
  

def plot_tc_features(results_dir, num_training, ori_list, train_only_str='', pre_post_scatter_flag=False):

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
                
    # Define indices for each group of cells
    E_sup = 648+numpy.linspace(0, 80, 81).astype(int) 
    I_sup = 648+numpy.linspace(81, 161, 81).astype(int) 
    E_mid = numpy.linspace(0, 647, 648).round().reshape(8, 81, -1)[0:9:2].ravel().astype(int) 
    I_mid = numpy.linspace(0, 647, 648).round().reshape(8, 81, -1)[1:9:2].ravel().astype(int) 
    indices = [E_sup, I_sup, E_mid, I_mid]

    #E_sup_centre = 648+numpy.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)
    #I_sup_centre = (E_sup_centre+81).astype(int)
    #E_mid_centre = numpy.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)
    #I_mid_centre = (E_mid_centre+81).astype(int)
    
    # Create labels for the plot
    labels = ['E_sup','I_sup','E_mid','I_mid']
    # Create legends for the plot
    patches = []
    cmap = plt.get_cmap('rainbow')
    colors = numpy.flip(cmap(numpy.linspace(0,1, 8)), axis = 0)
    bins = ['0-4', '4-12', '12-20', '20-28', '28-36', '36-44', '44-50', '+50']
    for j in range(0,len(colors)):
        patches.append(mpatches.Patch(color=colors[j], label=bins[j]))

    if pre_post_scatter_flag:
        
        # Plot slope
        if train_only_str=='':
            fig, axs = plt.subplots(4, 4, figsize=(15, 20)) 
            for j in range(len(indices)):
                title = 'Slope pretraining ' + labels[j]
                plot_pre_post_scatter(axs[j,0], data['norm_slope_prepre'] , data['norm_slope_postpre'] ,  data['orientations_prepre'],  indices[j],num_training, title = title,colors=colors)

                title = 'Slope training, ' + labels[j]
                plot_pre_post_scatter(axs[j,1], data['norm_slope_postpre'] , data['norm_slope_post'] ,  data['orientations_postpre'], indices[j],num_training, title = title,colors=colors)

                title = 'Fwhm pretraining ' + labels[j]
                plot_pre_post_scatter(axs[j,2],  data['fwhm_prepre'] ,  data['fwhm_postpre'] ,  data['orientations_prepre'], indices[j], num_training, title = title,colors=colors)

                title = 'Fwhm training, ' + labels[j] 
                plot_pre_post_scatter(axs[j,3], data['fwhm_postpre'] , data['fwhm_post'] ,data['orientations_postpre'], indices[j], num_training,title = title,colors=colors)
            axs[j,3].legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1), title='Pref ori - train ori')
        else:
            fig, axs = plt.subplots(4, 2, figsize=(10, 20)) 
            for j in range(len(indices)):    
                title = 'Slope training_only ' + labels[j]
                plot_pre_post_scatter(axs[j,0],  data['norm_slope_train_only_pre'] , data['norm_slope_train_only_post'] , data['orientations_train_only_pre'], indices[j], num_training, title = title,colors=colors)

                title = 'Fwhm training_only ' + labels[j] 
                plot_pre_post_scatter(axs[j,1],  data['fwhm_train_only_pre'] , data['fwhm_train_only_post'] ,data['orientations_train_only_pre'], indices[j], num_training, title = title,colors=colors)
            axs[j,1].legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1), title='Pref ori - train ori')
        plt.tight_layout()
        if results_dir[-4:]=='only':
            fig.savefig(os.path.dirname(results_dir) + "/figures/tc_features" + train_only_str +".png")
        else:
            fig.savefig(results_dir + "/figures/tc_features" + train_only_str +".png")
        plt.close()

    # Plots for CCN abstract
    else:
        colors = numpy.flip(cmap(numpy.linspace(0,1, 8)), axis = 0)
        fs_text = 40
        fs_ticks = 30
        ############# Plot fwhm before vs after training for E_sup and E_mid #############
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        for j in [0,2]:            
            # add a little jitter to x and y to avoid overlapping points
            x = numpy.random.normal(0, 0.3, data['fwhm_prepre'].shape) + data['fwhm_prepre']
            y = numpy.random.normal(0, 0.3, data['fwhm_post'].shape) + data['fwhm_post']

            plot_pre_post_scatter(axs[abs((2-j))//2,1], x , y ,data['orientations_postpre'], indices[j], num_training, '', colors=None)
            # Format axes
            axes_format(axs[abs((2-j))//2,1], fs_ticks)
        axs[0,1].set_title('Full width \n at half maximum (deg.)', fontsize=fs_text)
        axs[0,1].set_xlabel('')
        axs[1,1].set_xlabel('Pre FWHM', fontsize=fs_text, labelpad=20)
        axs[1,1].set_ylabel('Post FWHM', fontsize=fs_text)
        axs[0,1].set_ylabel('Post FWHM', fontsize=fs_text)
        
        ############# Plot orientation vs slope #############
        # Add slope difference before and after training to the data dictionary
        data['slope_diff'] = data['norm_slope_post'] - data['norm_slope_prepre']
        
        # Scatter slope, where x-axis is orientation and y-axis is the change in slope before and after training
        for j in [0,2]:
            #axes_flat[j].scatter(data['orientations_prepre'][:,indices[j]], (data['norm_slope_post'][:,indices[j]]-data['norm_slope_prepre'][:,indices[j]]), s=20, alpha=0.7)
            
            # Define x and y values for the scatter plot
            x= data['orientations_prepre'][:,indices[j]].flatten()
            #shift x to have 0 in its center (with circular orientation) and 180 at the end and apply the same shift to the slope_diff
            x = numpy.where(x>90, x-180, x)
            y= data['slope_diff'][:,indices[j]].flatten()
            lowess = sm.nonparametric.lowess(y, x, frac=0.15)  # Example with frac=0.2 for more local averaging
            axs[abs((2-j)) // 2,0].scatter(x, y, s=15, alpha=0.7)
            axs[abs((2-j)) // 2,0].plot(lowess[:, 0], lowess[:, 1], color='black', linewidth=3)
            axes_format(axs[abs((2-j)) // 2,0], fs_ticks)
            
        axs[0,0].set_title('Tuning curve slope \n at trained orientation', fontsize=fs_text)
        axs[0,0].set_xlabel('')
        axs[1,0].set_xlabel('pref. ori - trained ori', fontsize=fs_text, labelpad=20)
        axs[1,0].set_ylabel(r'$\Delta$ slope', fontsize=fs_text)
        axs[0,0].set_ylabel(r'$\Delta$ slope', fontsize=fs_text)
        plt.tight_layout(w_pad=10, h_pad=7)
        fig.savefig(results_dir + "/figures/tc_features" + train_only_str +".png", bbox_inches='tight')
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


def plot_results_from_csv(
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
    categories_params = ['Jm_EE', 'Jm_IE', 'Jm_EI', 'Jm_II', 'Js_EE', 'Js_IE', 'Js_EI', 'Js_II']
    categories_metrics = [ 'c_E', 'c_I', 'f_E', 'f_I', 'acc', 'offset', 'rm_E', 'rm_I', 'rs_E','rs_I']
    rel_par_changes,_ = rel_changes(df) # 0 is pretraining and 1 is training in the second dimensions
    for i_train_pretrain in range(2):
        values_params = 100 * rel_par_changes[0:8,i_train_pretrain]
        values_metrics = 100 * rel_par_changes[8:18,i_train_pretrain]

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


def plot_results_from_csvs(folder_path, num_runs=3, num_rnd_cells=5, folder_to_save=None, starting_run=0):
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
            plot_results_from_csv(results_filename,results_fig_filename)

################### CORRELATION ANALYSIS ###################

def plot_correlations(folder, num_training, num_time_inds=3):
    import matplotlib.pyplot as plt
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
