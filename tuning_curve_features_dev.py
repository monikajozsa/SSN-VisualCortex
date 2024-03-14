## This code will plot key features of the tuning curves but it is under development
## This code needs testing!

import jax.numpy as np
import numpy
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def norm_slope(file_name, ori_list=numpy.arange(0,180,6), expand_dims=False):
    
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

def plot_pre_post_scatter(ax, x_axis, y_axis, orientations, indices_to_plot,N_runs, title, colors):
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
        '''
        for idx in indices_to_plot:
            # Select bin and colour based on orientation
            if np.abs(orientations[run_ind, idx]) < 4:
                colour = colors[0]
            elif np.abs(orientations[run_ind, idx]) > 50:
                colour = colors[-1]
            else:
                colour = colors[int(1 + np.floor((np.abs(orientations[run_ind, idx]) - 4) / 8))]
            ax.scatter(x_axis[run_ind, idx], y_axis[run_ind, idx], color=colour, s=10, alpha=0.7)
        '''
    
    # Plot x = y line
    xpoints = ypoints = ax.get_xlim()
    ax.plot(xpoints, ypoints, linestyle='--', color='gold', linewidth=1)
    ax.set_xlabel('Pre training')
    ax.set_ylabel('Post training')
    ax.set_title(title)
  

start_time = time.time()

#Load response matrices - use tuning_curves
results_dir= 'C:/Users/jozsa/Dropbox (Cambridge University)/Postdoc 2023-2024/Clara-Monika/results_Mar06_v6'
N_runs = 7
d_theta=6
ori_list=np.arange(0,180,d_theta)

# Initialize dictionaries to store the data arrays
data = {
    'norm_slope_prepre': [],
    'norm_slope_postpre': [],
    'norm_slope_post': [],
    'norm_slope_train_only_pre': [],
    'norm_slope_train_only_post': [],
    'fwhm_prepre': [],
    'fwhm_postpre': [],
    'fwhm_post': [],
    'fwhm_train_only_pre': [],
    'fwhm_train_only_post': [],
    'orientations_prepre': [],
    'orientations_postpre': [],
    'orientations_post': [],
    'orientations_train_only_pre': [],
    'orientations_train_only_post': []
}

for i in range(N_runs):
    # File names associated with each data type
    file_names = {
        'prepre': results_dir + f'tc_prepre_{i}.csv',
        'postpre':  results_dir + f'tc_postpre_{i}.csv',
        'post': results_dir + f'tc_post_{i}.csv',
        'train_only_pre': results_dir + f'train_only/tc_train_only_{i}.csv',
        'train_only_post': results_dir + f'train_only/tc_train_only_post_{i}.csv',
    }

    # Loop through each file name to process and store data
    for key, file_name in file_names.items():
        # Load data from file
        slope, fwhm, orientations = norm_slope(file_name, expand_dims=True)
        
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

#Create saving directory
save_dir='results/Mar06_v6'
# Create legend
patches = []
cmap = plt.get_cmap('rainbow')
colors = numpy.flip(cmap(numpy.linspace(0,1, 8)), axis = 0)
bins = ['0-4', '4-12', '12-20', '20-28', '28-36', '36-44', '44-50', '+50']
for j in range(0,len(colors)):
    patches.append(mpatches.Patch(color=colors[j], label=bins[j]))

# Plot slope
fig, axs = plt.subplots(4, 3, figsize=(15, 20))  # 2 rows, 3 columns
for j in range(len(indices)):
    title = 'Pretraining ' + labels[j]
    plot_pre_post_scatter(axs[j,0], data['norm_slope_prepre'] , data['norm_slope_postpre'] ,  data['orientations_prepre'],  indices[j],N_runs, title = title,colors=colors)

    title = 'Training, ' + labels[j]
    plot_pre_post_scatter(axs[j,1], data['norm_slope_postpre'] , data['norm_slope_post'] ,  data['orientations_postpre'], indices[j],N_runs, title = title,colors=colors)
    
    title = 'Training_only ' + labels[j]
    plot_pre_post_scatter(axs[j,2],  data['norm_slope_train_only_pre'] , data['norm_slope_train_only_post'] , data['orientations_train_only_pre'], indices[j], N_runs, title = title,colors=colors)
    print(time.time()-start_time)

axs[j,2].legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1), title='Pref ori - train ori')
plt.tight_layout()
fig.savefig("results/Mar06_v6/figures/tc_slope.png")

# Plot full-width-half-maximum
fig, axs = plt.subplots(4, 3, figsize=(15, 20))  # 2 rows, 3 columns
for j in range(len(indices)):
    title = 'Pretraining ' + labels[j]
    plot_pre_post_scatter(axs[j,0],  data['fwhm_prepre'] ,  data['fwhm_postpre'] ,  data['orientations_prepre'], indices[j], N_runs, title = title,colors=colors)

    title = 'Training, ' + labels[j] 
    plot_pre_post_scatter(axs[j,1], data['fwhm_postpre'] , data['fwhm_post'] ,data['orientations_postpre'], indices[j], N_runs,title = title,colors=colors)
    
    title = 'Training_only ' + labels[j] 
    plot_pre_post_scatter(axs[j,2],  data['fwhm_train_only_pre'] , data['fwhm_train_only_post'] ,data['orientations_train_only_pre'], indices[j], N_runs, title = title,colors=colors)
    print(time.time()-start_time)

axs[j,2].legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1), title='Pref ori - train ori')
plt.tight_layout()
fig.savefig("results/Mar06_v6/figures/tc_fwhm.png")

  
'''
# Schoups plot
#Bin neurons into preferred orientations
bins = ['0-4', '4-12', '12-20', '20-28', '28-36', '36-44', '44-50', '+50']
labels = []
orientations = norm_oris_sup
for idx in E_indices_sup:
        #Select bin and colour
    if np.abs(orientations[idx]) <4:
        labels.append(0)
    elif np.abs(orientations[idx]) >50:
        labels.append(7)
    else:
        labels.append(int(1+np.floor((np.abs(orientations[idx]) -4)/8) ))
labels = np.asarray(labels)
'''