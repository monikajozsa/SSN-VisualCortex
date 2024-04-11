import pandas as pd
import numpy
import time
import os

from training import batch_loss_ori_discr, generate_noise, mean_training_task_acc_test, offset_at_baseline_acc
from util import load_parameters
from util_gabor import init_untrained_pars
from util import create_grating_training
from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    readout_pars,
    ssn_pars,
    ssn_layer_pars,
    conv_pars,
    training_pars,
    loss_pars,
    pretrain_pars # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
)
from visualization import calculate_relative_change

def corr_from_csvs(folder,num_trainings, batch_size = 100):
    '''read CSV files and calculate the correlation between the changes of accuracy and J (J_II and J_EI are summed up and J_EE and J_IE are summed up) for each file'''
    file_pattern = folder + '/results_{}'

    # Initialize variables to store results in
    offsets = numpy.zeros((num_trainings,2))
    offset_th = numpy.zeros((num_trainings,2))
    offset_diff = numpy.zeros(num_trainings)
    offset_th_diff = numpy.zeros(num_trainings)
    
    J_m_diff = numpy.zeros((num_trainings,4))
    J_s_diff = numpy.zeros((num_trainings,4))
    f_diff = numpy.zeros((num_trainings,2))
    c_diff = numpy.zeros((num_trainings,2))
    # Initialize the test offset vector for the threshold calculation
    test_offset_vec = numpy.array([1, 2, 3, 4, 6]) 

    start_time = time.time()
    for i in range(num_trainings):
        # Construct the file name
        file_name = file_pattern.format(i) + '.csv'
        orimap_filename = folder + '/orimap_{}.npy'.format(i)
        
        # Read the file
        df = pd.read_csv(file_name)

        # Calculate the J differences (J_m_EE	J_m_EI	J_m_IE	J_m_II	J_s_EE	J_s_EI	J_s_IE	J_s_II) at start and end of training
        relative_changes, training_start, training_end,_,_=calculate_relative_change(df)
        J_m_diff[i,:] = relative_changes[0:4,1] *100
        J_s_diff[i,:] = relative_changes[4:8,1] *100
        c_diff[i,:] = relative_changes[8:10,1] *100
        f_diff[i,:] = relative_changes[10:12,1] *100
        
        # Check if offset_th.csv is already present
        if 'offset_th.csv' in os.listdir(folder):
            offset_th = numpy.loadtxt(folder + '/offset_th.csv', delimiter=',')
            offset_th_diff = -(offset_th[:,0] - offset_th[:,1]) / offset_th[:,0]
        else:
            # Calculate the offset difference at start and end of training
            offsets[i,:] = numpy.array([df['offset'][training_start],df['offset'][training_end]])
            offset_diff[i] = -(offsets[i,1] - offsets[i,0]) / offsets[i,0] *100

            # get accuracy metrics for training_start
            loaded_orimap =  numpy.load(orimap_filename)
            untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                            loss_pars, training_pars, pretrain_pars, readout_pars, None, orimap_loaded=loaded_orimap)
            untrained_pars.pretrain_pars.is_on = False
            
            trained_pars_stage1, trained_pars_stage2, _ = load_parameters(file_name, iloc_ind = training_start)
            acc_mean, _, _ = mean_training_task_acc_test(trained_pars_stage2, trained_pars_stage1, untrained_pars, jit_on=True, offset_vec=test_offset_vec, sample_size = 1 )
            offset_th[i,0] = offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec,  baseline_acc= 0.85)[0]

            trained_pars_stage1, trained_pars_stage2, _ = load_parameters(file_name, iloc_ind = training_end)
            acc_mean, _, _ = mean_training_task_acc_test(trained_pars_stage2, trained_pars_stage1, untrained_pars, jit_on=True, offset_vec=test_offset_vec )
            offset_th[i,1] = offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= 0.85)[0]
            offset_th_diff[i] = -(offset_th[i,1] - offset_th[i,0]) / offset_th[i,0] *100
            
        print('Finished reading file', i, 'time elapsed:', time.time() - start_time)

    # save out offset_th in a csv
    numpy.savetxt(folder + '/offset_th.csv', offset_th, delimiter=',')

    # Check if the offset difference is less valid (180 is a default value for when offset_th is not found within range)    
    mesh_offset_th=numpy.sum(offset_th, axis=1)<180
    offset_th = offset_th[mesh_offset_th,:]
    offset_th_diff = offset_th_diff[mesh_offset_th]
    J_m_diff = J_m_diff[mesh_offset_th,:]
    J_s_diff = J_s_diff[mesh_offset_th,:]
    offset_diff = offset_diff[mesh_offset_th]
    f_diff = f_diff[mesh_offset_th,:]
    c_diff = c_diff[mesh_offset_th,:]
    
    print('Number of samples:', sum(mesh_offset_th))
    
    return offset_diff, offset_th_diff, J_m_diff, J_s_diff, f_diff, c_diff

folder='results/Apr10_v1'
N_training=50
offset_diff, offset_th_diff, J_m_diff, J_s_diff, f_diff, c_diff = corr_from_csvs(folder, N_training)

import scipy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Convert relative parameter differences to pandas DataFrame
data = pd.DataFrame({'offset_th_diff': offset_th_diff, 'J_m_EE_diff': J_m_diff[:, 0], 'J_m_IE_diff': J_m_diff[:, 1], 'J_m_EI_diff': J_m_diff[:, 2], 'J_m_II_diff': J_m_diff[:, 3], 'J_s_EE_diff': J_s_diff[:, 0], 'J_s_IE_diff': J_s_diff[:, 1], 'J_s_EI_diff': J_s_diff[:, 2], 'J_s_II_diff': J_s_diff[:, 3], 'f_E_diff': f_diff[:, 0], 'f_I_diff': f_diff[:, 1], 'c_E_diff': c_diff[:, 0], 'c_I_diff': c_diff[:, 1]})

##################### Plot 1: Correlate offset_th_diff with the J_m_EE_diff, J_m_IE_diff, J_m_EI_diff, J_m_II_diff, J_s_EE_diff, J_s_IE_diff, J_s_EI_diff, J_s_II_diff #####################
# Create a 2x4 subplot grid
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
axes_flat = axes.flatten()

# x-axis labels
x_labels = ['J_m_EE_diff', 'J_m_IE_diff', 'J_m_EI_diff', 'J_m_II_diff', 'J_s_EE_diff', 'J_s_IE_diff', 'J_s_EI_diff', 'J_s_II_diff']
E_indices = [0,1,4,5]

for i in range(8):
    # Create lmplot for each pair of variables
    if i in E_indices:
        sns.regplot(x=x_labels[i], y='offset_th_diff', data=data, ax=axes_flat[i], ci=95, color='red', 
            line_kws={'color':'darkred'}, scatter_kws={'alpha':0.3, 'color':'red'})
    else:
        sns.regplot(x=x_labels[i], y='offset_th_diff', data=data, ax=axes_flat[i], ci=95, color='blue', 
            line_kws={'color':'darkblue'}, scatter_kws={'alpha':0.3, 'color':'blue'})
    # Calculate the Pearson correlation coefficient and the p-value
    corr, p_value = scipy.stats.pearsonr(data['offset_th_diff'], data[x_labels[i]])
    
    # Close the lmplot's figure to prevent overlapping
    axes_flat[i].set_title( f'Corr: {corr:.2f}, p-val: {p_value:.2f}')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig(folder + "/figures/Offset_corr_Jall.png")
plt.close()

# Create a boxplot for the J_m and J_s differences separately
J_box_colors = ['tab:red','tab:red','tab:blue','tab:blue']
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 8))

bp = axes[0].boxplot(data[['J_m_EE_diff', 'J_m_IE_diff', 'J_m_EI_diff', 'J_m_II_diff']], labels=['J_m_EE_diff', 'J_m_IE_diff', 'J_m_EI_diff', 'J_m_II_diff'], patch_artist=True, showfliers=False)
for box, color in zip(bp['boxes'], J_box_colors):
    box.set_facecolor(color)
axes[0].axhline(y=0, color='black', linestyle='--')

bp = axes[1].boxplot(data[['J_s_EE_diff', 'J_s_IE_diff', 'J_s_EI_diff', 'J_s_II_diff']],labels= ['J_s_EE_diff', 'J_s_IE_diff', 'J_s_EI_diff', 'J_s_II_diff'], patch_artist=True, showfliers=False)
for box, color in zip(bp['boxes'], J_box_colors):
    box.set_facecolor(color)
axes[1].axhline(y=0, color='black', linestyle='--')

plt.savefig(folder + "/figures/J_changes.png")
plt.close()

##################### Plot 2: Correlate offset_th_diff with the combintation of the J_m_EE and J_m_IE, J_m_EI and J_m_II, etc. #####################
## Create a 2x2 subplot grid
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
axes_flat = axes.flatten()

# x-axis labels
x_labels2 = ['J_m_E_diff', 'J_m_I_diff', 'J_s_E_diff', 'J_s_I_diff']
E_indices = [0,2]

# combine the J_m_EE and J_m_IE, J_m_EI and J_m_II, J_s_EE and J_s_IE, J_s_EI and J_s_II and add them to the data
data['J_m_E_diff'] = J_m_diff[:, 0] + J_m_diff[:, 1]
data['J_m_I_diff'] = J_m_diff[:, 2] + J_m_diff[:, 3]
data['J_s_E_diff'] = J_s_diff[:, 0] + J_s_diff[:, 1]
data['J_s_I_diff'] = J_s_diff[:, 2] + J_s_diff[:, 3]

for i in range(4):
    # Create lmplot for each pair of variables
    if i in E_indices:
        sns.regplot(x=x_labels2[i], y='offset_th_diff', data=data, ax=axes_flat[i], ci=95, color='red', 
            line_kws={'color':'darkred'}, scatter_kws={'alpha':0.3, 'color':'red'})
    else:
        sns.regplot(x=x_labels2[i], y='offset_th_diff', data=data, ax=axes_flat[i], ci=95, color='blue', 
            line_kws={'color':'darkblue'}, scatter_kws={'alpha':0.3, 'color':'blue'})
    # Calculate the Pearson correlation coefficient and the p-value
    corr, p_value = scipy.stats.pearsonr(data['offset_th_diff'], data[x_labels2[i]])
    
    # Close the lmplot's figure to prevent overlapping
    axes_flat[i].set_title( f'Corr: {corr:.2f}, p-val: {p_value:.2f}')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig(folder + "/figures/Offset_corr_J_IE.png")
plt.close()

##################### Plot 3: Correlate offset_th_diff with the ratio J_m_I_diff/J_m_E_diff and J_s_I_diff/J_s_E_diff #####################
## Create a 2x2 subplot grid
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 8))

# x-axis labels
x_labels3 = ['J_m_ratio', 'J_s_ratio']

# calculate ratio J_m_I_diff/J_m_E_diff and J_s_I_diff/J_s_E_diff and add them to the data
data['J_m_ratio'] = data['J_m_I_diff'] / data['J_m_E_diff']
data['J_s_ratio'] = data['J_s_I_diff'] / data['J_s_E_diff']

for i in range(2):
    # Create lmplot for each pair of variables
    sns.regplot(x=x_labels3[i], y='offset_th_diff', data=data, ax=axes[i], ci=95)
    # Calculate the Pearson correlation coefficient and the p-value
    corr, p_value = scipy.stats.pearsonr(data['offset_th_diff'], data[x_labels3[i]])
    
    # Close the lmplot's figure to prevent overlapping
    axes[i].set_title( f'Corr: {corr:.2f}, p-val: {p_value:.2f}')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig(folder + "/figures/Offset_corr_Jratio.png")
plt.close()

##################### Plot 4: Correlate offset_th_diff with f_diff and c_diff #####################
## Create a 2x2 subplot grid
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
axes_flat = axes.flatten()

# x-axis labels
x_labels4 = ['f_E_diff','f_I_diff', 'c_E_diff', 'c_I_diff']
E_indices = [0,2]
for i in range(len(x_labels4)):
    # Create lmplot for each pair of variables
    if i in E_indices:
        sns.regplot(x=x_labels4[i], y='offset_th_diff', data=data, ax=axes_flat[i], ci=95, color='red', 
            line_kws={'color':'darkred'}, scatter_kws={'alpha':0.3, 'color':'red'})
    else:
        sns.regplot(x=x_labels4[i], y='offset_th_diff', data=data, ax=axes_flat[i], ci=95, color='blue',
            line_kws={'color':'darkblue'}, scatter_kws={'alpha':0.3, 'color':'blue'})
    # Calculate the Pearson correlation coefficient and the p-value
    corr, p_value = scipy.stats.pearsonr(data['offset_th_diff'], data[x_labels4[i]])
    
    # Close the lmplot's figure to prevent overlapping
    axes_flat[i].set_title( f'Corr: {corr:.2f}, p-val: {p_value:.2f}')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig(folder + "/figures/Offset_corr_f_c.png")
plt.close()


# Create a boxplot for the f and c differences
box_colors = ['tab:red','tab:blue', 'tab:red','tab:blue']
labels=['f_E_diff', 'f_I_diff', 'c_E_diff', 'c_I_diff']
bp = plt.boxplot(data[labels], labels=labels, patch_artist=True, showfliers=False)
for box, color in zip(bp['boxes'], box_colors):
    box.set_facecolor(color)
plt.axhline(y=0, color='black', linestyle='--')

plt.savefig(folder + "/figures/fc_changes.png")
plt.close()

