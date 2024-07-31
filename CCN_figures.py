import pandas as pd
import jax.numpy as np
import numpy
import os
import time
import matplotlib.pyplot as plt

from training.training_functions import mean_training_task_acc_test, offset_at_baseline_acc
from util import filter_for_run, load_parameters
from training.util_gabor import init_untrained_pars
from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    readout_pars,
    ssn_pars,
    trained_pars,
    conv_pars,
    training_pars,
    loss_pars,
    pretraining_pars, # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
)
from analysis.analysis_functions import load_orientation_map

######### Overwrite parameters #########
#ssn_pars.p_local_s = [1.0, 1.0] # no horizontal connections in the superficial layer
if hasattr(trained_pars, 'J_2x2_s'):
    trained_pars.J_2x2_s = (np.array([[1.82650658, -0.68194475], [2.06815311, -0.5106321]]) * np.pi * 0.774) 
else:
    ssn_pars.J_2x2_s = (np.array([[1.82650658, -0.68194475], [2.06815311, -0.5106321]]) * np.pi * 0.774)
if hasattr(trained_pars, 'J_2x2_m'):
    trained_pars.J_2x2_m = np.array([[2.5, -1.3], [4.7, -2.2]]) * 0.774 
else:
    ssn_pars.J_2x2_m = np.array([[2.5, -1.3], [4.7, -2.2]]) * 0.774
pretraining_pars.ori_dist_int = [10, 20]
pretraining_pars.offset_threshold = [0,6]
pretraining_pars.SGD_steps = 500
pretraining_pars.min_acc_check_ind = 10
training_pars.eta = 2*10e-4
training_pars.first_stage_acc_th = 0.55
loss_pars.lambda_r_mean = 0
stimuli_pars.std = 200.0

start_time_in_main= time.time()
num_training = 50
results_file = os.path.join('results','Apr10_v1','results.csv')
main_folder = os.path.join('results','Apr10_v1')

'''
def stoichiometric_offsets_calc(results_file, num_training, start_time_in_main=start_time_in_main, step_indices=[0,-1], ref_ori=None, orimap_folder=None):
    # This needs to be updated if gE and gI are randomized (read them from init_params file and save them in the filter_pars)!
    pretraining_pars.is_on = False
    results_df = pd.read_csv(results_file)
    stoichiometric_offsets = numpy.zeros((num_training, len(step_indices)))
    acc_mean_all = []
    test_offset_vec = numpy.array([1, 2, 4, 6, 8, 10]) 
    jit_on= True
    if ref_ori is not None:
        stimuli_pars.ref_ori = ref_ori
    for run_index in range(num_training):
        df_i = filter_for_run(results_df, run_index)
        df_i = df_i[df_i['stage']==2]
        orimap_i =  load_orientation_map(orimap_folder, run_index)
        
        untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretraining_pars, readout_pars, orimap_loaded=orimap_i)
        for j in range(len(step_indices)):
            # Find the row that matches the given values
            readout_pars_dict, trained_pars_dict, untrained_pars, _,_ = load_parameters(df_i, iloc_ind = step_indices[j],untrained_pars = untrained_pars)
            acc_mean, _, _ = mean_training_task_acc_test(trained_pars_dict, readout_pars_dict, untrained_pars, jit_on, test_offset_vec, sample_size=10)
            # fit log-linear curve to acc_mean_max and test_offset_vec and find where it crosses baseline_acc=0.794
            stoich_offset = offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc=0.7)
            acc_mean_flipped=1-acc_mean
            #acc_mean_all.append(acc_mean)
            stoich_offset_flipped = offset_at_baseline_acc(acc_mean_flipped, offset_vec=test_offset_vec, baseline_acc=0.7)
            stoichiometric_offsets[run_index,j] = numpy.min([np.array(stoich_offset).item(), np.array(stoich_offset_flipped).item()])
            print('Stoichiometric_offset', stoichiometric_offsets[run_index,j], 'from accuracies', acc_mean, 'for run ', run_index, 'and step', step_indices[j])
        print(f'Finished calculating stoichiometric offsets in {time.time()-start_time_in_main} seconds for run {run_index}')
    
    return stoichiometric_offsets, acc_mean_all

# Save the stoichiometric offsets
stoichiometric_offsets, acc_mean_all = stoichiometric_offsets_calc(results_file, num_training, step_indices=[1,-1], orimap_folder=main_folder)
stoichiometric_offsets_df = pd.DataFrame(stoichiometric_offsets)
stoichiometric_offsets_df.to_csv('stoichiometric_offsets_55.csv')

stoichiometric_offsets_125, acc_mean_all_125 = stoichiometric_offsets_calc(results_file, num_training, step_indices=[1,-1], ref_ori=125, orimap_loaded=main_folder)
stoichiometric_offsets_125_df = pd.DataFrame(stoichiometric_offsets_125)
stoichiometric_offsets_125_df.to_csv('stoichiometric_offsets_125.csv')

######### Load the csv files #########
stoichiometric_offsets_55 = pd.read_csv('stoichiometric_offsets_55.csv', index_col=0).to_numpy()
stoichiometric_offsets_125 = pd.read_csv('stoichiometric_offsets_125.csv', index_col=0).to_numpy()

# barplot of stoichiometric offsets at different stages and orientations
color_pretest = '#F3929A'
color_posttest = '#70BFD9'
colors_bar = [color_pretest, color_posttest]

# Data preparation
include_runs=[]
for i in range(num_training):
    if all(stoichiometric_offsets_55[i,:]<25) and all(stoichiometric_offsets_125[i,:]<25):
        include_runs.append(i)
print(' Number of runs included in the plot:', len(include_runs))
categories = ['Trained', 'Untrained']
values_55 = [np.mean(stoichiometric_offsets_55[include_runs,0]), np.mean(stoichiometric_offsets_55[include_runs,-1])]
errors_55 = [np.std(stoichiometric_offsets_55[include_runs, 0]), np.std(stoichiometric_offsets_55[include_runs, -1])]

values_125 = [np.mean(stoichiometric_offsets_125[include_runs, 0]), np.mean(stoichiometric_offsets_125[include_runs, -1])]
errors_125 = [np.std(stoichiometric_offsets_125[include_runs, 0]), np.std(stoichiometric_offsets_125[include_runs, -1])]

# X locations for the groups
x = np.arange(4)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Plotting the bars
bar_width = 0.7  # Width of the bars

# First group of bars (df_a)
bars_a = ax.bar(x[:2] - bar_width/2, values_55, bar_width, yerr=errors_55, capsize=5, label='Trained \n orientation', color=colors_bar)

# Second group of bars (df_b)
bars_b = ax.bar(x[2:] + bar_width/2, values_125, bar_width, yerr=errors_125, capsize=5, label='Untrained \n orientation', color=colors_bar)
# remove the xticks

ax.set_xticks([0.2, 2.8])
ax.set_xticklabels(['Trained \n orientation', 'Untrained \n orientation'])
ax.set_ylabel('Mean stoichiometric offset')

plt.show()

# Save the stoichiometric offset plot
plt.savefig('stoichiometric_offsets.png')

plt.close()
'''
####################
import seaborn as sns
import scipy
from analysis.analysis_functions import rel_changes_from_csvs
'''
sigma_filter = 2
folder_to_save=os.path.join(main_folder, 'figures')
offset_th_diff, offset_th_diff_125,offset_staircase_diff, J_m_diff, J_s_diff, J_ms_ratio_diff, f_diff, c_diff, mesh_offset_th = rel_changes_from_csvs(main_folder, num_training, 2, mesh_for_valid_offset=False)
data = pd.DataFrame({'offset_th_diff': offset_th_diff, 'offset_th_diff_125': offset_th_diff_125, 'offset_staircase_diff': offset_staircase_diff,  'J_m_EE_diff': J_m_diff[:, 0], 'J_m_IE_diff': J_m_diff[:, 1], 'J_m_EI_diff': J_m_diff[:, 2], 'J_m_II_diff': J_m_diff[:, 3], 'J_s_EE_diff': J_s_diff[:, 0], 'J_s_IE_diff': J_s_diff[:, 1], 'J_s_EI_diff': J_s_diff[:, 2], 'J_s_II_diff': J_s_diff[:, 3], 'f_E_diff': f_diff[:, 0], 'f_I_diff': f_diff[:, 1], 'c_E_diff': c_diff[:, 0], 'c_I_diff': c_diff[:, 1]})

# combine the J_m_EE and J_m_IE, J_m_EI and J_m_II, J_s_EE and J_s_IE, J_s_EI and J_s_II and add them to the data
data['J_m_E_diff'] = J_m_diff[:, 4]
data['J_m_I_diff'] = J_m_diff[:, 5]
data['J_s_E_diff'] = J_s_diff[:, 4]
data['J_s_I_diff'] = J_s_diff[:, 5]
data['J_m_ratio_diff'] = J_m_diff[:, 6]
data['J_s_ratio_diff'] = J_s_diff[:, 6]
data['J_ms_ratio_diff'] = J_ms_ratio_diff[:,0]
data['J_ms_ratio_pre'] = J_ms_ratio_diff[:,1]
data['J_ms_ratio_post'] = J_ms_ratio_diff[:,2]
data['offset_staircase_diff']=-1*data['offset_staircase_diff']

data_sup_55 = pd.DataFrame({
    'JsI/JsE': data['J_s_ratio_diff']*100,
    'offset_th': data['offset_staircase_diff']
})
data_mid_55 = pd.DataFrame({
    'JmI/JmE': data['J_m_ratio_diff']*100,
    'offset_th': data['offset_staircase_diff']
})
data_midsup_55 = pd.DataFrame({
    'JmsI/JmsE': data['J_ms_ratio_diff']*100,
    'JmsI/JmsE_pre': data['J_ms_ratio_pre']*100,
    'JmsI/JmsE_post': data['J_ms_ratio_post']*100,
    'offset_th': data['offset_staircase_diff']
})
# save data_midsup_55
#data_midsup_55.to_csv(main_folder +'/data_midsup_55.csv')

# Plot the correlation between the relative changes in JmsI/JmsE and the offset threshold
plt.figure(figsize=(4, 4))
sns.regplot(x='JmsI/JmsE', y='offset_th', data=data_midsup_55)
corr, p_val = scipy.stats.pearsonr(data_midsup_55['JmsI/JmsE'], data_midsup_55['offset_th'])
x_label = r'$\Delta (J^{\text{tot}}_{I} /J^{\text{tot}}_{E})$'
y_label = r'$\Delta \text{ offset threshold (deg)}$'
plt.text(0.75, 0.1, f'corr={corr:.2f}\n p={p_val:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
plt.xlabel(x_label, fontsize=20)
plt.ylabel(y_label, fontsize=20)
plt.tight_layout()
plt.savefig(main_folder + f"/figures/corr_JmsIperJmsE_offset_midsup_55.png")
plt.show()
plt.close()


sns.regplot(x='JsI/JsE', y='offset_th', data=data_sup_55)
corr, p_val = scipy.stats.pearsonr(data_sup_55['JsI/JsE'], data_sup_55['offset_th'])
x_label = r'$\Delta (J^{\text{tot}}_{I} /J^{\text{tot}}_{E})$'
y_label = r'$\Delta \text{ offset threshold (deg)}$'
# Add correlation and p-value to the right bottom of the plot
plt.text(0.75, 0.1, f'corr={corr:.2f}\n p={p_val:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)

plt.xlabel(x_label, fontsize=20)
plt.ylabel(y_label, fontsize=20)
plt.tight_layout()
plt.savefig(main_folder + f"/figures/corr_JsIperJsE_offset_sup_55.png")
plt.show()

plt.close()

sns.regplot(x='JmI/JmE', y='offset_th', data=data_mid_55)
corr, p_val = scipy.stats.pearsonr(data_mid_55['JmI/JmE'], data_mid_55['offset_th'])
x_label = r'$\Delta (J^{\text{tot}}_{I} /J^{\text{tot}}_{E})$'
y_label = r'$\Delta \text{ offset threshold (deg)}$'
plt.text(0.75, 0.1, f'corr={corr:.2f} \n p={p_val:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
plt.xlabel(x_label, fontsize=20)
plt.ylabel(y_label, fontsize=20)
plt.tight_layout()
plt.savefig(main_folder + f"/figures/corr_JIperJE_offset_mid_55.png")
plt.show()

plt.close()
'''

# load data_midsup_55
data_midsup_55 = pd.read_csv(main_folder +'/data_midsup_55.csv', index_col=0)

# Barplot of the JmsI/JmsE before and after training
color_pretest = '#F3929A'
color_posttest = '#70BFD9'
colors_bar = [color_pretest, color_posttest]
darker_colors = ['#91575C', '#385F6C']
# Create the figure and axis
fig, ax = plt.subplots(figsize=(4, 4))

# Plotting the bars
data_midsup_55['JmsI/JmsE_pre'] = data_midsup_55['JmsI/JmsE_pre']/100
data_midsup_55['JmsI/JmsE_post'] = data_midsup_55['JmsI/JmsE_post']/100
bar_width = 0.4  # Width of the bars
std_Jms_ratio = [numpy.std(data_midsup_55['JmsI/JmsE_pre'], axis=0), numpy.std(data_midsup_55['JmsI/JmsE_post'], axis=0)]
mean_Jms_ratio = [numpy.mean(data_midsup_55['JmsI/JmsE_pre'], axis=0), numpy.mean(data_midsup_55['JmsI/JmsE_post'], axis=0)]
#ax.bar(np.arange(2), mean_Jms_ratio, bar_width, yerr=std_Jms_ratio, capsize=5, color=colors_bar)
# Plot the bar chart with error bars, applying colors individually
bars = []
for i, (mean, std, color) in enumerate(zip(mean_Jms_ratio, std_Jms_ratio, colors_bar)):
    bars.append(ax.bar(i, mean, bar_width, yerr=std, capsize=5, 
                       color=color, ecolor=color, error_kw=dict(ecolor=darker_colors[i], alpha=0.9, lw=2, capsize=5, capthick=2)))
ax.set_xticks([0.0, 1.0])
ax.set_xticklabels(['Pretrest', 'Posttest'], fontsize=20)
ax.set_ylabel(r'$J^{\text{tot}}_{I} /J^{\text{tot}}_{E}$', fontsize=20)
ax.tick_params(axis='y', which='both', labelsize=18)
ax.yaxis.set_tick_params(width=2, length=8)  # Customize tick size
plt.tight_layout()
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0.3, 0.55)

plt.savefig(main_folder + f"/figures/JmsIperJmsE_barplot.png")
plt.close()

# Set all the text sizes to 20
############################ tc_features ############################
import statsmodels.api as sm
from analysis.visualization import axes_format
def plot_tc_features_simplified(results_dir):
    def shift_x_data(x_data, indices, shift_value=90):
        # Shift x_data by shift_value and center it around 0 (that is, around shift_value)
        x_data_shifted = x_data[:, indices].flatten() - shift_value
        x_data_shifted = numpy.where(x_data_shifted > 90, x_data_shifted - 180, x_data_shifted)
        x_data_shifted = numpy.where(x_data_shifted < -90, x_data_shifted + 180, x_data_shifted)
        return x_data_shifted
    
    # load data from file
    saved_keys = ['fwhm_1', 'fwhm_2', 'slopediff_55_1', 'preforis_0', 'preforis_1']
    data = {}
    for key in saved_keys:
        data[key] = numpy.loadtxt(results_dir + f'/{key}.csv', delimiter=',')


    ############## Plots about changes before vs after training and pretraining (per layer and per centered or all) ##############
             
    # Define indices for each group of cells
    E_sup = 648+numpy.linspace(0, 80, 81).astype(int) 
    I_sup = 648+numpy.linspace(81, 161, 81).astype(int) 
    E_mid_array = numpy.linspace(0, 647, 648).round().reshape(4, 2, 81).astype(int)
    E_mid = E_mid_array[:,0,:].ravel().astype(int)
    I_mid_array = numpy.linspace(0, 647, 648).round().reshape(4, 2, 81).astype(int)
    I_mid = I_mid_array[:,1,:].ravel().astype(int)
    indices = [E_sup, I_sup, E_mid, I_mid]
    
    # Create legends for the plot
    cmap = plt.get_cmap('rainbow')
    colors = numpy.flip(cmap(numpy.linspace(0,1, 8)), axis = 0)
    
    #############################################################################
    ######### Schoups-style scatter plots - coloring based on cell type #########
    #############################################################################
    fs_text = 40
    fs_ticks = 30
    
    # Scatter slope, where x-axis is orientation and y-axis is the change in slope before and after training
    stage_labels = ['pretrain', 'train']
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))         
    ##### Plot fwhm before vs after training for E_sup and E_mid #####
    # add a little jitter to x and y to avoid overlapping points
    x = data[f'fwhm_1']+numpy.random.normal(0, 0.5, data[f'fwhm_1'].shape) 
    y = data[f'fwhm_2']+numpy.random.normal(0, 0.5, data[f'fwhm_2'].shape) 
    ax = axs[1]
    # Scatter on a log scale
    ax.scatter(x[:,E_mid], y[:,E_mid], s=30, alpha=0.3, color='peru')
    ax.scatter(x[:,E_sup], y[:,E_sup], s=30, alpha=0.5, color='seagreen')
    
    #ax.scatter(x[:,E_mid], y[:,E_mid], s=30, alpha=0.5, color='orange')
    #ax.scatter(x[:,E_sup], y[:,E_sup], s=30, alpha=0.5, color='red')
    # Add local averaging lines
    #lowess_E_mid = sm.nonparametric.lowess(y[:,E_mid].flatten(), x[:,E_mid].flatten(), frac=0.15)
    #ax.plot(lowess_E_mid[:, 0], lowess_E_mid[:, 1], color='saddlebrown', linewidth=8)
    #lowess_E_sup = sm.nonparametric.lowess(y[:,E_sup].flatten(), x[:,E_sup].flatten(), frac=0.15)
    #ax.plot(lowess_E_sup[:, 0], lowess_E_sup[:, 1], color='darkred', linewidth=8)

    xpoints = ypoints = ax.get_xlim()
    ax.plot(xpoints, ypoints, color='black', linewidth=4)
    ax.set_xlabel('Pre training')
    ax.set_ylabel('Post training')
    
    # Format axes
    axes_format(ax, fs_ticks)
    
    ##### Plot orientation vs slope #####
    # Middle layer scatter plots
    y_E_mid= data[f'slopediff_55_1'][:,E_mid].flatten()
    x_E_mid = shift_x_data(data[f'preforis_1'], E_mid, shift_value=55)
    axs[0].scatter(x_E_mid, y_E_mid, s=30, alpha=0.5, color='peru')
    # Superficial layer scatter plots
    y_E_sup= data[f'slopediff_55_1'][:,E_sup].flatten()
    x_E_sup= shift_x_data(data[f'preforis_1'], E_sup, shift_value=55)
    axs[0].scatter(x_E_sup, y_E_sup, s=30, alpha=0.7, color='seagreen')

    # Line plots for both layers: define x and y values and shift x to have 0 in its center
    lowess_E_mid = sm.nonparametric.lowess(y_E_mid, x_E_mid, frac=0.15)  # Example with frac=0.2 for more local averaging
    axs[0].plot(lowess_E_mid[:, 0], lowess_E_mid[:, 1], color='saddlebrown', linewidth=8)
    lowess_E_sup = sm.nonparametric.lowess(y_E_sup, x_E_sup, frac=0.15)
    axs[0].plot(lowess_E_sup[:, 0], lowess_E_sup[:, 1], color='darkgreen', linewidth=8)
    # Add labels to the lines
    axs[0].text(65, 0.8, 'mid.',color='saddlebrown', fontsize=fs_text)
    axs[0].text(65, 0.7, 'sup.',color='darkgreen', fontsize=fs_text)

    axes_format(axs[0], fs_ticks)
    
    axs[1].set_title('Full width \n at half maximum (deg.)', fontsize=fs_text)
    axs[1].set_xlabel('Pre FWHM', fontsize=fs_text, labelpad=20)
    axs[1].set_ylabel('Post FWHM', fontsize=fs_text)
    
    axs[0].set_title('Tuning curve slope\n'+ 'at trained orientation', fontsize=fs_text)
    axs[0].set_xlabel('pref. ori - trained ori', fontsize=fs_text, labelpad=20)
    axs[0].set_ylabel(r'$\Delta$ slope', fontsize=fs_text)

    plt.tight_layout(w_pad=10, h_pad=7)
    fig.savefig(results_dir + f"/figures/tc_features_simplified.png", bbox_inches='tight')
    plt.close()

start_time = time.time()
results_dir = os.path.join('results','Apr10_v1')
plot_tc_features_simplified(results_dir)
print(f'Finished plotting tuning curve features in {time.time()-start_time} seconds')
