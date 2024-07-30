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
def stoichiometric_offsets_calc(results_file, num_training, start_time_in_main=start_time_in_main, step_indices=[0,-1], ref_ori=None, orimap_loaded=None):
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
        orimap_i = orimap_loaded[orimap_loaded['run_index']==run_index].to_numpy()
        orimap_i_9x9 = np.reshape(orimap_i[0][1:], (9,9))

        untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretraining_pars, readout_pars, orimap_loaded=orimap_i_9x9)
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


num_training = 50
results_file = os.path.join('results','Apr10_v1','results.csv')
orimap_loaded = pd.read_csv(os.path.join('results','Apr10_v1','orimap.csv'))

# Save the stoichiometric offsets
stoichiometric_offsets, acc_mean_all = stoichiometric_offsets_calc(results_file, num_training, step_indices=[1,-1], orimap_loaded=orimap_loaded)
stoichiometric_offsets_df = pd.DataFrame(stoichiometric_offsets)
stoichiometric_offsets_df.to_csv('stoichiometric_offsets_55.csv')

stoichiometric_offsets_125, acc_mean_all_125 = stoichiometric_offsets_calc(results_file, num_training, step_indices=[1,-1], ref_ori=125, orimap_loaded=orimap_loaded)
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
