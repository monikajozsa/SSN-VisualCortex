import pandas as pd
import numpy
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
numpy.random.seed(0)

from training.util_gabor import init_untrained_pars
from analysis.analysis_functions import tuning_curve, SGD_indices_at_stages, tuning_curve, rel_change_for_runs
from util import load_parameters, filter_for_run_and_stage
from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    readout_pars,
    ssn_pars,
    conv_pars,
    training_pars,
    loss_pars,
    pretraining_pars # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on itP
)
from analysis.visualization import plot_results_from_csvs, boxplots_from_csvs, plot_tuning_curves, plot_tc_features, plot_corr_triangle
from analysis.MVPA_Mahal_combined import MVPA_Mahal_from_csv, plot_MVPA

# Checking that pretrain_pars.is_on is on
if not pretraining_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to True in parameters.py to run training with pretraining!')

num_training = 2
final_folder_path = os.path.join('results','Aug06_v1')
g_randomized_flag = False
start_time_in_main= time.time()

results_filename = os.path.join(final_folder_path, f"results.csv")
orimap_filename = os.path.join(final_folder_path, f"orimap.csv")
init_params_filename = os.path.join(final_folder_path, f"initial_parameters.csv")
orimap_loaded = pd.read_csv(orimap_filename)
results_df = pd.read_csv(results_filename)
if g_randomized_flag:
    init_params_df = pd.read_csv(init_params_filename)


######### PLOT RESULTS ON PARAMETERS ############

start_time=time.time()
tc_cells=[10,40,100,130,650,690,740,760]

## Pretraining + training
folder_to_save = os.path.join(final_folder_path, 'figures')
boxplot_file_name = 'boxplot_pretraining'
mahal_file_name = 'Mahal_dist'
num_SGD_inds = 3
sigma_filter = 2

plot_results_from_csvs(final_folder_path, num_training, folder_to_save=folder_to_save)

#########################################################################################
###### If based on the plots from plots_results_from_csvs, some runs are excluded, ######
#### run the following two lines adjusted to the run numbers that should be excluded ####
#########################################################################################
#excluded_run_inds = [0,8,14,16,17,24,28,34,35,36,37,40,42,43,44,45,47,48]
#exclude_runs(final_folder_path, excluded_run_inds)
#num_training=num_training-len(excluded_run_inds)

boxplots_from_csvs(final_folder_path, folder_to_save, boxplot_file_name, num_time_inds = num_SGD_inds, num_training=num_training)

######### CALCULATE MVPA AND PLOT CORRELATIONS ############

MVPA_Mahal_from_csv(final_folder_path, num_training, num_SGD_inds,sigma_filter=sigma_filter,r_noise=True, plot_flag=True, recalc=True)

folder_to_save=os.path.join(final_folder_path, 'figures')
data_rel_changes, _ = rel_change_for_runs(final_folder_path, num_indices=3)
data_rel_changes['staircase_offset']=-data_rel_changes['staircase_offset']

plot_MVPA(final_folder_path + f'/sigmafilt_{sigma_filter}',num_training)

MVPA_scores = numpy.load(final_folder_path + f'/sigmafilt_{sigma_filter}/MVPA_scores.npy') # MVPA_scores - num_trainings x layer x SGD_ind x ori_ind (sup layer = 0)
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

print(f'Finished calculating and plotting MVPA results in {time.time()-start_time_in_main} seconds')

########## CALCULATE TUNING CURVES ############
start_time_in_main = time.time()
tc_ori_list = numpy.arange(0,180,6)

# Define the filename for the tuning curves 
tc_filename = os.path.join(final_folder_path, 'tuning_curves.csv')
# Define the header for the tuning curves
tc_headers = []
tc_headers.append('run_index')
tc_headers.append('training_stage')
# Headers for middle layer cells - order matches the gabor filters
type_str = ['_E_','_I_']
for phase_ind in range(ssn_pars.phases):
    for type_ind in range(2):
        for i in range(grid_pars.gridsize_Nx**2):
            tc_header = 'G'+ str(i+1) + type_str[type_ind] + 'Ph' + str(phase_ind) + '_M'
            tc_headers.append(tc_header)
# Headers for superficial layer cells
for type_ind in range(2):
    for i in range(grid_pars.gridsize_Nx**2):
        tc_header = 'G'+str(i+1) + type_str[type_ind] +'S'
        tc_headers.append(tc_header)

# Loop over the different runs
from analysis.analysis_functions import load_orientation_map
for i in range(0,num_training):
    orimap_i = load_orientation_map(final_folder_path, i)
    df_i = filter_for_run_and_stage(results_df, i)
    SGD_step_inds = SGD_indices_at_stages(df_i, 3)
    if g_randomized_flag:
        g_randomized = dict(g_E = init_params_df['gE'][i], g_I = init_params_df['gI'][i])
    else:
        g_randomized = None

    # Load fixed parameters
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretraining_pars, readout_pars, orimap_loaded=orimap_i, randomize_g = g_randomized)

    # Loop over the different stages (before pretraining, after pretraining, after training) and calculate and save tuning curves
    for stage in range(3):
        trained_pars_stage1, trained_pars_stage2, _, _, _ = load_parameters(df_i, iloc_ind = SGD_step_inds[stage])
        tc_sup, tc_mid = tuning_curve(untrained_pars, trained_pars_stage2, tc_filename, ori_vec=tc_ori_list, training_stage=stage, run_index=i, header=tc_headers)
        tc_headers = False
        
    print(f'Finished calculating tuning curves for training {i} in {time.time()-start_time_in_main} seconds')

######### PLOT TUNING CURVES ############
start_time_in_main = time.time()
plot_tuning_curves(final_folder_path,tc_cells,num_training,folder_to_save)
plot_tc_features(final_folder_path, num_training, tc_ori_list)
print(f'Finished plotting tuning curves and features in {time.time()-start_time_in_main} seconds')