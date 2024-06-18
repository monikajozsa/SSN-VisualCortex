import pandas as pd
import numpy
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
numpy.random.seed(0)

from training.util_gabor import init_untrained_pars
from analysis_functions import tuning_curve, SGD_step_indices, tuning_curve
from util import load_parameters, filter_for_run
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

import jax.numpy as np
from analysis_functions import gabor_tuning
import matplotlib.pyplot as plt
tc_ori_list = numpy.arange(0,180,6)
num_training = 2
final_folder_path = os.path.join('results','Jun18_v0')


start_time_in_main= time.time()

results_filename = os.path.join(final_folder_path, f"results.csv")
orimap_filename = os.path.join(final_folder_path, f"orimap.csv")
orimap_loaded = pd.read_csv(orimap_filename)
df = pd.read_csv(results_filename)

########## Calculate and save tuning curves ############

tc_filename = os.path.join(final_folder_path, 'tuning_curves.csv')
tc_header = []
tc_header.append('run_index')
tc_header.append('training_stage')
# Middle layer cells
for i in range(grid_pars.gridsize_Nx**2):
    for phase_ind in range(ssn_pars.phases):
        for type_ind in range(2):
            cell_id = 1000*(i+1) + 100*phase_ind  + 10*type_ind 
            tc_header.append(str(cell_id))
# Superficial layer cells
for type_ind in range(2):
    for i in range(grid_pars.gridsize_Nx**2):
        cell_id = 1000*(i+1) + 10*type_ind +1
        tc_header.append(str(cell_id))

# Loop over the different runs
for i in range(0,num_training):
    mesh_i = orimap_loaded['run_index']==i
    orimap_i = orimap_loaded[mesh_i][1:]
    df_i = filter_for_run(df, i)
    SGD_step_inds = SGD_step_indices(df_i, 3)

    # Load parameters and calculate (and save) tuning curves
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, orimap_loaded=orimap_i)
    trained_pars_stage1, trained_pars_stage2, offset_last = load_parameters(df_i, iloc_ind = SGD_step_inds[0])
    tc_prepre, _ = tuning_curve(untrained_pars, trained_pars_stage2, tc_filename, ori_vec=tc_ori_list, training_stage=0, run_index=i, header=tc_header)
    tc_header = False
    trained_pars_stage1, trained_pars_stage2, offset_last = load_parameters(df_i, iloc_ind = SGD_step_inds[1], trained_pars_keys=trained_pars_stage2.keys())
    tc_postpre, _ = tuning_curve(untrained_pars, trained_pars_stage2, tc_filename, ori_vec=tc_ori_list, training_stage=1, run_index=i)
    _, trained_pars_stage2, _ = load_parameters(df_i, iloc_ind = SGD_step_inds[2], trained_pars_keys=trained_pars_stage2.keys())
    tc_post, _ = tuning_curve(untrained_pars, trained_pars_stage2, tc_filename, ori_vec=tc_ori_list, training_stage=2, run_index=i)
    
    print(f'Finished calculating tuning curves for training {i} in {time.time()-start_time_in_main} seconds')

######### PLOT RESULTS ############

from visualization import plot_results_from_csvs, boxplots_from_csvs, plot_tuning_curves, plot_tc_features, plot_corr_triangle
from MVPA_Mahal_combined import MVPA_Mahal_from_csv
from analysis_functions import MVPA_param_offset_correlations

start_time=time.time()
tc_cells=[10,40,100,130,650,690,740,760]

## Pretraining + training
folder_to_save = os.path.join(final_folder_path, 'figures')
boxplot_file_name = 'boxplot_pretraining'
mahal_file_name = 'Mahal_dist'
num_SGD_inds = 3
sigma_filter = 2

plot_results_from_csvs(final_folder_path, num_training, folder_to_save=folder_to_save)
boxplots_from_csvs(final_folder_path, folder_to_save, boxplot_file_name, num_time_inds = num_SGD_inds, num_training=num_training)
plot_tc_features(final_folder_path, num_training, tc_ori_list)
plot_tuning_curves(final_folder_path,tc_cells,num_training,folder_to_save, train_only_str='')

MVPA_Mahal_from_csv(final_folder_path, num_training, num_SGD_inds,sigma_filter=sigma_filter,r_noise=True, plot_flag=True)

folder_to_save=os.path.join(final_folder_path, 'figures')
data_rel_changes = MVPA_param_offset_correlations(final_folder_path, num_training, num_time_inds=3, x_labels=None,mesh_for_valid_offset=False, data_only=True) #J_m_ratio_diff, J_s_ratio_diff, offset_staircase_diff
data_rel_changes['offset_staircase_diff']=-1*data_rel_changes['offset_staircase_diff']

MVPA_scores = numpy.load(final_folder_path + f'/sigmafilt_{sigma_filter}/MVPA_scores.npy') # MVPA_scores - num_trainings x layer x SGD_ind x ori_ind (sup layer = 0)
data_sup_55 = pd.DataFrame({
    'MVPA': (MVPA_scores[:,0,-1,0]- MVPA_scores[:,0,-2,0])/MVPA_scores[:,0,-2,0],
    'JsI/JsE': data_rel_changes['J_s_ratio_diff'],
    'offset': data_rel_changes['offset_staircase_diff']
})
plot_corr_triangle(data_sup_55, folder_to_save, 'corr_triangle_sup_55')
data_sup_125 = pd.DataFrame({
    'MVPA': (MVPA_scores[:,0,-1,1]- MVPA_scores[:,0,-2,1])/MVPA_scores[:,0,-2,1],
    'JsI/JsE': data_rel_changes['J_s_ratio_diff'],
    'offset': data_rel_changes['offset_staircase_diff']
})
plot_corr_triangle(data_sup_125, folder_to_save, 'corr_triangle_sup_125')
data_mid_55 = pd.DataFrame({
    'MVPA': (MVPA_scores[:,1,-1,0]- MVPA_scores[:,1,-2,0])/MVPA_scores[:,1,-2,0],
    'JmI/JmE': data_rel_changes['J_m_ratio_diff'],
    'offset': data_rel_changes['offset_staircase_diff']
})
plot_corr_triangle(data_mid_55, folder_to_save, 'corr_triangle_mid_55')
data_mid_125 = pd.DataFrame({
    'MVPA': (MVPA_scores[:,1,-1,1]- MVPA_scores[:,1,-2,1])/MVPA_scores[:,1,-2,1],
    'JmI/JmE': data_rel_changes['J_m_ratio_diff'],
    'offset': data_rel_changes['offset_staircase_diff']
})
plot_corr_triangle(data_mid_125, folder_to_save, 'corr_triangle_mid_125')
