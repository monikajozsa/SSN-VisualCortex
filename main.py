# This code runs pretraining for a general and training for a fine orientation discrimination task with a two-layer neural network model, where each layer is an SSN.

import numpy
import time
import os

'''
from util_gabor import init_untrained_pars
from util import save_code, load_parameters
from training import train_ori_discr
from analysis import tuning_curve
from perturb_params import perturb_params
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
    pretrain_pars # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
)

# Checking that pretrain_pars.is_on is on
if not pretrain_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to True in parameters.py to run training with pretraining!')

# Setting train_only_flag to True will run an additional training without pretraining
train_only_flag = False

########## Initialize orientation map and gabor filters ############

# Save out initial offset and reference orientation
ref_ori_saved = float(stimuli_pars.ref_ori)
offset_saved = float(stimuli_pars.offset)

# Save scripts into scripts folder and create figures and train_only folders
results_filename, final_folder_path = save_code(train_only_flag=train_only_flag)

# Run num_training number of pretraining + training
tc_ori_list = numpy.arange(0,180,2)
num_training = 4
starting_time_in_main= time.time()
num_FailedRuns = 0
i=0
while i < num_training and num_FailedRuns < 20:
    numpy.random.seed(i)

    # Set pretraining flag to False
    pretrain_pars.is_on=True
    # Set offset and reference orientation to their initial values
    stimuli_pars.offset=offset_saved
    stimuli_pars.ref_ori=ref_ori_saved

    # Create file names
    results_filename = os.path.join(final_folder_path, f"results_{i}.csv")
    results_filename_train_only = os.path.join(final_folder_path, 'train_only', f"results_train_only{i}.csv")
    tc_prepre_filename = os.path.join(final_folder_path, f"tc_prepre_{i}.csv")
    tc_postpre_filename = os.path.join(final_folder_path, f"tc_postpre_{i}.csv")
    tc_post_filename = os.path.join(final_folder_path, f"tc_post_{i}.csv")
    tc_pre_train_only_filename = os.path.join(final_folder_path, 'train_only', f"tc_pre_train_only_{i}.csv")
    tc_post_train_only_filename = os.path.join(final_folder_path, 'train_only', f"tc_post_train_only_{i}.csv")
    orimap_filename = os.path.join(final_folder_path, f"orimap_{i}.npy")

    # Initialize untrained parameters (calculate gabor filters, orientation map related variables)
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, orimap_filename)

    ##### PRETRAINING: GENERAL ORIENTAION DISCRIMINATION #####

    # Perturb readout_pars and trained_pars by percent % and collect them into two dictionaries for the two stages of the pretraining
    # Note that orimap is regenerated if conditions do not hold!
    trained_pars_stage1, trained_pars_stage2, untrained_pars = perturb_params(readout_pars, trained_pars, untrained_pars, percent=0.1, orimap_filename=orimap_filename)
    # Calculate and save tuning curves
    tc_prepre, _ = tuning_curve(untrained_pars, trained_pars_stage2, tc_prepre_filename, ori_vec=tc_ori_list)
    
    # Run pre-training
    training_output_df, pretraining_final_step = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            untrained_pars,
            results_filename=results_filename,
            jit_on=True,
            offset_step = 0.1
        )
    
    # Handle the case when pretraining failed (possible reason can be the divergence of ssn diff equations)
    if training_output_df is None:
        print('######### Stopped run {} because of NaN values  - num failed runs = {} #########'.format(i, num_FailedRuns))
        num_FailedRuns = num_FailedRuns + 1
        continue
    
    ##### FINE DISCRIMINATION #####
    
    # Set pretraining flag to False
    untrained_pars.pretrain_pars.is_on = False
    # Load the last parameters from the pretraining
    trained_pars_stage1, trained_pars_stage2, offset_last = load_parameters(results_filename, iloc_ind = pretraining_final_step, trained_pars_keys=trained_pars_stage2.keys())
    # Calculate and save tuning curves
    tc_postpre, _ = tuning_curve(untrained_pars, trained_pars_stage2, tc_postpre_filename, ori_vec=tc_ori_list)
    # Set the offset to the offset, where a threshold accuracy is achieved with the parameters from the last SGD step (loaded as offset_last)
    untrained_pars.stimuli_pars.offset = min(offset_last,10)
    # Run training
    training_output_df, _ = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            untrained_pars,
            results_filename=results_filename,
            jit_on=True,
            offset_step=0.1
        )

    # Calculate and save tuning curves
    _, trained_pars_stage2, _ = load_parameters(results_filename, iloc_ind = -1)
    tc_post, _ = tuning_curve(untrained_pars, trained_pars_stage2, tc_post_filename, ori_vec=tc_ori_list)
    
    ########## TRAINING ONLY with the same initialization and orimap ##########
    if train_only_flag:
        # Load the first parameters that pretraining started with
        trained_pars_stage1, trained_pars_stage2, _ = load_parameters(results_filename, iloc_ind = 0)
        # Set the offset to the original offset that pretraining started with
        untrained_pars.stimuli_pars.offset = offset_saved
        # Set the reference orientation to the original one that pretraining started with
        untrained_pars.stimuli_pars.ref_ori = ref_ori_saved
        
        # Run training
        training_output_df, _ = train_ori_discr(
                trained_pars_stage1,
                trained_pars_stage2,
                untrained_pars,
                results_filename=results_filename_train_only,
                jit_on=True
            )
        
        # Calculate and save tuning curves
        trained_pars_stage1, trained_pars_stage2, _ = load_parameters(results_filename_train_only, iloc_ind = 0)
        _, _ = tuning_curve(untrained_pars, trained_pars_stage2, tc_pre_train_only_filename, ori_vec=tc_ori_list)
        trained_pars_stage1, trained_pars_stage2, _ = load_parameters(results_filename_train_only, iloc_ind = -1)
        _, _ = tuning_curve(untrained_pars, trained_pars_stage2, tc_post_train_only_filename, ori_vec=tc_ori_list)
        
    i = i + 1
    print('runtime of {} pretraining + training run(s)'.format(i), time.time()-starting_time_in_main)
    print('number of failed runs = ', num_FailedRuns)
'''

######### PLOT RESULTS ############

from visualization import plot_results_from_csvs, boxplots_from_csvs, plot_tuning_curves, plot_tc_features, plot_correlations
from MVPA_Mahal_combined import MVPA_Mahal_from_csv
from analysis import MVPA_param_offset_correlations

numpy.random.seed(0)
start_time=time.time()
final_folder_path=os.path.join('results','May10_v0')
num_training = 2
tc_ori_list = numpy.arange(0,180,2)
tc_cells=[10,40,100,130,650,690,740,760]

## Pretraining + training
folder_to_save = os.path.join(final_folder_path, 'figures')
boxplot_file_name = 'boxplot_pretraining'
mahal_file_name = 'Mahal_dist'
num_SGD_inds = 3
sigma_filter = 2

plot_results_from_csvs(final_folder_path, num_training, folder_to_save=folder_to_save)#, starting_run=10)
boxplots_from_csvs(final_folder_path, folder_to_save, boxplot_file_name, num_time_inds = num_SGD_inds, num_training=num_training)
plot_tc_features(final_folder_path, num_training, tc_ori_list)
plot_tuning_curves(final_folder_path,tc_cells,num_training,folder_to_save)
#MVPA_Mahal_from_csv(final_folder_path, num_training, num_SGD_inds,sigma_filter=sigma_filter,r_noise=True, plot_flag=True)

#MVPA_param_offset_correlations(final_folder_path, num_training, num_time_inds=3, x_labels=None)
#plot_correlations(final_folder_path, num_training, num_time_inds=3)

'''
## Training only
#final_folder_path_train_only = final_folder_path + '/train_only'
#boxplot_file_name_train_only = 'boxplot_train_only'
#mahal_file_name_train_only = 'Mahal_dist_train_only'
#plot_results_from_csvs(final_folder_path_train_only, num_training, folder_to_save=folder_to_save)
#boxplots_from_csvs(final_folder_path_train_only,folder_to_save, boxplot_file_name_train_only)
#Mahal_dist_from_csv(final_folder_path_train_only,num_training, folder_to_save, mahal_file_name_train_only)
#plot_tc_features(final_folder_path_train_only, num_training, tc_ori_list, train_only_str='train_only_')
#plot_tuning_curves(final_folder_path_train_only,tc_cells,num_training,folder_to_save,train_only_str='train_only_')

print('runtime of plotting', time.time()-start_time)


# Recalculating and replotting tuning curves if ori list is different than default
import pandas as pd
final_folder_path='results/Mar22_v0'
num_training=5
tc_ori_list = numpy.arange(0,180,2)

orimap_filename = final_folder_path+ '/orimap_0.npy'
untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, orimap_filename)
for i in range(num_training):
    results_filename=final_folder_path+f'/results_{i}.csv'
    tc_prepre_filename = f"{final_folder_path}/tc_prepre_{i}.csv"
    tc_postpre_filename = f"{final_folder_path}/tc_postpre_{i}.csv"
    tc_post_filename = f"{final_folder_path}/tc_post_{i}.csv"
    trained_pars_stage1, trained_pars_stage2, _ = load_parameters(results_filename, iloc_ind = 0)
    _, _ = tuning_curve(untrained_pars, trained_pars_stage2, tc_prepre_filename, ori_vec=tc_ori_list)
    df = pd.read_csv(results_filename)
    training_start_ind = df.index[df['stage'] == 1][0]
    trained_pars_stage1, trained_pars_stage2, _ = load_parameters(results_filename, iloc_ind = training_start_ind)
    _, _ = tuning_curve(untrained_pars, trained_pars_stage2, tc_postpre_filename, ori_vec=tc_ori_list)
    trained_pars_stage1, trained_pars_stage2, _ = load_parameters(results_filename, iloc_ind = -1)
    _, _ = tuning_curve(untrained_pars, trained_pars_stage2, tc_post_filename, ori_vec=tc_ori_list)
'''