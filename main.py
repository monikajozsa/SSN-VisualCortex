# This code runs pretraining for a general orientation task on a 2-layer SSN that models the middle and superficial layers of V1. 
# After pretraining, it trains the model in several configurations and saves the results in separate folders.

import os
import time
import pandas as pd
import numpy

from util import configure_parameters_file, save_code, set_up_config_folder
from training.main_training import main_pretraining, main_training
from analysis.main_analysis import main_tuning_curves, plot_results_on_parameters

## Set up number of runs and starting time
num_training = 2
starting_time_in_main= time.time()

# Set up results folder and save note and scripts
note=f'Different ablation cases'
folder_path = os.path.join('results','Aug28_v1')
#folder_path = save_code(note=note)

## Run pretraining
#main_pretraining(folder_path, num_training, starting_time_in_main=starting_time_in_main)

## Define the configurations for training
conf_0 = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_2x2_m', 'J_2x2_s', 'kappa'], [1.0, 0.0], False) # training all parameters (baseline case)
conf_1 = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_2x2_m', 'J_2x2_s', 'kappa'], [1.0, 0.0], True) # training with general discriminatiuon task (control case)
'''conf_2 = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_2x2_m', 'J_2x2_s', 'kappa'], [0.0, 1.0], False) # reading out from middle layer (ablation)
conf_3 = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_2x2_m', 'J_2x2_s'], [1.0, 0.0], False) # training all parameters but kappa (ablation)
conf_4 = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_2x2_m', 'kappa'], [1.0, 0.0], False) # training all but J_2x2_s (ablation)
conf_5 = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_2x2_s', 'kappa'], [1.0, 0.0], False) # training all but J_2x2_m (ablation)
conf_6 = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'J_2x2_m', 'J_2x2_s', 'kappa'], [1.0, 0.0], False) # training all but f_E, f_I (ablation)
conf_7 = (['cE_m', 'cI_m', 'f_E', 'f_I', 'J_2x2_m', 'J_2x2_s', 'kappa'], [1.0, 0.0], False) # training all but  cE_s, cI_s (ablation)
conf_8 = (['f_E', 'f_I', 'J_2x2_m', 'J_2x2_s', 'kappa'], [1.0, 0.0], False) # training all but cE_m, cI_m (ablation)
'''
conf_names = ['conf_0', 'conf_1']#, 'conf_2', 'conf_3', 'conf_4', 'conf_5', 'conf_6', 'conf_7', 'conf_8']

## Run training and analysis on the different configurations
i = 0
tc_ori_list = numpy.arange(0,180,60)
for conf in [conf_0, conf_1]:#, conf_2, conf_3, conf_4, conf_5, conf_6, conf_7, conf_8]:
    
    # create a configuration folder and copy relevant files to it
    config_folder = set_up_config_folder(folder_path, conf_names[i])
    
    # configure the parameters.py file
    configure_parameters_file(config_folder, conf[0], conf[1], conf[2]) 
    
    # run training
    main_training(config_folder, num_training, starting_time_training=time.time())
    
    # calculate tuning curves
    if i == 0:
        main_tuning_curves(config_folder, num_training, starting_time_in_main, stage_inds = range(3), tc_ori_list = tc_ori_list, add_header=True)
        tc_df = pd.read_csv(f'{config_folder}/tuning_curves.csv')
        mesh_i = tc_df['training_stage'] < 2
        tc_df = tc_df[mesh_i]
        tc_df = tc_df.reset_index(drop=True)
        tc_df.to_csv(f'{folder_path}/pretraining_tuning_curves.csv', index=False)
    else:
        # copy pretraining tuning curve file as tuning curve file from the first configuration to avoid multiple calculation of tuning curves before and after pretraining
        source_file = os.path.join(folder_path, 'pretraining_tuning_curves.csv')
        destination_file = os.path.join(config_folder, 'tuning_curves.csv')
        os.system(f'copy "{source_file}" "{destination_file}"')
        main_tuning_curves(config_folder, num_training, starting_time_in_main, stage_inds = range(2,3), tc_ori_list = tc_ori_list, add_header=False) 
    
    # plot results on parameters and tuning curves
    plot_results_on_parameters(config_folder, num_training, starting_time_in_main, tc_ori_list = numpy.arange(0,180,6))
    i += 1
