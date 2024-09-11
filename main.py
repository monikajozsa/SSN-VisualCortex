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
num_training = 49
starting_time_in_main= time.time()

# Set up results folder and save note and scripts
note=f'Getting as much data with corrected kappas over the weekend as possible'
root_folder = os.path.dirname(__file__)
#folder_path = save_code(note=note)
folder_path = os.path.join(root_folder, 'results', 'Aug15_v0')

## Run pretraining
#main_pretraining(folder_path, num_training, starting_time_in_main=starting_time_in_main)

## Define the configurations for training
conf_baseline = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], False) # training all parameters (baseline case)
#conf_gentask = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], True) # training with general discrimination task (control case)
#conf_midonly = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [0.0, 1.0], False) # reading out from middle layer (ablation)
# Additional case where labels are shuffled is handled with a separate util/py file

## Training all parameters but a few
#conf_kappa_a = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s'], [1.0, 0.0], False) # training all parameters but kappa (ablation)
conf_cms_excluded = (['f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], False) # training all but cE_m, cI_m, cE_s, cI_s (ablation)
conf_JI_excluded = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_IE_m', 'J_EE_s', 'J_IE_s', 'kappa'], [1.0, 0.0], False) # training all but JI (ablation)
conf_JE_excluded = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_II_m', 'J_EI_m', 'J_II_s', 'J_EI_s', 'kappa'], [1.0, 0.0], False) # training all but JI (ablation)
conf_Jm_excluded = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], False) # training all but Jm (ablation)
conf_Js_excluded = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'kappa'], [1.0, 0.0], False) # training all but Js (ablation)
conf_f_excluded = (['cE_m', 'cI_m', 'cE_s', 'cI_s', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], False) # training all but f_E, f_I (ablation)

## Training only a few
conf_cms_only = (['cE_m', 'cI_m', 'cE_s', 'cI_s'], [1.0, 0.0], False) # training only cE_m, cI_m, cE_s, cI_s (ablation)
conf_JI_only = (['J_II_m', 'J_EI_m', 'J_II_s', 'J_EI_s'], [1.0, 0.0], False) # training only JI
conf_JE_only = (['J_EE_m', 'J_IE_m', 'J_EE_s', 'J_IE_s'], [1.0, 0.0], False) # training only JE
conf_Jm_only = (['J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m'], [1.0, 0.0], False) # training only Jm
conf_Js_only = (['J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s'], [1.0, 0.0], False) # training only Js
conf_f_only = (['f_E','f_I'], [1.0, 0.0], False) # training only f

conf_names = ['conf_Js_excluded', 'conf_Jm_only', 'conf_Js_only', 'conf_JE_only', 'conf_f_only']

## Run training and analysis on the different configurations
i = 0
tc_ori_list = numpy.arange(0,180,6)
for conf in [conf_Js_excluded, conf_Jm_only, conf_Js_only, conf_JE_only, conf_f_only]:
    # create a configuration folder and copy relevant files to it
    config_folder = set_up_config_folder(folder_path, conf_names[i])
    
    # configure the parameters.py file
    configure_parameters_file(root_folder, conf[0], conf[1], conf[2]) 
    
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
    plot_results_on_parameters(config_folder, num_training, starting_time_in_main, tc_ori_list = tc_ori_list)
    i += 1
    
    print('\n')
    print(f'Configuration {i} done')
    print('\n')

# delete parameters.py file and make the parameters.py.bak file the parameters.py file
os.system(f'del "{root_folder}/parameters.py"')
os.system(f'rename "{root_folder}/parameters.py.bak" "{root_folder}/parameters.py"')

print('Finished all configurations')
