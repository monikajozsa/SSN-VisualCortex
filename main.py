# This code runs pretraining for a general orientation task on a 2-layer SSN that models the middle and superficial layers of V1. 
# After pretraining, it trains the model in several configurations and saves the results in separate folders.
# It then calculates tuning curves and features for both pretraining and training with different configurations.

import os
import time
import numpy
import shutil
import subprocess
# fix random seed
numpy.random.seed(0)

from configurations import config
from util import configure_parameters_file, save_code, set_up_config_folder
from training.main_pretraining import main_pretraining
from analysis.analysis_functions import save_tc_features, main_tuning_curves
from analysis.visualization import plot_results_from_csvs, barplots_from_csvs
from analysis.main_analysis import main_analysis

## Set up number of runs and starting time
num_pretraining = 50
starting_time_in_main= time.time()

# Set up results folder and save note and scripts
note=f'Shortened training but pretrain_stage_1_acc_th raised to 0.65, xtol changed to 1e-2'
root_folder = os.path.dirname(__file__)
folder_path = save_code(note=note)
#folder_path = os.path.join(root_folder, 'results', 'Dec30_v12')

########## ########## ########## 
######### Pretraining ##########
########## ########## ##########
from parameters import PretrainingPars
pretraining_pars = PretrainingPars() # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
if not pretraining_pars.is_on:
    raise ValueError('Pretraining is not on. Please set pretraining_pars.is_on=True in parameters.py')
num_training = main_pretraining(folder_path, num_pretraining, starting_time_in_main=starting_time_in_main)
#num_training = num_pretraining
########## ########## ##########
##########  Training  ##########
########## ########## ##########
conf_dict, conf_names, conf_list = config(['special'])

for i, conf in enumerate(conf_list):
    
    # create a configuration folder and copy relevant files to it
    config_folder = set_up_config_folder(folder_path, conf_names[i])
    #config_folder = os.path.join(folder_path, conf_names[i])
    
    # configure the parameters.py file and copy it as a backup file in the config folder
    configure_parameters_file(root_folder, conf)
    time.sleep(1) # wait for the file to be saved
    shutil.copy('parameters.py', os.path.join(config_folder,  f'parameters_{conf_names[i]}.py'))
    
    # run training with the configured parameters.py file
    main_training_source = os.path.join(root_folder, "training", "main_training.py")
    subprocess.run(["python3", str(main_training_source), config_folder, str(num_training), str(time.time())])
    
    # plot results on parameters
    plot_results_from_csvs(config_folder, num_training)
    barplots_from_csvs(config_folder)

    print(f'Configuration {i} done')
        
print('Finished all configurations')

# Make the parameters.py.bak file the parameters.py file
#if os.path.exists(os.path.join(root_folder,'parameters.py.bak')):
#    shutil.copy(os.path.join(root_folder,'parameters.py.bak'), os.path.join(root_folder,'parameters.py'))
#time.sleep(1) # wait for the file to be saved

########## ########## ##########
######### Tuning curves ########
########## ########## ##########

tc_ori_list = numpy.arange(0,360,5)
start_time = time.time()

# calculate tuning curves and features for before and after pretraining
main_tuning_curves(folder_path, num_training, starting_time_in_main, stage_inds = range(2), tc_ori_list = tc_ori_list, add_header=True, filename='pretraining_tuning_curves.csv')
tc_file_name = os.path.join(folder_path, 'pretraining_tuning_curves.csv')
save_tc_features(tc_file_name, num_runs=num_training, ori_list=tc_ori_list, ori_to_center_slope=[55, 125], stages=[0, 1], filename = 'pretraining_tuning_curve_features.csv')

# calculate tuning curves and features for the different configurations
for i, conf in enumerate(conf_names):
    config_folder = os.path.join(folder_path, conf)
    # calculate tuning curves for after training
    main_tuning_curves(config_folder, num_training, starting_time_in_main, stage_inds = range(2,3), tc_ori_list = tc_ori_list, add_header=False) 
    
    # calculate tuning curve features
    tc_file_name = os.path.join(config_folder, 'tuning_curves.csv')
    save_tc_features(tc_file_name, num_runs=num_training, ori_list=tc_ori_list, ori_to_center_slope=[55, 125])

main_analysis(folder_path, num_training, conf_names)