# This code runs pretraining for a general orientation task on a 2-layer SSN that models the middle and superficial layers of V1. 
# After pretraining, it trains the model in several configurations and saves the results in separate folders.

# to test this code, change SGD_steps in TrainingPars and PretrainingPars in parameters.py and set tc_ori_list = numpy.arange(0,360,20) and 
import os
import time
import pandas as pd
import numpy
import shutil
import subprocess

from util import configure_parameters_file, save_code, set_up_config_folder, del_pretrain_files_from_config_folder
from training.main_pretraining import main_pretraining
from analysis.main_analysis import main_tuning_curves, main_MVPA
from analysis.analysis_functions import save_tc_features, MVPA_anova
from analysis.visualization import plot_tuning_curves, plot_corr_triangles, plot_results_on_parameters, plot_tc_features, plot_param_offset_correlations

## Set up number of runs and starting time
num_training = 50
starting_time_in_main= time.time()

# Set up results folder and save note and scripts
note=f'Getting as much data with corrected kappas over the weekend as possible'
root_folder = os.path.dirname(__file__)
#folder_path = save_code(note=note)
folder_path = os.path.join(root_folder, 'results', 'Oct23_v6')

########## ########## ########## 
######### Pretraining ##########
########## ########## ##########  
main_pretraining(folder_path, num_training, starting_time_in_main=starting_time_in_main)

########## ########## ########## ##########
########  Training configurations  ########
########## ########## ########## ##########
## Define the configurations for training.
## Each configuration list contains the following elements:
## 1. Training parameters (e.g., 'cE_m', 'cI_m', etc.). Required, no default.
## 2. Readout contribution from superficial and middle layers ([superficial, middle]). (default: [1.0, 0.0])
## 3. Task type: False = fine discrimination, True = general discrimination. (default: False)
## 4. p_local_s: relative strength of local E projections in the superficial layer ([1, 1] = no local part). (default: [0.4, 0.7])
## 5. shuffle_labels: whether to shuffle the labels for the training task or not. (default: False)
## 6. opt_readout_before_training: whether there is a logistic regression optimization for readout parameters or not before training (default: False)

# special cases
conf_baseline = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa']] # training all parameters (baseline case)
conf_gentask = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], True] # training with general discrimination task (control case)
conf_no_horiconn = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], False, [1.0, 1.0]] 
conf_shuffled_labels = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], False, [0.4, 0.7], True]
conf_special_dict = {'conf_baseline': conf_baseline,
                    'conf_gentask': conf_gentask,
                    'conf_no_horiconn': conf_no_horiconn,
                    'conf_shuffled_labels': conf_shuffled_labels
                    }

# changed readout configurations - readout is optimized with logistic regression before training
conf_suponly_readout = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], False, [0.4, 0.7], False, True] # reading out from middle layer (ablation)
conf_mixed_readout = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [0.5, 0.5], False, [0.4, 0.7], False, True] # training all parameters but reading out from both middle and superficial layers
conf_midonly_readout = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [0.0, 1.0], False, [0.4, 0.7], False, True] # reading out from middle layer (ablation)
conf_suponly_no_hori_readout = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], False, [1.0, 1.0], False, True] # reading out from middle layer (ablation)
conf_readout_dict = {'conf_suponly_readout': conf_suponly_readout,
                    'conf_mixed_readout': conf_mixed_readout,
                    'conf_midonly_readout': conf_midonly_readout,
                    'conf_suponly_no_hori_readout': conf_suponly_no_hori_readout
                    }

# training with all parameters but a few
conf_kappa_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s']] # training all parameters but kappa (ablation)
conf_cms_excluded = [['f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa']] # training all but cE_m, cI_m, cE_s, cI_s (ablation)
conf_JI_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_IE_m', 'J_EE_s', 'J_IE_s', 'kappa']] # training all but JI (ablation)
conf_JE_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_II_m', 'J_EI_m', 'J_II_s', 'J_EI_s', 'kappa']] # training all but JI (ablation)
conf_Jm_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa']] # training all but Jm (ablation)
conf_Js_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'kappa']] # training all but Js (ablation)
conf_f_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa']] # training all but f_E, f_I (ablation)
conf_excluded_dict = {'conf_kappa_excluded': conf_kappa_excluded,
                    'conf_cms_excluded': conf_cms_excluded,
                    'conf_JI_excluded': conf_JI_excluded,
                    'conf_JE_excluded': conf_JE_excluded,
                    'conf_Jm_excluded': conf_Jm_excluded,
                    'conf_Js_excluded': conf_Js_excluded,
                    'conf_f_excluded': conf_f_excluded
                    }

# training with only a few parameters
conf_kappa_only = [['kappa']] # training only kappa (ablation)
conf_cms_only = [['cE_m', 'cI_m', 'cE_s', 'cI_s']] # training only cE_m, cI_m, cE_s, cI_s (ablation)
conf_JI_only = [['J_II_m', 'J_EI_m', 'J_II_s', 'J_EI_s']] # training only JI
conf_JE_only = [['J_EE_m', 'J_IE_m', 'J_EE_s', 'J_IE_s']] # training only JE
conf_Jm_only = [['J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m']] # training only Jm
conf_Js_only = [['J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s']] # training only Js
conf_f_only = [['f_E','f_I']] # training only f
conf_only_dict = {'conf_kappa_only': conf_kappa_only,
                'conf_cms_only': conf_cms_only,
                'conf_JI_only': conf_JI_only,
                'conf_JE_only': conf_JE_only,
                'conf_Jm_only': conf_Jm_only,
                'conf_Js_only': conf_Js_only,
                'conf_f_only': conf_f_only
                }

# create dictionary of configurations to loop over
conf_dict = {**conf_special_dict, **conf_excluded_dict, **conf_only_dict, **conf_readout_dict}

conf_names = list(conf_dict.keys())
conf_list = list(conf_dict.values())

########## ########## ##########
##########  Training  ##########
########## ########## ########## 
for i, conf in enumerate(conf_list):
    
    # create a configuration folder and copy relevant files to it
    config_folder = set_up_config_folder(folder_path, conf_names[i])
    #config_folder = os.path.join(folder_path, conf_names[i])
    
    # configure the parameters.py file and copy it as a backup fil in the config folder
    configure_parameters_file(root_folder, conf)
    shutil.copy('parameters.py', os.path.join(config_folder,  f'parameters_{conf_names[i]}.py'))
    
    # run training with the configured parameters.py file
    main_training_source = os.path.join(root_folder, "training", "main_training.py")
    subprocess.run(["python3", str(main_training_source), config_folder, str(num_training), str(time.time())])
    
    # plot results on parameters
    plot_results_on_parameters(config_folder, num_training, plot_per_run=True)
    plot_param_offset_correlations(config_folder)

    print('\n')
    print(f'Configuration {i} done')
    print('\n')

print('Finished all configurations')

## Make the parameters.py.bak file the parameters.py file
#if os.path.exists(os.path.join(root_folder,'parameters.py.bak')):
#    shutil.copy(os.path.join(root_folder,'parameters.py.bak'), os.path.join(root_folder,'parameters.py'))


########## ########## ##########
######### Tuning curves ########
########## ########## ##########

tc_ori_list = numpy.arange(0,360,20)
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
    # plot tuning curves and features
    tc_cells=[10,40,100,130,172,202,262,292,334,364,424,454,496,526,586,616,650,690,740,760] 
    # these are indices of representative cells from the different layers and types: every pair is for off center and center from 
    # mEph0(1-2), mIph0(3-4), mEph1(5-6), mIph1(7-8), mEph2(9-10), mIph2(11-12), mEph3(13-14), mIph3(15-16), sE(17-18), sI(19-20)
    folder_to_save=os.path.join(config_folder, 'figures')
    plot_tuning_curves(config_folder, tc_cells, num_training, folder_to_save)
    if i == 0:
        stages = [0,1,2]
    else:
        stages = [1,2]
    # plot_tc_features(config_folder, stages=stages, color_by='type', only_slope_plot=True)
    plot_tc_features(config_folder, stages=stages, color_by='type', add_cross=True)
    plot_tc_features(config_folder, stages=stages, color_by='run_index')
    plot_tc_features(config_folder, stages=stages, color_by='pref_ori')
    plot_tc_features(config_folder, stages=stages, color_by='phase')
    print('\n')
    print(f'Finished calculating tuning curves and features for {conf_names[i]} in {time.time()-start_time} seconds')
    print('\n')


########## ########## ##########
##########    MVPA    ##########
########## ########## ##########

for i, conf in enumerate(conf_names):
    start_time = time.time()
    config_folder = os.path.join(folder_path, conf)
    folder_to_save = config_folder + f'/figures'
    main_MVPA(config_folder, num_training, folder_to_save, 3, sigma_filter=2, r_noise=True, num_noisy_trials=200, plot_flag=True) 
    MVPA_anova(config_folder)
    plot_corr_triangles(config_folder, folder_to_save)
    print('Done with MVPA for configuration ', conf, ' in ', time.time()-start_time, ' seconds')


########## ########## ########## ########## ############
######### Clean up files about excluding runs ##########
########## ########## ########## ########## ############
'''
for i, conf in enumerate(conf_names):
    config_folder = os.path.join(folder_path, conf)
    # delete pretraining related files from the configuration folder
    del_pretrain_files_from_config_folder(config_folder)
'''
