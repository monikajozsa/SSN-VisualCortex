# this code is under development for the analysis on offsets
# t-test offsets before and after training for 55, ANOVA for offsets before and after training for 55, 125 and 0
# offset is calculated (one session is 16 runs, offset is per run (16), each run has 15 trials = reversals for the staircase), 
# paired t-test MPI for 55 and 125 ( [pre-test threshold – post-test threshold]/pre-test threshold * 100) 

import numpy
import os
import pandas as pd

from util_gabor import init_untrained_pars
from training import mean_training_task_acc_test, offset_at_baseline_acc
from util import load_parameters
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

def offset_th(num_training, folder, num_SGD_inds=3, th_accuracy=0.795):
    '''This function calculates the offset where th_accuracy is achieved from results_{i}.csv files at SGD step 0 (start of pretraining), -1 (end of training) and where pretraining ended if num_SGD_inds is 3'''
    
    test_offset_vec = numpy.array([1, 2, 4, 6, 9, 12, 15, 20])
    offset_at_bl_acc = numpy.zeros(num_training,num_SGD_inds)
    for run_ind in range(num_training):
        file_name = f"{folder}/results_{run_ind}.csv"
        orimap_filename = f"{folder}/orimap_{run_ind}.npy"
        if not os.path.exists(file_name):
            file_name = f"{folder}/results_train_only{run_ind}.csv"
            orimap_filename = os.path.dirname(folder)+f'/orimap_{run_ind}.npy'
        
        loaded_orimap =  numpy.load(orimap_filename)
        untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                        loss_pars, training_pars, pretrain_pars, readout_pars, None, orimap_loaded=loaded_orimap)
        df = pd.read_csv(file_name)
        train_start_ind = df.index[df['stage'] == 1][0]
        if num_SGD_inds==3:        
            if numpy.min(df['stage'])==0:
                pretrain_start_ind = df.index[df['stage'] == 0][0]
                SGD_step_inds=[pretrain_start_ind, train_start_ind, -1]
            else:
                print('Warning: There is no 0 stage but Mahal dist was asked to be calculated for pretraining!')
                SGD_step_inds=[train_start_ind, -1]
        else:
            SGD_step_inds=[train_start_ind, -1]

        # Iterate overs SGD_step indices (default is before and after training)
        for step_ind in SGD_step_inds:
            # Load parameters from csv for given epoch
            readout_pars_dict, ssn_layer_pars_dict, _ = load_parameters(file_name, iloc_ind = step_ind)
            # calculate mean accuracy for the given parameter set
            acc_mean, _, _ = mean_training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, untrained_pars, True, test_offset_vec, sample_size =10)
            # calculate offset by fitting non-linear curve
            offset_at_bl_acc[run_ind, step_ind] = offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= th_accuracy)
    
    return offset_at_bl_acc
