import pandas as pd
import numpy
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
numpy.random.seed(0)
import jax.numpy as np

from analysis.analysis_functions import tuning_curve, MVPA_Mahal_analysis
from util import load_parameters
from analysis.visualization import plot_MVPA_or_Mahal_scores
from parameters import GridPars, SSNPars, PretrainingPars
grid_pars, ssn_pars, pretraining_pars = GridPars(), SSNPars(), PretrainingPars()

# Checking that pretrain_pars.is_on is on
if not pretraining_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to True in parameters.py to run training with pretraining!')

########## CALCULATE TUNING CURVES ############
def main_tuning_curves(folder_path, num_training, start_time_in_main, stage_inds = range(3), tc_ori_list = numpy.arange(0,180,6), add_header=True, filename=None):
    """ Calculate tuning curves for the different runs and different stages in each run """
    from parameters import GridPars, SSNPars
    grid_pars, ssn_pars = GridPars(), SSNPars()
    # Define the filename for the tuning curves 
    if filename is not None:
        tc_file_path = os.path.join(folder_path, filename)
    else:
        tc_file_path = os.path.join(folder_path, 'tuning_curves.csv')

    if add_header:
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
    else:
        tc_headers = False

    # Loop over the different runs
    iloc_ind_vec = [0,-1,-1]
    stages = [0,0,2]
    for i in range(0, num_training):    
        # Loop over the different stages (before pretraining, after pretraining, after training) and calculate and save tuning curves
        for stage_ind in stage_inds:
            _, trained_pars_dict, untrained_pars = load_parameters(folder_path, run_index=i, stage=stages[stage_ind], iloc_ind=iloc_ind_vec[stage_ind])
            _, _ = tuning_curve(untrained_pars, trained_pars_dict, tc_file_path, ori_vec=tc_ori_list, training_stage=stage_ind, run_index=i, header=tc_headers)
            tc_headers = False
            
        print(f'Finished calculating tuning curves for training {i} in {time.time()-start_time_in_main} seconds')


########## CALCULATE MVPA SCORES AND MAHALANOBIS DISTANCES ############
def save_numpy_to_csv(numpy_array, file_name):
    """ Save mul numpy array to csv file """
    data = []
    for run_index in range(numpy.shape(numpy_array)[0]):
        for layer in range(numpy.shape(numpy_array)[1]):
            for stage in range(numpy.shape(numpy_array)[2]):
                for ori_index in range(numpy.shape(numpy_array)[3]):
                    # Add the new row to the dictionary
                    data.append({f'run_index': run_index, f'layer': layer, f'stage': stage, f'ori': ori_index, f'score': numpy_array[run_index, layer, stage, ori_index]})
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    return df

def csv_to_numpy(file_name):
    """ Reads a CSV file and converts it back to a 4D NumPy array."""
    # Read the CSV into a DataFrame
    df = pd.read_csv(file_name)
    
    # Get the maximum values for each index to determine the shape of the 4D array
    num_runs = df['run_index'].max() + 1
    num_layers = df['layer'].max() + 1
    num_stages = df['stage'].max() + 1
    num_oris = df['ori'].max() + 1
    
    # Initialize an empty NumPy array with the determined shape
    numpy_array = numpy.zeros((num_runs, num_layers, num_stages, num_oris))
    
    # Iterate over the DataFrame and populate the NumPy array
    for _, row in df.iterrows():
        run_index = int(row['run_index'])
        layer = int(row['layer'])
        stage = int(row['stage'])
        ori = int(row['ori'])
        score = row['score']
        
        # Assign the score to the correct position in the 4D array
        numpy_array[run_index, layer, stage, ori] = score
    
    return numpy_array

def main_MVPA(folder, num_training, folder_to_save=None, num_stage_inds=2, sigma_filter=5, r_noise=True, num_noisy_trials=100, plot_flag=False):
    """ Calculate MVPA scores for before pretraining, after pretraining and after training - score should increase for trained ori more than for other two oris especially in superficial layer"""
    if folder_to_save is None:
        # save the output into folder_to_save as npy files
        folder_to_save = folder + f'/MVPA'
        # create the folder if it does not exist
        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)
    
    if not os.path.exists(folder +'/MVPA_scores.csv'):
        MVPA_scores, Mahal_scores = MVPA_Mahal_analysis(folder,num_training, num_stage_inds, r_noise = r_noise, sigma_filter=sigma_filter, num_noisy_trials=num_noisy_trials, plot_flag=plot_flag)
        _ = save_numpy_to_csv(MVPA_scores, folder + '/MVPA_scores.csv')
        _ = save_numpy_to_csv(Mahal_scores, folder + '/Mahal_scores.csv')
    else:
        MVPA_scores = csv_to_numpy(folder +'/MVPA_scores.csv')
        Mahal_scores = csv_to_numpy(folder +'/Mahal_scores.csv')
    
    print('MVPA Pre-pre, pre and post training for 55~0, sup layer:',[np.mean(MVPA_scores[:,0,0,0]),np.mean(MVPA_scores[:,0,1,0]),np.mean(MVPA_scores[:,0,-1,0])])
    print('MVPA Pre-pre, pre and post training for 55~0, mid layer:',[np.mean(MVPA_scores[:,1,0,0]),np.mean(MVPA_scores[:,1,1,0]),np.mean(MVPA_scores[:,1,-1,0])])
    print('MVPA Pre-pre, pre and post training for 125~0, sup layer:',[np.mean(MVPA_scores[:,0,0,1]),np.mean(MVPA_scores[:,0,1,1]),np.mean(MVPA_scores[:,0,-1,1])])
    print('MVPA Pre-pre, pre and post training for 125~0, mid layer:',[np.mean(MVPA_scores[:,1,0,1]),np.mean(MVPA_scores[:,1,1,1]),np.mean(MVPA_scores[:,1,-1,1])])

    print('Mahal Pre-pre, pre and post training for 55~0, sup layer:',[np.mean(Mahal_scores[:,0,0,0]),np.mean(Mahal_scores[:,0,1,0]),np.mean(Mahal_scores[:,0,-1,0])])
    print('Mahal Pre-pre, pre and post training for 55~0, mid layer:',[np.mean(Mahal_scores[:,1,0,0]),np.mean(Mahal_scores[:,1,1,0]),np.mean(Mahal_scores[:,1,-1,0])])
    print('Mahal Pre-pre, pre and post training for 125~0, sup layer:',[np.mean(Mahal_scores[:,0,0,1]),np.mean(Mahal_scores[:,0,1,1]),np.mean(Mahal_scores[:,0,-1,1])])
    print('Mahal Pre-pre, pre and post training for 125~0, mid layer:',[np.mean(Mahal_scores[:,1,0,1]),np.mean(Mahal_scores[:,1,1,1]),np.mean(Mahal_scores[:,1,-1,1])])

    # Plot histograms of the LMI acorss the runs
    if plot_flag:
        plot_MVPA_or_Mahal_scores(folder, num_training, MVPA_scores, 'MVPA_scores')
        plot_MVPA_or_Mahal_scores(folder, num_training, Mahal_scores, 'Mahal_scores')
