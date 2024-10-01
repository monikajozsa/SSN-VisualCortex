import time
import os
import sys
import numpy
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
import argparse

from util import load_parameters
from training_functions import train_ori_discr
from main_pretraining import readout_pars_from_regr

############### TRAINING ###############
def main_training(folder_path, num_training, starting_time_training=0, run_indices=None):
    """ Run training on the discrimination task with the configuration specified in the parameters.py file and initial_parameters from the pretraining. """
    if run_indices is None:
        run_indices = range(num_training)

    for i in run_indices:
        # Setting the same seed for different configurations
        numpy.random.seed(i)

        # Load the last parameters from the pretraining
        pretrained_readout_pars_dict, trained_pars_dict, untrained_pars, offset_last, meanr_vec = load_parameters(folder_path, run_index=i, stage=0, iloc_ind = -1, for_training=True)
        if untrained_pars.training_pars.opt_readout_before_training:
            # Apply logistic regression to the readout weights and bias and save them to pretrained_readout_pars_dict
            # This needs to be adjusted if we mix the layers because the readout weights are just 5x5 at the moment - check if readout_pars_from_regr needs to be adjusted for reading out from middle layer or if contrib variable takes care of it already 
            pretrained_readout_pars_dict = readout_pars_from_regr(trained_pars_dict, untrained_pars, N=1000, for_training=True)
            # Save the readout parameters to the folder
            if untrained_pars.sup_mid_readout_contrib[0] == 0:
                columns = {f'w{i}_mid': pretrained_readout_pars_dict['w_sig'][i] for i in range(len(pretrained_readout_pars_dict['w_sig']))}
            elif untrained_pars.sup_mid_readout_contrib[1] == 0:
                columns = {f'w{i}_sup': pretrained_readout_pars_dict['w_sig'][i] for i in range(len(pretrained_readout_pars_dict['w_sig']))}
            else:
                columns = {f'w{i}_mid': pretrained_readout_pars_dict['w_sig'][i] for i in range(len(pretrained_readout_pars_dict['w_sig'])//2)}
                columns.update({f'w{i}_sup': pretrained_readout_pars_dict['w_sig'][i] for i in range(len(pretrained_readout_pars_dict['w_sig'])//2,len(pretrained_readout_pars_dict['w_sig']))})
            columns['b'] = pretrained_readout_pars_dict['b_sig']
            df = pd.DataFrame([columns])
            df.to_csv(os.path.join(folder_path,'readout_params.csv'), index=False)

        # Change mean rate homeostatic loss
        if meanr_vec is not None:
            untrained_pars.loss_pars.lambda_r_mean = 0.25
            untrained_pars.loss_pars.Rmean_E = meanr_vec[0]
            untrained_pars.loss_pars.Rmean_I = meanr_vec[1]

        # Set the offset to the offset threshold, where a given accuracy is achieved with the parameters from the last SGD step (loaded as offset_last)
        untrained_pars.stimuli_pars.offset = min(offset_last,untrained_pars.stimuli_pars.max_train_offset)

        # Run training
        results_filename = os.path.join(folder_path,'training_results.csv')
        _, _ = train_ori_discr(
                pretrained_readout_pars_dict,
                trained_pars_dict,
                untrained_pars,
                results_filename=results_filename,
                jit_on=True,
                offset_step=0.1,
                run_index = i
            )
        print('runtime of {} training'.format(i), time.time()-starting_time_training)

# Main_training is called with subprocesses and so we take command-line arguments to run it. 
# This serves the purpose of reloading parameters.py that defines the configurations. 
# Otherwise, jax-jit would freeze it to the very first configuration.

if __name__ == "__main__":
    # Use argparse to accept command-line arguments
    parser = argparse.ArgumentParser(description='Run main training with specified parameters.')
    
    parser.add_argument('folder_path', type=str, help='Path to the configuration folder.')
    parser.add_argument('num_training', type=int, help='Number of training iterations.')
    parser.add_argument('starting_time_training', type=float, help='Starting time of the training for a specific configuration.')

    args = parser.parse_args()

    # Call the main_training function with parsed arguments
    main_training(args.folder_path, args.num_training, args.starting_time_training)