## This code will be testing what output and stimulus noise level we need to achieve chance-level performance for random initialization
## Do not run this code yet
import numpy

from pretraining_supp import load_parameters
from util import create_grating_pretraining
from util_gabor import init_untrained_pars
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
from Mahal_distances import filtered_model_response
from training import batch_loss_ori_discr, generate_noise

N_trainings=2
folder= 'results/Mar01_v20'
stimuli_pars_std_list = [30,75,120,165,210]
N_readout_noise_list = [2,5,14,125]
for run_ind in range(N_trainings):
    file_name = f"{folder}/results_{run_ind}.csv"
    orimap_filename = f"{folder}/orimap_{run_ind}.npy"
    loaded_orimap =  numpy.load(orimap_filename)
    trained_pars_stage1, trained_pars_stage2, _ = load_parameters(file_name, iloc_ind = 1)
    for i in range(len(stimuli_pars_std_list)):
        stimuli_pars.std=stimuli_pars_std_list[i]
        untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                        loss_pars, training_pars, pretrain_pars, readout_pars, None, loaded_orimap)
        
        # Generate data
        test_data = create_grating_pretraining(pretrain_pars, 100, untrained_pars.BW_image_jax_inp, numRnd_ori1=1)
        
        for j in range(len(N_readout_noise_list)):
            
            noise_ref = generate_noise(100, trained_pars_stage1["w_sig"].shape[0], N_readout=N_readout_noise_list[j])
            noise_target = generate_noise(100,  trained_pars_stage1["w_sig"].shape[0], N_readout=N_readout_noise_list[j])
            loss, [all_losses, true_accuracy, sig_input, sig_output, max_rates] = batch_loss_ori_discr(trained_pars_stage2, trained_pars_stage1, untrained_pars, test_data, noise_ref, noise_target, jit_on=True)