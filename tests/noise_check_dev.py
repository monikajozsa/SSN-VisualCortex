## This code will be testing what output and stimulus noise level we need to achieve chance-level performance for random initialization
## Do not run this code yet
import numpy
import matplotlib.pyplot as plt
import time

from perturb_params import perturb_params
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
from training import batch_loss_ori_discr, generate_noise
if pretrain_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to False in parameters.py to run training without pretraining!')
start_time = time.time()

N_trainings=50
stimuli_pars_std_list = [30,75,120,210,400]
N_readout_noise_list = [5,30,125,500]
batch_size=100

# Generate accuracies for different input and output noise levels and seave them to noise_check_true_acc.npy
true_accuracy_all = numpy.zeros((len(stimuli_pars_std_list), len(N_readout_noise_list), batch_size))
for run_ind in range(N_trainings):
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, loss_pars, training_pars, pretrain_pars, readout_pars, None)
    trained_pars_stage1, trained_pars_stage2, untrained_pars = perturb_params(readout_pars, ssn_layer_pars, untrained_pars, percent=0.1, orimap_filename='noise_test_orimap')
    for i in range(len(stimuli_pars_std_list)):        
        untrained_pars.stimuli_pars.std=stimuli_pars_std_list[i]
        
        # Generate data
        test_data = create_grating_pretraining(pretrain_pars, batch_size, untrained_pars.BW_image_jax_inp, numRnd_ori1=1)        
        for j in range(len(N_readout_noise_list)):            
            noise_ref = generate_noise(batch_size, trained_pars_stage1["w_sig"].shape[0], num_readout_noise=N_readout_noise_list[j])
            noise_target = generate_noise(batch_size,  trained_pars_stage1["w_sig"].shape[0], num_readout_noise=N_readout_noise_list[j])
            loss, [all_losses, true_accuracy, sig_input, sig_output, max_rates] = batch_loss_ori_discr(trained_pars_stage2, trained_pars_stage1, untrained_pars, test_data, noise_ref, noise_target, jit_on=True)
            true_accuracy_all[i,j,run_ind] = true_accuracy
    print(time.time()-start_time)
numpy.save(f'noise_check_true_acc',true_accuracy_all)

# true_accuracy_all=numpy.load('noise_check_true_acc.npy')
fig, axs = plt.subplots(len(stimuli_pars_std_list), len(N_readout_noise_list), figsize=(20, 25))
for i in range(len(stimuli_pars_std_list)): 
    for j in range(len(N_readout_noise_list)):
        mask=true_accuracy_all[i,j,:]>0
        axs[i, j].hist(true_accuracy_all[i,j,mask])
        sig_noise=1/numpy.sqrt(N_readout_noise_list[j] * 0.2)
        axs[i, j].set_title('output noise {:.3f} (N={}), input noise {}'.format(sig_noise,N_readout_noise_list[j],stimuli_pars_std_list[i]))
        axs[i, j].set_xlabel('accuracy')


fig.savefig('Noise_check_accuracy')

