import os 
import jax

import util
from util import init_set_func, load_param_from_csv
import matplotlib.pyplot as plt
from pdb import set_trace
import jax.numpy as np
import numpy

from training import train_ori_discr
#from training_staircase import train_model_staircase
from parameters import *
import analysis
from SSN_classes_middle import SSN2DTopoV1_ONOFF_local
from SSN_classes_superficial import SSN2DTopoV1
numpy.random.seed(0)


################### PARAMETER SPECIFICATION #################
#SSN layer parameter initialisation
init_set_m ='C'
init_set_s=1
J_2x2_s, s_2x2, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
J_2x2_m, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)

#Excitatory and inhibitory constants for extra synaptic GABA
c_E = 5.0
c_I = 5.0

#Superficial layer W parameters
sigma_oris = np.asarray([90.0, 90.0])
kappa_pre = np.asarray([ 0.0, 0.0])
kappa_post = np.asarray([ 0.0, 0.0])

#Feedforwards connections
f_E =  np.log(1.11)
f_I = np.log(0.7)

#Constants for Gabor filters
gE = [gE_m, gE_s]
gI = [gI_m, gI_s]
#Sigmoid layer parameters
N_neurons = 25
w_sig = numpy.random.normal(scale = 0.25, size = (N_neurons,)) / np.sqrt(N_neurons)

b_sig = 0.0

#Load orientation map
ssn_ori_map_loaded = np.load(os.path.join(os.getcwd(), 'ssn_map_uniform_good.npy'))


#Collect training terms into corresponding dictionaries
readout_pars = dict(w_sig = w_sig, b_sig = b_sig)
ssn_layer_pars = dict(J_2x2_m = J_2x2_m, J_2x2_s = J_2x2_s, c_E = c_E, c_I = c_I, f_E=f_E, f_I = f_I, kappa_pre = kappa_pre, kappa_post = kappa_post)
 
#Find normalization constant of Gabor filters
ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE[0], gI=gI[0], ori_map = ssn_ori_map_loaded)
ssn_pars.A = ssn_mid.A
ssn_pars.A2 = ssn_mid.A2

'''#testing evaluate_model_response
ssn_sup=SSN2DTopoV1(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_s, J_2x2=J_2x2_s, s_2x2=s_2x2, sigma_oris = sigma_oris, ori_map = ssn_ori_map_loaded, train_ori = 55, kappa_post = kappa_post, kappa_pre = kappa_pre)

import jax.numpy as np
from model import two_layer_model
from util import constant_to_vec
from util import create_grating_pairs
train_data = create_grating_pairs(1, stimuli_pars)
stimuli = train_data['target']
stimuli = numpy.reshape(stimuli,numpy.shape(stimuli)[1])

constant_vector = constant_to_vec(c_E=c_E, c_I=c_I, ssn=ssn_mid)
constant_vector_sup = constant_to_vec(c_E=c_E, c_I=c_I, ssn=ssn_sup, sup=True)

r_ref, [r_max_ref_mid, r_max_ref_sup], [avg_dx_ref_mid, avg_dx_ref_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = two_layer_model(ssn_mid, ssn_sup, stimuli, conv_pars, constant_vector, constant_vector_sup, ssn_layer_pars['f_E'], ssn_layer_pars['f_I'])
'''
#Collect constant parameters into single class
class constant_pars:
    ssn_pars =ssn_pars
    s_2x2 = s_2x2
    sigma_oris = sigma_oris
    grid_pars = grid_pars
    conn_pars_m = conn_pars_m
    conn_pars_s = conn_pars_s
    gE = gE
    gI = gI
    filter_pars = filter_pars
    noise_type = 'poisson'
    ssn_ori_map = ssn_ori_map_loaded
    ref_ori = stimuli_pars.ref_ori
    conv_pars = conv_pars
    loss_pars= loss_pars

    
################### RESULTS DIRECTORY #################
#Name of results csv
home_dir = os.getcwd()

#Specify folder to save results
results_dir = os.path.join(home_dir, 'results', '27-11', 'test_conv_pars_st')
if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)
        
run_dir = os.path.join(results_dir,'set_'+str(init_set_m)+'_sig_noise_'+str(training_pars.sig_noise))
results_filename = os.path.join(run_dir+'_results.csv')

    
##################### TRA#INING ############

[ssn_layer_pars, readout_pars], val_loss_per_epoch, training_losses, training_accs, train_sig_inputs, train_sig_outputs, val_sig_inputs, val_sig_outputs, epochs_plot, save_w_sigs = train_ori_discr(ssn_layer_pars, readout_pars, constant_pars, training_pars, stimuli_pars, results_filename = results_filename, results_dir = run_dir)

#Staircase training
#[ssn_layer_pars, readout_pars], val_loss_per_epoch, training_losses, training_accs, train_sig_inputs, train_sig_outputs, val_sig_inputs, val_sig_outputs, epochs_plot, save_w_sigs, saved_offsets = train_model_staircase(ssn_layer_pars, readout_pars, constant_pars, training_pars, performance_pars, results_filename = results_filename, results_dir = run_dir)

#Plot offsets
#threshold_dir = os.path.join(run_dir+'_threshold')
#analysis.plot_offset(saved_offsets, epochs_plot = epochs_plot, save = threshold_dir)

print(ssn_layer_pars)
print(readout_pars)

#Save training and validation losses
#np.save(os.path.join(run_dir+'_training_losses.npy'), training_losses)
#np.save(os.path.join(run_dir+'_validation_losses.npy'), val_loss_per_epoch)

#Plot losses
losses_dir = os.path.join(run_dir+'_losses')
analysis.plot_losses_two_stage(training_losses, val_loss_per_epoch, epochs_plot = epochs_plot, save = losses_dir, inset=False)

#Plot results
results_plot_dir =  os.path.join(run_dir+'_results')
analysis.plot_results_two_layers(results_filename = results_filename, epochs_plot = epochs_plot, save= results_plot_dir)

#Plot sigmoid
sig_dir = os.path.join(run_dir+'_sigmoid')
analysis.plot_sigmoid_outputs( train_sig_input= train_sig_inputs, val_sig_input =  val_sig_inputs, train_sig_output = train_sig_outputs, val_sig_output = val_sig_outputs, epochs_plot = epochs_plot, save=sig_dir)

    
#Plot training_accs
training_accs_dir = os.path.join(run_dir+'_training_accs')
analysis.plot_training_accs(training_accs, epochs_plot = epochs_plot, save = training_accs_dir)
