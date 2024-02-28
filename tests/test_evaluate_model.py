import os
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy
numpy.random.seed(0)

import util
from training import train_ori_discr
from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    sig_pars,
    ssn_pars,
    conn_pars_m,
    conn_pars_s,
    ssn_layer_pars,
    conv_pars,
    training_pars,
    loss_pars,
)
from save_code import save_code
from pretraining_supp import randomize_params
import visualization

ssn_ori_map_loaded = np.load(os.path.join(os.getcwd(), "ssn_map_uniform_good.npy"))

# Find normalization constant of Gabor filters
ssn_pars.A = util.find_A(
    filter_pars.k,
    filter_pars.sigma_g,
    filter_pars.edge_deg,
    filter_pars.degree_per_pixel,
    indices=np.sort(ssn_ori_map_loaded.ravel()),
    phase=0,
)
if ssn_pars.phases == 4:
    ssn_pars.A2 = util.find_A(
        filter_pars.k,
        filter_pars.sigma_g,
        filter_pars.edge_deg,
        filter_pars.degree_per_pixel,
        indices=np.sort(ssn_ori_map_loaded.ravel()),
        phase=np.pi / 2,
    )

##### testing evaluate_model_response
from SSN_classes import SSN_mid_local, SSN_sup
ssn_mid=SSN_mid_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=ssn_layer_pars.J_2x2_m, gE = ssn_layer_pars.gE[0], gI=ssn_layer_pars.gI[0], ori_map = ssn_ori_map_loaded)
ssn_sup=SSN_sup(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_s, J_2x2=ssn_layer_pars.J_2x2_s, s_2x2=ssn_layer_pars.s_2x2_s, sigma_oris = ssn_layer_pars.sigma_oris, ori_map = ssn_ori_map_loaded, train_ori = 55, kappa_post = ssn_layer_pars.kappa_post, kappa_pre = ssn_layer_pars.kappa_pre)

#import jax.numpy as np
from model import evaluate_model_response
#stimuli = np.load('traindata_fortest.npz')
from util import create_grating_training
train_data = create_grating_training(stimuli_pars, 1)
stimuli = train_data['target']
stimuli = numpy.reshape(stimuli,numpy.shape(stimuli)[1])

r_ref, [r_max_ref_mid, r_max_ref_sup], [avg_dx_ref_mid, avg_dx_ref_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = evaluate_model_response(ssn_mid, ssn_sup, stimuli, conv_pars, ssn_layer_pars.c_E, ssn_layer_pars.c_I, ssn_layer_pars.f_E, ssn_layer_pars.f_I)


''' test script for Clara's code:
import os 
import jax

import util
from util import init_set_func, load_param_from_csv
import matplotlib.pyplot as plt
from pdb import set_trace
import jax.numpy as np
import numpy

from training import train_model
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

#testing evaluate_model_response
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