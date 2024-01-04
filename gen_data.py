# Generate synthetic observed data
import numpy
import os
import jax
from jax import numpy as np

from util import create_grating_pairs, cosdiff_ring
from util_gabor import create_gabor_filters_util
from parameters import stimuli_pars, training_pars,ssn_layer_pars, ssn_pars, grid_pars, stimuli_pars, conv_pars,filter_pars, readout_pars
from SSN_classes import SSN_mid, SSN_sup
from model import evaluate_model_response

def generate_noise(sig_noise,  batch_size, length):
    '''
    Creates vectors of neural noise. Function creates N vectors, where N = batch_size, each vector of length = length. 
    '''
    return sig_noise*numpy.random.randn(batch_size, length)


input_data = create_grating_pairs(stimuli_pars, training_pars.batch_size)
ssn_ori_map_loaded = numpy.load(os.path.join(os.getcwd(), "ssn_map_uniform_good.npy"))
gabor_filters, A, A2 = create_gabor_filters_util(ssn_ori_map_loaded, ssn_pars.phases, filter_pars, grid_pars, ssn_layer_pars.gE_m, ssn_layer_pars.gI_m)

J_2x2_m = ssn_layer_pars.J_2x2_m
J_2x2_s = ssn_layer_pars.J_2x2_s
p_local_s = ssn_layer_pars.p_local_s
oris = ssn_ori_map_loaded.ravel()[:, None]
ori_dist = cosdiff_ring(oris - oris.T, 180)
sigma_oris = ssn_layer_pars.sigma_oris
s_2x2 = ssn_layer_pars.s_2x2_s
kappa_pre = ssn_layer_pars.kappa_pre
kappa_post = ssn_layer_pars.kappa_post
c_E = ssn_layer_pars.c_E
c_I = ssn_layer_pars.c_I
f_E = ssn_layer_pars.f_E
f_I = ssn_layer_pars.f_I

# Create middle and superficial SSN layers *** this is something that would be great to change - to call the ssn classes from inside the training
ssn_mid=SSN_mid(ssn_pars=ssn_pars, grid_pars=grid_pars, J_2x2=J_2x2_m)
ssn_sup=SSN_sup(ssn_pars=ssn_pars, grid_pars=grid_pars, J_2x2=J_2x2_s, p_local=p_local_s, oris=oris, s_2x2=s_2x2, sigma_oris = sigma_oris, ori_dist = ori_dist, train_ori = stimuli_pars.ref_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)

#Run reference and targetthrough two layer model
r_ref, _, [r_max_ref_mid, r_max_ref_sup], [avg_dx_ref_mid, avg_dx_ref_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = evaluate_model_response(ssn_mid, ssn_sup, input_data['ref'], conv_pars, c_E, c_I, f_E, f_I, gabor_filters)
r_target, _, [r_max_target_mid, r_max_target_sup], [avg_dx_target_mid, avg_dx_target_sup], _, _= evaluate_model_response(ssn_mid, ssn_sup, input_data['target'], conv_pars, c_E, c_I, f_E, f_I, gabor_filters)

# Generate noise
noise_ref = generate_noise(
    training_pars.sig_noise, training_pars.batch_size, readout_pars.w_sig.shape[0]
)
noise_target = generate_noise(
    training_pars.sig_noise, training_pars.batch_size, readout_pars.w_sig.shape[0]
)
r_ref_box = r_ref + noise_ref*np.sqrt(jax.nn.softplus(r_ref))
r_target_box = r_target + noise_target*np.sqrt(jax.nn.softplus(r_target))

# Calculate readout - no sigmoid now!
sig_input = np.dot(readout_pars.w_sig, r_ref_box) + readout_pars.b_sig
sig_output = sig_input

# save sig_output into file
sig_output_numpy = numpy.array(sig_output)
numpy.save('simulated_output.npy', sig_output_numpy)