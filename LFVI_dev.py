import numpy
import os
import jax

from model import evaluate_model_response
from parameters import stimuli_pars, training_pars,ssn_layer_pars, ssn_pars, grid_pars, stimuli_pars, conv_pars,filter_pars, readout_pars
from util_gabor import create_gabor_filters_util
from util import cosdiff_ring, create_grating_pairs
from SSN_classes import SSN_mid, SSN_sup
from pretraining_supp import perturb_params

ssn_ori_map_loaded = numpy.load(os.path.join(os.getcwd(), "ssn_map_uniform_good.npy"))
gabor_filters, A, A2 = create_gabor_filters_util(ssn_ori_map_loaded, ssn_pars.phases, filter_pars, grid_pars, ssn_layer_pars.gE_m, ssn_layer_pars.gI_m)

class ConstantPars:
    ori_map = ssn_ori_map_loaded
    gabor_filters = gabor_filters
    p_local_s = ssn_layer_pars.p_local_s
    oris = ssn_ori_map_loaded.ravel()[:, None]
    ori_dist = cosdiff_ring(oris - oris.T, 180)
    sigma_oris = ssn_layer_pars.sigma_oris
    s_2x2 = ssn_layer_pars.s_2x2_s
    kappa_pre = ssn_layer_pars.kappa_pre
    kappa_post = ssn_layer_pars.kappa_post
    
constant_pars = ConstantPars

true_other_pars = dict( J_2x2_m = ssn_layer_pars.J_2x2_m, J_2x2_s = ssn_layer_pars.J_2x2_s, c_E = ssn_layer_pars.c_E, c_I = ssn_layer_pars.c_I, f_E = ssn_layer_pars.f_E, f_I=ssn_layer_pars.f_I)

def my_model(other_pars, constant_pars, input_data):
    J_2x2_m = other_pars['J_2x2_m']
    J_2x2_s = other_pars['J_2x2_s']
    c_E = other_pars['c_E']
    c_I = other_pars['c_I']
    f_E = other_pars['f_E']
    f_I = other_pars['f_I']

    # Create middle and superficial SSN layers *** this is something that would be great to change - to call the ssn classes from inside the training
    ssn_mid=SSN_mid(ssn_pars=ssn_pars, grid_pars=grid_pars, J_2x2=J_2x2_m)
    ssn_sup=SSN_sup(ssn_pars=ssn_pars, grid_pars=grid_pars, J_2x2=J_2x2_s, p_local=constant_pars.p_local_s, oris=constant_pars.oris, s_2x2=constant_pars.s_2x2, sigma_oris = constant_pars.sigma_oris, ori_dist = constant_pars.ori_dist, train_ori = stimuli_pars.ref_ori, kappa_post = constant_pars.kappa_post, kappa_pre = constant_pars.kappa_pre)

    #Run reference and targetthrough two layer model
    r_ref, _, [_,_], [_,_],[_,_,_,_], _ = evaluate_model_response(ssn_mid, ssn_sup, input_data, conv_pars, c_E, c_I, f_E, f_I, gabor_filters)

    # Calculate readout - no sigmoid now!
    output = numpy.dot(readout_pars.w_sig, r_ref) + readout_pars.b_sig
    
    return output

# Define a function to simulate data from your model
def simulate_data(other_pars, constant_pars, input_data):
    return my_model(other_pars, constant_pars, input_data)

def sample_from_surrogate(surrogate_params):
    '''
    This function samples parameters from a Gaussian surrogate posterior.
    Input: surrogate_params - dictionary where each item is a tuple (mean, std)
    Output: sampled_params - samples for each item in surrogate_params
    '''
    sampled_params = {}
    for param_name, (mean, std) in surrogate_params.items():
        sampled_params[param_name] = numpy.random.normal(mean, std, size=mean.shape)
    return sampled_params

# Define a direct discrepancy function
def direct_discrepancy(simulated_data, observed_data):
    # Implement a measure of discrepancy (e.g., mean squared error)
    return numpy.mean((simulated_data - observed_data) ** 2)

def indirect_discrepancy(surrogate_params, constant_pars, observed_data, input_data, num_simulations=100):
    total_discrepancy = 0
    for _ in range(num_simulations):
        sampled_params = sample_from_surrogate(surrogate_params)
        simulated_data = simulate_data(sampled_params, constant_pars, input_data)
        total_discrepancy += direct_discrepancy(simulated_data, observed_data)
    return total_discrepancy / num_simulations

calculate_gradients = jax.value_and_grad(indirect_discrepancy, argnums=0)


def update_params(surrogate_params, constant_pars, observed_data, input_data, learning_rate=0.01):
    loss, gradients = calculate_gradients(surrogate_params, constant_pars, observed_data, input_data)
    updated_params = {}
    for param_name in surrogate_params:
        updated_params[param_name] = surrogate_params[param_name] - learning_rate * gradients[param_name]
    return updated_params

# Optimization loop
num_optimization_steps=10
J_2x2_m = perturb_params(true_other_pars.J_2x2_m, percent = 0.1)
J_2x2_s = perturb_params(true_other_pars.J_2x2_s, percent = 0.1)
J_2x2_m = perturb_params(true_other_pars, percent = 0.1)
J_2x2_m = perturb_params(true_other_pars, percent = 0.1)
J_2x2_m = perturb_params(true_other_pars, percent = 0.1)
surrogate_params = dict(J_2x2_m = J_2x2_m)

for step in range(num_optimization_steps):
    # Sample parameters from the surrogate posterior
    sampled_params = sample_from_surrogate(surrogate_params)
    
    # Generate input to the model
    input_data = create_grating_pairs(stimuli_pars, training_pars.batch_size)

    # Simulate data using sampled parameters
    simulated_data = simulate_data(sampled_params, constant_pars, input_data['ref'])
    observed_data = simulate_data(true_other_pars, constant_pars, input_data['ref'])
    
    # Update surrogate parameters to minimize the discrepancy
    # This step will require calculating gradients and performing optimization step
    # Since we are not using a framework like TensorFlow or PyTorch,
    # this has to be implemented manually or with optimization libraries
    surrogate_params = update_params(surrogate_params, constant_pars, observed_data, input_data)
