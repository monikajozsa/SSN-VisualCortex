import jax.numpy as np
from jax import vmap

def evaluate_model_response(
    ssn_mid, ssn_sup, stimuli, conv_pars, c_E, c_I, f_E, f_I, gabor_filters
):
    '''
    Run individual stimulus through two layer model. 
    
    Inputs:
     ssn_mid, ssn_sup: middle and superficial layer classes
     stimuli: stimuli to pass through network
     conv_pars: convergence parameters for ssn 
     c_E, c_I: baseline inhibition for middle and superficial layers
     f_E, f_I: feedforward connections between layers
    
    Outputs:
     r_sup - fixed point of centre neurons in superficial layer (default 5x5)
     loss related terms (wrt to middle and superficial layer) :
         - r_max_": loss minimising maximum rates
         - avg_dx_": loss minimising number of steps taken during convergence 
     max_(E/I)_(mid/sup): maximum rate for each type of neuron in each layer 
     
    '''
    # Create vector using extrasynaptic constants
    constant_vector = constant_to_vec(c_E=c_E, c_I=c_I, ssn=ssn_mid)
    constant_vector_sup = constant_to_vec(c_E=c_E, c_I=c_I, ssn=ssn_sup, sup=True)

    # Apply Gabor filters to stimuli to create input of middle layer
    if stimuli.shape[0]==1:
        # handling the case when there are no batches
        input_mid = np.reshape(np.matmul(gabor_filters, np.transpose(stimuli)),len(constant_vector))
    else:
        input_mid = np.matmul(gabor_filters, stimuli)

    # Rectify middle layer input before fix point calculation
    SSN_mid_input = np.maximum(0, input_mid) + constant_vector

    # Calculate steady state response of middle layer
    r_mid, r_max_mid, avg_dx_mid, fp_mid, max_E_mid, max_I_mid = middle_layer_fixed_point(ssn_mid, SSN_mid_input, conv_pars, return_fp=True)

    # Create input to (I and E neurons in) superficial layer
    sup_input_ref = np.hstack([r_mid * f_E, r_mid * f_I]) + constant_vector_sup

    # Calculate steady state response of superficial layer
    r_sup, r_max_sup, avg_dx_sup, fp_sup, max_E_sup, max_I_sup = superficial_layer_fixed_point(
        ssn_sup, sup_input_ref, conv_pars, return_fp=True
    )

    return r_sup, r_mid, [r_max_mid, r_max_sup], [avg_dx_mid, avg_dx_sup], [max_E_mid, max_I_mid, max_E_sup, max_I_sup], [fp_mid, fp_sup]

vmap_evaluate_model_response = vmap(evaluate_model_response, in_axes = (None, None, 0, None, None, None, None, None, None))


def obtain_fixed_point(
    ssn, ssn_input, conv_pars
):
    r_init = np.zeros(ssn_input.shape[0])
    dt = conv_pars.dt
    xtol = conv_pars.xtol
    Tmax = conv_pars.Tmax

    # Find fixed point
    r_fp, avg_dx = ssn.fixed_point_r(
        ssn_input,
        r_init=r_init,
        dt=dt,
        xtol=xtol,
        Tmax=Tmax,
    )

    avg_dx = np.maximum(0, (avg_dx - 1))

    return r_fp, avg_dx

import numpy
def middle_layer_fixed_point(
    ssn,
    ssn_input,
    conv_pars,
    Rmax_E=40,
    Rmax_I=80,
    return_fp=False,
):    
    fp, avg_dx = obtain_fixed_point(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars)

    map_numbers_E = np.arange(1, 2 * ssn.phases, 2)
    map_numbers_I = np.arange(2, 2 * ssn.phases + 1, 2)
    #fp_E_on = ssn.select_type(fp, map_number = 1)
    #fp_E_off = ssn.select_type(fp, map_number = (ssn.phases+1))
    #fp_E_on_pi2 = ssn.select_type(fp, map_number = 3)
    #fp_E_off_pi2 = ssn.select_type(fp, map_number = 7)
    # tested the match and it is ok layer_output_cp = fp_E_on + fp_E_off + fp_E_on_pi2 + fp_E_off_pi2
    fp_E=ssn.select_type_mj(fp, map_numbers_E)
    fp_I=ssn.select_type_mj(fp, map_numbers = map_numbers_I)
 
    #Define output as sum of E neurons
    layer_output = np.sum(fp_E, axis=0)
    
    #Find maximum rates
    max_E =  np.max(fp_E)
    max_I = np.max(fp_I)

    #Loss for high rates
    r_max = np.maximum(0, (max_E/Rmax_E - 1)) + np.maximum(0, (max_I/Rmax_I - 1))
    #r_max = leaky_relu(max_E, R_thresh = Rmax_E, slope = 1/Rmax_E) + leaky_relu(max_I, R_thresh = Rmax_I, slope = 1/Rmax_I)
    
    #layer_output = layer_output/ssn.phases
    if return_fp ==True:
            return layer_output, r_max, avg_dx, fp, max_E, max_I
    else:
        return layer_output, r_max, avg_dx


def superficial_layer_fixed_point(
    ssn,
    ssn_input,
    conv_pars,
    Rmax_E=40,
    Rmax_I=80,
    return_fp=False,
):    
    fp, avg_dx = obtain_fixed_point(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars)
     
    #Define output as sum of E neurons
    layer_output = fp[: ssn.Ne]
    
    #Find maximum rates
    max_E =  np.max(fp[: ssn.Ne])
    max_I = np.max(fp[ssn.Ne:-1])

    #Loss for high rates
    r_max = np.maximum(0, (max_E/Rmax_E - 1)) + np.maximum(0, (max_I/Rmax_I - 1))
    #r_max = leaky_relu(max_E, R_thresh = Rmax_E, slope = 1/Rmax_E) + leaky_relu(max_I, R_thresh = Rmax_I, slope = 1/Rmax_I)
    
    if return_fp ==True:
            return layer_output, r_max, avg_dx, fp, max_E, max_I
    else:
        return layer_output, r_max, avg_dx


def constant_to_vec(c_E, c_I, ssn, sup=False):
    edge_length = ssn.grid_pars.gridsize_Nx

    matrix_E = np.ones((edge_length, edge_length)) * c_E
    vec_E = np.ravel(matrix_E)

    matrix_I = np.ones((edge_length, edge_length)) * c_I
    vec_I = np.ravel(matrix_I)

    constant_vec = np.hstack((vec_E, vec_I, vec_E, vec_I))

    if sup == False and ssn.phases == 4:
        constant_vec = np.kron(np.asarray([1, 1]), constant_vec)

    if sup:
        constant_vec = np.hstack((vec_E, vec_I))

    return constant_vec