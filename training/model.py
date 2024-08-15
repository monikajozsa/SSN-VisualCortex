import jax.numpy as np
from jax import vmap

def evaluate_model_response(
    ssn_mid, ssn_sup, stimuli, conv_pars,  cE_m, cI_m, cE_s, cI_s, f_E, f_I, gabor_filters
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
    constant_vector = constant_to_vec(c_E=cE_m, c_I=cI_m, ssn=ssn_mid)
    constant_vector_sup = constant_to_vec(c_E=cE_s, c_I=cI_s, ssn=ssn_sup, sup=True)

    # Apply Gabor filters to stimuli to create input of middle layer
    if stimuli.shape[0]==1:
        # handling the case when there are no batches
        input_mid = np.reshape(np.matmul(gabor_filters, np.transpose(stimuli)),len(constant_vector))
    else:
        input_mid = np.matmul(gabor_filters, stimuli)
    input_mid = input_mid.ravel()

    # Rectify middle layer input before fix point calculation
    SSN_mid_input = np.maximum(0, input_mid) + constant_vector

    # Calculate steady state response of middle layer
    r_mid, fp_mid, avg_dx_mid, max_E_mid, max_I_mid, mean_E_mid, mean_I_mid = middle_layer_fixed_point(ssn_mid, SSN_mid_input, conv_pars, return_fp=True)

    # Create input to (I and E neurons in) superficial layer
    sup_input_ref = np.hstack([r_mid * f_E, r_mid * f_I]) + constant_vector_sup

    # Calculate steady state response of superficial layer
    r_sup, fp_sup, avg_dx_sup, max_E_sup, max_I_sup, mean_E_sup, mean_I_sup = superficial_layer_fixed_point(
        ssn_sup, sup_input_ref, conv_pars, return_fp=True
    )

    return [r_sup, r_mid], [fp_mid, fp_sup], [avg_dx_mid, avg_dx_sup], [max_E_mid, max_I_mid, max_E_sup, max_I_sup], [mean_E_mid, mean_I_mid, mean_E_sup, mean_I_sup]

vmap_evaluate_model_response = vmap(evaluate_model_response, in_axes = (None, None, 0, None, None, None, None, None, None, None, None))


def evaluate_model_response_mid(
    ssn_mid, stimuli, conv_pars, c_E, c_I, gabor_filters
):
    '''
    Run individual stimulus through one layer model. 
    
    Inputs:
     ssn_mid: middle layer class
     stimuli: stimuli to pass through network
     conv_pars: convergence parameters for ssn 
     c_E, c_I: baseline inhibition for middle layer
    
    Outputs:
     r_mid - fixed point of centre neurons in middle layer (default 5x5)
     loss related terms (wrt to middle and superficial layer) :
         - r_max_": loss minimising maximum rates
         - avg_dx_": loss minimising number of steps taken during convergence 
     max_(E/I)_(mid/sup): maximum rate for each type of neuron in each layer 
     
    '''
    # Create vector using extrasynaptic constants
    constant_vector = constant_to_vec(c_E=c_E, c_I=c_I, ssn=ssn_mid)

    # Apply Gabor filters to stimuli to create input of middle layer
    if stimuli.shape[0]==1:
        # handling the case when there are no batches
        input_mid = np.reshape(np.matmul(gabor_filters, np.transpose(stimuli)),len(constant_vector))
    else:
        input_mid = np.matmul(gabor_filters, stimuli)

    # Rectify middle layer input before fix point calculation
    SSN_mid_input = np.maximum(0, input_mid) + constant_vector

    # Calculate steady state response of middle layer
    r_mid, fp_mid, avg_dx_mid, max_E_mid, max_I_mid, mean_E_mid, mean_I_mid = middle_layer_fixed_point(ssn_mid, SSN_mid_input, conv_pars, return_fp=True)

    return r_mid, fp_mid, avg_dx_mid, max_E_mid, max_I_mid, mean_E_mid, mean_I_mid

vmap_evaluate_model_response_mid = vmap(evaluate_model_response_mid, in_axes = (None, 0, None, None, None, None))


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

def middle_layer_fixed_point(
    ssn,
    ssn_input,
    conv_pars,
    return_fp=False,
):    
    # Calculate layer response and SSN convergence level
    fp, avg_dx = obtain_fixed_point(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars)
    
    # Define excitatory and inhibitory cell indices and responses
    map_numbers_E = np.arange(1, 2 * ssn.phases, 2) # 1,3,5,7
    map_numbers_I = np.arange(2, 2 * ssn.phases + 1, 2) # 2,4,6,8
    fp_E=ssn.select_type(fp, map_numbers_E)
    fp_I=ssn.select_type(fp, map_numbers = map_numbers_I)
 
    # Define output as sum of E neurons
    layer_output = np.sum(fp_E, axis=0)
    
    # Find maximum and mean rates
    maxr_E =  np.max(fp_E)
    maxr_I = np.max(fp_I)
    meanr_E =  np.mean(fp_E)
    meanr_I = np.mean(fp_I)

    if return_fp ==True:
        return layer_output, fp, avg_dx, maxr_E, maxr_I,  meanr_E, meanr_I
    else:
        return layer_output, avg_dx


def superficial_layer_fixed_point(
    ssn,
    ssn_input,
    conv_pars,
    return_fp=False,
):    
    # Calculate layer response and SSN convergence level
    fp, avg_dx = obtain_fixed_point(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars)

    # Define output as sum of E neurons
    layer_output = fp[: ssn.Ne]
    
    # Find maximum and mean rates
    maxr_E =  np.max(fp[: ssn.Ne])
    maxr_I = np.max(fp[ssn.Ne:-1])
    meanr_E = np.mean(fp[: ssn.Ne])
    meanr_I = np.mean(fp[ssn.Ne:-1])

    if return_fp ==True:
        return layer_output, fp, avg_dx,  maxr_E, maxr_I, meanr_E, meanr_I
    else:
        return layer_output, avg_dx


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