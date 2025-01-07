import jax.numpy as jnp
from jax import vmap

def evaluate_model_response(
    ssn_mid, ssn_sup, stimuli, conv_pars,  cE_m, cI_m, cE_s, cI_s, f_E, f_I, gabor_filters, distance_from_single_ori, kappa_f=jnp.array([0.0,0.0]), kappa_range=90
):
    """
    This function runs individual stimulus through two layer model.    
    Inputs:
        ssn_mid, ssn_sup: instances of SSN_mid and SSN_sup classes
        stimuli: stimuli to pass through network
        conv_pars: instance of ConvPars class containing convergence parameters for ssn 
        c_E, c_I: baseline inhibition for middle and superficial layers
        f_E, f_I: feedforward connections between layers    
    Outputs:
        r_sup, r_mid: fixed point of output neurons in middle and superficial layers (excitatory neurons in middle and all neurons that contribute to readout from superficial layer)
        fp_mid, fp_sup: fixed point of all neurons in middle and superficial layers
        avg_dx_mid, avg_dx_sup: average change in the second half of the fixed point calculation
        max_E_mid, max_I_mid, max_E_sup, max_I_sup: maximum rates of excitatory and inhibitory neurons in middle and superficial layers
        mean_E_mid, mean_I_mid, mean_E_sup, mean_I_sup: mean rates of excitatory and inhibitory neurons in middle and superficial layers
    """
    # Create vector using extrasynaptic constants
    constant_vector = constant_to_vec(c_E=cE_m, c_I=cI_m, ssn=ssn_mid)
    if ssn_sup.couple_c_ms:
        constant_vector_sup = constant_to_vec(c_E=cE_m, c_I=cI_m, ssn=ssn_sup, sup=True) # if c_m = c_s then this is needed so that the gradient of c_m reflects the correct effect
    else:    
        constant_vector_sup = constant_to_vec(c_E=cE_s, c_I=cI_s, ssn=ssn_sup, sup=True)

    # Apply Gabor filters to stimuli to create input of middle layer
    if stimuli.shape[0]==1:
        # handling the case when there are no batches
        input_mid = jnp.reshape(jnp.matmul(gabor_filters, jnp.transpose(stimuli)),len(constant_vector))
    else:
        input_mid = jnp.matmul(gabor_filters, stimuli)
    input_mid = input_mid.ravel()

    # Rectify middle layer input before fix point calculation
    SSN_mid_input = jnp.maximum(0, input_mid) + constant_vector

    # Calculate steady state response of middle layer
    r_mid, fp_mid, avg_dx_mid, max_E_mid, max_I_mid, mean_E_mid, mean_I_mid = middle_layer_fixed_point(ssn_mid, SSN_mid_input, conv_pars, return_fp=True)

    # Create input to (I and E neurons in) superficial layer
    # sup_input_ref = jnp.hstack([r_mid * f_E, r_mid * f_I]) + constant_vector_sup # this is the original line without kappa_f
    tanh_kappa_f = jnp.tanh(kappa_f) # kappa_f needs to be an input to evaluate_model_response
    sup_input_ref = jnp.hstack([f_E * jnp.exp(tanh_kappa_f[0] * distance_from_single_ori[:,0]**2/(2*kappa_range**2)) * r_mid, f_I * jnp.exp(tanh_kappa_f[1] * distance_from_single_ori[:,0]) * r_mid]) + constant_vector_sup

    # Calculate steady state response of superficial layer
    r_sup, fp_sup, avg_dx_sup, max_E_sup, max_I_sup, mean_E_sup, mean_I_sup = superficial_layer_fixed_point(
        ssn_sup, sup_input_ref, conv_pars, return_fp=True
    )

    return [r_sup, r_mid], [fp_mid, fp_sup], [avg_dx_mid, avg_dx_sup], [max_E_mid, max_I_mid, max_E_sup, max_I_sup], [mean_E_mid, mean_I_mid, mean_E_sup, mean_I_sup]

# vmap evaluate_model_response along bach dimension
vmap_evaluate_model_response = vmap(evaluate_model_response, in_axes = (None, None, 0, None, None, None, None, None, None, None, None, None, None, None))


def evaluate_model_response_mid(
    ssn_mid, stimuli, conv_pars, c_E, c_I, gabor_filters
):
    """
    This function runs individual stimulus through the first (referred to as middle) layer of the model.   
    Inputs:
        ssn_mid: instance of SSN_mid class
        stimuli: stimuli to pass through network
        conv_pars: instance of ConvPars class containing convergence parameters for ssn 
        c_E, c_I: baseline inhibition for middle layer    
    Outputs:
        r_mid: fixed point of excitatory neurons
        fp_mid: fixed point of all neurons
        avg_dx_mid: average change in the second half of the fixed point calculation
        max_E_mid, max_I_mid: maximum rates of excitatory and inhibitory neurons
        mean_E_mid, mean_I_mid: mean rates of excitatory and inhibitory neurons
    """
    # Create vector using extrasynaptic constants
    constant_vector = constant_to_vec(c_E=c_E, c_I=c_I, ssn=ssn_mid)

    # Apply Gabor filters to stimuli to create input of middle layer
    if stimuli.shape[0]==1:
        # handling the case when there are no batches
        input_mid = jnp.reshape(jnp.matmul(gabor_filters, jnp.transpose(stimuli)),len(constant_vector))
    else:
        input_mid = jnp.matmul(gabor_filters, stimuli)

    # Rectify middle layer input before fix point calculation
    SSN_mid_input = jnp.maximum(0, input_mid) + constant_vector

    # Calculate steady state response of middle layer
    r_mid, fp_mid, avg_dx_mid, max_E_mid, max_I_mid, mean_E_mid, mean_I_mid = middle_layer_fixed_point(ssn_mid, SSN_mid_input, conv_pars, return_fp=True)

    return r_mid, fp_mid, avg_dx_mid, max_E_mid, max_I_mid, mean_E_mid, mean_I_mid
# vmap evaluate_model_response_mid along bach dimension
vmap_evaluate_model_response_mid = vmap(evaluate_model_response_mid, in_axes = (None, 0, None, None, None, None))


def obtain_fixed_point(
    ssn, ssn_input, conv_pars
):
    """ This function calculates the fixed point of an SSN model. 
    Inputs:
        ssn: instance of _SSN_Base class (either SSN_mid or SSN_sup)
        ssn_input: input to the SSN model
        conv_pars: instance of ConvPars class containing convergence parameters for SSN model
    Outputs:
        r_fp: final values of the fixed point calculation
        avg_dx: average change in the second half of the fixed point calculation
    """
    r_init = jnp.zeros(ssn_input.shape[0])
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

    avg_dx = jnp.maximum(0, (avg_dx - 1))

    return r_fp, avg_dx


def middle_layer_fixed_point(
    ssn,
    ssn_input,
    conv_pars,
    return_fp=False,
):
    """ This function calculates the fixed point of the middle layer of the SSN model.
    Inputs:
        ssn: instance of SSN_mid class
        ssn_input: input to the middle layer
        conv_pars: instance of ConvPars class containing convergence parameters for ssn
    Outputs:
        layer_output: output of the middle layer
        avg_dx: average change in the second half of the fixed point calculation
        maxr_E, maxr_I: maximum rate of excitatory and inhibitory neurons
        meanr_E, meanr_I: mean rate of excitatory and inhibitory neurons
    """
    # Calculate layer response and SSN convergence level
    fp, avg_dx = obtain_fixed_point(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars)
    
    # Define excitatory and inhibitory cell indices and responses
    map_numbers_E = jnp.arange(1, 2 * ssn.phases, 2) # 1,3,5,7
    map_numbers_I = jnp.arange(2, 2 * ssn.phases + 1, 2) # 2,4,6,8
    fp_E=ssn.select_type(fp, map_numbers_E)
    fp_I=ssn.select_type(fp, map_numbers = map_numbers_I)
 
    # Define output as sum of E neurons (sum over phases)
    layer_output = jnp.sum(fp_E, axis=0)
    
    # Find maximum and mean rates
    maxr_E =  jnp.max(fp_E)
    maxr_I = jnp.max(fp_I)
    meanr_E =  jnp.mean(fp_E)
    meanr_I = jnp.mean(fp_I)

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
    """ This function calculates the fixed point of the superficial layer of the SSN model.
    Inputs:
        ssn: instance of SSN_sup class
        ssn_input: input to the superficial layer
        conv_pars: instance of ConvPars class containing convergence parameters for ssn
    Outputs:
        layer_output: output of the superficial layer
        avg_dx: average change in the second half of the fixed point calculation
        maxr_E, maxr_I: maximum rate of excitatory and inhibitory neurons
        meanr_E, meanr_I: mean rate of excitatory and inhibitory neurons
    """
    # Calculate layer response and SSN convergence level
    fp, avg_dx = obtain_fixed_point(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars)

    # Define output as sum of E neurons
    layer_output = fp[: ssn.Ne]
    
    # Find maximum and mean rates
    maxr_E =  jnp.max(fp[: ssn.Ne])
    maxr_I = jnp.max(fp[ssn.Ne:-1])
    meanr_E = jnp.mean(fp[: ssn.Ne])
    meanr_I = jnp.mean(fp[ssn.Ne:-1])

    if return_fp ==True:
        return layer_output, fp, avg_dx,  maxr_E, maxr_I, meanr_E, meanr_I
    else:
        return layer_output, avg_dx


def constant_to_vec(c_E, c_I, ssn, sup=False):
    """ This function create a vector from the baseline inihibitory and excitatory constants for the SSN model.
    Inputs:
        c_E, c_I: baseline excitation and inhibition added to the fixed points of middle and/or superficial layers
        ssn: instance of SSN_mid or SSN_sup class
        sup: bool indicating if the vector is for the superficial layer
    Outputs:
        constant_vec: vector of baseline inhibition for the SSN model
    """
    edge_length = ssn.grid_pars.gridsize_Nx

    matrix_E = jnp.ones((edge_length, edge_length)) * c_E
    vec_E = jnp.ravel(matrix_E)

    matrix_I = jnp.ones((edge_length, edge_length)) * c_I
    vec_I = jnp.ravel(matrix_I)

    constant_vec = jnp.hstack((vec_E, vec_I, vec_E, vec_I))

    if sup == False and ssn.phases == 4:
        constant_vec = jnp.kron(jnp.asarray([1, 1]), constant_vec)

    if sup:
        constant_vec = jnp.hstack((vec_E, vec_I))

    return constant_vec