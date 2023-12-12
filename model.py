import jax.numpy as np
import numpy

from util import constant_to_vec, leaky_relu


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
    # in order to use the gabor_filters input, I need to multiply it by gE and gI in script_pretraining...
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

    return r_sup, [r_max_mid, r_max_sup], [avg_dx_mid, avg_dx_sup], [max_E_mid, max_I_mid, max_E_sup, max_I_sup], [fp_mid, fp_sup]


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
    Rmax_E=40,
    Rmax_I=80,
    return_fp=False,
):    
    fp, avg_dx = obtain_fixed_point(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars)
    
    #Add responses from E and I neurons
    fp_E_1=ssn.select_type(fp, map_number = 1)
    fp_E=np.empty((ssn.phases,*fp_E_1.shape))
    fp_I=np.empty_like(fp_E)
    for phase_i in range(ssn.phases):
        fp_E = fp_E.at[phase_i,:].set(ssn.select_type(fp, map_number = (phase_i-1)*2+1))
        fp_I = fp_I.at[phase_i,:].set(ssn.select_type(fp, map_number = (phase_i+1)*2))
    
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
    layer_output = fp[: ssn.Ne]# ***this ravels and does not let me apply map_vec = jax.lax.dynamic_slice(map_vec, (start, start), (size, size))
    
    #Find maximum rates
    max_E =  np.max(fp[: ssn.Ne])
    max_I = np.max(fp[ssn.Ne:-1])

    #Loss for high rates
    r_max = np.maximum(0, (max_E/Rmax_E - 1)) + np.maximum(0, (max_I/Rmax_I - 1))
    #r_max = leaky_relu(max_E, R_thresh = Rmax_E, slope = 1/Rmax_E) + leaky_relu(max_I, R_thresh = Rmax_I, slope = 1/Rmax_I)
    
    #layer_output = layer_output/ssn.phases
    if return_fp ==True:
            return layer_output, r_max, avg_dx, fp, max_E, max_I
    else:
        return layer_output, r_max, avg_dx

     
'''
def obtain_fixed_point_centre_E(
    ssn,
    ssn_input,
    conv_pars,
    inhibition=False,
    PLOT=False,
    save=None,
    inds=None,
    return_fp=False,
):
    # Obtain fixed point
    fp, avg_dx = obtain_fixed_point(
        ssn=ssn,
        ssn_input=ssn_input,
        conv_pars=conv_pars,
        PLOT=PLOT,
        save=save,
        inds=inds,
    )
    
    # Apply bounding box to data
    r_box = (ssn.apply_bounding_box(fp, size=3.2)).ravel()

    # Obtain inhibitory response
    if inhibition == True:
        r_box_i = ssn.apply_bounding_box(fp, size=3.2, select="I_ON").ravel()
        r_box = [r_box, r_box_i]

    max_E = np.max(fp[: ssn.Ne])
    max_I = np.max(fp[ssn.Ne : -1])

    r_max = leaky_relu(max_E, R_thresh=conv_pars.Rmax_E, slope=1 / conv_pars.Rmax_E) + leaky_relu(
        max_I, R_thresh=conv_pars.Rmax_I, slope=1 / conv_pars.Rmax_I
    )

    if return_fp == True:
        return r_box, r_max, avg_dx, fp, max_E, max_I
    else:
        return r_box, r_max, avg_dx
'''