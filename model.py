import jax.numpy as np
import numpy

from util import constant_to_vec, leaky_relu

def two_layer_model(ssn_m, ssn_s, stimuli, conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I):
    '''
    Run individual stimulus through two layer model. 
    
    Inputs:
     ssn_mid, ssn_sup: middle and superficial layer classes
     stimuli: stimuli to pass through network
     conv_pars: convergence parameters for ssn 
     constant_vector_mid, constant_vector_sup: extra synaptic constants for middle and superficial layer
     f_E, f_I: feedforward connections between layers
    
    Outputs:
     r_sup - fixed point of centre neurons (5x5) in superficial layer
     loss related terms (wrt to middle and superficial layer) :
         - r_max_": loss minimising maximum rates
         - avg_dx_": loss minimising number of steps taken during convergence 
     max_(E/I)_(mid/sup): maximum rate for each type of neuron in each layer 
     
    '''
    
    #Find input of middle layer
    stimuli_gabor=np.matmul(ssn_m.gabor_filters, stimuli)
 
    #Rectify input
    SSN_mid_input = np.maximum(0, stimuli_gabor) + constant_vector_mid
    
    #Calculate steady state response of middle layer
    r_mid, r_max_mid, avg_dx_mid, fp_mid, max_E_mid, max_I_mid = middle_layer_fixed_point(ssn_m, SSN_mid_input, conv_pars, return_fp = True)
    
    #Concatenate input to superficial layer
    sup_input_ref = np.hstack([r_mid*f_E, r_mid*f_I]) + constant_vector_sup
    
    #Calculate steady state response of superficial layer
    r_sup, r_max_sup, avg_dx_sup, fp_sup, max_E_sup, max_I_sup= obtain_fixed_point_centre_E(ssn_s, sup_input_ref, conv_pars, return_fp= True)
    return r_sup, [r_max_mid, r_max_sup], [avg_dx_mid, avg_dx_sup], [max_E_mid, max_I_mid, max_E_sup, max_I_sup], [fp_mid, fp_sup]


def middle_layer_fixed_point(
    ssn,
    ssn_input,
    conv_pars,
    Rmax_E=40,
    Rmax_I=80,
    inhibition=False,
    PLOT=False,
    save=None,
    inds=None,
    return_fp=False,
    print_dt=False,
):
    fp, avg_dx = obtain_fixed_point(
        ssn=ssn,
        ssn_input=ssn_input,
        conv_pars=conv_pars,
        PLOT=PLOT,
        save=save,
        inds=inds,
        print_dt=print_dt,
    )

    # Add responses from E and I neurons
    fp_E_on = ssn.select_type(fp, map_number=1)
    fp_E_off = ssn.select_type(fp, map_number=(ssn.phases + 1))

    layer_output = fp_E_on + fp_E_off

    # Find maximum rate
    max_E = np.max(np.asarray([fp_E_on, fp_E_off]))
    max_I = np.maximum(
        np.max(fp[3 * int(ssn.Ne / 2) : -1]), np.max(fp[int(ssn.Ne / 2) : ssn.Ne])
    )

    if ssn.phases == 4:
        fp_E_on_pi2 = ssn.select_type(fp, map_number=3)
        fp_E_off_pi2 = ssn.select_type(fp, map_number=7)

        # Changes
        layer_output = layer_output + fp_E_on_pi2 + fp_E_off_pi2
        max_E = np.max(np.asarray([fp_E_on, fp_E_off, fp_E_on_pi2, fp_E_off_pi2]))
        max_I = np.max(
            np.asarray([fp[int(x) : int(x) + 80] for x in numpy.linspace(81, 567, 4)])
        )

    # Loss for high rates
    r_max = leaky_relu(max_E, R_thresh=Rmax_E, slope=1 / Rmax_E) + leaky_relu(
        max_I, R_thresh=Rmax_I, slope=1 / Rmax_I
    )

    # layer_output = layer_output/ssn.phases
    if return_fp == True:
        return layer_output, r_max, avg_dx, fp, max_E, max_I
    else:
        return layer_output, r_max, avg_dx


def obtain_fixed_point(
    ssn, ssn_input, conv_pars, PLOT=False, save=None, inds=None, print_dt=False
):
    r_init = np.zeros(ssn_input.shape[0])
    dt = conv_pars.dt
    xtol = conv_pars.xtol
    Tmax = conv_pars.Tmax

    # Find fixed point
    if PLOT == True:
        fp, avg_dx = ssn.fixed_point_r_plot(
            ssn_input,
            r_init=r_init,
            dt=dt,
            xtol=xtol,
            Tmax=Tmax,
            PLOT=PLOT,
            save=save,
            inds=inds,
        )
    else:
        fp, _, avg_dx = ssn.fixed_point_r(
            ssn_input,
            r_init=r_init,
            dt=dt,
            xtol=xtol,
            Tmax=Tmax,
            PLOT=PLOT,
            save=save,
        )

    avg_dx = np.maximum(0, (avg_dx - 1))

    return fp, avg_dx


def obtain_fixed_point_centre_E(
    ssn,
    ssn_input,
    conv_pars,
    Rmax_E=40,
    Rmax_I=80,
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

    r_max = leaky_relu(max_E, R_thresh=Rmax_E, slope=1 / Rmax_E) + leaky_relu(
        max_I, R_thresh=Rmax_I, slope=1 / Rmax_I
    )

    if return_fp == True:
        return r_box, r_max, avg_dx, fp, max_E, max_I
    else:
        return r_box, r_max, avg_dx


def evaluate_model_response(
    ssn_mid, ssn_sup, c_E, c_I, f_E, f_I, conv_pars, train_data
):
    # Create vector using extrasynaptic constants
    constant_vector = constant_to_vec(c_E=c_E, c_I=c_I, ssn=ssn_mid)
    constant_vector_sup = constant_to_vec(c_E=c_E, c_I=c_I, ssn=ssn_sup, sup=True)

    # Apply Gabor filters to stimuli
    output_mid = np.matmul(ssn_mid.gabor_filters, train_data)

    # Rectify output
    SSN_input = np.maximum(0, output_mid) + constant_vector

    # Find fixed point for middle layer
    r_ref_mid, _, _, fp, _, _ = middle_layer_fixed_point(
        ssn_mid, SSN_input, conv_pars, return_fp=True
    )

    # Input to superficial layer
    sup_input_ref = np.hstack([r_ref_mid * f_E, r_ref_mid * f_I]) + constant_vector_sup

    # Find fixed point for superficial layer
    r_ref, _, _, fp_sup, _, _ = obtain_fixed_point_centre_E(
        ssn_sup, sup_input_ref, conv_pars, return_fp=True
    )

    # Evaluate total E and I input to neurons
    W_E_mid = ssn_mid.W.at[:, 1 : ssn_mid.phases * 2 : 2].set(0)
    W_I_mid = ssn_mid.W.at[:, 0 : ssn_mid.phases * 2 - 1 : 2].set(0)
    W_E_sup = ssn_sup.W.at[:, 81:].set(0)
    W_I_sup = ssn_sup.W.at[:, :81].set(0)

    fp = np.reshape(fp, (-1, ssn_mid.Nc))
    E_mid_input = np.ravel(W_E_mid @ fp) + SSN_input
    E_sup_input = W_E_sup @ fp_sup + sup_input_ref

    I_mid_input = np.ravel(W_I_mid @ fp)
    I_sup_input = W_I_sup @ fp_sup

    return r_ref, E_mid_input, E_sup_input, I_mid_input, I_sup_input
