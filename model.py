import jax
import jax.numpy as np
import numpy

from util import sep_exponentiate, constant_to_vec, leaky_relu, sigmoid, binary_loss
from SSN_classes import SSN2DTopoV1_ONOFF_local, SSN2DTopoV1


def two_layer_model(
    logJ_2x2,
    logs_2x2,
    c_E,
    c_I,
    f_E,
    f_I,
    w_sig,
    b_sig,
    sigma_oris,
    kappa_pre,
    kappa_post,
    ssn_mid_ori_map,
    ssn_sup_ori_map,
    train_data,
    ssn_pars,
    grid_pars,
    conn_pars_m,
    conn_pars_s,
    gE_m,
    gI_m,
    gE_s,
    gI_s,
    filter_pars,
    conv_pars,
    loss_pars,
    noise_ref,
    noise_target,
    noise_type="poisson",
    train_ori=55,
    debug_flag=False,
):
    """
    SSN two-layer model. SSN layers are regenerated every run. Static arguments in jit specify parameters that stay constant throughout training. Static parameters cant be dictionaries
    Inputs:
        individual parameters - having taken logs of differentiable parameters
        noise_type: select different noise models
        debug_flag: to be used in pdb mode allowing debugging inside function
    Outputs:
        losses to take gradient with respect to
        sig_input, x: I/O values for sigmoid layer
    """

    J_2x2_m = sep_exponentiate(logJ_2x2[0])
    J_2x2_s = sep_exponentiate(logJ_2x2[1])
    s_2x2_s = np.exp(logs_2x2)
    sigma_oris_s = np.exp(sigma_oris)

    _f_E = np.exp(f_E)
    _f_I = np.exp(f_I)

    kappa_pre = np.tanh(kappa_pre)
    kappa_post = np.tanh(kappa_post)

    # Initialise network
    ssn_mid = SSN2DTopoV1_ONOFF_local(
        ssn_pars=ssn_pars,
        grid_pars=grid_pars,
        conn_pars=conn_pars_m,
        filter_pars=filter_pars,
        J_2x2=J_2x2_m,
        gE=gE_m,
        gI=gI_m,
        ori_map=ssn_mid_ori_map,
    )
    ssn_sup = SSN2DTopoV1(
        ssn_pars=ssn_pars,
        grid_pars=grid_pars,
        conn_pars=conn_pars_s,
        filter_pars=filter_pars,
        J_2x2=J_2x2_s,
        s_2x2=s_2x2_s,
        gE=gE_s,
        gI=gI_s,
        sigma_oris=sigma_oris_s,
        ori_map=ssn_sup_ori_map,
        train_ori=train_ori,
        kappa_post=kappa_post,
        kappa_pre=kappa_pre,
    )

    # Create vector using extrasynaptic constants
    constant_vector = constant_to_vec(c_E=c_E, c_I=c_I, ssn=ssn_mid)
    constant_vector_sup = constant_to_vec(c_E=c_E, c_I=c_I, ssn=ssn_sup, sup=True)

    # Apply Gabor filters to stimuli
    output_ref = np.matmul(ssn_mid.gabor_filters, train_data["ref"])
    output_target = np.matmul(ssn_mid.gabor_filters, train_data["target"])

    # Rectify output
    SSN_input_ref = np.maximum(0, output_ref) + constant_vector
    SSN_input_target = np.maximum(0, output_target) + constant_vector

    # Find fixed point for middle layer
    (
        r_ref_mid,
        r_max_ref_mid,
        avg_dx_ref_mid,
        _,
        max_E_mid,
        max_I_mid,
    ) = middle_layer_fixed_point(ssn_mid, SSN_input_ref, conv_pars, return_fp=True)
    r_target_mid, r_max_target_mid, avg_dx_target_mid = middle_layer_fixed_point(
        ssn_mid, SSN_input_target, conv_pars
    )

    # Input to superficial layer
    sup_input_ref = (
        np.hstack([r_ref_mid * _f_E, r_ref_mid * _f_I]) + constant_vector_sup
    )
    sup_input_target = (
        np.hstack([r_target_mid * _f_E, r_target_mid * _f_I]) + constant_vector_sup
    )

    # Find fixed point for superficial layer
    (
        r_ref,
        r_max_ref_sup,
        avg_dx_ref_sup,
        _,
        max_E_sup,
        max_I_sup,
    ) = obtain_fixed_point_centre_E(ssn_sup, sup_input_ref, conv_pars, return_fp=True)
    r_target, r_max_target_sup, avg_dx_target_sup = obtain_fixed_point_centre_E(
        ssn_sup, sup_input_target, conv_pars
    )

    # Add noise
    if noise_type == "additive":
        r_ref = r_ref + noise_ref
        r_target = r_target + noise_target

    elif noise_type == "multiplicative":
        r_ref = r_ref * (1 + noise_ref)
        r_target = r_target * (1 + noise_target)

    elif noise_type == "poisson":
        r_ref = r_ref + noise_ref * np.sqrt(jax.nn.softplus(r_ref))
        r_target = r_target + noise_target * np.sqrt(jax.nn.softplus(r_target))

    delta_x = r_ref - r_target
    sig_input = np.dot(w_sig, (delta_x)) + b_sig

    # Apply sigmoid function - combine ref and target
    x = sigmoid(sig_input)

    # Calculate losses
    loss_binary = binary_loss(train_data["label"], x)
    loss_avg_dx = (
        loss_pars.lambda_dx
        * (avg_dx_ref_mid + avg_dx_target_mid + avg_dx_ref_sup + avg_dx_target_sup)
        / 4
    )
    loss_r_max = (
        loss_pars.lambda_r_max
        * (r_max_ref_mid + r_max_target_mid + r_max_ref_sup + r_max_target_sup)
        / 4
    )
    loss_w = loss_pars.lambda_w * (np.linalg.norm(w_sig) ** 2)
    loss_b = loss_pars.lambda_b * (b_sig**2)

    # Combine all losses
    loss = loss_binary + loss_w + loss_b + loss_avg_dx + loss_r_max
    all_losses = np.vstack((loss_binary, loss_avg_dx, loss_r_max, loss_w, loss_b, loss))
    pred_label = np.round(x)

    return (
        loss,
        all_losses,
        pred_label,
        sig_input,
        x,
        [max_E_mid, max_I_mid, max_E_sup, max_I_sup],
    )


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
    verbose = conv_pars.verbose
    silent = conv_pars.silent

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

    # r_max = np.maximum(0, (max_E/Rmax_E - 1)) + np.maximum(0, (max_I/Rmax_I - 1))
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
