import os
import jax.numpy as np
import numpy

numpy.random.seed(0)

import util
from training import train_model
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

####################### TRAINING PARAMETERS #######################

# Collect constant parameters into single class
class constant_pars:
    ssn_pars = ssn_pars
    s_2x2 = ssn_layer_pars.s_2x2_s
    sigma_oris = ssn_layer_pars.sigma_oris
    grid_pars = grid_pars
    conn_pars_m = conn_pars_m
    conn_pars_s = conn_pars_s
    gE = ssn_layer_pars.gE
    gI = ssn_layer_pars.gI
    filter_pars = filter_pars
    noise_type = "poisson"
    ssn_ori_map = ssn_ori_map_loaded
    ref_ori = stimuli_pars.ref_ori


# Collect training parameters into two dictionaries
readout_pars_dict = dict(w_sig=sig_pars.w_sig, b_sig=sig_pars.b_sig)
ssn_layer_pars_dict = dict(
    J_2x2_m=ssn_layer_pars.J_2x2_m,
    J_2x2_s=ssn_layer_pars.J_2x2_s,
    c_E=ssn_layer_pars.c_E,
    c_I=ssn_layer_pars.c_I,
    f_E=ssn_layer_pars.f_E,
    f_I=ssn_layer_pars.f_I,
    kappa_pre=ssn_layer_pars.kappa_pre,
    kappa_post=ssn_layer_pars.kappa_post,
)

####################### TRAINING #######################
from save_code import save_code

results_filename = save_code()
(
    [ssn_layer_pars, readout_pars],
    val_loss_per_epoch,
    all_losses,
    train_accs,
    train_sig_input,
    train_sig_output,
    val_sig_input,
    val_sig_output,
    epochs_plot,
    save_w_sigs,
) = train_model(
    ssn_layer_pars_dict,
    readout_pars_dict,
    constant_pars,
    conv_pars,
    loss_pars,
    training_pars,
    stimuli_pars,
    results_filename,
)
