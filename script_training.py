import os
import jax.numpy as np
import matplotlib.pyplot as plt
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

''' testing evaluate_model_response
from SSN_classes import SSN_mid_local, SSN_sup
ssn_mid=SSN_mid_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=ssn_layer_pars.J_2x2_m, gE = ssn_layer_pars.gE[0], gI=ssn_layer_pars.gI[0], ori_map = ssn_ori_map_loaded)
ssn_sup=SSN_sup(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_s, J_2x2=ssn_layer_pars.J_2x2_s, s_2x2=ssn_layer_pars.s_2x2_s, sigma_oris = ssn_layer_pars.sigma_oris, ori_map = ssn_ori_map_loaded, train_ori = 55, kappa_post = ssn_layer_pars.kappa_post, kappa_pre = ssn_layer_pars.kappa_pre)

#import jax.numpy as np
from model import evaluate_model_response
#stimuli = np.load('traindata_fortest.npz')
from util import create_grating_pairs
train_data = create_grating_pairs(stimuli_pars, 1)
stimuli = train_data['target']
stimuli = numpy.reshape(stimuli,numpy.shape(stimuli)[1])

r_ref, [r_max_ref_mid, r_max_ref_sup], [avg_dx_ref_mid, avg_dx_ref_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = evaluate_model_response(ssn_mid, ssn_sup, stimuli, conv_pars, ssn_layer_pars.c_E, ssn_layer_pars.c_I, ssn_layer_pars.f_E, ssn_layer_pars.f_I)

print(r_max_ref_mid)

'''    
####################### TRAINING PARAMETERS #######################
# randomize_params(ssn_layer_pars, stimuli_pars, percent=0.2)
# Collect constant parameters into single class
class ConstantPars:
    grid_pars = grid_pars
    stimuli_pars = stimuli_pars
    conn_pars_m = conn_pars_m
    conn_pars_s = conn_pars_s
    filter_pars = filter_pars
    ssn_ori_map = ssn_ori_map_loaded
    ssn_pars = ssn_pars
    ssn_layer_pars = ssn_layer_pars
    conv_pars = conv_pars
    loss_pars = loss_pars
    training_pars = training_pars

constant_pars = ConstantPars()


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
results_filename, final_folder_path = save_code()
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
        results_filename,
        jit_on=False
    )


############ PLOTS ################
# Save training and validation losses
np.save(os.path.join(final_folder_path , "training_losses.npy"), all_losses)
np.save(os.path.join(final_folder_path, "validation_losses.npy"), val_loss_per_epoch)

# Plot J, c, f, kappa,
results_plot_dir = os.path.join(final_folder_path, "plot_results")
visualization.plot_results_two_layers(
    results_filename, bernoulli=False, epoch_c=epochs_plot, save=results_plot_dir
)

# Plot losses
losses_dir = os.path.join(final_folder_path, "plot_losses")
visualization.plot_losses_two_stage(
    all_losses, val_loss_per_epoch, epoch_c=epochs_plot, save=losses_dir, inset=False
)

# Plot training_accs
training_accs_dir = os.path.join(final_folder_path, "plot_training_accs")
visualization.plot_training_accs(train_accs, epoch_c=epochs_plot, save=training_accs_dir)

# Plot sigmoid layer parameters
sig_dir = os.path.join(final_folder_path, "plot_sigmoid")
visualization.plot_sigmoid_outputs(
    train_sig_input=train_sig_input,
    val_sig_input=val_sig_input,
    train_sig_output=train_sig_output,
    val_sig_output=val_sig_output,
    epoch_c=epochs_plot,
    save=sig_dir,
)