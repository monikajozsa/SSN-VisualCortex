import os
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy
import pandas as pd

numpy.random.seed(0)

from util import find_A, take_log
from training import train_ori_discr
from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    readout_pars,
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
#randomize_params(ssn_layer_pars, stimuli_pars, percent=0.2)

# Find normalization constant of Gabor filters
ssn_pars.A = find_A(
    filter_pars.k,
    filter_pars.sigma_g,
    filter_pars.edge_deg,
    filter_pars.degree_per_pixel,
    indices=np.sort(ssn_ori_map_loaded.ravel()),
    phase=0,
)
if ssn_pars.phases == 4:
    ssn_pars.A2 = find_A(
        filter_pars.k,
        filter_pars.sigma_g,
        filter_pars.edge_deg,
        filter_pars.degree_per_pixel,
        indices=np.sort(ssn_ori_map_loaded.ravel()),
        phase=np.pi / 2,
    )
 
####################### TRAINING PARAMETERS #######################

#randomize_params(ssn_layer_pars, stimuli_pars, percent=0.1)
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
    pretraining=False

constant_pars = ConstantPars()

# Collect training parameters into two dictionaries
trained_pars_stage1 = dict(w_sig=readout_pars.w_sig, b_sig=readout_pars.b_sig)

trained_pars_stage2 = dict(
    log_J_2x2_m= take_log(ssn_layer_pars.J_2x2_m),
    log_J_2x2_s= take_log(ssn_layer_pars.J_2x2_s),
    c_E=ssn_layer_pars.c_E,
    c_I=ssn_layer_pars.c_I,
    f_E=ssn_layer_pars.f_E,
    f_I=ssn_layer_pars.f_I,
    kappa_pre=ssn_layer_pars.kappa_pre,
    kappa_post=ssn_layer_pars.kappa_post,
)

####################### TRAINING #######################
results_filename, final_folder_path = save_code()
training_output_df = train_ori_discr(
        trained_pars_stage1,
        trained_pars_stage2,
        constant_pars,
        results_filename,
        jit_on=False
    )

######### PLOT RESULTS ############
# Load the DataFrame from the CSV file
df = pd.read_csv(results_filename)
fig_filename = os.path.join(final_folder_path,'results_fig')

visualization.plot_results_from_csv(results_filename,fig_filename)