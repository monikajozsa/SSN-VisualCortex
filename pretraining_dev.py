import os
import csv
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
from save_code import save_code
from pretraining_supp import randomize_params
import visualization

ssn_ori_map_loaded = np.load(os.path.join(os.getcwd(), "ssn_map_uniform_good.npy"))
randomize_params(ssn_layer_pars, stimuli_pars, percent=0.2)

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
results_filename, results_folder_path = save_code()
print(results_folder_path)
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

####################### Pre-TRAINING #######################
#initialize ref_ori_list and offset_list
ref_ori_list = []
offset_list = []

n_pretrain_loops=50
for i in range(n_pretrain_loops):
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
        results_filename
    )

    stimuli_pars.ref_ori = numpy.random.uniform(low=0, high=180)
    stimuli_pars.offset = numpy.random.uniform(low=4, high=5)
    ref_ori_list.append(stimuli_pars.ref_ori)
    offset_list.append(stimuli_pars.offset)
    constant_pars = ConstantPars()
    constant_pars.stimuli_pars = stimuli_pars
    ssn_layer_pars = ssn_layer_pars_dict
    readout_pars = readout_pars_dict
    print(i)

############ PLOTS ################
# Save training and validation losses
np.save(os.path.join(results_folder_path , "training_losses.npy"), all_losses)
np.save(os.path.join(results_folder_path, "validation_losses.npy"), val_loss_per_epoch)

# Plot J, c, f, kappa,
results_plot_dir = os.path.join(results_folder_path, "plot_results")
visualization.plot_results_two_layers(
    results_filename, bernoulli=False, epochs_plot=epochs_plot, save=results_plot_dir
)

# Plot losses
losses_dir = os.path.join(results_folder_path, "plot_losses")
visualization.plot_losses_two_stage(
    all_losses, val_loss_per_epoch, epochs_plot, save=losses_dir, inset=False
)

# Plot training_accs
training_accs_dir = os.path.join(results_folder_path, "plot_training_accs")
visualization.plot_training_accs(train_accs, epochs_plot, save=training_accs_dir)

# Plot sigmoid layer parameters
sig_dir = os.path.join(results_folder_path, "plot_sigmoid")
visualization.plot_sigmoid_outputs(
    train_sig_input=train_sig_input,
    val_sig_input=val_sig_input,
    train_sig_output=train_sig_output,
    val_sig_output=val_sig_output,
    epochs_plot=epochs_plot,
    save=sig_dir,
)

# save ref_ori_list and offset_list into a file
combined_stim_pars = list(zip(ref_ori_list, offset_list))
with open(os.path.join(results_folder_path,'stim_list.csv'), 'w', newline='') as f:
    # Create a CSV writer
    writer = csv.writer(f)
    # Write the data to the CSV file
    writer.writerows(combined_stim_pars)

'''
# plotting changes in J
import pandas as pd
import matplotlib.pyplot as plt

def plot_variable_against_epoch(file_path, var_names, xlabel='Epoch', ylabel='J values', title='J vs Epoch',islabel=False):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Plot each variable in J_values against 'epoch'
    if islabel:
        for var_name in var_names:
            plt.plot(df['epoch'], df[var_name], label=var_name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
    else:
        for var_name in var_names:
            plt.plot(df['epoch'], df[var_name],label='_nolegend_')


file_path = 'C:/Users/jozsa/Desktop/Postdoc 2023-24/ABL-MJ/results/Nov27_v4/Nov27_v4_results.csv'
J_values = ['J_EE_m', 'J_IE_m', 'J_EI_m', 'J_II_m','J_EE_s', 'J_IE_s', 'J_EI_s', 'J_II_s']  # Replace with the actual column names in your CSV file
plot_variable_against_epoch(file_path, J_values)
file_path = 'C:/Users/jozsa/Desktop/Postdoc 2023-24/ABL-MJ/results/Nov27_v3/Nov27_v3_results.csv'
plot_variable_against_epoch(file_path, J_values, islabel=True)
plt.legend(loc=2)  # Add a legend to differentiate between different J values
plt.show()
'''