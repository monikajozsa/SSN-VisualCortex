## This code will plot key features of the tuning curves but it is under development
## This code needs testing!

import jax.numpy as np
import numpy
import pandas as pd
import time
import matplotlib.pyplot as plt

from visualization import plot_pre_post_scatter
from analysis import tuning_curves
from pretraining_supp import load_parameters
from util_gabor import init_untrained_pars
'''
from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    readout_pars,
    ssn_pars,
    ssn_layer_pars,
    conv_pars,
    training_pars,
    loss_pars,
    pretrain_pars # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
)

for i in range(7):
    results_filename = f'results/Mar06_v6/train_only/results_train_only{i}.csv'
    tuning_curve_train_only= f'results/Mar06_v6/train_only/tc_train_only_post_{i}.csv'
    trained_pars_stage1, trained_pars_stage2, _ = load_parameters(results_filename, iloc_ind = -1)
    orimap_loaded = np.load(f'results/Mar06_v6/orimap_{i}.npy')
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, None, orimap_loaded)
    responses_sup_postpre, responses_mid_postpre = tuning_curves(untrained_pars, trained_pars_stage2, tuning_curve_train_only)
'''

def tc_slope(tuning_curve, x_axis, x1, x2, normalised=False):
    """
    Calculates slope of normalized tuning_curve between points x1 and x2. tuning_curve is given at x_axis points.
    """
    #Remove baseline if normalising
    if normalised == True:
        tuning_curve = (tuning_curve - tuning_curve.min())/tuning_curve.max()
    
    #Find indices corresponding to desired x values
    idx_1 = (np.abs(x_axis - x1)).argmin()
    idx_2 = (np.abs(x_axis - x2)).argmin()
    
    grad =(np.abs(tuning_curve[idx_2] - tuning_curve[idx_1]))/(x2-x1)
    
    return grad

def full_width_half_max(vector, d_theta):
    
    #Remove baseline
    vector = vector-vector.min()
    half_height = vector.max()/2
    points_above = len(vector[vector>half_height])

    distance = d_theta * points_above
    
    return distance

def norm_slope(file_name, ori_list=np.arange(0,180,6), expand_dims=False):
    
    # Tuning curve of given cell indices
    tuning_curve = numpy.array(pd.read_csv(file_name))
    num_cells = tuning_curve.shape[1]
    
    # Find preferred orientation and center it at 55
    pref_ori = ori_list[np.argmax(tuning_curve, axis = 0)]
    norm_pref_ori = pref_ori -55

    # Full width half height
    full_width_half_max_vec = numpy.zeros(num_cells) 
    d_theta = ori_list[1]-ori_list[0]
    for i in range(0, num_cells):
        full_width_half_max_vec[i] = full_width_half_max(tuning_curve[:,i], d_theta = d_theta)

    # Norm slope
    avg_slope_vec =numpy.zeros(num_cells) 
    for i in range(num_cells):
        avg_slope_vec[i] = tc_slope(tuning_curve[:, i], x_axis = ori_list, x1 = 52, x2 = 58, normalised =True)
    if expand_dims:
        avg_slope_vec = numpy.expand_dims(avg_slope_vec, axis=0)
        full_width_half_max_vec = numpy.expand_dims(full_width_half_max_vec, axis=0)
        norm_pref_ori = numpy.expand_dims(norm_pref_ori, axis=0)

    return avg_slope_vec, full_width_half_max_vec, norm_pref_ori

start_time = time.time()

#Load response matrices - use tuning_curves
results_dir='results/Mar06_v6/'

d_theta=6
ori_list=np.arange(0,180,d_theta)
N_runs = 7

for i in range(N_runs):
    file_name_prepre = results_dir + f'tc_prepre_{i}.csv'
    file_name_post_pre = results_dir + f'tc_postpre_{i}.csv'
    file_name_post = results_dir + f'tc_post_{i}.csv'
    file_name_train_only = results_dir + f'train_only/tc_train_only_{i}.csv'
    file_name_train_only_post = results_dir + f'train_only/tc_train_only_post_{i}.csv'
    if i==0:
        norm_slope_prepre, fwhm_prepre, orientations_prepre = norm_slope(file_name_prepre, expand_dims=True)
        norm_slope_postpre, fwhm_postpre, orientations_postpre = norm_slope(file_name_post_pre, expand_dims=True)
        norm_slope_post, fwhm_post, orientations_post = norm_slope(file_name_post, expand_dims=True)
        norm_slope_train_only_pre, fwhm_train_only_pre, orientations_train_only_pre = norm_slope(file_name_train_only, expand_dims=True)
        norm_slope_train_only_post, fwhm_train_only_post, orientations_train_only_post = norm_slope(file_name_train_only_post, expand_dims=True)
    else:
        norm_slope_prepre_i, fwhm_prepre_i, orientations_prepre_i = norm_slope(file_name_prepre, expand_dims=True)
        norm_slope_postpre_i, fwhm_postpre_i, orientations_postpre_i = norm_slope(file_name_post_pre, expand_dims=True)
        norm_slope_post_i, fwhm_post_i, orientations_post_i = norm_slope(file_name_post, expand_dims=True)
        norm_slope_prepre=numpy.concatenate((norm_slope_prepre,norm_slope_prepre_i),axis=0)
        norm_slope_postpre=numpy.concatenate((norm_slope_postpre,norm_slope_postpre_i),axis=0)
        norm_slope_post=numpy.concatenate((norm_slope_post,norm_slope_post_i),axis=0)
        fwhm_prepre=numpy.concatenate((fwhm_prepre,fwhm_prepre_i),axis=0)
        fwhm_postpre=numpy.concatenate((fwhm_postpre,fwhm_postpre_i),axis=0)
        fwhm_post=numpy.concatenate((fwhm_post,fwhm_post_i),axis=0)
        orientations_prepre=numpy.concatenate((orientations_prepre,orientations_prepre_i),axis=0)
        orientations_postpre=numpy.concatenate((orientations_postpre,orientations_postpre_i),axis=0)
        orientations_post=numpy.concatenate((orientations_post,orientations_post_i),axis=0)
        norm_slope_train_only_pre_i, fwhm_train_only_pre_i, orientations_train_only_pre_i = norm_slope(file_name_train_only, expand_dims=True)
        norm_slope_train_only_post_i, fwhm_train_only_post_i, orientations_train_only_post_i = norm_slope(file_name_train_only_post, expand_dims=True)
        norm_slope_train_only_pre=numpy.concatenate((norm_slope_train_only_pre,norm_slope_train_only_pre_i),axis=0)
        norm_slope_train_only_post=numpy.concatenate((norm_slope_train_only_post,norm_slope_train_only_post_i),axis=0)
        fwhm_train_only_pre=numpy.concatenate((fwhm_train_only_pre,fwhm_train_only_pre_i),axis=0)
        fwhm_train_only_post=numpy.concatenate((fwhm_train_only_post,fwhm_train_only_post_i),axis=0)
        orientations_train_only_pre=numpy.concatenate((orientations_train_only_pre,orientations_train_only_pre_i),axis=0)
        orientations_train_only_post=numpy.concatenate((orientations_train_only_post,orientations_train_only_post_i),axis=0)


# Plots about changes before vs after training and pretraining and training only (per layer and per centered or all)
# Plotting indices
E_sup = 648+np.linspace(0, 80, 81).astype(int)
I_sup = 648+np.linspace(81, 161, 81).astype(int)
E_sup_centre = 648+np.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)
I_sup_centre = (E_sup_centre+81).astype(int)
E_mid = np.linspace(0, 647, 648).round().reshape(8, 81, -1)[0:9:2].ravel().astype(int)
I_mid =np.linspace(0, 647, 648).round().reshape(8, 81, -1)[1:9:2].ravel().astype(int)
E_mid_centre = np.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)
I_mid_centre = (E_mid_centre+81).astype(int)

labels=['E_sup','I_sup','E_sup_centre','I_sup_centre','E_mid','I_mid','E_mid_centre','I_mid_centre']
indices=[E_sup, I_sup, E_sup_centre, I_sup_centre, E_mid, I_mid, E_mid_centre, I_mid_centre]
#Create saving directory
save_dir='results/Mar06_v6'
for i in range(3):#N_runs but it is 16 plots at the moment per run
    for j in range(len(indices)):
        # Affect of pretraining on tuning curve slopes in middle layer
        
        title = 'Pretraining ' + labels[j] + f', run {i}'
        save_file = save_dir + '/figures/slope_' + labels[j] + f'_pretrain_{i}'
        plot_pre_post_scatter(x_axis = norm_slope_prepre[i,:] , y_axis = norm_slope_postpre[i,:] , orientations = orientations_prepre[i,:], indices_to_plot = indices[j], title = title, save_file = save_file)

        title = 'Training, ' + labels[j] + f', run {i}'
        save_file = save_dir + '/figures/slope_' + labels[j] + f'_train_{i}'
        plot_pre_post_scatter(x_axis = norm_slope_postpre[i,:] , y_axis = norm_slope_post[i,:] , orientations = orientations_postpre[i,:], indices_to_plot = indices[j], title = title, save_file = save_file)
        
        title = 'Training_only ' + labels[j] + f', run {i}'
        save_file = save_dir + '/train_only/slope_' + labels[j] + f'_train_only_{i}'
        plot_pre_post_scatter(x_axis = norm_slope_train_only_pre[i,:] , y_axis = norm_slope_train_only_post[i,:] , orientations = orientations_train_only_pre[i,:], indices_to_plot = indices[j], title = title, save_file = save_file)


'''
# Schoups plot
#Bin neurons into preferred orientations
bins = ['0-4', '4-12', '12-20', '20-28', '28-36', '36-44', '44-50', '+50']
labels = []
orientations = norm_oris_sup
for idx in E_indices_sup:
        #Select bin and colour
    if np.abs(orientations[idx]) <4:
        labels.append(0)
    elif np.abs(orientations[idx]) >50:
        labels.append(7)
    else:
        labels.append(int(1+np.floor((np.abs(orientations[idx]) -4)/8) ))
labels = np.asarray(labels)


#Calculate mean slope for each bin
mean_pre = []
mean_post = []
for i in range(0, len(bins)):
    print(i)
    mean_pre.append(norm_slope_sup_pre[np.argwhere(labels==i)].mean())
    mean_post.append(norm_slope_sup_post[np.argwhere(labels==i)].mean())

#Plot mean slope pre and post training
plt.scatter(np.linspace(0, 7, 8), mean_pre, label = 'pre')
plt.scatter(np.linspace(0, 7, 8), mean_post, label = 'post')
plt.xticks(ticks=np.linspace(0, 7, 8), labels=bins)
plt.legend()
plt.ylabel('Normalised slope')
plt.xlabel('PO - TO')
plt.savefig(os.path.join(save_dir, 'Schoups_plot.png'))
'''