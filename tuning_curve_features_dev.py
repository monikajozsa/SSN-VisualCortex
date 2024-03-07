## This code will plot key features of the tuning curves but it is under development
## This code needs testing!
import os
import jax.numpy as np
import numpy
import pandas as pd
import time

from visualization import plot_pre_post_scatter

def tc_slope(tuning_curve, x_axis, x1, x2, normalised=False):
    '''
    Calculates slope of normalized tuning_curve between points x1 and x2. tuning_curve is given at x_axis points.
    '''
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

def norm_slope(file_name, ori_list=np.arange(0,180,6)):
    
    # Tuning curve of given cell indices
    tuning_curve = numpy.array(pd.read_csv(file_name))
    num_cells = tuning_curve.shape[1]
    
    # Find preferred orientation and center it at 55
    pref_ori = ori_list[np.argmax(tuning_curve, axis = 0)]
    norm_pref_ori = pref_ori -55

    # Full width half height
    full_width_half_max_vec = []
    d_theta = ori_list[1]-ori_list[0]
    for i in range(0, num_cells):
        full_width_half_max_vec.append(full_width_half_max(tuning_curve[:,i], d_theta = d_theta))
    full_width_half_max_vec =  np.asarray(full_width_half_max_vec) 

    # Norm slope
    avg_slope_vec = []
    for i in range(0, num_cells):
        avg_slope_vec.append(tc_slope(tuning_curve[:, i], x_axis = ori_list, x1 = 52, x2 = 58, normalised =True))

    avg_slope_vec =  np.asarray(avg_slope_vec) 

    return avg_slope_vec, full_width_half_max_vec, norm_pref_ori

start_time = time.time()

#Load response matrices - use tuning_curves
results_dir='results/Mar06_v6/'

d_theta=6
ori_list=np.arange(0,180,d_theta)

file_name = results_dir + 'tc_prepre_1.csv'
norm_slope_prepre, fwhm_prepre, orientations_prepre = norm_slope(file_name)
file_name = results_dir + 'tc_postpre_1.csv'
norm_slope_postpre, fwhm_postpre, orientations_postpre = norm_slope(file_name)
file_name = results_dir + 'tc_post_1.csv'
norm_slope_post, fwhm_post, orientations_post = norm_slope(file_name)


# Plots about changes before vs after training and pretraining per layer (and also centered or not centered neurons)
# colors represent orientation tuning
# Plotting indices
centre_E_indices = 648+np.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)
centre_I_indices = 648+(centre_E_indices+81).astype(int)
E_indices_mid = np.linspace(0, 647, 648).round().reshape(8, 81, -1)[0:9:2].ravel().astype(int)
I_indices_mid =np.linspace(0, 647, 648).round().reshape(8, 81, -1)[1:9:2].ravel().astype(int)
E_indices_sup = 648+np.linspace(0, 80, 81).astype(int)
I_indices_sup = 648+np.linspace(81, 161, 81).astype(int)

#Create saving directory
save_dir='results/Mar06_v6/figures'

# Affect of pretraining on tuning curve slopes in middle layer
title = 'Slope for E_mid'
save_file = save_dir + '/slope_E_mid_pretrain'
plot_pre_post_scatter(x_axis = norm_slope_prepre , y_axis = norm_slope_postpre , orientations = orientations_prepre, indices_to_plot = E_indices_mid, title = title, save_file = save_file)

title = 'Slope for E_centre'
save_file = save_dir + '/slope_E_centre_pretrain'
plot_pre_post_scatter(x_axis = norm_slope_prepre , y_axis = norm_slope_postpre , orientations = orientations_prepre, indices_to_plot = centre_E_indices, title = title, save_file = save_file)

title = 'Slope for I_mid'
save_file = save_dir + '/slope_I_mid_pretrain'
plot_pre_post_scatter(x_axis = norm_slope_prepre , y_axis = norm_slope_postpre , orientations = orientations_prepre, indices_to_plot = I_indices_mid, title = title, save_file = save_file)

title = 'Slope for I_centre'
save_file = save_dir + '/slope_I_centre_pretrain'
plot_pre_post_scatter(x_axis = norm_slope_prepre , y_axis = norm_slope_postpre , orientations = orientations_prepre, indices_to_plot = centre_I_indices, title = title, save_file = save_file)

# Taining
title = 'Slope for E_mid'
save_file = save_dir + '/slope_E_mid_train'
plot_pre_post_scatter(x_axis = norm_slope_postpre , y_axis = norm_slope_post , orientations = orientations_postpre, indices_to_plot = E_indices_mid, title = title, save_file = save_file)

title = 'Slope for E_centre'
save_file = save_dir + '/slope_E_centre_train'
plot_pre_post_scatter(x_axis = norm_slope_postpre , y_axis = norm_slope_post , orientations = orientations_postpre, indices_to_plot = centre_E_indices, title = title, save_file = save_file)

title = 'Slope for I_mid'
save_file = save_dir + '/slope_I_mid_train'
plot_pre_post_scatter(x_axis = norm_slope_postpre , y_axis = norm_slope_post , orientations = orientations_postpre, indices_to_plot = I_indices_mid, title = title, save_file = save_file)

title = 'Slope for I_centre'
save_file = save_dir + '/slope_I_centre_train'
plot_pre_post_scatter(x_axis = norm_slope_postpre , y_axis = norm_slope_post , orientations = orientations_postpre, indices_to_plot = centre_I_indices, title = title, save_file = save_file)
print(time.time()-start_time)
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