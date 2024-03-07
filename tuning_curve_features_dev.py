## This code will plot key features of the tuning curves but it is under development
## Do not run this code yet
import os
import jax.numpy as np
import pandas as pd

from visualization import plot_pre_post_scatter

def avg_slope(vector, x_axis, x1, x2, normalised=False):
    '''
    Calculates average slope between points x1 and x2. x1 and x2  given in absolute values, then converted to indices in function 
    '''
    #Remove baseline if normalising
    if normalised == True:
        vector = (vector - vector.min())/vector.max()
    
    #Find indices corresponding to desired x values
    idx_1 = (np.abs(x_axis - x1)).argmin()
    idx_2 = (np.abs(x_axis - x2)).argmin()
    
    grad =(np.abs(vector[idx_2] - vector[idx_1]))/(x2-x1)
    
    return grad

def full_width_half_max(vector, d_theta):
    
    #Remove baseline
    vector = vector-vector.min()
    half_height = vector.max()/2
    points_above = len(vector[vector>half_height])

    distance = d_theta * points_above
    
    return distance

#Load response matrices - use tuning_curves
results_dir='results/Mar06_v6'

d_theta=6
ori_list=np.arange(0,180,d_theta)

def norm_slope(file_name, layer, ori_list=np.arange(0,180,6), SGD_step_ind = 1):
   
    tuning_curve = np.load(os.path.join(results_dir, 'response_epoch'+str(SGD_step_ind)+'_mid.npy')).squeeze() # ***
    
    #0. Find preferred orientation to bin neurons
    pref_ori = ori_list[np.argmax(tuning_curve, axis = 1)]

    #Normalise preferred orientations
    norm_oris_mid = pref_ori -55

    #1. Calculate baseline value
    min_val =tuning_curve.min(axis = 1)

    #2. Calculate maximum value
    max_val =tuning_curve.max(axis = 1)

    #3. Full width half height
    #Middle layer
    full_width_half_max_vec = []
    for i in range(0, len(tuning_curve)):
        full_width_half_max_vec.append(full_width_half_max(tuning_curve[i, :], d_theta = d_theta))
    full_width_half_max_vec =  np.asarray(full_width_half_max_vec)  ## CHANGE HERE

    #5. Norm slope
    avg_slope_vec = []
    for i in range(0, len(tuning_curve)):
        avg_slope_vec.append(avg_slope(tuning_curve[i, :], x_axis = ori_list, x1 = 52, x2 = 58, normalised =True))

    avg_slope_vec =  np.asarray(avg_slope_vec) ## CHANGE HERE

    return avg_slope_vec, norm_oris_mid

# 1. I need to use 0 and 1 to define indices of middle layer and superficial layer
norm_slope_mid_prepre, orientations = norm_slope('tc_prepre_1.csv', 0)
norm_slope_sup_prepre, orientations = norm_slope('tc_prepre_1.csv', 1)
norm_slope_mid_postpre = norm_slope('tc_postpre_1.csv', 0)
norm_slope_sup_postpre = norm_slope('tc_postpre_1.csv', 1)
norm_slope_mid_post = norm_slope('tc_post_1.csv', 0)
norm_slope_sup_post = norm_slope('tc_post_1.csv', 1)

norm_slopes = [norm_slope_mid_prepre, norm_slope_mid_postpre, norm_slope_mid_post, norm_slope_sup_prepre, norm_slope_sup_postpre, norm_slope_sup_post]

# Plots about changes before vs after training and pretraining per layer (and also centered or not centered neurons)
# colors represent orientation tuning
# Plotting indices
centre_E_indices = np.linspace(0, 80, 81).reshape(9,9)[2:7, 2:7].ravel().astype(int)
centre_I_indices = (centre_E_indices+81).astype(int)
E_indices_mid = np.linspace(0, 647, 648).round().reshape(8, 81, -1)[0:9:2].ravel().astype(int)
I_indices_mid =np.linspace(0, 647, 648).round().reshape(8, 81, -1)[1:9:2].ravel().astype(int)
E_indices_sup = np.linspace(0, 80, 81).astype(int)
I_indices_sup = np.linspace(81, 161, 81).astype(int)

#Create saving directory
save_dir='results/Mar6_06'
saving_dir = os.path.join(save_dir, 'norm_slope')  ## CHANGE HERE
if os.path.exists(saving_dir) == False:
       os.makedirs(saving_dir)

# Affect of pretraining on tuning curves in middle layer - change charac_to_plot indices to 
x_axis = norm_slopes[0] 
y_axis =norm_slopes[1] 
indices_to_plot = E_indices_mid
title = 'E_mid_all'
plot_pre_post_scatter(x_axis = x_axis , y_axis = y_axis , orientations = orientations, indices_to_plot = indices_to_plot, title = title, save_dir = saving_dir)

indices_to_plot = centre_E_indices
title = 'E_mid_centre'
plot_pre_post_scatter(x_axis = x_axis , y_axis = y_axis , orientations = orientations, indices_to_plot = indices_to_plot, title = title, save_dir = saving_dir)

indices_to_plot = centre_I_indices
title = 'I_mid_centre'
plot_pre_post_scatter(x_axis = x_axis , y_axis = y_axis , orientations = orientations, indices_to_plot = indices_to_plot, title = title, save_dir = saving_dir)

indices_to_plot = I_indices_mid
title = 'I_mid_all'
plot_pre_post_scatter(x_axis = x_axis , y_axis = y_axis , orientations = orientations, indices_to_plot = indices_to_plot, title = title, save_dir = saving_dir)

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