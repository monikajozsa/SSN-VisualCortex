import jax.numpy as np

from util_gabor import init_untrained_pars
from filtered_model_response_dev import filtered_model_response
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

file_name = 'C:/Users/jozsa/Postdoc YA/SSN-VisualCortex/results/Feb29_v5/results_0.csv'
loaded_orimap = np.load('C:/Users/jozsa/Postdoc YA/SSN-VisualCortex/results/Feb29_v5/orimap_0.npy')
untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, None, loaded_orimap)


import numpy as np
import scipy.io as sio
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.covariance import MahalanobisDistances
import matplotlib.pyplot as plt
from pathlib import Path

# Specify number of epochs (pre and post training)
num_epochs = 2

# Specify number of layers (superficial and superficial)
num_layers = 2

# Specify number of components to use
PC_used = 15

# Initialise arrays to store results
finalSD_results = np.zeros((num_epochs, 3))  # To include control in SD calculation
finalME_results = np.zeros((num_epochs, 2))
finalSNR_results = []

# Assuming 'centre_E_indices' and 'untrained_pers' need not be converted as they're not directly used

# Iterate over number of epochs
count = 0
for seed_n in range(1, 21):
    count += 1
    filename = f'Users/clarapecci/Desktop/ssn_modelling/ssn-simulator/results/11-12/stair_results/stair_noise200.0gE0.3_{seed_n}/noisy_respose_ori_map/noisy_response.mat'
    smooth_all = sio.loadmat(filename)
    labels = smooth_all['labels'][0, :900]
    
    for epoch in range(num_epochs):
        # Specify layer (superficial/middle)!!!
        curr_data = np.squeeze(smooth_all['superficial'][epoch, :, :])

        # Normalise data
        z_score_data = zscore(curr_data, axis=0)

        # PCA
        pca = PCA(n_components=PC_used)
        score = pca.fit_transform(z_score_data)

        # Select components
        Curr_data_used_pca = score[:, :PC_used]

        # Separate data into orientation conditions
        train_data = Curr_data_used_pca[labels == 55, :]
        untrain_data = Curr_data_used_pca[labels == 125, :]
        control_data = Curr_data_used_pca[labels == 0, :]

        # Calculate Mahalanobis distance
        md = MahalanobisDistances()  # Assuming a placeholder for correct Mahalanobis distance calculation
        train_dis_mahal = np.sqrt(md.fit(control_data).distances_(train_data))
        untrain_dis_mahal = np.sqrt(md.fit(control_data).distances_(untrain_data))

        # Mean
        train_dis_mahal_mean = np.mean(train_dis_mahal)
        untrain_dis_mahal_mean = np.mean(untrain_dis_mahal)

        # Calculate the standard deviation
        train_data_size = train_data.shape[0]
        distanceSD_train = np.zeros(train_data_size)
        distanceSD_untrain = np.zeros(train_data_size)
        distanceSD_control = np.zeros(train_data_size)

        for cross_size_i in range(train_data_size):
            # Create temporary copies excluding one sample for cross-validation
            mask = np.ones(train_data_size, dtype=bool)
            mask[cross_size_i] = False
            train_data_temp = train_data[mask]
            untrain_data_temp = untrain_data[mask]
            control_data_temp = control_data[mask]

            # Calculate distances
            distanceSD_train[cross_size_i] = np.sqrt(md.fit(train_data_temp).distances_([train_data[cross_size_i]]))
            distanceSD_untrain[cross_size_i] = np.sqrt(md.fit(untrain_data_temp).distances_([untrain_data[cross_size_i]]))
            distanceSD_control[cross_size_i] = np.sqrt(md.fit(control_data_temp).distances_([control_data[cross_size_i]]))

        # Calculate SD for each condition
        train_dis_mahal_sd = np.mean(distanceSD_train)
        untrain_dis_mahal_sd = np.mean(distanceSD_untrain)
        control_dis_mahal_sd = np.mean(distanceSD_control)

        # Calculate SNR
        train_SNR = train_dis_mahal / distanceSD_train
        untrain_SNR = untrain_dis_mahal / distanceSD_untrain
        train_SNR_mean = np.mean(train_SNR)
        untrain_SNR_mean = np.mean(untrain_SNR)

        # Plotting (replacing MATLAB figures with Matplotlib)
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.hist(train_dis_mahal)
        plt.title('Trained ori')
        plt.subplot(2, 1, 2)
        plt.hist(untrain_dis_mahal)
        plt.title


filtered_r_mid_all, filtered_r_sup_all = filtered_model_response(file_name, untrained_pars, ori_list= np.asarray([55, 125, 0]), n_noisy_trials = 300, step_inds = np.asarray([1, -1]) )