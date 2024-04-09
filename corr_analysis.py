import pandas as pd
import numpy
import time

from training import batch_loss_ori_discr, generate_noise, mean_training_task_acc_test, offset_at_baseline_acc
from util import load_parameters
from util_gabor import init_untrained_pars
from util import create_grating_training
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

def corr_from_csvs(folder,num_trainings, batch_size = 100):
    '''read CSV files and calculate the correlation between the changes of accuracy and J (J_II and J_EI are summed up and J_EE and J_IE are summed up) for each file'''
    file_pattern = folder + '/results_{}'

    # Initialize variables to store results in
    offsets = numpy.zeros((num_trainings,3))
    offset_th = numpy.zeros((num_trainings,2))
    offset_diff = numpy.zeros(num_trainings)
    offset_th_diff = numpy.zeros(num_trainings)
    
    Jm_start = numpy.zeros((num_trainings, 4))
    Jm_middle = numpy.zeros((num_trainings, 4))
    Jm_end = numpy.zeros((num_trainings, 4))
    Js_start = numpy.zeros((num_trainings, 4))
    Js_middle = numpy.zeros((num_trainings, 4))
    Js_end = numpy.zeros((num_trainings, 4))
    J_m_diff = numpy.zeros((num_trainings,4))
    J_s_diff = numpy.zeros((num_trainings,4))

    corr_offset_J = numpy.zeros((2,4))
    corr_offset_th_J = numpy.zeros((2,4))
    
    # Initialize the test offset vector for the threshold calculation
    test_offset_vec = numpy.array([1, 2, 3, 4, 6]) 

    start_time = time.time()
    for i in range(num_trainings):
        # Construct the file name
        file_name = file_pattern.format(i) + '.csv'
        orimap_filename = folder + '/orimap_{}.npy'.format(i)
        
        # Read the file
        df = pd.read_csv(file_name)
        
        # Find index of the last row where 'stage' is 0 and 2 (started and finished training)
        training_start = df[df['stage'] == 0].index[-1]+10
        training_end = df[df['stage'] == 2].index[-1]
        training_middle = (training_end - training_start) // 2 + training_start

        # Calculate the offset difference at start and end of training
        offsets[i,:] = numpy.array([df['offset'][training_start], df['offset'][training_middle], df['offset'][training_end]])
        offset_diff[i] = -(offsets[i,2] - offsets[i,0]) / offsets[i,0] *100

        # get accuracy metrics for training_start
        loaded_orimap =  numpy.load(orimap_filename)
        untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                        loss_pars, training_pars, pretrain_pars, readout_pars, None, orimap_loaded=loaded_orimap)
        untrained_pars.pretrain_pars.is_on = False
        
        #noise_ref = generate_noise(pretrain_pars.batch_size, untrained_pars.middle_grid_ind.shape[0], N_readout = untrained_pars.N_readout_noise)
        #noise_target = generate_noise(pretrain_pars.batch_size, untrained_pars.middle_grid_ind.shape[0], N_readout = untrained_pars.N_readout_noise)
        #train_data = create_grating_training(untrained_pars.stimuli_pars, batch_size, untrained_pars.BW_image_jax_inp)
        trained_pars_stage1, trained_pars_stage2, _ = load_parameters(file_name, iloc_ind = training_start)
        acc_mean, _, _ = mean_training_task_acc_test(trained_pars_stage2, trained_pars_stage1, untrained_pars, jit_on=True, offset_vec=test_offset_vec, sample_size = 1 )
        offset_th[i,0] = offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec,  baseline_acc= 0.85)

        trained_pars_stage1, trained_pars_stage2, _ = load_parameters(file_name, iloc_ind = training_end)
        acc_mean, _, _ = mean_training_task_acc_test(trained_pars_stage2, trained_pars_stage1, untrained_pars, jit_on=True, offset_vec=test_offset_vec )
        offset_th[i,1] = offset_at_baseline_acc(acc_mean, offset_vec=test_offset_vec, baseline_acc= 0.85)
        offset_th_diff[i] = -(offset_th[i,1] - offset_th[i,0]) / offset_th[i,0] *100
        
        # Calculate the J differences (J_m_EE	J_m_EI	J_m_IE	J_m_II	J_s_EE	J_s_EI	J_s_IE	J_s_II) at start and end of training
        Jm_start[i,:] = df[['J_m_EE', 'J_m_IE', 'J_m_EI', 'J_m_II']].iloc[training_start].to_numpy()
        Jm_middle[i,:] = df[['J_m_EE', 'J_m_IE', 'J_m_EI', 'J_m_II']].iloc[training_middle].to_numpy()
        Jm_end[i,:] = df[['J_m_EE', 'J_m_IE', 'J_m_EI', 'J_m_II']].iloc[training_end].to_numpy()
        Js_start[i,:] = df[['J_s_EE', 'J_s_IE', 'J_s_EI', 'J_s_II']].iloc[training_start].to_numpy()
        Js_middle[i,:] = df[['J_s_EE', 'J_s_IE', 'J_s_EI', 'J_s_II']].iloc[training_middle].to_numpy()
        Js_end[i,:] = df[['J_s_EE', 'J_s_IE', 'J_s_EI', 'J_s_II']].iloc[training_end].to_numpy()
        
        J_m_diff[i,:] = numpy.abs(Jm_start[i,:] - Jm_end[i,:]) / numpy.abs(Jm_start[i,:]) *100
        J_s_diff[i,:] = numpy.abs(Js_start[i,:] - Js_end[i,:]) / numpy.abs(Js_start[i,:]) *100

        print('Finished reading file', i, 'time elapsed:', time.time() - start_time)
    # Correlate the accuracy difference with the J differences
    mesh_offset_th=numpy.sum(offset_th, axis=1)<180
    offset_th = offset_th[mesh_offset_th,:]
    offset_th_diff = offset_th_diff[mesh_offset_th]
    J_m_diff = J_m_diff[mesh_offset_th,:]
    J_s_diff = J_s_diff[mesh_offset_th,:]
    offset_diff = offset_diff[mesh_offset_th]

    #corr_offset_J[0,:] = numpy.array([numpy.corrcoef(offset_diff,J_m_diff[:,0])[0,1], numpy.corrcoef(offset_diff,J_m_diff[:,1])[0,1],numpy.corrcoef(offset_diff,J_m_diff[:,2])[0,1], numpy.corrcoef(offset_diff,J_m_diff[:,3])[0,1]])
    #corr_offset_J[1,:] = numpy.array([numpy.corrcoef(offset_diff,J_s_diff[:,0])[0,1], numpy.corrcoef(offset_diff,J_s_diff[:,1])[0,1], numpy.corrcoef(offset_diff,J_s_diff[:,2])[0,1], numpy.corrcoef(offset_diff,J_s_diff[:,3])[0,1]])
    #corr_offset_th_J[0,:] = numpy.array([numpy.corrcoef(offset_th_diff,J_m_diff[:,0])[0,1], numpy.corrcoef(offset_th_diff,J_m_diff[:,1])[0,1],numpy.corrcoef(offset_th_diff,J_m_diff[:,2])[0,1], numpy.corrcoef(offset_th_diff,J_m_diff[:,3])[0,1]]) 
    #corr_offset_th_J[1,:] = numpy.array([numpy.corrcoef(offset_th_diff,J_s_diff[:,0])[0,1], numpy.corrcoef(offset_th_diff,J_s_diff[:,1])[0,1], numpy.corrcoef(offset_th_diff,J_s_diff[:,2])[0,1], numpy.corrcoef(offset_th_diff,J_s_diff[:,3])[0,1]])

    #corr_accuracy_J = [numpy.corrcoef(accuracy_difference,J_m_E_difference), numpy.corrcoef(accuracy_difference,J_m_I_difference), numpy.corrcoef(accuracy_difference,J_s_E_difference), numpy.corrcoef(accuracy_difference,J_s_I_difference)]
    print('Number of samples:', sum(mesh_offset_th))
    #print(corr_offset_J)
    #print(corr_offset_th_J)
    return offset_diff, offset_th_diff, J_m_diff, J_s_diff

#corr_offset_J, corr_offset_th_J, offset_diff, offset_th_diff, J_m_diff, J_s_diff = corr_from_csvs('results/Apr07_v0', 18)
offset_diff, offset_th_diff, J_m_diff, J_s_diff = corr_from_csvs('results/Mar27_v0', 30)

import scipy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Convert relative parameter differences to pandas DataFrame
data = pd.DataFrame({'offset_th_diff': offset_th_diff, 'J_m_EE_diff': J_m_diff[:, 0], 'J_m_IE_diff': J_m_diff[:, 1], 'J_m_EI_diff': J_m_diff[:, 2], 'J_m_II_diff': J_m_diff[:, 3], 'J_s_EE_diff': J_s_diff[:, 0], 'J_s_IE_diff': J_s_diff[:, 1], 'J_s_EI_diff': J_s_diff[:, 2], 'J_s_II_diff': J_s_diff[:, 3]})

# Create a 2x4 subplot grid
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
axes_flat = axes.flatten()

# y-axis labels
x_labels = ['J_m_EE_diff', 'J_m_IE_diff', 'J_m_EI_diff', 'J_m_II_diff', 'J_s_EE_diff', 'J_s_IE_diff', 'J_s_EI_diff', 'J_s_II_diff']

for i in range(8):
    # Create lmplot for each pair of variables
    sns.regplot(x=x_labels[i], y='offset_th_diff', data=data, ax=axes_flat[i], ci=95)
    # Calculate the Pearson correlation coefficient and the p-value
    corr, p_value = scipy.stats.pearsonr(data['offset_th_diff'], data[x_labels[i]])
    
    # Close the lmplot's figure to prevent overlapping
    axes_flat[i].set_title( f'Corr: {corr:.2f}, p-val: {p_value:.2f}')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("Offset_correlation.png")