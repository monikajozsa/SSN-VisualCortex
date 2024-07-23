import jax
import jax.numpy as np
import numpy
import os
import shutil
from datetime import datetime
import pandas as pd
import re

from training.util_gabor import BW_image_jit_noisy


##### Functions to create training data #####
def create_grating_training(stimuli_pars, batch_size, BW_image_jit_inp_all):
    '''
    Create input stimuli gratings. Both the refence and the target are jitted by the same angle. 
    Input:
       stimuli pars
       batch_size - batch size
    
    Output:
        dictionary containing reference target and label 
    '''
    
    #initialise empty arrays
    ref_ori = stimuli_pars.ref_ori
    offset = stimuli_pars.offset
    data_dict = {'ref':[], 'target': [], 'label':[]}

    # Vectorize target_ori calculation, label and jitter generation 
    uniform_dist_value = numpy.random.uniform(low = 0, high = 1, size = batch_size)
    mask = uniform_dist_value < 0.5
    target_ori_vec = np.where(mask, ref_ori - offset, ref_ori + offset) # 1 when ref> target
    labels = mask.astype(int)  # Converts True/False to 1/0
    jitter_val = stimuli_pars.jitter_val
    jitter_vec = np.array(numpy.random.uniform(low = -jitter_val, high = jitter_val, size=batch_size))

    # Create reference and target gratings
    ref_ori_vec = np.ones(batch_size)*ref_ori
    x = BW_image_jit_inp_all[4]
    y = BW_image_jit_inp_all[5]
    alpha_channel = BW_image_jit_inp_all[6]
    mask = BW_image_jit_inp_all[7]
    ref = BW_image_jit_noisy(BW_image_jit_inp_all[0:4], x, y, alpha_channel, mask, ref_ori_vec, jitter_vec)
    target = BW_image_jit_noisy(BW_image_jit_inp_all[0:4], x, y, alpha_channel, mask, target_ori_vec, jitter_vec)
    data_dict['ref']=ref
    data_dict['target']=target
    data_dict['label']=labels
            
    return data_dict


def generate_random_pairs(min_value, max_value, min_distance, max_distance=None, batch_size=1, tot_angle=180, numRnd_ori1=1, numRnd_dist=None):
    '''
    Create batch_size number of pairs of numbers between min_value and max_value with ori distance between min_distance and max_distance. numRnd_ori1 is the number of different values for the first number. numRnd_dist is the number of different distances.
    '''
    if max_distance==None:
        max_distance = max_value - min_value
    if numRnd_dist==None:
        numRnd_dist = batch_size
    # Generate the first numbers
    rnd_numbers = numpy.random.uniform(min_value, max_value, numRnd_ori1) #numpy.random.randint(low=min_value, high=max_value, size=numRnd_ori1, dtype=int)
    num1 = numpy.repeat(rnd_numbers, int(batch_size/numRnd_ori1))

    # Generate random distances within specified range (numRnd_dist is the number of different distances)
    rnd_dist_samples = numpy.random.choice([-1, 1], batch_size) * numpy.random.uniform(min_distance,max_distance ,numRnd_dist) #numpy.random.randint(low=min_distance,high=max_distance, size=batch_size, dtype=int)
    rnd_distances = numpy.repeat(rnd_dist_samples, int(batch_size/numRnd_dist))
    numpy.random.shuffle(rnd_distances)

    # Generate the second numbers with correction if they are out of the specified range
    num2 = num1 - rnd_distances # order and sign are important!

    # Create a mask where flip_numbers equals 1
    swap_numbers = numpy.random.choice([0, 1], batch_size) 
    mask = swap_numbers == 1

    # Swap values where mask is True
    temp_num1 = np.copy(num1[mask]) # temporary array to hold the values of num1 where the mask is True
    num1[mask] = num2[mask]
    num2[mask] = temp_num1
    rnd_distances[mask] = -rnd_distances[mask]
    
    return np.array(num1), np.array(num2), rnd_distances


def create_grating_pretraining(pretrain_pars, batch_size, BW_image_jit_inp_all, numRnd_ori1=1):
    '''
    Create input stimuli gratings for pretraining by randomizing ref_ori for both reference and target (with random difference between them)
    Output:
        dictionary containing grating1, grating2 and difference between gratings that is calculated from features
    '''
    
    # Initialise empty data dictionary - names are not describing the purpose of the variables but this allows for reusing code
    data_dict = {'ref': [], 'target': [], 'label':[]}

    # Randomize orientations for stimulus 1 and stimulus 2
    ori1, ori2, ori1_minus_ori2 = generate_random_pairs(min_value=pretrain_pars.ref_ori_int[0], max_value=pretrain_pars.ref_ori_int[1], min_distance=pretrain_pars.ori_dist_int[0], max_distance=pretrain_pars.ori_dist_int[1], batch_size=batch_size, numRnd_ori1=numRnd_ori1)

    x = BW_image_jit_inp_all[4]
    y = BW_image_jit_inp_all[5]
    alpha_channel = BW_image_jit_inp_all[6]
    mask = BW_image_jit_inp_all[7]
    
    # Generate noisy stimulus1 and stimulus2 with no jitter
    stim1 = BW_image_jit_noisy(BW_image_jit_inp_all[0:4], x, y, alpha_channel, mask, ori1, jitter=np.zeros_like(ori1))
    stim2 = BW_image_jit_noisy(BW_image_jit_inp_all[0:4], x, y, alpha_channel, mask, ori2, jitter=np.zeros_like(ori1))
    data_dict['ref']=stim1
    data_dict['target']=stim2

    # Define label as the normalized signed difference in angle
    label = np.zeros_like(ori1_minus_ori2)
    data_dict['label'] = label.at[ori1_minus_ori2 > 0].set(1) # 1 when ref> target and 0 when ref<=target
    
    return data_dict

##### Other helper functions #####
def sigmoid(x, epsilon=0.01):
    """
    Introduction of epsilon stops asymptote from reaching 1 (avoids NaN)
    """
    sig_x = 1 / (1 + np.exp(-x))
    return (1 - 2 * epsilon) * sig_x + epsilon


def take_log(J_2x2):
    signs = np.array([[1, -1], [1, -1]])
    logJ_2x2 = np.log(J_2x2 * signs)

    return logJ_2x2


def sep_exponentiate(J_s):
    signs = np.array([[1, -1], [1, -1]])
    new_J = np.exp(np.array(J_s, dtype = float)) * signs

    return new_J


def x_greater_than(x, constant, slope, height):
    return np.maximum(0, (x * slope - (1 - height)))


def x_less_than(x, constant, slope, height):
    return constant * (x**2)


def leaky_relu(x, R_thresh, slope, height=0.15):
    """Customized relu function for regulating the rates"""
    constant = height / (R_thresh**2)
    # jax.lax.cond(cond, func1, func2, args - same for both functions) meaning if cond then apply func1, if not then apply func2 with the given arguments
    y = jax.lax.cond(
        (x < R_thresh), x_less_than, x_greater_than, x, constant, slope, height
    )

    return y


def save_code(final_folder_path=None, train_only_flag=False, note=None):
    '''
    This code is used to save code files to make results replicable.
    1) It copies specific code files into a folder called 'script'
    3) Returns the path to save the results into
    '''
    # Get the current date
    current_date = datetime.now().strftime("%b%d")

    # Create a folder name based on the current date
    if final_folder_path is None:
        folder_name = f"results/{current_date}_v"
        # Find the next available script version
        version = 0
        while os.path.exists(f"{folder_name}{version}"):
            version += 1
        final_folder_path = f"{folder_name}{version}"
        # Create the folder for the results    
        os.makedirs(final_folder_path)
    else:
        # version is after '_v' in final_folder_path
        match = re.search('_v', final_folder_path)
        if match:
            # Extract the number from the match object
            version = final_folder_path[match.endpos-1:]
        else:
            version = 0

    # Save note to final_folder_path as csv
    if note is not None:
        with open(os.path.join(final_folder_path, 'note.txt'), 'w') as f:
            f.write(note)

    # Create a folder for the scripts and a folder for the figures
    script_folder = os.path.join(final_folder_path, 'scripts')
    if not os.path.exists(script_folder):
        os.makedirs(script_folder)
    figure_folder = os.path.join(final_folder_path, 'figures')
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    
    # Get the path to the script's directory
    script_from_folder = os.path.dirname(os.path.realpath(__file__))

    # List of root files to be copied
    file_names_root = ['parameters.py', 'util.py']

    # Copy root files
    for file_name in file_names_root:
        source_path = os.path.join(script_from_folder, file_name)
        destination_path = os.path.join(script_folder, file_name)
        shutil.copyfile(source_path, destination_path)

    # Create the 'training' folder in the destination folder
    training_folder_dest = os.path.join(script_folder, 'training')
    os.makedirs(training_folder_dest, exist_ok=True)

    # Get the list of all .py files in the source 'training' folder
    training_folder_source = os.path.join(script_from_folder, 'training')
    file_names_training = [f for f in os.listdir(training_folder_source) if f.endswith('.py')]

    # Copy training files
    for file_name in file_names_training:
        source_path = os.path.join(training_folder_source, file_name)
        destination_path = os.path.join(training_folder_dest, file_name)
        shutil.copyfile(source_path, destination_path)

    # Create the 'analysis' folder in the destination folder
    analysis_folder_dest = os.path.join(script_folder, 'analysis')
    os.makedirs(analysis_folder_dest, exist_ok=True)

    # Get the list of all .py files in the source 'analysis' folder
    analysis_folder_source = os.path.join(script_from_folder, 'analysis')
    file_names_analysis = [f for f in os.listdir(analysis_folder_source) if f.endswith('.py')]

    # Copy analysis files
    for file_name in file_names_analysis:
        source_path = os.path.join(analysis_folder_source, file_name)
        destination_path = os.path.join(analysis_folder_dest, file_name)
        shutil.copyfile(source_path, destination_path)

    print(f"Script files copied successfully to: {script_folder}")

    # return path (inclusing filename) to save results into
    results_filename = os.path.join(final_folder_path,"results.csv")

    return results_filename, final_folder_path


def load_parameters(df, readout_grid_size=5, iloc_ind=-1, trained_pars_keys=['log_J_2x2_m', 'log_J_2x2_s', 'c_E', 'c_I', 'log_f_E', 'log_f_I']):

    # Get the last row of the given csv file
    selected_row = df.iloc[int(iloc_ind)]

    # Extract stage 1 parameters from df
    w_sig_keys = [f'w_sig_{i}' for i in range(1, readout_grid_size*readout_grid_size+1)] 
    w_sig_values = selected_row[w_sig_keys].values
    pars_stage1 = dict(w_sig=w_sig_values, b_sig=selected_row['b_sig'])

    # Extract stage 2 parameters from df
    log_J_m_keys = ['log_J_m_EE','log_J_m_EI','log_J_m_IE','log_J_m_II'] 
    log_J_s_keys = ['log_J_s_EE','log_J_s_EI','log_J_s_IE','log_J_s_II']
    J_m_values = selected_row[log_J_m_keys].values.reshape(2, 2)
    J_s_values = selected_row[log_J_s_keys].values.reshape(2, 2)
    
    # Create a dictionary with the trained parameters
    pars_stage2 = dict(
        log_J_2x2_m = J_m_values,
        log_J_2x2_s = J_s_values
    )
    if 'c_E' in trained_pars_keys:
        pars_stage2['c_E'] = selected_row['c_E']
        pars_stage2['c_I'] = selected_row['c_I']
    if 'log_f_E' in trained_pars_keys:
        pars_stage2['log_f_E'] = selected_row['log_f_E']
        pars_stage2['log_f_I'] = selected_row['log_f_I']

    offsets  = df['stoichiometric_offsets'].dropna().reset_index(drop=True)
    offset_last = offsets[len(offsets)-1]

    if 'meanr_E_mid' in df.columns:
        meanr_vec=[[df['meanr_E_mid'][len(df)-1], df['meanr_E_sup'][len(df)-1]], [df['meanr_I_mid'][len(df)-1], df['meanr_I_sup'][len(df)-1]]]
    else:
        meanr_vec = None
        
    return pars_stage1, pars_stage2, offset_last, meanr_vec


def filter_for_run(df,run_index):
    df['run_index'] = pd.to_numeric(df['run_index'], errors='coerce')
    mesh_i = df['run_index'] == run_index
    df_i = df[mesh_i]
    df_i = df_i.drop(columns=['run_index'])
    df_i = df_i.reset_index(drop=True)

    return df_i

