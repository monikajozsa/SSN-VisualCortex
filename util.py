import jax
import jax.numpy as np
import numpy
import shutil
from datetime import datetime
import pandas as pd
from pathlib import Path
import os

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


def save_code(final_folder_path=None, note=None):
    """
    This function saves code files to make results replicable.
    1) Copies specific code files into a folder called 'scripts'.
    2) Returns the path to save the results into.
    """

    def create_versioned_folder(base_path):
        version = 0
        while base_path.with_name(f"{base_path.name}_v{version}").exists():
            version += 1
        versioned_folder = base_path.with_name(f"{base_path.name}_v{version}")
        versioned_folder.mkdir(parents=True, exist_ok=True)
        return versioned_folder

    def copy_files(source_folder, destination_folder, file_pattern):
        destination_folder.mkdir(parents=True, exist_ok=True)
        for file in source_folder.glob(file_pattern):
            shutil.copy(file, destination_folder / file.name)

    # Get the current date
    current_date = datetime.now().strftime("%b%d")

    # Determine the final folder path
    if final_folder_path is None:
        base_folder = Path("results") / current_date
        final_folder_path = create_versioned_folder(base_folder)
    else:
        final_folder_path = Path(final_folder_path)

    # Save note if provided
    if note:
        with open(final_folder_path / 'note.txt', 'w') as f:
            f.write(note)

    # Create subfolders
    script_folder = final_folder_path / 'scripts'
    figure_folder = final_folder_path / 'figures'
    figure_folder.mkdir(parents=True, exist_ok=True)

    # Define source folder
    script_from_folder = Path(__file__).parent

    # Copy root files, 'training' files and 'analysis' files
    copy_files(script_from_folder, script_folder, '*.py')
    copy_files(script_from_folder / 'training', script_folder / 'training', '*.py')
    copy_files(script_from_folder / 'analysis', script_folder / 'analysis', '*.py')

    print(f"Script files copied successfully to: {script_folder}")

    # Return path to save results
    results_filename = final_folder_path / "results.csv"
    return str(results_filename), str(final_folder_path)

def load_orientation_map(folder, run_ind):
    '''Loads the orientation map from the folder for the training indexed by run_ind.'''
    orimap_filename = os.path.join(folder, f"orimap.csv")
    if not os.path.exists(orimap_filename):
        orimap_filename = os.path.join(folder, f"orimap_{run_ind}.npy")
        orimap = np.load(orimap_filename)
    else:
        orimaps = pd.read_csv(orimap_filename, header=0)
        mesh_run = orimaps['run_index']==float(run_ind)
        orimap = orimaps[mesh_run].to_numpy()
        orimap = orimap[0][1:]

    return orimap

def load_parameters(folder_path, run_index, stage=0, iloc_ind=-1, for_training=False):
    from training.util_gabor import init_untrained_pars

    # Define parameter keys that will be trained depending on the stage
    from parameters import SSNPars, ReadoutPars, TrainedSSNPars, PretrainedSSNPars, GridPars, FilterPars, StimuliPars, ConvPars, TrainingPars, LossPars, PretrainingPars
    ssn_pars, readout_pars, trained_pars, pretrained_pars = SSNPars(), ReadoutPars(), TrainedSSNPars(), PretrainedSSNPars()
    grid_pars, filter_pars, stimuli_pars = GridPars(), FilterPars(), StimuliPars()
    conv_pars, training_pars, loss_pars, pretraining_pars = ConvPars(), TrainingPars(), LossPars(), PretrainingPars()
    if for_training==0:
        par_keys = {attr: getattr(trained_pars, attr) for attr in dir(trained_pars)}
    else:
        par_keys = {attr: getattr(pretrained_pars, attr) for attr in dir(pretrained_pars)}

    # Get the last row of the given csv file
    df_all = pd.read_csv(os.path.join(folder_path, 'results.csv'))
    df = filter_for_run_and_stage(df_all, run_index, stage)
    selected_row = df.iloc[int(iloc_ind)]

    # Extract readout parameters from df and save it to readout pars
    if for_training:
        middle_grid_inds = readout_pars.middle_grid_ind
    else:
        middle_grid_inds = range(readout_pars.readout_grid_size[0]**2)
    w_sig_keys = [f'w_sig_{middle_grid_inds[i]}' for i in range(len(middle_grid_inds))] 
    w_sig_values = selected_row[w_sig_keys].values
    readout_pars_loaded = dict(w_sig=w_sig_values, b_sig=selected_row['b_sig'])
    readout_pars.b_sig = selected_row['b_sig']
    readout_pars.w_sig = w_sig_values

    # Define keys for J parameters
    log_J_m_keys = ['log_J_m_EE','log_J_m_EI','log_J_m_IE','log_J_m_II'] 
    log_J_s_keys = ['log_J_s_EE','log_J_s_EI','log_J_s_IE','log_J_s_II']
    J_m_keys = ['J_m_EE','J_m_EI','J_m_IE','J_m_II'] 
    J_s_keys = ['J_s_EE','J_s_EI','J_s_IE','J_s_II']

    # Create a dictionary with the trained parameters and update untrained parameters if J, c or f are not trained
    pars_dict = {}
    if 'log_J_2x2_m' in par_keys or 'J_2x2_m' in par_keys:
        pars_dict['log_J_2x2_m'] = selected_row[log_J_m_keys].values.reshape(2, 2)
    else:
        ssn_pars.J_2x2_m = selected_row[J_m_keys].values.reshape(2, 2)
    if 'log_J_2x2_s' in par_keys or 'J_2x2_s' in par_keys:
        pars_dict['log_J_2x2_s'] = selected_row[log_J_s_keys].values.reshape(2, 2)
    else:
        ssn_pars.J_2x2_s = selected_row[J_s_keys].values.reshape(2, 2)
    if 'c_E' in par_keys:
        pars_dict['c_E'] = selected_row['c_E']
        pars_dict['c_I'] = selected_row['c_I']
    else:
        ssn_pars.c_E = selected_row['c_E']
        ssn_pars.c_I = selected_row['c_I']
    if 'log_f_E' in par_keys or 'f_E' in par_keys:
        pars_dict['log_f_E'] = selected_row['log_f_E']
        pars_dict['log_f_I'] = selected_row['log_f_I']
    else:
        ssn_pars.f_E = selected_row['f_E']
        ssn_pars.f_I = selected_row['f_I']
    
    # Set the randomized gE, gI and eta parameters in relevant classes from initial_parameters.csv
    df_init_pars_all = pd.read_csv(os.path.join(folder_path, 'initial_parameters.csv'))
    df_init_pars = filter_for_run_and_stage(df_init_pars_all, run_index, stage)
 
    training_pars.eta = df_init_pars['eta'].iloc[0]
    filter_pars.gE_m = df_init_pars['gE'].iloc[0]
    filter_pars.gI_m = df_init_pars['gI'].iloc[0]

    # Load orientation map
    loaded_orimap = load_orientation_map(folder_path, run_index)

    # Initialize untrained parameters with the loaded values
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                    loss_pars, training_pars, pretraining_pars, readout_pars, run_index, orimap_loaded=loaded_orimap)
    # Get other metrics needed for training
    if for_training:
        untrained_pars.pretrain_pars.is_on = False
        if 'psychometric_offset' in df.keys():
            offsets  = df['psychometric_offset'].dropna().reset_index(drop=True)
        else:
            for keys in df.keys():
                if 'ric_offset' in keys:
                    offsets = df[keys].dropna().reset_index(drop=True)
        offset_last = offsets[len(offsets)-1]

        if 'meanr_E_mid' in df.columns:
            meanr_vec=[[df['meanr_E_mid'][len(df)-1], df['meanr_E_sup'][len(df)-1]], [df['meanr_I_mid'][len(df)-1], df['meanr_I_sup'][len(df)-1]]]
        else:
            meanr_vec = None
           
        return readout_pars_loaded, pars_dict, untrained_pars, offset_last, meanr_vec
    else:
        return readout_pars_loaded, pars_dict, untrained_pars


def filter_for_run_and_stage(df,run_index, stage=None):
    df['run_index'] = pd.to_numeric(df['run_index'], errors='coerce')
    mesh_i = df['run_index'] == run_index
    df_i = df[mesh_i]
    df_i = df_i.drop(columns=['run_index'])
    df_i = df_i.reset_index(drop=True)

    if stage is not None:
        mesh_i = df_i['stage'] == stage
        df_i = df_i[mesh_i]
        df_i = df_i.drop(columns=['stage'])
        df_i = df_i.reset_index(drop=True)

    return df_i

