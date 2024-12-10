import jax
import jax.numpy as jnp
import numpy
import shutil
from datetime import datetime
import pandas as pd
from pathlib import Path
import os

from training.util_gabor import BW_image_jit_noisy

def check_header(filename):
    """Checking if csv file has a header or not"""
    first_row = pd.read_csv(filename, nrows=1, header=None).iloc[0]
    if first_row.apply(lambda x: isinstance(x, str)).any():
        header = 0
    else:
        header = None
    return header


def unpack_ssn_parameters(trained_pars, ssn_pars, as_log_list=False, return_kappas= True):
    """Unpacks the trained parameters and the untrained parameters. If as_log_list is True, then the J and f parameters are returned as a list of logs. 
    If return_kappas is True, then the kappa_Jsup, kappa_Jmid, kappa_f parameters are returned."""

    def get_J(par_name, trained_pars, ssn_pars):
        """
        Returns the value of J based on the provided parameters.
        """
        # Check if 'log_' version of the parameter is in trained_pars
        log_key = f'log_{par_name}'
        if log_key in trained_pars:
            if 'I_' in par_name:
                J = -jnp.exp(trained_pars[log_key]) 
            else:
                J = jnp.exp(trained_pars[log_key])
        # Check if the parameter itself is in trained_pars
        elif par_name in trained_pars:
            J = trained_pars[par_name]
        # Return default value from untrained_pars
        else:
            J = getattr(ssn_pars, par_name)

        return J

    J_II_m = get_J('J_II_m', trained_pars, ssn_pars)
    J_IE_m = get_J('J_IE_m', trained_pars, ssn_pars)
    J_EI_m = get_J('J_EI_m', trained_pars, ssn_pars)
    J_EE_m = get_J('J_EE_m', trained_pars, ssn_pars)
    J_2x2_m = jnp.array([[J_EE_m, J_EI_m], [J_IE_m, J_II_m]])
    J_II_s = get_J('J_II_s', trained_pars, ssn_pars)
    J_IE_s = get_J('J_IE_s', trained_pars, ssn_pars)
    J_EI_s = get_J('J_EI_s', trained_pars, ssn_pars)
    J_EE_s = get_J('J_EE_s', trained_pars, ssn_pars)
    J_2x2_s = jnp.array([[J_EE_s, J_EI_s], [J_IE_s, J_II_s]])

    if 'cE_m' in trained_pars:
        cE_m = trained_pars['cE_m']
        cI_m = trained_pars['cI_m']
    else:
        cE_m = ssn_pars.cE_m
        cI_m = ssn_pars.cI_m
    if 'cE_s' in trained_pars:
        cE_s = trained_pars['cE_s']
        cI_s = trained_pars['cI_s']
    else:
        cE_s = ssn_pars.cE_s
        cI_s = ssn_pars.cI_s
    if 'log_f_E' in trained_pars:  
        f_E = jnp.exp(trained_pars['log_f_E'])
        f_I = jnp.exp(trained_pars['log_f_I'])
    elif 'f_E' in trained_pars:
        f_E = trained_pars['f_E']
        f_I = trained_pars['f_I']
    else:
        f_E = ssn_pars.f_E
        f_I = ssn_pars.f_I
    if return_kappas: 
        if 'kappa_Jsup' in trained_pars:
            kappa_Jsup = trained_pars['kappa_Jsup']
        else:
            if hasattr(ssn_pars, 'kappa_Jsup'): # case when during pretraining we check training task accuracy
                kappa_Jsup = ssn_pars.kappa_Jsup
            else:
                kappa_Jsup = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        if 'kappa_Jmid' in trained_pars:
            kappa_Jmid = trained_pars['kappa_Jmid']
        else:
            if hasattr(ssn_pars, 'kappa_Jmid'): # case when during pretraining we check training task accuracy
                kappa_Jmid = ssn_pars.kappa_Jmid
            else:
                kappa_Jmid = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        if 'kappa_f' in trained_pars:
            kappa_f = trained_pars['kappa_f']
        else:
            if hasattr(ssn_pars, 'kappa_f'):
                kappa_f = ssn_pars.kappa_f
            else:
                kappa_f = jnp.array([0.0, 0.0])
    else:
        kappa_Jsup = None
        kappa_Jmid = None
        kappa_f = None
    if as_log_list:
        log_J_2x2_m = take_log(J_2x2_m)
        log_J_2x2_s = take_log(J_2x2_s)
        log_f_E = jnp.log(f_E)
        log_f_I = jnp.log(f_I)
        return [log_J_2x2_m.ravel()], [log_J_2x2_s.ravel()], [cE_m], [cI_m], [cE_s], [cI_s], [log_f_E], [log_f_I], [kappa_Jsup.ravel()], [kappa_Jmid.ravel()], [kappa_f]
    else:
        return J_2x2_m, J_2x2_s, cE_m, cI_m, cE_s, cI_s, f_E, f_I, kappa_Jsup, kappa_Jmid, kappa_f


def cosdiff_ring(d_x, L):
    """
    Calculate the cosine-based distance.
    Parameters:
    d_x: The difference in the angular position.
    L: The total angle.
    """
    # Calculate the cosine of the scaled angular difference
    cos_angle = jnp.cos(d_x * 2 * jnp.pi / L)

    # Calculate scaled distance
    distance = jnp.sqrt( (1 - cos_angle) * 2) * L / (2 * jnp.pi)

    return distance


##### Functions to create training data #####
def create_grating_training(stimuli_pars, batch_size, BW_image_jit_inp_all, shuffle_labels=False):
    """
    Create input stimuli gratings. Both the refence and the target are jitted by the same angle. 
    Input:
       stimuli pars
       batch_size - batch size
    
    Output:
        dictionary containing reference target and label 
    """
    
    #initialise empty arrays
    ref_ori = stimuli_pars.ref_ori
    offset = stimuli_pars.offset
    data_dict = {'ref':[], 'target': [], 'label':[]}

    # Vectorize target_ori calculation, label and jitter generation 
    uniform_dist_value = numpy.random.uniform(low = 0, high = 1, size = batch_size)
    mask = uniform_dist_value < 0.5
    target_ori_vec = jnp.where(mask, ref_ori - offset, ref_ori + offset) # 1 when ref> target
    labels = mask.astype(int)  # Converts True/False to 1/0
    jitter_val = stimuli_pars.jitter_val
    jitter_vec = jnp.array(numpy.random.uniform(low = -jitter_val, high = jitter_val, size=batch_size))

    # Create reference and target gratings
    ref_ori_vec = jnp.ones(batch_size)*ref_ori
    x = BW_image_jit_inp_all[4]
    y = BW_image_jit_inp_all[5]
    alpha_channel = BW_image_jit_inp_all[6]
    mask = BW_image_jit_inp_all[7]
    ref = BW_image_jit_noisy(BW_image_jit_inp_all[0:4], x, y, alpha_channel, mask, ref_ori_vec, jitter_vec)
    target = BW_image_jit_noisy(BW_image_jit_inp_all[0:4], x, y, alpha_channel, mask, target_ori_vec, jitter_vec)
    data_dict['ref']=ref
    data_dict['target']=target

    if shuffle_labels:
        dlabel_length = len(labels)
        labels = jnp.array(numpy.random.randint(2, size=dlabel_length))
    
    data_dict['label']=labels
            
    return data_dict


def generate_random_pairs(min_value, max_value, min_distance, max_distance=None, batch_size=1, numRnd_ori1=1, numRnd_dist=None):
    """
    Create batch_size number of pairs of numbers between min_value and max_value with ori distance between min_distance and max_distance. numRnd_ori1 is the number of different values for the first number. numRnd_dist is the number of different distances.
    """
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
    temp_num1 = jnp.copy(num1[mask]) # temporary array to hold the values of num1 where the mask is True
    num1[mask] = num2[mask]
    num2[mask] = temp_num1
    rnd_distances[mask] = -rnd_distances[mask]
    
    return jnp.array(num1), jnp.array(num2), rnd_distances


def create_grating_pretraining(pretrain_pars, batch_size, BW_image_jit_inp_all, numRnd_ori1=1):
    """
    Create input stimuli gratings for pretraining by randomizing ref_ori for both reference and target (with random difference between them)
    Output:
        dictionary containing grating1, grating2 and difference between gratings that is calculated from features
    """
    
    # Initialise empty data dictionary - names are not describing the purpose of the variables but this allows for reusing code
    data_dict = {'ref': [], 'target': [], 'label':[]}

    # Randomize orientations for stimulus 1 and stimulus 2
    ori1, ori2, ori1_minus_ori2 = generate_random_pairs(min_value=pretrain_pars.ref_ori_int[0], max_value=pretrain_pars.ref_ori_int[1], min_distance=pretrain_pars.ori_dist_int[0], max_distance=pretrain_pars.ori_dist_int[1], batch_size=batch_size, numRnd_ori1=numRnd_ori1)

    x = BW_image_jit_inp_all[4]
    y = BW_image_jit_inp_all[5]
    alpha_channel = BW_image_jit_inp_all[6]
    mask = BW_image_jit_inp_all[7]
    
    # Generate noisy stimulus1 and stimulus2 with no jitter
    stim1 = BW_image_jit_noisy(BW_image_jit_inp_all[0:4], x, y, alpha_channel, mask, ori1, jitter=jnp.zeros_like(ori1))
    stim2 = BW_image_jit_noisy(BW_image_jit_inp_all[0:4], x, y, alpha_channel, mask, ori2, jitter=jnp.zeros_like(ori1))
    data_dict['ref']=stim1
    data_dict['target']=stim2

    # Define label as the normalized signed difference in angle
    label = jnp.zeros_like(ori1_minus_ori2)
    data_dict['label'] = label.at[ori1_minus_ori2 > 0].set(1) # 1 when ref> target and 0 when ref<=target
    
    return data_dict

##### Other helper functions #####
def sigmoid(x, epsilon=0.01):
    """
    Introduction of epsilon stops asymptote from reaching 1 (avoids NaN)
    """
    sig_x = 1 / (1 + jnp.exp(-x))
    return (1 - 2 * epsilon) * sig_x + epsilon


def take_log(J_2x2):
    """Take the log of the 2x2 matrix J"""
    signs = jnp.array([[1, -1], [1, -1]])
    logJ_2x2 = jnp.log(J_2x2 * signs)

    return logJ_2x2


def sep_exponentiate(J_2x2):
    """Exponentiate the J matrix"""
    signs = jnp.array([[1, -1], [1, -1]])
    new_J = jnp.exp(jnp.array(J_2x2, dtype = float)) * signs

    return new_J


def leaky_relu(x, R_thresh, slope, height=0.15):
    """ Customized relu function for regulating the rates. """
    def x_greater_than(x, constant, slope, height):
        return jnp.maximum(0, (x * slope - (1 - height)))


    def x_less_than(x, constant, slope, height):
        return constant * (x**2)
    constant = height / (R_thresh**2)
    # jax.lax.cond(cond, func1, func2, args - same for both functions) meaning if cond then apply func1, if not then apply func2 with the given arguments
    y = jax.lax.cond(
        (x < R_thresh), x_less_than, x_greater_than, x, constant, slope, height
    )

    return y


def save_code(final_folder_path=None, note=None):
    """
    This function saves code files to make results replicable.
    1) Creates a folder for results, scripts and figures.
    2) Copies specific code files into a folder called 'scripts'.
    3) Returns the path to save the results into.
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

    # Define source  and destination folders
    script_from_folder = Path(__file__).parent


    # Copy root files, 'training' files and 'analysis' files
    copy_files(script_from_folder, script_folder, '*.py')
    copy_files(script_from_folder / 'training', script_folder / 'training', '*.py')
    copy_files(script_from_folder / 'analysis', script_folder / 'analysis', '*.py')

    print(f"Script files copied successfully to: {script_folder}")

    return str(final_folder_path)


def save_numpy_to_csv(numpy_array, file_name):
    """ Save mul numpy array to csv file """
    data = []
    for run_index in range(numpy.shape(numpy_array)[0]):
        for layer in range(numpy.shape(numpy_array)[1]):
            for stage in range(numpy.shape(numpy_array)[2]):
                for ori_index in range(numpy.shape(numpy_array)[3]):
                    # Add the new row to the dictionary
                    data.append({f'run_index': run_index, f'layer': layer, f'stage': stage, f'ori': ori_index, f'score': numpy_array[run_index, layer, stage, ori_index]})
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    return df


def csv_to_numpy(file_name):
    """ Reads a CSV file and converts it back to a 4D NumPy array."""
    # Read the CSV into a DataFrame
    df = pd.read_csv(file_name)
    
    # Get the maximum values for each index to determine the shape of the 4D array
    num_runs = df['run_index'].max() + 1
    num_layers = df['layer'].max() + 1
    num_stages = df['stage'].max() + 1
    num_oris = df['ori'].max() + 1
    
    # Initialize an empty NumPy array with the determined shape
    numpy_array = numpy.zeros((num_runs, num_layers, num_stages, num_oris))
    
    # Iterate over the DataFrame and populate the NumPy array
    for _, row in df.iterrows():
        run_index = int(row['run_index'])
        layer = int(row['layer'])
        stage = int(row['stage'])
        ori = int(row['ori'])
        score = row['score']
        
        # Assign the score to the correct position in the 4D array
        numpy_array[run_index, layer, stage, ori] = score
    
    return numpy_array


def load_orientation_map(folder, run_ind):
    """Loads the orientation map from the folder for the training indexed by run_ind."""
    orimap_filename = os.path.join(folder, "orimap.csv")
    orimaps = pd.read_csv(orimap_filename, header=0)
    mesh_run = orimaps['run_index']==float(run_ind)
    orimap = orimaps[mesh_run].to_numpy()
    orimap = orimap[0][1:]

    return orimap


def load_parameters(folder_path, run_index, stage=1, iloc_ind=-1, for_training=False):
    """Loads the parameters from the pretraining_results.csv or training_results.csv file depending on the stage. 
    If for_training is True, then the last row of pretraining_results.csv is loaded as readout parameters are not trained during training. 
    If for_training is False, then the full last row of training_results.csv is loaded. 
    The parameters are then used to initialize the untrained parameters and readout parameters.
    The offset_last is the last offset value from the psychometric offsets."""

    from training.util_gabor import init_untrained_pars
    from parameters import SSNPars, ReadoutPars, TrainedSSNPars, PretrainedSSNPars, GridPars, FilterPars, StimuliPars, ConvPars, TrainingPars, LossPars, PretrainingPars
    ssn_pars, readout_pars, trained_pars, pretrained_pars = SSNPars(), ReadoutPars(), TrainedSSNPars(), PretrainedSSNPars()
    grid_pars, filter_pars, stimuli_pars = GridPars(), FilterPars(), StimuliPars()
    conv_pars, training_pars, loss_pars, pretraining_pars = ConvPars(), TrainingPars(), LossPars(), PretrainingPars()
    
    ###### Set the J, f and c parameters from pretraining_results.csv or training_results.csv depending on stage ######
    # define what keys the output pars_dict should have
    if for_training:
        par_keys = {attr: getattr(trained_pars, attr) for attr in dir(trained_pars) if not callable(getattr(trained_pars, attr)) and not attr.startswith("__") and not attr.startswith("_")}
    else:
        par_keys = {attr: getattr(pretrained_pars, attr) for attr in dir(pretrained_pars) if not callable(getattr(pretrained_pars, attr)) and not attr.startswith("__") and not attr.startswith("_")}

    # Get the iloc_ind row of the pretraining_results or training_results csv file depending on stage
    if stage<2:
        if os.path.exists(os.path.join(os.path.dirname(folder_path), 'pretraining_results.csv')):
            df_all = pd.read_csv(os.path.join(os.path.dirname(folder_path), 'pretraining_results.csv'))
        else: #this happens when load_parameters is called for stage 1 during pretraining
            df_all = pd.read_csv(os.path.join(folder_path, 'pretraining_results.csv'))
    else:
        df_all = pd.read_csv(os.path.join(folder_path, 'training_results.csv'))
    df = filter_for_run_and_stage(df_all, run_index, stage)
    if df.empty:
        print(f'Empty dataframe for run index {run_index} and stage {stage}.')
        if for_training:
            return None, None, None, None, None
        else:
            return None, None, None
    selected_row = df.iloc[int(iloc_ind)]

    # Define parameter keys
    log_J_keys = ['log_J_EE_m','log_J_EI_m','log_J_IE_m','log_J_II_m', 'log_J_EE_s','log_J_EI_s','log_J_IE_s','log_J_II_s'] 
    J_keys = ['J_EE_m','J_EI_m','J_IE_m','J_II_m', 'J_EE_s','J_EI_s','J_IE_s','J_II_s']
    kappa_Jsup_keys = ['kappa_Jsup_EE','kappa_Jsup_EI','kappa_Jsup_IE','kappa_Jsup_II']
    kappa_Jmid_keys = ['kappa_Jmid_EE','kappa_Jmid_EI','kappa_Jmid_IE','kappa_Jmid_II']
    kappa_f_keys = ['kappa_f_E','kappa_f_I']
    # Initialize the trained parameters dictionary with the J parameters
    trained_pars_dict = {}
    for i in range(len(log_J_keys)):
        log_J_key = log_J_keys[i]
        J_key = J_keys[i]
        if log_J_key in par_keys or J_key in par_keys:
            trained_pars_dict[log_J_key] = selected_row[log_J_key]
        else:
            # Set the attribute J_key in ssn_pars to the value from the selected row
            setattr(ssn_pars, J_key, selected_row[J_key])

    # Set c, f and kappa_Jsup parameters
    for param in ['cE_m', 'cI_m', 'cE_s', 'cI_s']:
        if param in par_keys or param in par_keys:
            trained_pars_dict[param] = selected_row[param]
        else:
            setattr(ssn_pars, param, selected_row[param])
    if 'log_f_E' in par_keys or 'f_E' in par_keys:
        trained_pars_dict['log_f_E'] = selected_row['log_f_E']
    else:
        ssn_pars.f_E = selected_row['f_E']
    if 'log_f_E' in par_keys or 'f_E' in par_keys:
        trained_pars_dict['log_f_I'] = selected_row['log_f_I']
    else:
        ssn_pars.f_I = selected_row['f_I']
    if 'kappa_Jsup' in par_keys:
        if kappa_Jsup_keys[0] in selected_row.keys():
            kappa_Jsup_values = [selected_row[key] for key in kappa_Jsup_keys]
            trained_pars_dict['kappa_Jsup'] = jnp.array(kappa_Jsup_values).reshape(2, 2)
        else: # case when kappa_Jsup are not in selected_row but they are required in trained_pars (beginning of training)
            trained_pars_dict['kappa_Jsup'] = trained_pars.kappa_Jsup
    if 'kappa_Jmid' in par_keys:
        if kappa_Jmid_keys[0] in selected_row.keys():
            kappa_Jmid_values = [selected_row[key] for key in kappa_Jmid_keys]
            trained_pars_dict['kappa_Jmid'] = jnp.array(kappa_Jmid_values).reshape(2, 2)
        else: # case when kappa_Jsup are not in selected_row but they are required in trained_pars (beginning of training)
            trained_pars_dict['kappa_Jmid'] = trained_pars.kappa_Jmid
    if 'kappa_f' in par_keys:
        if kappa_f_keys[0] in selected_row.keys():
            kappa_f_values = [selected_row[key] for key in kappa_f_keys]
            trained_pars_dict['kappa_f'] = jnp.array(kappa_f_values).reshape(2, 2)
        else: # case when kappa_Jsup are not in selected_row but they are required in trained_pars (beginning of training)
            trained_pars_dict['kappa_f'] = trained_pars.kappa_f
    
    ###### Extract readout parameters from pretraining.csv and save it to readout pars ######
    # If stage is >0, then load the last row of pretraining_results.csv as readout parameters are not trained during training
    if stage == 2:
        df_all = pd.read_csv(os.path.join(os.path.dirname(folder_path), 'pretraining_results.csv'))
        df = filter_for_run_and_stage(df_all, run_index)
        selected_row = df.iloc[-1]
   
    # Load the whole or the middle readout parameters depending on if the load is for training or pretraining
    if for_training:
        middle_grid_inds = readout_pars.middle_grid_ind
    else:
        middle_grid_inds = range(readout_pars.readout_grid_size[0]**2)
    w_sig_keys = [f'w_sig_{middle_grid_inds[i]}' for i in range(len(middle_grid_inds))] 
    w_sig_list = [float(selected_row[key]) for key in w_sig_keys]
    w_sig_values = jnp.array(w_sig_list)
    b_sig_value = float(selected_row['b_sig'])
    readout_pars_loaded = dict(w_sig=w_sig_values, b_sig=b_sig_value)
    readout_pars.b_sig = b_sig_value
    readout_pars.w_sig = w_sig_values

    ###### Set the gE, gI and eta parameters from initial_parameters.csv ######
    if os.path.exists(os.path.join(os.path.dirname(folder_path), 'initial_parameters.csv')):
        df_init_pars_all = pd.read_csv(os.path.join(os.path.dirname(folder_path), 'initial_parameters.csv'))
    else:
        df_init_pars_all = pd.read_csv(os.path.join(folder_path, 'initial_parameters.csv'))
    stage_for_init_pars=min(1,stage)
    df_init_pars = filter_for_run_and_stage(df_init_pars_all, run_index, stage_for_init_pars)
 
    training_pars.eta = df_init_pars['eta'].iloc[0]
    filter_pars.gE_m = df_init_pars['gE'].iloc[0]
    filter_pars.gI_m = df_init_pars['gI'].iloc[0]

    ###### Initialize untrained parameters with the loaded values ######
    # Load orientation map
    if os.path.exists(os.path.join(os.path.dirname(folder_path),"orimap.csv")):
        loaded_orimap = load_orientation_map(os.path.dirname(folder_path), run_index)
    else:
        loaded_orimap = load_orientation_map(folder_path, run_index)
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                    loss_pars, training_pars, pretraining_pars, readout_pars, orimap_loaded=loaded_orimap)
    
    ###### Get other metrics needed for training ######
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
           
        return readout_pars_loaded, trained_pars_dict, untrained_pars, offset_last, meanr_vec
    else:
        return readout_pars_loaded, trained_pars_dict, untrained_pars


def filter_for_run_and_stage(df, run_index, stage=None):
    """Filters the dataframe for the run_index and stage."""

    # Convert run_index to numeric and filter the dataframe for the run_index
    df['run_index'] = pd.to_numeric(df['run_index'], errors='coerce')
    mesh_run = df['run_index'] == run_index
    if not any(mesh_run):
        print('Returning empty dataframe from filter_for_run_and_stage for run index', run_index)
    df_run = df[mesh_run]
    df_run = df_run.drop(columns=['run_index'])
    df_run = df_run.reset_index(drop=True)

    # If stage is provided, then filter the dataframe for the stage
    if stage is not None:
        mesh_stage = df_run['stage'] == stage
        if any(mesh_stage):
            df_stage = df_run[mesh_stage]
            df_stage = df_stage.drop(columns=['stage'])
            df_stage = df_stage.reset_index(drop=True)
        elif stage==1: # case when stage 1 is required but there is no stage 1 in the dataframe - return stage 0
            mesh_stage = df_run['stage'] == stage-1
            df_stage = df_run[mesh_stage]
            df_stage = df_stage.drop(columns=['stage'])
            df_stage = df_stage.reset_index(drop=True)
        else:
            print('Returning empty dataframe from filter_for_run_and_stage')
            df_stage = df_run[mesh_stage]
            df_stage = df_stage.drop(columns=['stage'])
            df_stage = df_stage.reset_index(drop=True)
        return df_stage
    else:
        return df_run


def set_up_config_folder(results_folder, conf_name):
    """Create a folder for the training configuration and copy the necessary files to it."""
    config_folder = Path(results_folder + '/' + conf_name)
    config_folder.mkdir(parents=True, exist_ok=True)
    figure_folder = config_folder / 'figures'
    figure_folder.mkdir(parents=True, exist_ok=True)
    return config_folder


def configure_parameters_file(root_folder, conf):
    """
    Load parameters.py, change the parameters according to the input of this function, and then save parameters.py with these new parameters.
    """
    def extract_attributes(lines, class_names, keys):
        """Extract attributes and descriptions from a specific class based on keys."""
        attributes = []
        in_class = False

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if any([line.startswith(f"class {class_name}") for class_name in class_names]):
                in_class = True
            elif line.startswith("class "): # switch off in_class when we get to a new class that is not of interest
                in_class = False

            if in_class and any(line.startswith(key) for key in keys):
                description = lines[i + 1].strip()
                attributes.append((line, description))
                i += 1  # Skip the next line (description)
            
            i += 1

        return attributes

    def remove_attributes(lines, class_name, keys):
        """Remove specific attributes from a class based on keys."""
        in_class = False
        updated_lines = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith(f"class {class_name}"):
                in_class = True
            elif line.startswith("class "):
                in_class = False

            # Skip attributes that should be removed
            if in_class and any(line.startswith(key) for key in keys):
                i += 2  # Skip the attribute and its description
                continue

            updated_lines.append(lines[i])
            i += 1

        return updated_lines

    def add_attributes(lines, class_name, attributes):
        """Add attributes and descriptions to a specific class."""
        in_class = False
        updated_lines = []
        added = False

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith(f"class {class_name}"):
                in_class = True
            elif line.startswith("class "):
                in_class = False

            updated_lines.append(lines[i])

            # Add attributes at the start of the class if they haven't been added yet
            if in_class and not added:
                for declaration, description in attributes:
                    updated_lines.append(f"    {declaration}\n")
                    updated_lines.append(f"    {description}\n")
                added = True

            i += 1

        return updated_lines
    
    # Extract input - handles default values
    trained_pars_list, sup_mid_readout_contrib, pretraining_task, p_local_s, shuffle_labels, opt_readout_before_training = conf + ([[1.0, 0.0], False, [0.4, 0.7], False, False])[len(conf)-1:6]

    # Load the parameters.py file content
    params_file_path = Path(os.path.join(root_folder,"parameters.py"))
    if not params_file_path.exists():
        raise FileNotFoundError(f"{params_file_path} does not exist.")
    
    with open(params_file_path, "r") as file:
        lines = file.readlines()

    # Create a backup of the original parameters.py
    bak_file_path = params_file_path.with_suffix(".py.bak")
    
    # Copy backup if the backup file does not exist
    if not os.path.exists(bak_file_path):
        shutil.copy(params_file_path, bak_file_path)

    #### Update the ReadoutPars and TrainingPars parameters ####
    updated_lines = []
    for line in lines:
        # Update sup_mid_readout_contrib
        if "sup_mid_readout_contrib" in line:
            line = f"    sup_mid_readout_contrib = {sup_mid_readout_contrib}\n"
        
        # Update pretraining_task in TrainingPars
        if "pretraining_task:" in line:
            line = f"    pretraining_task: bool = {pretraining_task}\n"

        # Update p_local_s in SSNPars
        if "p_local_s" in line:
            line = f"    p_local_s = {p_local_s}\n"

        # Update shuffle_labels in TrainingPars
        if "shuffle_labels" in line:
            line = f"    shuffle_labels: bool = {shuffle_labels}\n"

        # Update opt_readout_before_training in TrainingPars
        if "opt_readout_before_training" in line:
            line = f"    opt_readout_before_training: bool = {opt_readout_before_training}\n"

        # Update is_on in PretrainingPars
        if "is_on: bool = True" in line:
            line = f"    is_on: bool = False\n"
        
        updated_lines.append(line)

    #### Extract, Remove, and Add Attributes ####
    all_keys = ['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_II_m', 'J_EI_m', 'J_IE_m', 'J_EE_m', 'J_II_s', 'J_EI_s', 'J_IE_s', 'J_EE_s', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f']
    ssnpars_keys = [key for key in all_keys if key not in trained_pars_list]
    
    # Extract attributes from both classes
    ssnpars_attributes = extract_attributes(lines, ["SSNPars", "TrainedSSNPars"], ssnpars_keys)
    trainedpars_attributes = extract_attributes(lines, ["SSNPars", "TrainedSSNPars"], trained_pars_list)

    # Remove attributes from the original classes
    updated_lines = remove_attributes(updated_lines, "SSNPars", all_keys)
    updated_lines = remove_attributes(updated_lines, "TrainedSSNPars", all_keys)

    # Add attributes to the opposite classes
    updated_lines = add_attributes(updated_lines, "SSNPars", ssnpars_attributes)
    updated_lines = add_attributes(updated_lines, "TrainedSSNPars", trainedpars_attributes)

    # Save the updated parameters.py file
    with open(params_file_path, "w") as file:
        file.writelines(updated_lines)

    print(f"Updated parameters.py with new parameters.")
