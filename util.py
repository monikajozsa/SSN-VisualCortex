import jax
import jax.numpy as np
import matplotlib
matplotlib.use('Agg')
import numpy
import os
import shutil
from datetime import datetime
from scipy import ndimage

from util_gabor import BW_Grating, BW_image_jit_noisy

def smooth_data(vector, sigma = 1):

    '''
    Smooth fixed point. Data is reshaped into 9x9 grid
    '''
    
    new_data = []
    for trial_response in vector:

        trial_response = trial_response.reshape(9,9,-1)
        smoothed_data = numpy.asarray([ndimage.gaussian_filter(numpy.reshape(trial_response[:, :, i], (9,9)), sigma = sigma) for i in range(0, trial_response.shape[2])]).ravel()
        new_data.append(smoothed_data)
    
    return np.vstack(np.asarray(new_data))  

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


def exponentiate(opt_pars):
    signs = np.array([[1, -1], [1, -1]])
    J_2x2 = np.exp(opt_pars["logJ_2x2"]) * signs
    s_2x2 = np.exp(opt_pars["logs_2x2"])

    return J_2x2, s_2x2


def sep_exponentiate(J_s):
    signs = np.array([[1, -1], [1, -1]])
    new_J = np.exp(np.array(J_s, dtype = float)) * signs

    return new_J


def create_grating_training(stimuli_pars, batch_size, jit_inp_all= None):
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
    target_ori_vec = np.where(mask, ref_ori - offset, ref_ori + offset)
    labels = mask.astype(int)  # Converts True/False to 1/0
    jitter_val = stimuli_pars.jitter_val
    jitters = numpy.random.uniform(low = -jitter_val, high = jitter_val, size=batch_size)

    # Create reference and target gratings
    # if jit_inp_all is given then use the jit-compatible version of BW_image. Otherwise, use the original BW_gratings class
    if jit_inp_all is None:
        for i in range(batch_size):                     
            
            ref = BW_Grating(ori_deg = ref_ori, jitter=jitters[i], stimuli_pars = stimuli_pars).BW_image().ravel()
            target = BW_Grating(ori_deg = target_ori_vec[i], jitter=jitters[i], stimuli_pars = stimuli_pars).BW_image().ravel()
                
            data_dict['ref'].append(ref)
            data_dict['target'].append(target)
            data_dict['label'].append(labels)
    else:
        ref_ori_vec = np.ones(batch_size)*ref_ori
        x = jit_inp_all[5]
        y = jit_inp_all[6]
        alpha_channel = jit_inp_all[7]
        mask_jax = jit_inp_all[8]
        background = jit_inp_all[9]
        roi =jit_inp_all[10]
        
        ref = BW_image_jit_noisy(jit_inp_all[0:5],x,y,alpha_channel,mask_jax, background, roi, ref_ori_vec, jitters)
        target = BW_image_jit_noisy(jit_inp_all[0:5],x,y,alpha_channel,mask_jax, background, roi, target_ori_vec, jitters)
        data_dict['ref']=ref
        data_dict['target']=target
        data_dict['label']=labels
            
    data_dict['ref'] = np.asarray(data_dict['ref'])
    data_dict['target'] = np.asarray(data_dict['target'])
    data_dict['label'] = np.asarray(data_dict['label'])

    return data_dict


def generate_random_pairs(min_value, max_value, min_distance, max_distance=None, batch_size=1, tot_angle=180, numRnd_ori1=1):
    '''
    Create batch_size number of pairs of numbers between min_value and max_value with minimum distance min_distance and maximum distance max_distance.
    If tot_angle is provided, values wrap around between 0 and tot_angle.
    '''
    if max_distance==None:
        max_distance = max_value - min_value

    # Generate the first numbers
    rnd_numbers = numpy.random.uniform(min_value, max_value, numRnd_ori1) #numpy.random.randint(low=min_value, high=max_value, size=numRnd_ori1, dtype=int)
    num1 = numpy.repeat(rnd_numbers, int(batch_size/numRnd_ori1))

    # Generate a random distance within specified range
    random_distance = numpy.random.choice([-1, 1], batch_size) * numpy.random.uniform(min_distance,max_distance ,batch_size) #numpy.random.randint(low=min_distance,high=max_distance, size=batch_size, dtype=int)

    # Generate the second numbers with correction if they are out of the specified range
    num2 = num1 - random_distance #order and sign are important!

    # Create a mask where flip_numbers equals 1
    swap_numbers = numpy.random.choice([0, 1], batch_size) 
    mask = swap_numbers == 1

    # Swap values where mask is True
    # We'll use a temporary array to hold the values of num1 where the mask is True
    temp_num1 = np.copy(num1[mask])
    num1[mask] = num2[mask]
    num2[mask] = temp_num1
    random_distance[mask] = -random_distance[mask]
    
    # Apply wrap-around logic
    #num2[num2 > tot_angle] = num2[num2 > tot_angle] - tot_angle
    #num2[num2 < 0] = num2[num2 < 0] + tot_angle
    
    return np.array(num1), np.array(num2), random_distance


def create_grating_pretraining(pretrain_pars, batch_size, jit_inp_all, numRnd_ori1=1):
    '''
    Create input stimuli gratings for pretraining by randomizing ref_ori for both reference and target (with random difference between them)
    Output:
        dictionary containing grating1, grating2 and difference between gratings that is calculated from features
    '''
    
    # Initialise empty data dictionary - names are not describing the purpose of the variables but this allows for reusing code
    data_dict = {'ref': [], 'target': [], 'label':[]}

    # Randomize orientations for stimulus 1 and stimulus 2
    L_ring = 180
    min_ori_dist = pretrain_pars.min_ori_dist
    max_ori_dist = pretrain_pars.max_ori_dist
    ori1, ori2, ori_diff = generate_random_pairs(min_value=30, max_value=150, min_distance=min_ori_dist, max_distance=max_ori_dist, batch_size=batch_size, tot_angle=L_ring, numRnd_ori1=numRnd_ori1)

    x = jit_inp_all[5]
    y = jit_inp_all[6]
    alpha_channel = jit_inp_all[7]
    mask_jax = jit_inp_all[8]
    background = jit_inp_all[9]
    roi =jit_inp_all[10]
    
    # Generate stimulus1 and stimulus2 with no jitter and no noise (seed needs to be randomized if we add noise!)
    stim1 = BW_image_jit_noisy(jit_inp_all[0:5], x, y, alpha_channel, mask_jax, background, roi, ori1, jitter=np.zeros_like(ori1))
    stim2 = BW_image_jit_noisy(jit_inp_all[0:5], x, y, alpha_channel, mask_jax, background, roi, ori2, jitter=np.zeros_like(ori1))
    data_dict['ref']=stim1
    data_dict['target']=stim2

    # Define label as the normalized signed difference in angle
    label = np.zeros_like(ori_diff)
    data_dict['label'] = label.at[ori_diff > 0].set(1)

    return data_dict


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


def save_code(folder_to_save=None):
    '''
    This code is used to save code files to make results replicable.
    1) It copies specific code files into a folder called 'script'
    3) Returns the path to save the results into
    '''
    # Get the current date
    current_date = datetime.now().strftime("%b%d")

    # Create a folder name based on the current date
    if folder_to_save is None:
        folder_name = f"results/{current_date}_v"
    else:
        folder_name = folder_to_save + f"/{current_date}_v"

    # Find the next available script version
    version = 0
    while os.path.exists(f"{folder_name}{version}"):
        version += 1

    # Create the folder for the results
    final_folder_path = f"{folder_name}{version}"
    os.makedirs(final_folder_path)

    # Create a subfolder for the scripts
    subfolder_script_path = f"{folder_name}{version}/scripts"
    os.makedirs(subfolder_script_path)

    # Get the path to the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))

    # Copy files into the folder
    file_names = ['main.py', 'util_gabor.py', 'pretraining_supp.py', 'parameters.py', 'training.py', 'model.py', 'util.py', 'SSN_classes.py', 'analysis.py', 'visualization.py']
    for file_name in file_names:
        source_path = os.path.join(script_directory, file_name)
        destination_path = os.path.join(subfolder_script_path, file_name)
        shutil.copyfile(source_path, destination_path)

    print(f"Script files copied successfully to: {subfolder_script_path}")

    # return path (inclusing filename) to save results into
    results_filename = os.path.join(final_folder_path,f"{current_date}_v{version}_results.csv")

    return results_filename, final_folder_path
