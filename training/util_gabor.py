#import matplotlib.pyplot as plt
import numpy
from numpy import random
import jax.numpy as np
from jax import jit, vmap
import pandas as pd
import os, sys
from scipy.stats import chi2_contingency
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from parameters import StimuliPars

###### Functions for generating and handling orimaps ###### 

def test_uniformity(numbers, num_bins=18, alpha=0.25):
    """
    This function assesses the uniformity of 'numbers' within the range [0, 180] by dividing the range into 'num_bins' 
    equally sized bins and comparing the observed frequencies in these bins against the expected frequencies for a uniform 
    distribution. The test is performed at a significance level 'alpha'.

    Parameters:
    - numbers (list or array-like): The set of numbers to test for uniformity.
    - num_bins (int): The number of bins to use for dividing the range [0, 180]. Default is 10.
    - alpha (float): The significance level for the chi-squared test. Default is 0.1.

    Returns:
    - bool: False if the null hypothesis (that the distribution is uniform) is rejected, True otherwise.
    """

    n = len(numbers)
    expected_freq = n / num_bins
    observed_freq = [0] * num_bins
    
    for number in numbers:
        if 0 <= number <= 180:  # Ensure the number is within the desired range
            bin_index = int((number / 180) * num_bins)
            observed_freq[bin_index] += 1
  
    # Perform the Chi-square test
    _, p_value, _, expected_freq = chi2_contingency(observed_freq)
    if p_value <= alpha and np.all(np.array(observed_freq) > np.array(expected_freq) / 3) and np.all(np.array(observed_freq) < np.array(expected_freq) * 3):
        return False
    else:
        return True


def make_orimap(X, Y, hyper_col=None, nn=30, deterministic=False):
	"""
	Makes the orientation map for the grid by superposition of plane-waves.

	Parameters:
	hyper_col: Hyper column length for the network in retinotopic degrees.
			   Determines the spatial frequency of the waves.
	nn: Number of plane waves used to construct the map (30 by default).
	X, Y: Coordinates for the grid points. If not provided, uses internal grid maps.

	Outputs/side-effects:
	ori_map: Orientation preference for each cell in the network.
	ori_vec: Vectorized version of ori_map.
	"""

	# Set or update the hyper column length
	if hyper_col is None:
		hyper_col = 3.2
	
	# Initialize a complex plane to accumulate wave contributions
	z = np.zeros_like(X, dtype=complex)

	# Loop to create and superimpose plane waves
	for j in range(nn):
		# Wave vector for j-th plane wave, direction varies with j
		kj = (numpy.array([numpy.cos(j * numpy.pi / nn), numpy.sin(j * numpy.pi / nn)]) * 2 * numpy.pi / hyper_col)

		# Determine sign (+1 or -1) to vary wave orientation
		if deterministic:
			sj = 1 if j % 2 == 0 else -1
		else:
			sj = (2 * random.randint(0, 2) - 1)

		# Define phase shift
		if deterministic:
			phij = 2 * numpy.pi * j/ nn
		else:
			phij = random.rand() * 2 * numpy.pi

		# Construct the j-th wave and add to the total
		tmp = (X * kj[0] + Y * kj[1]) * sj + phij
		z += np.exp(1j * tmp)

	# Convert the accumulated complex plane to orientation map; orientation values are in the range (0, 180] degrees
	ori_map = (np.angle(z) + np.pi) * 180 / (2 * np.pi)

	return ori_map


def save_orimap(untrained_pars, run_ind, folder_to_save=None):
    """Save the orimap to a csv file"""
    # ravel ssn_ori_map and add run_ind as a first element
    ssn_ori_map = untrained_pars.oris
    ssn_ori_map = numpy.insert(ssn_ori_map, 0, run_ind)

    # add header as the header of the file if run_ind == 0
    if run_ind == 0:
        orimap_header = []
        orimap_header.append('run_index')
        # Grid points
        for i in range(untrained_pars.grid_pars.gridsize_Nx**2):
            orimap_header.append(str(i))
    else:
        orimap_header = False

    # Save the orimap to a csv file
    ssn_ori_map_df = pd.DataFrame(ssn_ori_map.reshape(1, -1))
    ssn_ori_map_df.to_csv(folder_to_save + '/orimap.csv', mode='a', header=orimap_header, index=False, float_format='%.4f')
    print('Saved orimap to ' + folder_to_save + '/orimap.csv')


###### Functions for generating and handling gratings ###### 

def BW_image_jax_supp(stimuli_pars, x0 = 0, y0=0, phase=0.0, full_grating=False):
    """
    This function supports BW_image_jax (that generates grating images) by calculating variables that do not need to be recalculated in the training loop. 
    """     
    _WHITE = 255
    degree_per_pixel = stimuli_pars.degree_per_pixel
    pixel_per_degree = 1 / degree_per_pixel
    smooth_sd = pixel_per_degree / 6
    
    # Getting image coordinates
    x_mm, y_mm, N_pixs = calculate_shifted_coords_mm(stimuli_pars, x0, y0)
    
    ##### Calculating alpha_channel_jax, mask_bool and background_jax #####
    x_pix = np.round(2 * x_mm /(stimuli_pars.magnif_factor * degree_per_pixel))
    y_pix = np.round(2 * y_mm /(stimuli_pars.magnif_factor * degree_per_pixel))
    edge_control_dist = np.sqrt(np.power(x_pix, 2) + np.power(y_pix, 2))
    edge_control = np.divide(edge_control_dist, pixel_per_degree)
    # Define a matrix (alpha_channel) that is 255 (white) within the inner_radius and exponentially fades to 0 as the radius increases
    overrado = edge_control > stimuli_pars.inner_radius
    annulus = numpy.ones((N_pixs, N_pixs))
    if not full_grating:
        exponent_part = -1 * ((edge_control[overrado] - stimuli_pars.inner_radius) * pixel_per_degree) ** 2 / (2 * (smooth_sd**2))
        annulus[overrado] *= np.exp(exponent_part)
    alpha_channel = annulus.reshape(N_pixs,N_pixs) * _WHITE
    alpha_channel_jax = np.array(alpha_channel)

    # Define a boolean mask for outside the grating size - this will be used to set pixels outside the grating size to _GRAY
    if not full_grating:
        mask = (edge_control_dist > (N_pixs-1)//2).reshape((N_pixs,N_pixs))
    else:
        mask = np.zeros_like(alpha_channel_jax) # if we want full gratings with no masking with the background
    mask_bool = np.array(mask, dtype=bool)

    # Define input for BW_image_jax or 
    BW_image_const_inp = (stimuli_pars.k, stimuli_pars.grating_contrast, phase, stimuli_pars.std, x_mm, y_mm, alpha_channel_jax, mask_bool)
    
    return BW_image_const_inp


def BW_image_jax(BW_image_const_inp, x, y, alpha_channel, mask, ref_ori, jitter):
    """
    Creates grating images.
    
    Parameters:
    - BW_image_const_inp: Constants for generating the Gabor stimulus, including noise std and start_indices.
    - x, y: Meshgrid arrays for spatial coordinates.
    - alpha_channel: Alpha channel for blending.
    - mask: Binary mask for the stimulus.
    - ref_ori: Reference orientation for the Gabor stimulus.
    - jitter: Orientation jitter.
    
    Returns:
    - images: The final images
    """
    _GRAY = 128.0
    k = BW_image_const_inp[0]
    grating_contrast = BW_image_const_inp[1]
    phase = BW_image_const_inp[2]
      
    # Calculate the angle in radians, incorporating the orientation and jitter
    angle = (ref_ori + jitter) / 180 * np.pi

    # Compute the spatial component of the grating
    spatial_component = np.cos(2 * np.pi * k * (x * np.cos(angle) + y * np.sin(angle) ) + phase )

    # Generate the Gabor stimulus
    gabor_sti = _GRAY * (1 + grating_contrast * spatial_component)

    # Apply the mask, setting pixels outside to a neutral gray
    gabor_sti = np.where(mask, _GRAY, gabor_sti)

    # Blend the Gabor stimulus with the alpha channel in ROI
    final_image = np.floor(alpha_channel / 256 * gabor_sti + (1.0 - alpha_channel / 256) * _GRAY)

    return 3*final_image.ravel() # *** multiplied by 3 to match the SNR in Apr 10 run!

# Vectorize BW_image function to process batches
BW_image_vmap = vmap(BW_image_jax, in_axes=(None,None,None,None,None,0,0))

# Compile the vectorized functions for performance
BW_image_jit = jit(BW_image_vmap, static_argnums=[0])


def BW_image_jit_noisy(BW_image_const_inp, x, y, alpha_channel, mask, ref_ori, jitter):
    """
    Calls BW_image_jit function and adds Gaussian noise to its output.
    """
    # Generate the images
    images = BW_image_jit(BW_image_const_inp, x, y, alpha_channel, mask, ref_ori, jitter)
    
    # Add noise to the images
    if BW_image_const_inp[3]>0:
        noisy_images = images + np.array(random.normal(loc=0, scale=BW_image_const_inp[3], size=images.shape))
    else:
         noisy_images = images

    return noisy_images


###### Functions for generating and handling gabor filters ######

def calculate_shifted_coords_mm(class_pars, x0=0, y0=0):
    """Calculate the shifted coordinates of the grid points in mm centered at x0, y0."""
    # create image axis
    gridsize_deg = class_pars.gridsize_deg
    magnif_factor = class_pars.magnif_factor
    degree_per_pixel = class_pars.degree_per_pixel
    N_pixels = int(magnif_factor * gridsize_deg / degree_per_pixel) + 1
    x_1D = np.linspace(-gridsize_deg, gridsize_deg, N_pixels, endpoint=True)
    x_1D = np.reshape(x_1D, (N_pixels, 1))
    y_1D = np.linspace(-gridsize_deg, gridsize_deg, N_pixels, endpoint=True)
    y_1D = np.reshape(y_1D, (1, N_pixels))

    # convert to mm from degrees and radian from degree
    x0 = x0 / class_pars.magnif_factor
    y0 = y0 / class_pars.magnif_factor

    # Reshape the center coordinates into column vectors; repeat and reshape the center coordinates to allow calculating diff_x and diff_y
    x_2D = np.repeat(x_1D, N_pixels, axis=1)
    diff_x = x_2D - x0

    y_2D = np.repeat(y_1D, N_pixels, axis=0)
    diff_y = y_2D - y0

    return diff_x, diff_y, N_pixels


def gabor_filter(x0, y0,filter_pars,angle,phase=0):
    """
    Creates Gabor filters.
    Inputs:
        x_i, y_i: centre of the filter
        filter_pars: filter parameters including spatial frequency in cycles/degrees (radians) and other conversion constants
        angle: orientation of the Gabor filter
        phase: phase of the Gabor filter (default is 0)
    """

    k = filter_pars.k
    sigma_g = filter_pars.sigma_g
    diff_x, diff_y, _ = calculate_shifted_coords_mm(filter_pars, x0, y0)

    # Calculate the spatial component of the Gabor filter (same convention as stimuli)
    angle = angle * (np.pi / 180) # convert to mm from degrees and radian from degree
    spatial_component = np.cos(2 * np.pi * k  * (diff_x * np.cos(angle) + diff_y * np.sin(angle)) + phase)

    # Calculate the Gaussian component of the Gabor filter
    gaussian = np.exp(-0.5 * (diff_x**2 + diff_y**2) / sigma_g**2)

    # Multiply Gaussian and spatial_component to get the Gabor filter
    gabor_filter= np.array(gaussian * spatial_component)

    return  gabor_filter

        
def find_gabor_A(
    filter_pars,
    oris,
    phase=0
):
    """
    Find constant to multiply Gabor filters such that the maximum response .
    Input:
        gabor_pars: Filter parameters - centre already specified in function
        stimuli_pars: Stimuli parameters (high constrast and spanning all visual field)
        indices: List of orientatins in degrees to calculate filter and corresponding stimuli
    Output:
        A: value of constant so that contrast = 100
    """
    all_output_gabor = []
    # handling the scalar case
    if isinstance(oris, (int, float, numpy.integer, numpy.float16, numpy.float32)):
        oris = [oris]
    else:
        oris = oris.ravel()

    for ori in oris: 
        # Create local_stimui_pars to pass it to BW_Gratings
        local_stimuli_pars = StimuliPars()
        local_stimuli_pars.jitter_val = 0
        local_stimuli_pars.std = 0
        local_stimuli_pars.ref_ori = ori
        
        # Generate test stimuli at ori orientation
        BW_image_jit_inp_all = BW_image_jax_supp(local_stimuli_pars, phase = phase)
        x = BW_image_jit_inp_all[4]
        y = BW_image_jit_inp_all[5]
        alpha_channel = BW_image_jit_inp_all[6]
        mask = BW_image_jit_inp_all[7]
        
        test_stimuli = BW_image_jax(BW_image_jit_inp_all[0:5], x, y, alpha_channel, mask, local_stimuli_pars.ref_ori, 0) #BW_image_full(BW_image_jit_inp_all[0:5], x, y,  ori, 0)#
        
        # Generate Gabor filter at orientation
        gabor = gabor_filter(0, 0,filter_pars,ori,phase=phase)

        # Remove mean
        mean_removed_filter = gabor - gabor.mean()
        
        # Multiply filter and stimuli
        output_gabor = mean_removed_filter.ravel() @ test_stimuli

        # Create list of output_gabors
        all_output_gabor.append(output_gabor)

    # Find max value of all_output_gabor and define A to scale it to 100*local_stimuli_pars.grating_contrast
    A = 100*local_stimuli_pars.grating_contrast / np.mean(np.array(all_output_gabor))

    return A


def create_gabor_filters_ori_map(
    ori_map,
    num_phases,
    filter_pars,
    grid_pars, 
    flatten=False
):
    """Create Gabor filters for each orientation and phase in orimap."""
    k=int(num_phases//2)
    phases = np.linspace(0, np.pi, k, endpoint=False)
    grid_size_1D = grid_pars.gridsize_Nx
    grid_size_2D = grid_pars.gridsize_Nx**2
    if ori_map.shape[0] == grid_size_2D:
        ori_map = ori_map.reshape(grid_size_1D, grid_size_1D)
    image_size = int(((filter_pars.gridsize_deg*grid_size_1D**2)//2)**2)
    gabors_all = numpy.zeros((num_phases, 2, grid_size_2D, image_size))

    # Iterate over SSN map
    gabors_demean = numpy.zeros((k,grid_size_2D, image_size))
    for phases_ind in range(len(phases)):
        phase = phases[phases_ind]
        # Find scaling constant based on the middle cell
        A = find_gabor_A(filter_pars,oris=ori_map,phase=phase)
        for i in range(grid_size_1D):
            for j in range(grid_size_1D):
                gabor = gabor_filter(grid_pars.x_map[i, j], grid_pars.y_map[i, j], filter_pars, ori_map[i, j], phase)                
                gabors_demean[phases_ind,grid_size_1D*i+j,:] = A * (gabor.ravel()-np.mean(gabor)) # Demean and scale the Gabor filters
    gabors_all[0:k,0,:,:] = filter_pars.gE_m*gabors_demean # E filters phase 0 and pi/2
    gabors_all[k:2*k,0,:,:] = - filter_pars.gE_m*gabors_demean # E filters with opposite phases
    gabors_all[0:k,1,:,:] = filter_pars.gI_m*gabors_demean # I filters phase 0 and pi/2
    gabors_all[k:2*k,1,:,:] = - filter_pars.gI_m*gabors_demean # I filters with opposite phases
   
    if flatten: # flatten the first three dimensions of gabors_all
        gabors_all = gabors_all.reshape((num_phases*2*grid_size_2D, image_size))
    return np.array(gabors_all)


######### Class and functions for creating and handling untrained parameters #########
#########     These rely on gabor filter, grating and orimap generations     #########
# class definition to collect parameters that are not trained but used throughout training
class UntrainedPars:
    def __init__(self, grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, oris, ori_dist, gabor_filters, 
                 readout_pars, dist_from_single_ori):
        self.grid_pars = grid_pars
        self.stimuli_pars = stimuli_pars
        self.filter_pars = filter_pars
        self.oris = oris
        self.ori_dist = ori_dist
        self.ssn_pars = ssn_pars
        self.conv_pars = conv_pars
        self.loss_pars = loss_pars
        self.training_pars = training_pars
        self.gabor_filters = gabor_filters
        self.pretrain_pars = pretrain_pars
        self.BW_image_jax_inp = BW_image_jax_supp(stimuli_pars)
        self.readout_grid_size = readout_pars.readout_grid_size
        self.middle_grid_ind = readout_pars.middle_grid_ind
        self.sup_mid_readout_contrib = readout_pars.sup_mid_readout_contrib
        self.num_readout_noise = readout_pars.num_readout_noise
        self.dist_from_single_ori = dist_from_single_ori


def init_untrained_pars( grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, orimap_loaded=None, regen_extended_orimap=False):
    """
    Define untrained_pars with a randomly generated or given orientation map.
    """
    from util import cosdiff_ring
    if (orimap_loaded is not None):
        if (orimap_loaded.shape[0] == grid_pars.gridsize_Nx) or (orimap_loaded.shape[0] == grid_pars.gridsize_Nx**2):
            ssn_ori_map_flat=orimap_loaded
            if orimap_loaded.shape[0] == grid_pars.gridsize_Nx:
                ssn_ori_map_flat = ssn_ori_map_flat.ravel()         
        else:
            ValueError('The loaded orimap does not have the correct size')
    else:
        is_uniform = False
        map_gen_ind = 0
        X = grid_pars.x_map
        Y = grid_pars.y_map
        while not is_uniform:
            ssn_ori_map = make_orimap(X, Y, hyper_col=None, nn=30, deterministic=False)
            if (orimap_loaded is not None) and not regen_extended_orimap:
                # paste the loaded orimap in the middle of the generated orimap
                loaded_ori_size = orimap_loaded.shape[0]
                start_ind = (grid_pars.gridsize_Nx-loaded_ori_size)//2
                end_ind = start_ind+loaded_ori_size
                ssn_ori_map=ssn_ori_map.at[start_ind:end_ind,start_ind:end_ind].set(orimap_loaded)
            ssn_ori_map_flat = ssn_ori_map.ravel()
            is_uniform = test_uniformity(ssn_ori_map_flat[readout_pars.middle_grid_ind], num_bins=10, alpha=0.25)
            map_gen_ind = map_gen_ind+1
            if map_gen_ind>100:
                print(f'############## After {map_gen_ind} attempts the randomly generated maps did not pass the uniformity test ##############')
                break

    gabor_filters = create_gabor_filters_ori_map(ssn_ori_map_flat, ssn_pars.phases, filter_pars, grid_pars, flatten=True)
    oris = ssn_ori_map_flat[:, None]
    beta_rep = numpy.tile(ssn_pars.beta, (grid_pars.gridsize_Nx**2, 1))
    dist_from_single_ori = cosdiff_ring(oris - beta_rep.T, 180)
    ori_dist = cosdiff_ring(oris - oris.T, 180)
    
    # Collect parameters that are not trained into a single class
    untrained_pars = UntrainedPars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, ssn_ori_map_flat, ori_dist, gabor_filters, 
                 readout_pars, dist_from_single_ori)
    
    return untrained_pars


def update_untrained_pars(untrained_pars, readout_pars, gE_m, gI_m, eta= None):
    """Update the gE_m and gI_m filter parameters and the gabor filters (because their normalization depend on g)"""
    untrained_pars.filter_pars.gE_m = gE_m
    untrained_pars.filter_pars.gI_m = gI_m
    if eta is not None:
        untrained_pars.training_pars.eta = eta
    gabor_filters = create_gabor_filters_ori_map(untrained_pars.oris, untrained_pars.ssn_pars.phases, untrained_pars.filter_pars, untrained_pars.grid_pars, flatten=True)
    untrained_pars = UntrainedPars(untrained_pars.grid_pars, untrained_pars.stimuli_pars, untrained_pars.filter_pars, untrained_pars.ssn_pars, untrained_pars.conv_pars, 
                 untrained_pars.loss_pars, untrained_pars.training_pars, untrained_pars.pretrain_pars, untrained_pars.oris, untrained_pars.ori_dist, gabor_filters, 
                 readout_pars, untrained_pars.dist_from_single_ori)
    return untrained_pars
