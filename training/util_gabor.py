#import matplotlib.pyplot as plt
import numpy
from numpy import random
import jax.numpy as jnp
from jax import jit, vmap
import pandas as pd
import os, sys
from scipy.stats import chisquare
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from parameters import StimuliPars

###### Functions for generating and handling orimaps ###### 

def test_uniformity(numbers, num_bins=18, alpha=0.25):
    """
    This function assesses the uniformity of 'numbers' within the range [0, 180] by dividing the range into 'num_bins' 
    equally sized bins and comparing the observed frequencies in these bins against the expected frequencies for a uniform 
    distribution. The test is performed at a significance level 'alpha'.
    Inputs:
        numbers (list or array-like): The set of numbers to test for uniformity.
        num_bins (int): The number of bins to use for dividing the range [0, 180].
        alpha (float): The significance level for the chi-squared test.
    Output:
        bool: False if the null hypothesis (that the distribution is uniform) is rejected, True otherwise.
    """

    n = len(numbers)
    expected_freq = n / num_bins
    observed_freq = [0] * num_bins
    
    for number in numbers:
        if 0 <= number <= 180:  # Ensure the number is within the desired range
            bin_index = int((number / 180) * num_bins)
            observed_freq[bin_index] += 1
  
    # Perform the Chi-square test
    if any([observed_freq[i] == 0 for i in range(num_bins)]):
        return False
    else:
        expected_freq_val = n / num_bins
        expected_freq = expected_freq_val * numpy.ones_like(observed_freq)
        # Perform the Chi-square test
        stat, p_value = chisquare(f_obs=observed_freq, f_exp=expected_freq)
        if p_value > alpha:
            return True
        else:
            return False


def make_orimap(X, Y, hyper_col=None, nn=30):
	"""
	This function makes the orientation map for the grid by superposition of plane-waves.
	Inputs:
        X, Y: Coordinates for the grid points.
	    hyper_col: Hyper column length for the network in retinotopic degrees.
			   Determines the spatial frequency of the waves.
	    nn: Number of plane waves used to construct the map (30 by default).
	Output:
	    ori_map: Orientation preference for each cell in the network.
	"""

	# Set or update the hyper column length
	if hyper_col is None:
		hyper_col = 3.2
	
	# Initialize a complex plane to accumulate wave contributions
	z = jnp.zeros_like(X, dtype=complex)

	# Loop to create and superimpose plane waves
	for j in range(nn):
		# Wave vector for j-th plane wave, direction varies with j
		kj = (numpy.array([numpy.cos(j * numpy.pi / nn), numpy.sin(j * numpy.pi / nn)]) * 2 * numpy.pi / hyper_col)

		# Determine sign (+1 or -1) to vary wave orientation
		sj = (2 * random.randint(0, 2) - 1)

		# Define phase shift
		phij = random.rand() * 2 * numpy.pi

		# Construct the j-th wave and add to the total
		tmp = (X * kj[0] + Y * kj[1]) * sj + phij
		z += jnp.exp(1j * tmp)

	# Convert the accumulated complex plane to orientation map; orientation values are in the range (0, 180] degrees
	ori_map = (jnp.angle(z) + jnp.pi) * 180 / (2 * jnp.pi)

	return ori_map


def save_orimap(ssn_ori_map, run_ind, folder_to_save=None):
    """ 
    This function saves the orimap to a csv file 
    Inputs:
        orimap: The orientation map to save as a flattened array
        run_ind: The index of the run
        folder_to_save: The folder to save the orimap to
    """
    # add run_ind as a first element
    ssn_ori_map = numpy.insert(ssn_ori_map, 0, run_ind)

    # add header as the header of the file if run_ind == 0
    if run_ind == 0:
        orimap_header = []
        orimap_header.append('run_index')
        # Grid points
        for i in range(len(ssn_ori_map)-1):
            orimap_header.append(str(i))
    else:
        orimap_header = False

    # Save the orimap to a csv file
    ssn_ori_map_df = pd.DataFrame(ssn_ori_map.reshape(1, -1))
    ssn_ori_map_df.to_csv(folder_to_save + '/orimap.csv', mode='a', header=orimap_header, index=False, float_format='%.4f')
    print('Saved orimap to ' + folder_to_save + '/orimap.csv')


###### Functions for generating and handling images of gratings ###### 

def BW_image_jax_supp(stimuli_pars, x0 = 0, y0=0, phase=0.0, full_grating=False):
    """
    This function supports BW_image_jax (that generates grating images) by calculating variables that do not need to be recalculated in the training loop.
    Inputs:
        stimuli_pars: Stimuli parameters.
        x0, y0: Coordinates of the center of the image.
        phase: Phase of the grating.
        full_grating: Boolean indicating whether to generate full gratings (default is False).
    Output:
        BW_image_const_inp: tuple containing the variables supporting BW_image_jax.
    """     
    _WHITE = 255
    degree_per_pixel = stimuli_pars.degree_per_pixel
    pixel_per_degree = 1 / degree_per_pixel
    smooth_sd = pixel_per_degree / 6
    
    # Getting image coordinate arrays in mm and in pixels
    x_mm, y_mm, N_pixs = calculate_shifted_coords_mm(stimuli_pars, x0, y0)
    x_pix = jnp.round(2 * x_mm /(stimuli_pars.magnif_factor * degree_per_pixel))
    y_pix = jnp.round(2 * y_mm /(stimuli_pars.magnif_factor * degree_per_pixel))
    
    # Define a matrix (alpha_channel) that is 255 (white) within the inner_radius and if full_grating is False, it exponentially fades to 0 as the radius increases
    annulus = numpy.ones((N_pixs, N_pixs))
    if not full_grating:
        # Calculating spherical distance from the center of the image in degrees - grating will be in a circular region in the middle
        edge_control_dist = jnp.sqrt(jnp.power(x_pix, 2) + jnp.power(y_pix, 2))
        edge_control = jnp.divide(edge_control_dist, pixel_per_degree)
        overrado = edge_control > stimuli_pars.inner_radius
        exponent_part = -1 * ((edge_control[overrado] - stimuli_pars.inner_radius) * pixel_per_degree) ** 2 / (2 * (smooth_sd**2))
        annulus[overrado] *= jnp.exp(exponent_part)
    alpha_channel_jax = jnp.array(annulus * _WHITE)

    # Define a boolean mask (mask_bool) for outside the grating size - this will be used to set pixels outside the grating size to _GRAY
    if not full_grating:
        mask = (edge_control_dist > (N_pixs-1)//2).reshape((N_pixs,N_pixs))
    else:
        mask = jnp.zeros_like(alpha_channel_jax) # if we want full gratings with no masking with the background
    mask_bool = jnp.array(mask, dtype=bool)

    # Define input for BW_image_jax or 
    BW_image_const_inp = (stimuli_pars.k, stimuli_pars.grating_contrast, phase, stimuli_pars.std, x_mm, y_mm, alpha_channel_jax, mask_bool)
    
    return BW_image_const_inp


def BW_image_jax(BW_image_const_inp, x, y, alpha_channel, mask, ref_ori, jitter):
    """
    This function creates the grating images.    
    Inputs:
        BW_image_const_inp: Constants for generating the Gabor stimulus, including noise std and start_indices.
        x, y: Meshgrid arrays for spatial coordinates.
        alpha_channel: Alpha channel for blending.
        mask: Binary mask for the stimulus.
        ref_ori: Reference orientation for the Gabor stimulus.
        jitter: Orientation jitter.  
    Outout:
        images: The final images
    """
    _GRAY = 128.0
    k = BW_image_const_inp[0]
    grating_contrast = BW_image_const_inp[1]
    phase = BW_image_const_inp[2]
      
    # Calculate the angle in radians, incorporating the orientation and jitter
    angle = (ref_ori + jitter) / 180 * jnp.pi

    # Compute the spatial component of the grating
    spatial_component = jnp.cos(2 * jnp.pi * k * (x * jnp.cos(angle) + y * jnp.sin(angle) ) + phase )

    # Generate the Gabor stimulus
    gabor_sti = _GRAY * (1 + grating_contrast * spatial_component)

    # Apply the mask, setting pixels outside to a neutral gray
    gabor_sti = jnp.where(mask, _GRAY, gabor_sti)

    # Blend the Gabor stimulus with the alpha channel in ROI
    final_image = jnp.floor(alpha_channel / 256 * gabor_sti + (1.0 - alpha_channel / 256) * _GRAY)

    return 3*final_image.ravel() # For historical reasons (colored images), the output is 3 times the final_image

# Vectorize BW_image function to process batches
BW_image_vmap = vmap(BW_image_jax, in_axes=(None,None,None,None,None,0,0))

# Compile the vectorized functions for performance
BW_image_jit = jit(BW_image_vmap, static_argnums=[0])


def BW_image_jit_noisy(BW_image_const_inp, x, y, alpha_channel, mask, ref_ori, jitter):
    """
    This function calls BW_image_jit function and adds Gaussian noise to its output. NOTE: Noise adding is outside of jitted functions to avoid relying on pre-generated random numbers.
    Inputs:
        BW_image_const_inp: Constants, where BW_image_const_inp[3] is noise std and the rest are to be passed to BW_image_jit.
        x, y: Meshgrid arrays for spatial coordinates.
        alpha_channel: Alpha channel for blending with background.
        mask: Binary mask for the blending.
        ref_ori: Reference orientation.
        jitter: Small value to jitter the grating orientation with. 
    Output:
        noisy_images: the images with added noise.
    """
    # Generate the images
    images = BW_image_jit(BW_image_const_inp, x, y, alpha_channel, mask, ref_ori, jitter)
    
    # Add noise to the images
    if BW_image_const_inp[3]>0:
        noisy_images = images + jnp.array(random.normal(loc=0, scale=BW_image_const_inp[3], size=images.shape))
    else:
         noisy_images = images

    return noisy_images


###### Functions for generating and handling gabor filters ######

def calculate_shifted_coords_mm(class_pars, x0=0, y0=0):
    """
    This function calculates the shifted coordinates of the grid points in mm centered at x0, y0.
    Inputs:
        filter_pars: instance from the class FilterPars or StimuliPars containing parameters about the grid and conversions.
        x0, y0: Coordinates of the center of the image.    
    Outputs:
        diff_x, diff_y: The shifted coordinates in mm.
        N_pixels: The number of pixels in the image.
    """
    # create image axis
    gridsize_deg = class_pars.gridsize_deg
    magnif_factor = class_pars.magnif_factor
    degree_per_pixel = class_pars.degree_per_pixel
    N_pixels = int(magnif_factor * gridsize_deg / degree_per_pixel) + 1
    x_1D = jnp.linspace(-gridsize_deg, gridsize_deg, N_pixels, endpoint=True)
    x_1D = jnp.reshape(x_1D, (N_pixels, 1))
    y_1D = jnp.linspace(-gridsize_deg, gridsize_deg, N_pixels, endpoint=True)
    y_1D = jnp.reshape(y_1D, (1, N_pixels))

    # convert to mm from degrees and radian from degree
    x0 = x0 / class_pars.magnif_factor
    y0 = y0 / class_pars.magnif_factor

    # Reshape the center coordinates into column vectors; repeat and reshape the center coordinates to allow calculating diff_x and diff_y
    x_2D = jnp.repeat(x_1D, N_pixels, axis=1)
    diff_x = x_2D - x0

    y_2D = jnp.repeat(y_1D, N_pixels, axis=0)
    diff_y = y_2D - y0

    return diff_x, diff_y, N_pixels


def gabor_filter(x0, y0, filter_pars, angle, phase=0):
    """
    This function creates Gabor filters.
    Inputs:
        x_i, y_i: centre of the filter
        filter_pars: instance from the class FilterPars that includes spatial frequency in cycles/degrees (radians) and other parameters of the Gabor filters
        angle: orientation of the Gabor filter
        phase: phase of the Gabor filter (default is 0)
    Output:
        gabor_filter: The Gabor filters.
    """

    diff_x, diff_y, _ = calculate_shifted_coords_mm(filter_pars, x0, y0)

    # Calculate the spatial component of the Gabor filter (same convention as stimuli)
    angle = angle * (jnp.pi / 180) # convert to mm from degrees and radian from degree
    spatial_component = jnp.cos(2 * jnp.pi * filter_pars.k  * (diff_x * jnp.cos(angle) + diff_y * jnp.sin(angle)) + phase)

    # Calculate the Gaussian component of the Gabor filter
    gaussian = jnp.exp(-0.5 * (diff_x**2 + diff_y**2) / filter_pars.sigma_g**2)

    # Multiply Gaussian and spatial_component to get the Gabor filter
    gabor_filter= jnp.array(gaussian * spatial_component)

    return  gabor_filter

        
def find_gabor_A(
    filter_pars,
    oris,
    phase=0
):
    """
    This function calculates a constant to multiply Gabor filters with such that the maximum response has contrast defined by local_stimuli_pars.grating_contrast.
    Inputs:
        filter_pars: Filter parameters - centre already specified in function
        oris: Orientatin or list of orientatins in degrees
        phase: Phase of the Gabor filter
    Output:
        A: value of constant so that contrast is = 100*local_stimuli_pars.grating_contrast
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
    A = 100*local_stimuli_pars.grating_contrast / jnp.mean(jnp.array(all_output_gabor))

    return A


def create_gabor_filters_ori_map(
    ori_map,
    num_phases,
    filter_pars,
    grid_pars, 
    flatten=False
):
    """ 
    This function creates Gabor filters for each orientation and phase in orimap. 
    Inputs:
        ori_map: orientation map
        num_phases: number of phases to use
        filter_pars: filter parameters
        grid_pars: grid parameters
        flatten: if True, the output is flattened to 2D array
    Outputs:
        gabors_all: Gabor filters for each orientation and phase
    """
    # Extract input
    k=int(num_phases//2)
    grid_size_1D = grid_pars.gridsize_Nx
    grid_size_2D = grid_pars.gridsize_Nx**2
    if ori_map.shape[0] == grid_size_2D:
        ori_map = ori_map.reshape(grid_size_1D, grid_size_1D)
    # Initialize variables
    phases = jnp.linspace(0, jnp.pi, k, endpoint=False)
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
                gabors_demean[phases_ind,grid_size_1D*i+j,:] = A * (gabor.ravel()-jnp.mean(gabor)) # Demean and scale the Gabor filters
    gabors_all[0:k,0,:,:] = filter_pars.gE_m*gabors_demean # E filters phase 0 and pi/2
    gabors_all[k:2*k,0,:,:] = - filter_pars.gE_m*gabors_demean # E filters with opposite phases
    gabors_all[0:k,1,:,:] = filter_pars.gI_m*gabors_demean # I filters phase 0 and pi/2
    gabors_all[k:2*k,1,:,:] = - filter_pars.gI_m*gabors_demean # I filters with opposite phases
   
    if flatten: # flatten the first three dimensions of gabors_all
        gabors_all = gabors_all.reshape((num_phases*2*grid_size_2D, image_size))
    return jnp.array(gabors_all)


######### Class and functions for creating and handling untrained parameters #########
#########     These rely on gabor filter, grating and orimap generations     #########

class UntrainedPars:
    """ Collection of parameters that are not trained but used throughout training """
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


def init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, loss_pars, training_pars, 
                        pretrain_pars, readout_pars, orimap_loaded=None, regen_extended_orimap=False):
    """
    Define untrained_pars with a randomly generated or given orientation map.
    Inputs:
        grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, loss_pars, training_pars, pretrain_pars, readout_pars: instances of parameter classes
        orimap_loaded: loaded orientation map to use
        regen_extended_orimap: if True, the orientation map is regenerated with the loaded orimap in the center
    Output: 
        untrained_pars: instance of UntrainedPars class
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
            ssn_ori_map = make_orimap(X, Y, hyper_col=None, nn=30)
            if (orimap_loaded is not None) and not regen_extended_orimap:
                # paste the loaded orimap in the middle of the generated orimap
                loaded_ori_size = orimap_loaded.shape[0]
                start_ind = (grid_pars.gridsize_Nx-loaded_ori_size)//2
                end_ind = start_ind+loaded_ori_size
                ssn_ori_map = ssn_ori_map.at[start_ind:end_ind,start_ind:end_ind].set(orimap_loaded)
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
    """ Update the gE_m and gI_m filter parameters and the gabor filters (because their normalization depend on g) 
    Inputs:
        untrained_pars: instance of UntrainedPars
        readout_pars: instance of ReadoutPars
        gE_m, gI_m: new values for the filter parameters
        eta: new value for the learning rate
    Output:
        untrained_pars: updated instance of UntrainedPars
    """
    untrained_pars.filter_pars.gE_m = gE_m
    untrained_pars.filter_pars.gI_m = gI_m
    if eta is not None:
        untrained_pars.training_pars.eta = eta
    gabor_filters = create_gabor_filters_ori_map(untrained_pars.oris, untrained_pars.ssn_pars.phases, untrained_pars.filter_pars, untrained_pars.grid_pars, flatten=True)
    untrained_pars = UntrainedPars(untrained_pars.grid_pars, untrained_pars.stimuli_pars, untrained_pars.filter_pars, untrained_pars.ssn_pars, untrained_pars.conv_pars, 
                 untrained_pars.loss_pars, untrained_pars.training_pars, untrained_pars.pretrain_pars, untrained_pars.oris, untrained_pars.ori_dist, gabor_filters, 
                 readout_pars, untrained_pars.dist_from_single_ori)
    return untrained_pars