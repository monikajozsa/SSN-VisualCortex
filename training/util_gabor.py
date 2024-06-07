#import matplotlib.pyplot as plt
import numpy
import jax.numpy as np
from jax import jit, lax, vmap
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from parameters import StimuliPars

######### Orimap and initialization of untrained parameters #########
# class definition to collect parameters that are not trained
class UntrainedPars:
    def __init__(self, grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, ssn_ori_map, oris, ori_dist, gabor_filters, 
                 readout_pars):
        self.grid_pars = grid_pars
        self.stimuli_pars = stimuli_pars
        self.filter_pars = filter_pars
        self.ssn_ori_map = ssn_ori_map
        self.oris = oris
        self.ori_dist = ori_dist
        self.ssn_pars = ssn_pars
        self.conv_pars = conv_pars
        self.loss_pars = loss_pars
        self.training_pars = training_pars
        self.gabor_filters = gabor_filters
        self.readout_grid_size = readout_pars.readout_grid_size
        self.middle_grid_ind = readout_pars.middle_grid_ind
        self.num_readout_noise = readout_pars.num_readout_noise
        self.pretrain_pars = pretrain_pars
        self.BW_image_jax_inp = BW_image_jax_supp(stimuli_pars)


def cosdiff_ring(d_x, L):
    """
    Calculate the cosine-based distance.
    Parameters:
    d_x: The difference in the angular position.
    L: The total angle.
    """
    # Calculate the cosine of the scaled angular difference
    cos_angle = np.cos(d_x * 2 * np.pi / L)

    # Calculate scaled distance
    distance = np.sqrt( (1 - cos_angle) * 2) * L / (2 * np.pi)

    return distance


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
    
    chi_squared_stat = sum(((obs - expected_freq) ** 2) / expected_freq for obs in observed_freq)
    
    # Chi-square table for degrees of freedom 1-20 and significance level 0.1, 0.05, 0.025 and 0.01
    sig_levels = numpy.array([0.25, 0.1, 0.05, 0.025, 0.01])
    row_ind = num_bins-1 -1 # degree of freedom is bins -1 and index starts from 0
    col_ind = numpy.argmin(numpy.abs(numpy.ones_like(sig_levels)*alpha-sig_levels))
    
    # create chi-square table manually to avoid loading a package
    ChiSquareTable = numpy.array([[1.323,2.706, 3.841, 5.024, 6.635],
                                [2.773,4.605, 5.991, 7.378, 9.210],
                                [4.108, 6.251, 7.815, 9.348, 11.345],
                                [5.385, 7.779, 9.488, 11.143, 13.277],
                                [6.626, 9.236, 11.070, 12.833, 15.086],
                                [7.841, 10.645, 12.592, 14.449, 16.812],
                                [9.037, 12.017, 14.067, 16.013, 18.475],
                                [10.219, 13.362, 15.507, 17.535, 20.090],
                                [11.389, 14.684, 16.919, 19.023, 21.666],
                                [12.549, 15.987, 18.307, 20.483, 23.209],
                                [13.701, 17.275, 19.675, 21.920, 24.725],
                                [14.845, 18.549, 21.026, 23.337, 26.217],
                                [15.984, 19.812, 22.362, 24.736, 27.688],
                                [17.117, 21.064, 23.685, 26.119, 29.141],
                                [18.245, 22.307, 24.996, 27.488, 30.578],
                                [19.369, 23.542, 26.296, 28.845, 32.000],
                                [20.489, 24.769, 27.587, 30.191, 33.409],
                                [21.605, 25.989, 28.869, 31.526, 34.805],
                                [22.718, 27.204, 30.144, 32.852, 36.191],
                                [23.828, 28.412, 31.410, 34.170, 37.566]])
    
    chi_squared_critical = ChiSquareTable[row_ind,col_ind]

    if chi_squared_stat <= chi_squared_critical and all(numpy.array(observed_freq) > expected_freq/3) and all(numpy.array(observed_freq) < expected_freq*3):
        #Fail to reject the null hypothesis: The distribution may be uniform.
        return True
    else:
        #Reject the null hypothesis: The distribution is not uniform.        
        return False


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
			sj = (2 * numpy.random.randint(0, 2) - 1)

		# Define phase shift
		if deterministic:
			phij = 2 * numpy.pi * j/ nn
		else:
			phij = numpy.random.rand() * 2 * numpy.pi

		# Construct the j-th wave and add to the total
		tmp = (X * kj[0] + Y * kj[1]) * sj + phij
		z += np.exp(1j * tmp)

	# Convert the accumulated complex plane to orientation map
	# Orientation values are in the range (0, 180] degrees
	ori_map = (np.angle(z) + np.pi) * 180 / (2 * np.pi)

	return ori_map


def init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, file_to_save = None, orimap_loaded=None, regen_extended_orimap=False):
    """
    Define untrained_pars with a randomly generated or given orientation map.
    """

    if (orimap_loaded is not None) and (orimap_loaded.shape[0] == grid_pars.gridsize_Nx):
         ssn_ori_map=orimap_loaded
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
            if map_gen_ind>50:
                print('############## After 50 attempts the randomly generated maps did not pass the uniformity test ##############')
                break
    
    gabor_filters = create_gabor_filters_ori_map(ssn_ori_map, ssn_pars.phases, filter_pars, grid_pars, flatten=True)
    '''
    # testing the gabor output - should be below 100 - not quite because we do not do phase matching in find_gabor_A?
    ori=45
    BW_image_jit_inp_all = BW_image_jax_supp(stimuli_pars, phase = 0)
    x = BW_image_jit_inp_all[5]
    y = BW_image_jit_inp_all[6]
    alpha_channel = BW_image_jit_inp_all[7]
    mask = BW_image_jit_inp_all[8]
    background = BW_image_jit_inp_all[9]

    test_stimuli = BW_image_jax(BW_image_jit_inp_all[0:5], x, y, alpha_channel, mask, background, ori, 0)
    gabor_filters[0,:] @ test_stimuli
    '''
    oris = ssn_ori_map.ravel()[:, None]
    ori_dist = cosdiff_ring(oris - oris.T, 180)
    
    # Collect parameters that are not trained into a single class
    untrained_pars = UntrainedPars(grid_pars, stimuli_pars, filter_pars, ssn_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, ssn_ori_map, oris, ori_dist, gabor_filters, 
                 readout_pars)
    
    # Save orimap if file_name is specified
    if file_to_save is not None:
        numpy.save(file_to_save, ssn_ori_map)

    return untrained_pars


###### Functions for grating generation ###### ***
def BW_image_jax_supp(stimuli_pars, phase=0.0):
    """
    This function supports BW_image_jax (that generates grating images) by calculating variables that do not need to be recalculated in the training loop. 
    """     
    _BLACK = 0
    _WHITE = 255
    _GRAY = round((_WHITE + _BLACK) / 2)
    degree_per_pixel = stimuli_pars.degree_per_pixel
    pixel_per_degree = 1 / degree_per_pixel
    smooth_sd = pixel_per_degree / 6
    grating_size = round(stimuli_pars.outer_radius * pixel_per_degree)
    gridsize_deg = stimuli_pars.gridsize_deg
    size = int(stimuli_pars.gridsize_deg  * 2 * pixel_per_degree) + 1
    
    # following the same pattern as in gabor filter generation
    N_pixels = grating_size*2+1
    x_jax, y_jax = calculate_shifted_coords_mm(gridsize_deg, N_pixels)
    
    ##### Calculating alpha_channel_jax, mask_bool and background_jax #####
    x, y = numpy.mgrid[
        -grating_size : grating_size + 1.0,
        -grating_size : grating_size + 1.0,
    ]
    edge_control_dist = numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2))
    edge_control = numpy.divide(edge_control_dist, pixel_per_degree)

    # Define a matrix (alpha_channel) that is 255 (white) within the inner_radius and exponentially fades to 0 as the radius increases
    overrado = edge_control > stimuli_pars.inner_radius
    d = grating_size * 2 + 1
    annulus = numpy.ones((d, d))
    exponent_part = -1 * ((edge_control[overrado] - stimuli_pars.inner_radius) * pixel_per_degree) ** 2 / (2 * (smooth_sd**2))
    annulus[overrado] *= numpy.exp(exponent_part)
    alpha_channel = annulus.reshape(d,d) * _WHITE
    alpha_channel_jax = np.array(alpha_channel)

    # Define a boolean mask for outside the grating size - this will be used to set pixels outside the grating size to _GRAY
    mask = (edge_control_dist > grating_size).reshape((2 * int(grating_size) + 1,2 * int(grating_size) + 1))
    mask_bool = np.array(mask, dtype=bool)

    # Define indices for bounding box
    center_x, center_y = size // 2, size // 2
    bbox_height = np.abs(center_x - grating_size)
    bbox_width = np.abs(center_y - grating_size)
    start_indices = (int(bbox_height), int(bbox_width))

    # Define gray background
    background_jax = np.full((size, size), _GRAY, dtype=np.float32)

    # Define input for BW_image_jax or 
    BW_image_const_inp = (stimuli_pars.k, stimuli_pars.grating_contrast, phase, stimuli_pars.std, start_indices, x_jax, y_jax, alpha_channel_jax, mask_bool, background_jax)
    
    return BW_image_const_inp


def BW_image_jax(BW_image_const_inp, x, y, alpha_channel, mask, background, ref_ori, jitter):
    """
    Creates grating images.
    
    Parameters:
    - BW_image_const_inp: Constants for generating the Gabor stimulus, including noise std and start_indices.
    - x, y: Meshgrid arrays for spatial coordinates.
    - alpha_channel: Alpha channel for blending.
    - mask: Binary mask for the stimulus.
    - background: Background image.
    - ref_ori: Reference orientation for the Gabor stimulus.
    - jitter: Orientation jitter.
    
    Returns:
    - images: The final images
    """
    _GRAY = 128.0
    k = BW_image_const_inp[0]
    grating_contrast = BW_image_const_inp[1]
    phases = BW_image_const_inp[2]
    start_indices = BW_image_const_inp[4]
      
    # Calculate the angle in radians, incorporating the orientation and jitter
    angle = ((ref_ori + jitter) - 90) / 180 * np.pi

    # Compute the spatial component of the grating
    spatial_component = np.cos(2 * np.pi * k * (x * np.cos(angle) + y * np.sin(angle) ) + phases )
    
    # Generate the Gabor stimulus
    gabor_sti = _GRAY * (1 + grating_contrast * spatial_component)

    # Apply the mask, setting pixels outside to a neutral gray
    gabor_sti = np.where(mask, _GRAY, gabor_sti)

    # Blend the Gabor stimulus with the alpha channel in ROI
    if gabor_sti.shape[0]+2*start_indices[0]==background.shape[0]: # this is true for regular call of the function but not true for find_gabor_A, when gabor_sti is 257x257 and not 121x121
        result_roi = np.floor(alpha_channel / 256 * gabor_sti + (1.0 - alpha_channel / 256) * _GRAY)
    else:
        alpha_channel_resized = alpha_channel[start_indices[0]:-start_indices[0],start_indices[1]:-start_indices[1]] 
        gabor_sti_resized = gabor_sti[start_indices[0]:-start_indices[0],start_indices[1]:-start_indices[1]] 
        result_roi = np.floor(alpha_channel_resized / 256 * gabor_sti_resized + (1.0 - alpha_channel_resized / 256) * _GRAY)
    
    # Update the background image with result_roi
    combined_image = lax.dynamic_update_slice(background, result_roi, start_indices)

    return 3*combined_image.ravel()

def BW_image_full(BW_image_const_inp, x, y, ref_ori, jitter):
    '''Do what BW_image_jax does but without the GRAY backgroung'''
    _GRAY = 128.0
    k = BW_image_const_inp[0]
    grating_contrast = BW_image_const_inp[1]
    phases = BW_image_const_inp[2]

    # Calculate the angle in radians, incorporating the orientation and jitter
    angle = ((ref_ori + jitter) - 90) / 180 * np.pi

    # Compute the spatial component of the grating # spatial_freq=0.05 default
    spatial_component = np.cos(2 * np.pi * k *(x * np.cos(angle) + y * np.sin(angle) ) + phases )
    # Generate the Gabor stimulus
    gabor_sti = _GRAY * (1 + grating_contrast * spatial_component)
    
    return 3*gabor_sti.ravel()

# Vectorize BW_image function to process batches
BW_image_vmap = vmap(BW_image_jax, in_axes=(None,None,None,None,None,None,0,0))
BW_image_full_vmap = vmap(BW_image_full, in_axes=(None,None,None,0,0))
# Compile the vectorized functions for performance
BW_image_jit = jit(BW_image_vmap, static_argnums=[0])
BW_image_full_jit = jit(BW_image_full_vmap, static_argnums=[0])

def BW_image_jit_noisy(BW_image_const_inp, x, y, alpha_channel, mask, background, ref_ori, jitter, background_flag = True):
    """
    Calls BW_image_jit function and adds Gaussian noise to its output.
    """
    # Generate the images
    if background_flag:
        images = BW_image_jit(BW_image_const_inp, x, y, alpha_channel, mask, background, ref_ori, jitter)
    else:
        images = BW_image_full_jit(BW_image_const_inp, x, y, ref_ori, jitter)
    # Add noise to the images
    if BW_image_const_inp[3]>0:
        noisy_images = images + np.array(numpy.random.normal(loc=0, scale=BW_image_const_inp[3], size=images.shape))
    else:
         noisy_images = images
    return noisy_images


#### Functions for gabor filters ####

def calculate_shifted_coords_mm(gridsize_deg, N_pixels, x_i=0, y_i=0):
    # create image axis
    x_1D = np.linspace(-gridsize_deg, gridsize_deg, N_pixels, endpoint=True)
    x_1D = np.reshape(x_1D, (N_pixels, 1))
    y_1D = np.linspace(-gridsize_deg, gridsize_deg, N_pixels, endpoint=True)
    y_1D = np.reshape(y_1D, (1, N_pixels))

    # Reshape the center coordinates into column vectors; repeat and reshape the center coordinates to allow calculating diff_x and diff_y
    x_2D = np.repeat(x_1D, N_pixels, axis=1)
    diff_x = x_2D - x_i

    y_2D = np.repeat(y_1D, N_pixels, axis=0)
    diff_y = y_2D - y_i

    return diff_x, diff_y

def gabor_filter(x_i, y_i,filter_pars,angle,phase=0):
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
    gridsize_deg = filter_pars.gridsize_deg
    degree_per_pixel = filter_pars.degree_per_pixel
    magnif_factor=filter_pars.magnif_factor
    # convert to mm from degrees and radian from degree
    x_i = x_i / magnif_factor
    y_i = y_i / magnif_factor
    angle = angle * (np.pi / 180)
    # calculate the number of pixels by converting gridsize_deg to pixels
    N_pixels = int(magnif_factor * gridsize_deg / degree_per_pixel) + 1

    diff_x, diff_y = calculate_shifted_coords_mm(gridsize_deg, N_pixels, x_i, y_i)

    # Calculate the spatial component of the Gabor filter (same convention as stimuli)
    spatial_component = np.cos(2 * np.pi * k  * (diff_x * np.cos(angle) + diff_y * np.sin(angle)) + phase)
    # Calculate the Gaussian component of the Gabor filter
    gaussian = np.exp(-0.5 * (diff_x**2 + diff_y**2) / sigma_g**2)
    '''fig, axs = plt.subplots(1, 2, figsize=(5*2, 5*1))
    spatial_component2 = np.cos(2 * np.pi * k  * (x_2D * np.cos(angle) + y_2D * np.sin(angle)) + phase)
    gaussian2 = np.exp(-0.5 * (x_2D**2 + y_2D**2) / sigma_g**2)
    axs[1].imshow(spatial_component2+gaussian2)
    axs[0].imshow(spatial_component+gaussian)
    plt.savefig('tests/gaussian_spatial_shift.png')'''
    gabor_filter= np.array(gaussian * spatial_component)
    return  gabor_filter #np.array(gaussian * spatial_component[::-1]) 


        
def find_gabor_A(
    filter_pars,
    oris,
    phase=0
):
    """
    Find constant to multiply Gabor filters.
    Input:
        gabor_pars: Filter parameters - centre already specified in function
        stimuli_pars: Stimuli parameters (high constrast and spanning all visual field)
        indices: List of orientatins in degrees to calculate filter and corresponding stimuli
    Output:
        A: value of constant so that contrast = 100
    """
    k = filter_pars.k
    gridsize_deg = filter_pars.gridsize_deg
    all_A = []
    # handling the scalar case
    if isinstance(oris, (int, float, numpy.integer, numpy.float16, numpy.float32)):
        oris = [oris]

    for ori in oris: 
        # create local_stimui_pars to pass it to BW_Gratings
        local_stimuli_pars = StimuliPars()
        # extract original parameters
        #grating_size = round(local_stimuli_pars.outer_radius / local_stimuli_pars.degree_per_pixel)
        #N_pixels = int(grating_size*2+1)
        local_stimuli_pars.gridsize_deg = gridsize_deg
        local_stimuli_pars.k = k
        local_stimuli_pars.outer_radius = gridsize_deg * 2
        local_stimuli_pars.inner_radius = gridsize_deg * 2
        local_stimuli_pars.grating_contrast = 0.99
        local_stimuli_pars.jitter_val = 0
        local_stimuli_pars.std = 0
        local_stimuli_pars.ref_ori = ori
        
        # generate test stimuli at ori orientation
        BW_image_jit_inp_all = BW_image_jax_supp(local_stimuli_pars, phase = phase)
        x = BW_image_jit_inp_all[5]
        y = BW_image_jit_inp_all[6]
        ''' This is how x and y are calculated within BW_image_jax_supp
        x_1D = np.linspace(-gridsize_deg, gridsize_deg, N_pixels, endpoint=True)
        x_1D = np.reshape(x_1D, (N_pixels,1))
        y_1D = np.linspace(-gridsize_deg, gridsize_deg, N_pixels, endpoint=True)
        y_1D = np.reshape(y_1D, (1,N_pixels))
        x = np.repeat(x_1D, N_pixels, axis=1)    
        y = np.repeat(y_1D, N_pixels, axis=0)
        '''
        alpha_channel = BW_image_jit_inp_all[7]
        mask = BW_image_jit_inp_all[8]
        background = BW_image_jit_inp_all[9] # this is the only 129 x 129, all x, y, alpha_channel, mask are 257x257 and this makes test_stimuli the right size
        
        test_stimuli = BW_image_jax(BW_image_jit_inp_all[0:5], x, y, alpha_channel, mask, background, ori, 0) #BW_image_full(BW_image_jit_inp_all[0:5], x, y,  ori, 0)#
        
        # Generate Gabor filter at orientation
        gabor = gabor_filter(0, 0,filter_pars,ori,phase=phase)
        mean_removed_filter = gabor - gabor.mean()
        
        # multiply filter and stimuli
        output_gabor = mean_removed_filter.ravel() @ test_stimuli

        # calculate value of A
        A_value = 100 / (output_gabor)

        # create list of A
        all_A.append(A_value)

    # find average value of A
    all_A = np.array(all_A)
    A = all_A.mean()

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
    grid_size_1D = ori_map.shape[0]
    grid_size_2D = grid_size_1D*grid_size_1D
    image_size = int(((filter_pars.gridsize_deg*grid_size_1D**2)//2)**2)
    gabors_all = numpy.zeros((grid_size_2D, num_phases, 2, image_size))

    # Iterate over SSN map
    gabors_demean = numpy.zeros((grid_size_2D, k, image_size))
    for phases_ind in range(len(phases)):
        phase = phases[phases_ind]
        A = find_gabor_A(filter_pars,
                    oris=ori_map[4,4], # change! - we only do it for the middle 
                    phase=phase
                )
        for i in range(grid_size_1D):
            for j in range(grid_size_1D):
                gabor = gabor_filter(grid_pars.x_map[i, j], grid_pars.y_map[i, j], filter_pars, ori_map[i, j], phase)                
                gabors_demean[i+grid_size_1D*j,phases_ind,:] = A * (gabor.ravel()-np.mean(gabor)) # E filters
    gabors_all[:,0:k,0,:] = filter_pars.gE_m*gabors_demean # E filters phase 0 and pi/2
    gabors_all[:,k:2*k,0,:] = - filter_pars.gE_m*gabors_demean # E filters with opposite phases
    gabors_all[:,0:k,1,:] = filter_pars.gI_m*gabors_demean # I filters phase 0 and pi/2
    gabors_all[:,k:2*k,1,:] = - filter_pars.gI_m*gabors_demean # I filters with opposite phases
    if flatten: # flatten the first three dimensions of gabors_all
        gabors_all = gabors_all.reshape((grid_size_2D*num_phases*2, image_size))
    return np.array(gabors_all)