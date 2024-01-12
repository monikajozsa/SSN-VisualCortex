import numpy
from PIL import Image

from parameters import StimuliPars

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
	z = numpy.zeros_like(X, dtype=complex)

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
		z += numpy.exp(1j * tmp)

	# Convert the accumulated complex plane to orientation map
	# Orientation values are in the range (0, 180] degrees
	ori_map = (numpy.angle(z) + numpy.pi) * 180 / (2 * numpy.pi)

	return ori_map


class BW_Grating:
    """ """

    def __init__(
        self,
        ori_deg,
        stimuli_pars,
        jitter=0,
        phase=0,
        crop_f=None,
    ):
        self.ori_deg = ori_deg
        self.jitter = jitter
        self.outer_radius = stimuli_pars.outer_radius  # in degrees
        self.inner_radius = stimuli_pars.inner_radius  # in degrees
        self.grating_contrast = stimuli_pars.grating_contrast
        self.std = stimuli_pars.std
        degree_per_pixel = stimuli_pars.degree_per_pixel
        pixel_per_degree = 1 / degree_per_pixel
        self.pixel_per_degree = pixel_per_degree
        edge_deg = stimuli_pars.edge_deg
        size = int(edge_deg * 2 * pixel_per_degree) + 1
        self.size = size
        k = stimuli_pars.k
        spatial_frequency = k * degree_per_pixel  
        self.phase = phase
        self.crop_f = crop_f
        self.smooth_sd = self.pixel_per_degree / 6
        self.spatial_freq = spatial_frequency or (1 / self.pixel_per_degree)
        self.grating_size = round(self.outer_radius * self.pixel_per_degree)

    def BW_image(self):
        _BLACK = 0
        _WHITE = 255
        _GRAY = round((_WHITE + _BLACK) / 2)

        # Generate a 2D grid of coordinates
        x, y = numpy.mgrid[
            -self.grating_size : self.grating_size + 1.0,
            -self.grating_size : self.grating_size + 1.0,
        ]

        # Calculate the distance from the center for each pixel
        edge_control_dist = numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2))
        edge_control = numpy.divide(edge_control_dist, self.pixel_per_degree)

        # Create a matrix (alpha_channel) that is 255 (white) within the inner_radius and exponentially fades to 0 as the radius increases
        overrado = numpy.nonzero(edge_control > self.inner_radius)
        d = self.grating_size * 2 + 1
        annulus = numpy.ones((d, d))

        annulus[overrado] *= numpy.exp(
            -1
            * ((edge_control[overrado] - self.inner_radius) * self.pixel_per_degree)
            ** 2
            / (2 * (self.smooth_sd**2))
        )
        alpha_channel = annulus * _WHITE

        # Generate the grating pattern, which is a centered and tilted sinusoidal matrix
        #initialize output
        angle = ((self.ori_deg + self.jitter) - 90) / 180 * numpy.pi

        spatial_component = (
            2
            * numpy.pi
            * self.spatial_freq
            * (y * numpy.sin(angle) + x * numpy.cos(angle))
        )
        gabor_sti = _GRAY * (
            1 + self.grating_contrast * numpy.cos(spatial_component + self.phase)
        )

        # Set pixels outside the grating size to gray
        gabor_sti[edge_control_dist > self.grating_size] = _GRAY

        # Add Gaussian white noise to the grating
        if self.std>0:
            noisy_gabor_sti = gabor_sti + numpy.random.normal(loc=0, scale=self.std, size=gabor_sti.shape)
        else:
            noisy_gabor_sti = gabor_sti 

        # Expand the grating to have three colors and concatenate it with alpha_channel
        gabor_sti_final = numpy.repeat(noisy_gabor_sti[:, :, numpy.newaxis], 3, axis=-1)
        gabor_sti_final_with_alpha = numpy.concatenate(
            (gabor_sti_final, alpha_channel[:, :, numpy.newaxis]), axis=-1
        )
        gabor_sti_final_with_alpha_image = Image.fromarray(
            gabor_sti_final_with_alpha.astype(numpy.uint8)
        )

        # Create a background image filled with gray
        background = numpy.full((self.size, self.size, 3), _GRAY, dtype=numpy.uint8)
        final_image = Image.fromarray(background)

        # Paste the grating into the final image: paste the grating into a bounding box and apply the alpha channel as a mask
        center_x, center_y = self.size // 2, self.size // 2
        bounding_box = (center_x - self.grating_size, center_y - self.grating_size)
        final_image.paste(
            gabor_sti_final_with_alpha_image,
            box=bounding_box,
            mask=gabor_sti_final_with_alpha_image,
        )

        # Sum the image over color channels
        final_image_np = numpy.array(final_image, dtype=numpy.float16)
        image=numpy.sum(final_image_np, axis=2)

        # Crop the image if crop_f is specified
        if self.crop_f:
            image = image[self.crop_f : -self.crop_f, self.crop_f : -self.crop_f]     

        return image
import numpy
from PIL import Image
import jax.numpy as np
from jax import random, jit, lax

from parameters import StimuliPars

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
	z = numpy.zeros_like(X, dtype=complex)

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
		z += numpy.exp(1j * tmp)

	# Convert the accumulated complex plane to orientation map
	# Orientation values are in the range (0, 180] degrees
	ori_map = (numpy.angle(z) + numpy.pi) * 180 / (2 * numpy.pi)

	return ori_map


class BW_Grating:
    """ """

    def __init__(
        self,
        ori_deg,
        stimuli_pars,
        jitter=0,
        phase=0
    ):
        self.ori_deg = ori_deg
        self.jitter = jitter
        self.outer_radius = stimuli_pars.outer_radius  # in degrees
        self.inner_radius = stimuli_pars.inner_radius  # in degrees
        self.grating_contrast = stimuli_pars.grating_contrast
        self.std = stimuli_pars.std
        degree_per_pixel = stimuli_pars.degree_per_pixel
        pixel_per_degree = 1 / degree_per_pixel
        self.pixel_per_degree = pixel_per_degree
        edge_deg = stimuli_pars.edge_deg
        size = int(edge_deg * 2 * pixel_per_degree) + 1
        self.size = size
        k = stimuli_pars.k
        spatial_frequency = k * degree_per_pixel  
        self.phase = phase
        self.smooth_sd = self.pixel_per_degree / 6
        self.spatial_freq = spatial_frequency or (1 / self.pixel_per_degree)
        self.grating_size = round(self.outer_radius * self.pixel_per_degree)

    def BW_image(self):
        _BLACK = 0
        _WHITE = 255
        _GRAY = round((_WHITE + _BLACK) / 2)

        # Generate a 2D grid of coordinates
        x, y = numpy.mgrid[
            -self.grating_size : self.grating_size + 1.0,
            -self.grating_size : self.grating_size + 1.0,
        ]

        # Calculate the distance from the center for each pixel
        edge_control_dist = numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2))
        edge_control = numpy.divide(edge_control_dist, self.pixel_per_degree)

        # Create a matrix (alpha_channel) that is 255 (white) within the inner_radius and exponentially fades to 0 as the radius increases
        overrado = numpy.nonzero(edge_control > self.inner_radius)
        d = self.grating_size * 2 + 1
        annulus = numpy.ones((d, d))

        annulus[overrado] *= numpy.exp(
            -1
            * ((edge_control[overrado] - self.inner_radius) * self.pixel_per_degree)
            ** 2
            / (2 * (self.smooth_sd**2))
        )
        alpha_channel = annulus * _WHITE

        # Generate the grating pattern, which is a centered and tilted sinusoidal matrix
        #initialize output
        angle = ((self.ori_deg + self.jitter) - 90) / 180 * numpy.pi

        spatial_component = (
            2
            * numpy.pi
            * self.spatial_freq
            * (y * numpy.sin(angle) + x * numpy.cos(angle))
        )
        gabor_sti = _GRAY * (
            1 + self.grating_contrast * numpy.cos(spatial_component + self.phase)
        )

        # Set pixels outside the grating size to gray
        gabor_sti[edge_control_dist > self.grating_size] = _GRAY

        # Add Gaussian white noise to the grating
        if self.std>0:
            noisy_gabor_sti = gabor_sti + numpy.random.normal(loc=0, scale=self.std, size=gabor_sti.shape)
        else:
            noisy_gabor_sti = gabor_sti 

        # Expand the grating to have three colors and concatenate it with alpha_channel
        gabor_sti_final = numpy.repeat(noisy_gabor_sti[:, :, numpy.newaxis], 3, axis=-1)
        gabor_sti_final_with_alpha = numpy.concatenate(
            (gabor_sti_final, alpha_channel[:, :, numpy.newaxis]), axis=-1
        )
        gabor_sti_final_with_alpha_image = Image.fromarray(
            gabor_sti_final_with_alpha.astype(numpy.uint8)
        )

        # Create a background image filled with gray
        background = numpy.full((self.size, self.size, 3), _GRAY, dtype=numpy.uint8)
        final_image = Image.fromarray(background)

        # Paste the grating into the final image: paste the grating into a bounding box and apply the alpha channel as a mask
        center_x, center_y = self.size // 2, self.size // 2
        bounding_box = (center_x - self.grating_size, center_y - self.grating_size)
        final_image.paste(
            gabor_sti_final_with_alpha_image,
            box=bounding_box,
            mask=gabor_sti_final_with_alpha_image,
        )

        # Sum the image over color channels
        final_image_np = numpy.array(final_image, dtype=numpy.float16)
        image=numpy.sum(final_image_np, axis=2)
       
        return image
    
def BW_image_jax(jit_inp, ref_ori, jitter, seed):
    pixel_per_degree = jit_inp[0]
    inner_radius = jit_inp[1]
    outer_radius = jit_inp[2]
    edge_deg = jit_inp[3]
    k = jit_inp[4]
    phases = jit_inp[5]
    grating_contrast = jit_inp[6]
    std  = jit_inp[7]
    _WHITE = 255.0
    _GRAY = 128.0
    smooth_sd = pixel_per_degree / 6
    grating_size = outer_radius * pixel_per_degree
    size = int(edge_deg * 2 * pixel_per_degree) + 1
    spatial_freq = k / pixel_per_degree
    # Generate a 2D grid of coordinates
    x, y = np.mgrid[
        -grating_size : grating_size + 1.0,
        -grating_size : grating_size + 1.0,
    ]
    
    # Calculate the distance from the center for each pixel
    edge_control_dist = np.sqrt(np.power(x, 2) + np.power(y, 2))
    edge_control = edge_control_dist / pixel_per_degree

    # Create a matrix that fades to 0 as the radius increases

    # i) Calculate the exponential part for all elements
    exp_part = np.exp(-1 * ((edge_control - inner_radius) * pixel_per_degree) ** 2 / (2 * (smooth_sd**2)))

    # ii) Apply a mask to the exponential part, where edge_control <= inner_radius is multiplied by 0
    mask = (edge_control_dist > grating_size).reshape((2 * int(grating_size) + 1,2 * int(grating_size) + 1))
    mask = np.array(mask, dtype=bool)
    alpha_channel = np.where(mask, exp_part, 1) * _WHITE

    # Generate the grating pattern
    angle = ((ref_ori + jitter) - 90) / 180 * np.pi
    spatial_component = 2 * np.pi * spatial_freq * (y * np.sin(angle) + x * np.cos(angle))
    gabor_sti = _GRAY * (1 + grating_contrast * np.cos(spatial_component + phases))

    # Use concrete shapes in reshape
    #gabor_sti = gabor_sti.at[mask].set(_GRAY)
    gabor_sti = np.where(mask, _GRAY, gabor_sti)

    # Add Gaussian white noise to the grating
    rng_key=random.PRNGKey(seed)
    noisy_gabor_sti = gabor_sti + random.normal(rng_key, gabor_sti.shape) * std
    
    # Create a background image filled with gray
    background = np.full((size, size), _GRAY, dtype=np.float32)

    # Paste the grating into the final image: paste the grating into a bounding box and apply the alpha channel as a mask
    center_x, center_y = size // 2, size // 2
    bbox_height=center_x - grating_size
    bbox_width = center_y - grating_size        
    alpha_height, alpha_width = noisy_gabor_sti.shape

    # Calculate the region of interest (ROI) in the background
    start_indices = (int(bbox_height), int(bbox_width))
    slice_sizes = (int(alpha_height), int(alpha_width))

    roi = lax.dynamic_slice(background, start_indices, slice_sizes)
    
    result_roi = np.floor(alpha_channel/255 * noisy_gabor_sti + (1.0 - alpha_channel/255) * roi)

    # Place the modified ROI back into the background
    #combined_image = background.at[bbox_height:bbox_height+alpha_height, bbox_width:bbox_width+alpha_width].set(result_roi)
    combined_image = lax.dynamic_update_slice(background, result_roi, start_indices)
    
    ### consider inputting roi, mask , background and alpha_channel to minimize what is calculated iteratively
    return 3*combined_image

jit_BW_image_jax = jit(BW_image_jax, static_argnums = [0])

#### CREATE GABOR FILTERS ####
class GaborFilter:
    def __init__(
        self,
        x_i,
        y_i,
        k,
        sigma_g,
        theta,
        edge_deg,
        degree_per_pixel,
        phase=0,
        conv_factor=None,
    ):
        """
        Gabor filter class.
        Called from SSN_mid_local.create_gabor_filters() and SSN_mid.create_gabor_filters() whose outputs are gabor_filters and A (attributes of SSN_mid and SSN_mid_local)
        Inputs:
            x_i, y_i: centre of the filter
            k: preferred spatial frequency in cycles/degrees (radians)
            sigma_g: variance of Gaussian function
            theta: orientation map
            edge_deg: extent of the filter in degrees
            degree_per_pixel: resolution in degrees per pixel
            phase: phase of the Gabor filter (default is 0)
            conv_factor: conversion factor from degrees to mm
        """

        # convert to mm from degrees
        if conv_factor:
            self.conv_factor = conv_factor
            self.x_i = x_i / conv_factor
            self.y_i = y_i / conv_factor
        else:
            self.x_i = x_i
            self.y_i = y_i
        self.k = k
        self.theta = theta * (numpy.pi / 180)
        self.phase = phase
        self.sigma_g = sigma_g
        self.edge_deg = edge_deg
        self.degree_per_pixel = degree_per_pixel
        self.N_pixels = int(edge_deg * 2 / degree_per_pixel) + 1

        # create image axis
        x_axis = numpy.linspace(-edge_deg, edge_deg, self.N_pixels, endpoint=True)
        y_axis = numpy.linspace(-edge_deg, edge_deg, self.N_pixels, endpoint=True)

        # construct filter as an attribute
        self.filter = self.create_filter(x_axis, y_axis)

    def create_filter(self, x_axis, y_axis):
        """
        Create Gabor filters in vectorised form.
        """
        # Reshape the center coordinates into column vectors; repeat and reshape the center coordinates to allow calculating diff_x and diff_y
        x_axis = numpy.reshape(x_axis, (self.N_pixels, 1))
        x_i = numpy.repeat(self.x_i, self.N_pixels)
        x_i = numpy.reshape(x_i, (self.N_pixels, 1))
        diff_x = x_axis.T - x_i

        y_axis = numpy.reshape(y_axis, (self.N_pixels, 1))
        y_i = numpy.repeat(self.y_i, self.N_pixels)
        y_i = numpy.reshape(y_i, (self.N_pixels, 1))
        diff_y = y_axis - y_i.T

        # Calculate the spatial component of the Gabor filter
        spatial = numpy.cos(
            self.k
            * numpy.pi
            * 2
            * (diff_x * numpy.cos(self.theta) + diff_y * numpy.sin(self.theta))
            + self.phase
        )
        # Calculate the Gaussian component of the Gabor filter
        gaussian = numpy.exp(-0.5 * (diff_x**2 + diff_y**2) / self.sigma_g**2)

        return gaussian * spatial[::-1]  # same convention as stimuli

        
### FINDING CONSTANT FOR GABOR FILTERS ###
def find_A(
    k,
    sigma_g,
    edge_deg,
    degree_per_pixel,
    indices,
    phase=0,
    return_all=False,
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
    all_A = []
    all_gabors = []
    all_test_stimuli = []

    for ori in indices:
        # generate Gabor filter and stimuli at orientation
        gabor = GaborFilter(
            theta=ori,
            x_i=0,
            y_i=0,
            edge_deg=edge_deg,
            k=k,
            sigma_g=sigma_g,
            degree_per_pixel=degree_per_pixel,
            phase=phase,
        )
        # create local_stimui_pars to pass it to BW_Gratings
        local_stimuli_pars = StimuliPars()
        local_stimuli_pars.edge_deg = edge_deg
        local_stimuli_pars.k = k
        local_stimuli_pars.outer_radius = edge_deg * 2
        local_stimuli_pars.inner_radius = edge_deg * 2
        local_stimuli_pars.degree_per_pixel = degree_per_pixel
        local_stimuli_pars.grating_contrast = 0.99
        local_stimuli_pars.jitter_val = 0

        test_grating = BW_Grating(
            ori_deg=ori,
            stimuli_pars=local_stimuli_pars,
            phase=phase,
            jitter=local_stimuli_pars.jitter_val,
        )
        
        test_stimuli = test_grating.BW_image()

        mean_removed_filter = gabor.filter - gabor.filter.mean()
        # multiply filter and stimuli
        output_gabor = mean_removed_filter.ravel() @ test_stimuli.ravel()

        all_gabors.append(gabor.filter)
        all_test_stimuli.append(test_stimuli)

        # calculate value of A
        A_value = 100 / (output_gabor)

        # create list of A
        all_A.append(A_value)

    # find average value of A
    all_A = numpy.array(all_A)
    A = all_A.mean()

    all_gabors = numpy.array(all_gabors)
    all_test_stimuli = numpy.array(all_test_stimuli)

    if return_all == True:
        output = A, all_gabors, all_test_stimuli
    else:
        output = A

    return output


def create_gabor_filters_util(
    ori_map,
    phases,
    filter_pars,
    grid_pars,
    gE,
    gI
):
    e_filters = []
    if phases == 4:
        e_filters_pi2 = []

    # Iterate over SSN map
    for i in range(ori_map.shape[0]):
        for j in range(ori_map.shape[1]):
            gabor = GaborFilter(
                x_i=grid_pars.x_map[i, j],
                y_i=grid_pars.y_map[i, j],
                edge_deg=filter_pars.edge_deg,
                k=filter_pars.k,
                sigma_g=filter_pars.sigma_g,
                theta=ori_map[i, j],
                conv_factor=filter_pars.conv_factor,
                degree_per_pixel=filter_pars.degree_per_pixel,
            )

            e_filters.append(gabor.filter.ravel())

            if phases == 4:
                gabor_2 = GaborFilter(
                    x_i=grid_pars.x_map[i, j],
                    y_i=grid_pars.y_map[i, j],
                    edge_deg=filter_pars.edge_deg,
                    k=filter_pars.k,
                    sigma_g=filter_pars.sigma_g,
                    theta=ori_map[i, j],
                    conv_factor=filter_pars.conv_factor,
                    degree_per_pixel=filter_pars.degree_per_pixel,
                    phase=numpy.pi / 2,
                )
                e_filters_pi2.append(gabor_2.filter.ravel())

    e_filters_o = numpy.array(e_filters)
    e_filters = gE * e_filters_o
    i_filters = gI * e_filters_o

    # create filters with phase equal to pi
    e_off_filters = -e_filters
    i_off_filters = -i_filters

    if phases == 4:
        e_filters_o_pi2 = numpy.array(e_filters_pi2)

        e_filters_pi2 = gE * e_filters_o_pi2
        i_filters_pi2 = gI * e_filters_o_pi2

        # create filters with phase equal to -pi/2
        e_off_filters_pi2 = -e_filters_pi2
        i_off_filters_pi2 = -i_filters_pi2
        SSN_filters = numpy.vstack(
            [
                e_filters,
                i_filters,
                e_filters_pi2,
                i_filters_pi2,
                e_off_filters,
                i_off_filters,
                e_off_filters_pi2,
                i_off_filters_pi2,
            ]
        )

    else:
        SSN_filters = numpy.vstack(
            [e_filters, i_filters, e_off_filters, i_off_filters]
        )

    # Normalise Gabor filters
    A = find_A(
                k=filter_pars.k,
                sigma_g=filter_pars.sigma_g,
                edge_deg=filter_pars.edge_deg,
                degree_per_pixel=filter_pars.degree_per_pixel,
                indices=numpy.sort(ori_map.ravel()),
                phase=0,  
                return_all=False,
            )
    SSN_filters = SSN_filters * A

    if phases == 4:
        A2 = find_A(
            k=filter_pars.k,
            sigma_g=filter_pars.sigma_g,
            edge_deg=filter_pars.edge_deg,
            degree_per_pixel=filter_pars.degree_per_pixel,
            indices=numpy.sort(ori_map.ravel()),
            phase=numpy.pi / 2,
            return_all=False,
        )

        SSN_filters = numpy.vstack( #*** Do sg with it, I got out of memory for 500 epochs...
            [
                e_filters * A,
                i_filters * A,
                e_filters_pi2 * A2,
                i_filters_pi2 * A2,
                e_off_filters * A,
                i_off_filters * A,
                e_off_filters_pi2 * A2,
                i_off_filters_pi2 * A2,
            ]
        )

    # remove mean so that input to constant grating is 0
    gabor_filters = SSN_filters - numpy.mean(SSN_filters, axis=1)[:, None]

    return gabor_filters, A, A2
#### CREATE GABOR FILTERS ####
class GaborFilter:
    def __init__(
        self,
        x_i,
        y_i,
        k,
        sigma_g,
        theta,
        edge_deg,
        degree_per_pixel,
        phase=0,
        conv_factor=None,
    ):
        """
        Gabor filter class.
        Called from SSN_mid_local.create_gabor_filters() and SSN_mid.create_gabor_filters() whose outputs are gabor_filters and A (attributes of SSN_mid and SSN_mid_local)
        Inputs:
            x_i, y_i: centre of the filter
            k: preferred spatial frequency in cycles/degrees (radians)
            sigma_g: variance of Gaussian function
            theta: orientation map
            edge_deg: extent of the filter in degrees
            degree_per_pixel: resolution in degrees per pixel
            phase: phase of the Gabor filter (default is 0)
            conv_factor: conversion factor from degrees to mm
        """

        # convert to mm from degrees
        if conv_factor:
            self.conv_factor = conv_factor
            self.x_i = x_i / conv_factor
            self.y_i = y_i / conv_factor
        else:
            self.x_i = x_i
            self.y_i = y_i
        self.k = k
        self.theta = theta * (numpy.pi / 180)
        self.phase = phase
        self.sigma_g = sigma_g
        self.edge_deg = edge_deg
        self.degree_per_pixel = degree_per_pixel
        self.N_pixels = int(edge_deg * 2 / degree_per_pixel) + 1

        # create image axis
        x_axis = numpy.linspace(-edge_deg, edge_deg, self.N_pixels, endpoint=True)
        y_axis = numpy.linspace(-edge_deg, edge_deg, self.N_pixels, endpoint=True)

        # construct filter as an attribute
        self.filter = self.create_filter(x_axis, y_axis)

    def create_filter(self, x_axis, y_axis):
        """
        Create Gabor filters in vectorised form.
        """
        # Reshape the center coordinates into column vectors; repeat and reshape the center coordinates to allow calculating diff_x and diff_y
        x_axis = numpy.reshape(x_axis, (self.N_pixels, 1))
        x_i = numpy.repeat(self.x_i, self.N_pixels)
        x_i = numpy.reshape(x_i, (self.N_pixels, 1))
        diff_x = x_axis.T - x_i

        y_axis = numpy.reshape(y_axis, (self.N_pixels, 1))
        y_i = numpy.repeat(self.y_i, self.N_pixels)
        y_i = numpy.reshape(y_i, (self.N_pixels, 1))
        diff_y = y_axis - y_i.T

        # Calculate the spatial component of the Gabor filter
        spatial = numpy.cos(
            self.k
            * numpy.pi
            * 2
            * (diff_x * numpy.cos(self.theta) + diff_y * numpy.sin(self.theta))
            + self.phase
        )
        # Calculate the Gaussian component of the Gabor filter
        gaussian = numpy.exp(-0.5 * (diff_x**2 + diff_y**2) / self.sigma_g**2)

        return gaussian * spatial[::-1]  # same convention as stimuli

        
### FINDING CONSTANT FOR GABOR FILTERS ###
def find_A(
    k,
    sigma_g,
    edge_deg,
    degree_per_pixel,
    indices,
    phase=0,
    return_all=False,
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
    all_A = []
    all_gabors = []
    all_test_stimuli = []

    for ori in indices:
        # generate Gabor filter and stimuli at orientation
        gabor = GaborFilter(
            theta=ori,
            x_i=0,
            y_i=0,
            edge_deg=edge_deg,
            k=k,
            sigma_g=sigma_g,
            degree_per_pixel=degree_per_pixel,
            phase=phase,
        )
        # create local_stimui_pars to pass it to BW_Gratings
        local_stimuli_pars = StimuliPars()
        local_stimuli_pars.edge_deg = edge_deg
        local_stimuli_pars.k = k
        local_stimuli_pars.outer_radius = edge_deg * 2
        local_stimuli_pars.inner_radius = edge_deg * 2
        local_stimuli_pars.degree_per_pixel = degree_per_pixel
        local_stimuli_pars.grating_contrast = 0.99
        local_stimuli_pars.jitter_val = 0

        test_grating = BW_Grating(
            ori_deg=ori,
            stimuli_pars=local_stimuli_pars,
            phase=phase,
            jitter=local_stimuli_pars.jitter_val,
        )
        
        test_stimuli = test_grating.BW_image()

        mean_removed_filter = gabor.filter - gabor.filter.mean()
        # multiply filter and stimuli
        output_gabor = mean_removed_filter.ravel() @ test_stimuli.ravel()

        all_gabors.append(gabor.filter)
        all_test_stimuli.append(test_stimuli)

        # calculate value of A
        A_value = 100 / (output_gabor)

        # create list of A
        all_A.append(A_value)

    # find average value of A
    all_A = numpy.array(all_A)
    A = all_A.mean()

    all_gabors = numpy.array(all_gabors)
    all_test_stimuli = numpy.array(all_test_stimuli)

    if return_all == True:
        output = A, all_gabors, all_test_stimuli
    else:
        output = A

    return output


def create_gabor_filters_util(
    ori_map,
    phases,
    filter_pars,
    grid_pars,
    gE,
    gI
):
    e_filters = []
    if phases == 4:
        e_filters_pi2 = []

    # Iterate over SSN map
    for i in range(ori_map.shape[0]):
        for j in range(ori_map.shape[1]):
            gabor = GaborFilter(
                x_i=grid_pars.x_map[i, j],
                y_i=grid_pars.y_map[i, j],
                edge_deg=filter_pars.edge_deg,
                k=filter_pars.k,
                sigma_g=filter_pars.sigma_g,
                theta=ori_map[i, j],
                conv_factor=filter_pars.conv_factor,
                degree_per_pixel=filter_pars.degree_per_pixel,
            )

            e_filters.append(gabor.filter.ravel())

            if phases == 4:
                gabor_2 = GaborFilter(
                    x_i=grid_pars.x_map[i, j],
                    y_i=grid_pars.y_map[i, j],
                    edge_deg=filter_pars.edge_deg,
                    k=filter_pars.k,
                    sigma_g=filter_pars.sigma_g,
                    theta=ori_map[i, j],
                    conv_factor=filter_pars.conv_factor,
                    degree_per_pixel=filter_pars.degree_per_pixel,
                    phase=numpy.pi / 2,
                )
                e_filters_pi2.append(gabor_2.filter.ravel())

    e_filters_o = numpy.array(e_filters)
    e_filters = gE * e_filters_o
    i_filters = gI * e_filters_o

    # create filters with phase equal to pi
    e_off_filters = -e_filters
    i_off_filters = -i_filters

    if phases == 4:
        e_filters_o_pi2 = numpy.array(e_filters_pi2)

        e_filters_pi2 = gE * e_filters_o_pi2
        i_filters_pi2 = gI * e_filters_o_pi2

        # create filters with phase equal to -pi/2
        e_off_filters_pi2 = -e_filters_pi2
        i_off_filters_pi2 = -i_filters_pi2
        SSN_filters = numpy.vstack(
            [
                e_filters,
                i_filters,
                e_filters_pi2,
                i_filters_pi2,
                e_off_filters,
                i_off_filters,
                e_off_filters_pi2,
                i_off_filters_pi2,
            ]
        )

    else:
        SSN_filters = numpy.vstack(
            [e_filters, i_filters, e_off_filters, i_off_filters]
        )

    # Normalise Gabor filters
    A = find_A(
                k=filter_pars.k,
                sigma_g=filter_pars.sigma_g,
                edge_deg=filter_pars.edge_deg,
                degree_per_pixel=filter_pars.degree_per_pixel,
                indices=numpy.sort(ori_map.ravel()),
                phase=0,  
                return_all=False,
            )
    SSN_filters = SSN_filters * A

    if phases == 4:
        A2 = find_A(
            k=filter_pars.k,
            sigma_g=filter_pars.sigma_g,
            edge_deg=filter_pars.edge_deg,
            degree_per_pixel=filter_pars.degree_per_pixel,
            indices=numpy.sort(ori_map.ravel()),
            phase=numpy.pi / 2,
            return_all=False,
        )

        SSN_filters = numpy.vstack(
            [
                e_filters * A,
                i_filters * A,
                e_filters_pi2 * A2,
                i_filters_pi2 * A2,
                e_off_filters * A,
                i_off_filters * A,
                e_off_filters_pi2 * A2,
                i_off_filters_pi2 * A2,
            ]
        )

    # remove mean so that input to constant grating is 0
    gabor_filters = SSN_filters - numpy.mean(SSN_filters, axis=1)[:, None]

    return gabor_filters, A, A2
