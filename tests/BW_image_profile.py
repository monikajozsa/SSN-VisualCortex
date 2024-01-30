import jax.numpy as np
from jax import random

from parameters import stimuli_pars, ssn_pars

'''
@profile
def BW_image_jax(stimuli_pars,ssn_pars,seed):
    _BLACK = 0
    _WHITE = 255
    _GRAY = round((_WHITE + _BLACK) / 2)

    degree_per_pixel = stimuli_pars.degree_per_pixel
    pixel_per_degree = 1 / degree_per_pixel
    smooth_sd = pixel_per_degree / 6
    grating_size = round(stimuli_pars.outer_radius * pixel_per_degree)
    size = int(stimuli_pars.edge_deg * 2 * pixel_per_degree) + 1
    spatial_freq = stimuli_pars.k * degree_per_pixel  
    # Generate a 2D grid of coordinates
    x, y = np.mgrid[
        -grating_size : grating_size + 1.0,
        -grating_size : grating_size + 1.0,
    ]

    # Calculate the distance from the center for each pixel
    edge_control_dist = np.sqrt(np.power(x, 2) + np.power(y, 2))
    edge_control = edge_control_dist / pixel_per_degree

    # Create a matrix that fades to 0 as the radius increases
    overrado = np.nonzero(edge_control > stimuli_pars.inner_radius)
    d = grating_size * 2 + 1
    annulus = np.ones((d, d))

    annulus = annulus.at[overrado].multiply(
        np.exp(-1 * ((edge_control[overrado] - stimuli_pars.inner_radius) * pixel_per_degree) ** 2 / (2 * (smooth_sd**2)))
    )
    alpha_channel = annulus * _WHITE

    # Generate the grating pattern
    angle = ((stimuli_pars.ref_ori + 0) - 90) / 180 * np.pi
    spatial_component = 2 * np.pi * spatial_freq * (y * np.sin(angle) + x * np.cos(angle))
    gabor_sti = _GRAY * (1 + stimuli_pars.grating_contrast * np.cos(spatial_component + ssn_pars.phases))

    # Set pixels outside the grating size to gray
    gabor_sti = gabor_sti.at[edge_control_dist > grating_size].set(_GRAY)

    # Add Gaussian white noise to the grating
    rng_key=random.PRNGKey(seed)
    noisy_gabor_sti = gabor_sti + random.normal(rng_key, gabor_sti.shape) * stimuli_pars.std
    
    # Create a background image filled with gray
    background = np.full((size, size), _GRAY, dtype=np.float32)

    # Paste the grating into the final image: paste the grating into a bounding box and apply the alpha channel as a mask
    center_x, center_y = size // 2, size // 2
    bbox_height=center_x - grating_size
    bbox_width = center_y - grating_size        
    alpha_height, alpha_width = noisy_gabor_sti.shape

    # Calculate the region of interest (ROI) in the background
    roi = background[bbox_height:bbox_height+alpha_height, bbox_width:bbox_width+alpha_width]

    # Apply the alpha image as a mask to the ROI
    result_roi = alpha_channel[:, :]/255 * noisy_gabor_sti[:, :] + (1.0 - alpha_channel[:, :]/255) * roi[:,:]

    # Place the modified ROI back into the background
    combined_image = background.at[bbox_height:bbox_height+alpha_height, bbox_width:bbox_width+alpha_width].set(result_roi)
    
    return 3*combined_image

'''
import numpy
from PIL import Image

@profile
def BW_image(stimuli_pars, ssn_pars):
    _BLACK = 0
    _WHITE = 255
    _GRAY = round((_WHITE + _BLACK) / 2)
    degree_per_pixel = stimuli_pars.degree_per_pixel
    pixel_per_degree = 1 / degree_per_pixel
    smooth_sd = pixel_per_degree / 6
    grating_size = round(stimuli_pars.outer_radius * pixel_per_degree)
    size = int(stimuli_pars.edge_deg * 2 * pixel_per_degree) + 1
    spatial_freq = stimuli_pars.k * degree_per_pixel  
    # Generate a 2D grid of coordinates
    x, y = numpy.mgrid[
        -grating_size : grating_size + 1.0,
        -grating_size : grating_size + 1.0,
    ]

    # Calculate the distance from the center for each pixel
    edge_control_dist = numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2))
    edge_control = numpy.divide(edge_control_dist, pixel_per_degree)

    # Create a matrix (alpha_channel) that is 255 (white) within the inner_radius and exponentially fades to 0 as the radius increases
    overrado = numpy.nonzero(edge_control > stimuli_pars.inner_radius)
    d = grating_size * 2 + 1
    annulus = numpy.ones((d, d))

    annulus[overrado] *= numpy.exp(
        -1
        * ((edge_control[overrado] - stimuli_pars.inner_radius) * pixel_per_degree)
        ** 2
        / (2 * (smooth_sd**2))
    )
    alpha_channel = annulus * _WHITE

    # Generate the grating pattern, which is a centered and tilted sinusoidal matrix
    #initialize output
    angle = ((stimuli_pars.ref_ori + 0) - 90) / 180 * numpy.pi

    spatial_component = (
        2
        * numpy.pi
        * spatial_freq
        * (y * numpy.sin(angle) + x * numpy.cos(angle))
    )
    gabor_sti = _GRAY * (
        1 + stimuli_pars.grating_contrast * numpy.cos(spatial_component + ssn_pars.phases)
    )

    # Set pixels outside the grating size to gray
    gabor_sti[edge_control_dist > grating_size] = _GRAY

    # Add Gaussian white noise to the grating
    if stimuli_pars.std>0:
        noisy_gabor_sti = gabor_sti + numpy.random.normal(loc=0, scale=stimuli_pars.std, size=gabor_sti.shape)
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
    background = numpy.full((size, size, 3), _GRAY, dtype=numpy.uint8)
    final_image = Image.fromarray(background)

    # Paste the grating into the final image: paste the grating into a bounding box and apply the alpha channel as a mask
    center_x, center_y = size // 2, size // 2
    bounding_box = (center_x - grating_size, center_y - grating_size)
    final_image.paste(
        gabor_sti_final_with_alpha_image,
        box=bounding_box,
        mask=gabor_sti_final_with_alpha_image,
    )

    # Sum the image over color channels
    final_image_np = numpy.array(final_image, dtype=numpy.float16)
    image=numpy.sum(final_image_np, axis=2)
    
    return image
BW_image(stimuli_pars,ssn_pars)

#BW_image_jax(stimuli_pars,ssn_pars,1)




def BW_image_jax(grating_size, pixel_per_degree, inner_radius,smooth_sd, ori_deg, jitter, spatial_freq, phase, grating_contrast,std, size, seed):
    _BLACK = 0
    _WHITE = 255
    _GRAY = round((_WHITE + _BLACK) / 2)

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
    mask = edge_control > inner_radius
    alpha_channel = np.where(mask, exp_part, 1) * _WHITE

    # Generate the grating pattern
    angle = ((ori_deg + jitter) - 90) / 180 * np.pi
    spatial_component = 2 * np.pi * spatial_freq * (y * np.sin(angle) + x * np.cos(angle))
    gabor_sti = _GRAY * (1 + grating_contrast * np.cos(spatial_component + phase))

    # Set pixels outside the grating size to gray
    gabor_sti = gabor_sti.at[edge_control_dist > grating_size].set(_GRAY)

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
    roi = background[bbox_height:bbox_height+alpha_height, bbox_width:bbox_width+alpha_width]

    # Apply the alpha image as a mask to the ROI
    result_roi = np.floor(alpha_channel[:, :]/255 * noisy_gabor_sti[:, :] + (1.0 - alpha_channel[:, :]/255) * roi[:,:])

    # Place the modified ROI back into the background
    combined_image = background.at[bbox_height:bbox_height+alpha_height, bbox_width:bbox_width+alpha_width].set(result_roi)
    
    return 3*combined_image