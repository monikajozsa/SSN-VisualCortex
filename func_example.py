import numpy
import math
from PIL import Image
from dataclasses import dataclass

@profile
def BW_image(grating_size, pixel_per_degree, inner_radius, smooth_sd, spatial_freq, angle, grating_contrast, phase, std, size):
    _BLACK = 0
    _WHITE = 255
    _GRAY = round((_WHITE + _BLACK) / 2)
    
    # Generate a 2D grid of coordinates
    x, y = numpy.mgrid[-grating_size:grating_size + 1.0, -grating_size:grating_size + 1.0]

    # Calculate the distance from the center for each pixel
    edge_control_dist = numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2))
    edge_control = numpy.divide(edge_control_dist, pixel_per_degree)
    
    # Create a matrix (alpha_channel) that is 255 (white) within the inner_radius and exponentially fades to 0 as the radius increases
    overrado = numpy.nonzero(edge_control > inner_radius)
    d = grating_size * 2 + 1
    annulus = numpy.ones((d, d))

    annulus[overrado] *= numpy.exp(-1 * ((edge_control[overrado] - inner_radius) * pixel_per_degree)**2 / (2 * (smooth_sd**2)))
    alpha_channel = annulus * _WHITE

    # Generate the grating pattern, which is a centered and tilted sinusoidal matrix 
    spatial_component = 2 * math.pi * spatial_freq * (y * numpy.sin(angle) + x * numpy.cos(angle))
    gabor_sti = _GRAY * (1 + grating_contrast * numpy.cos(spatial_component + phase))

    # Set pixels outside the grating size to gray
    gabor_sti[edge_control_dist > grating_size] = _GRAY

    # Add Gaussian white noise to the grating
    noise = numpy.random.normal(loc=0, scale=std, size=gabor_sti.shape)
    noisy_gabor_sti = gabor_sti + noise

    # Expand the grating to have three colors andconcatenate it with alpha_channel
    gabor_sti_final = numpy.repeat(noisy_gabor_sti[:, :, numpy.newaxis], 3, axis=-1)        
    gabor_sti_final_with_alpha = numpy.concatenate((gabor_sti_final, alpha_channel[:, :, numpy.newaxis]), axis=-1)
    gabor_sti_final_with_alpha_image = Image.fromarray(gabor_sti_final_with_alpha.astype(numpy.uint8))

    # Create a background image filled with gray
    background = numpy.full((size, size, 3), _GRAY, dtype=numpy.uint8)
    final_image = Image.fromarray(background)

    # Paste the grating into the final image: paste the grating into a bounding box and apply the alpha channel as a mask
    center_x, center_y = size // 2, size // 2
    bounding_box = (center_x - grating_size, center_y - grating_size)
    final_image.paste(gabor_sti_final_with_alpha_image, box=bounding_box, mask=gabor_sti_final_with_alpha_image)

    # Sum the image over color channels
    final_image_np = numpy.array(final_image, dtype=numpy.float16)
    image = numpy.sum(final_image_np, axis=2)

    return image

@dataclass
class StimuliPars: #the attributes are changed within SSN_classes for a local instance
    inner_radius: float = 2.5 # inner radius of the stimulus
    outer_radius: float = 3.0 # outer radius of the stimulus: together with inner_radius, they define how the edge of the stimulus fades away to the gray background
    grating_contrast: float = 0.8 # from Current Biology 2020 Ke's paper
    std: float = 0.0 # no noise at the moment but this is a Gaussian white noise added to the stimulus
    jitter_val: float = 5.0 # uniform jitter between [-5, 5] to make the training stimulus vary
    k: float = 2 * 1.6   # It would be great to get rid of this because FILTER_PARS HAS IT but then it is used when it is passed to new_two_stage_training at BW_Grating
    edge_deg: float = 2 * 1.6   # same as for k
    degree_per_pixel = numpy.array(0.05)  # same as for k
    ref_ori: float = 55.0
    offset: float = 4.0
stimuli_pars = StimuliPars()
ori_deg = stimuli_pars.ref_ori
jitter = stimuli_pars.jitter_val
outer_radius = stimuli_pars.outer_radius  # in degrees
inner_radius = stimuli_pars.inner_radius  # in degrees
grating_contrast = stimuli_pars.grating_contrast
std = stimuli_pars.std
degree_per_pixel = stimuli_pars.degree_per_pixel
pixel_per_degree = 1 / degree_per_pixel
edge_deg = stimuli_pars.edge_deg
size = int(edge_deg * 2 * pixel_per_degree) + 1
k = stimuli_pars.k
spatial_freq = k * degree_per_pixel # 0.05235987755982988        
phase = 0
smooth_sd = pixel_per_degree / 6
grating_size = round(stimuli_pars.outer_radius * pixel_per_degree)
angle = ((ori_deg + jitter) - 90) / 180 * numpy.pi

BW_image(grating_size, pixel_per_degree, inner_radius, smooth_sd, spatial_freq, angle, grating_contrast, phase, std, size)