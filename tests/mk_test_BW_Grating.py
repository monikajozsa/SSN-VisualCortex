import numpy
from util import BW_Grating_Clara, BW_Grating

inner_radius = 2.5 # inner radius of the stimulus
outer_radius = 3.0 # outer radius of the stimulus: together with inner_radius, they define how the edge of the stimulus fades away to the gray background
grating_contrast = 0.8 # from Current Biology 2020 Ke's paper
std = 0.0 # no noise at the moment but this is a Gaussian white noise added to the stimulus
jitter = 5.0 # uniform jitter between [-5, 5] to make the training stimulus vary
k = numpy.pi/(6 * 0.5) # It would be great to get rid of this because FILTER_PARS HAS IT but then it is used when it is passed to new_two_stage_training at BW_Grating
edge_deg = 2 * 1.6   # same as for k
degree_per_pixel = numpy.array(0.05)
ref_ori = 55.0
offset: float = 4.0
testgrating = BW_Grating_Clara(ref_ori, outer_radius, inner_radius, degree_per_pixel, grating_contrast, edge_deg, phase=0, jitter=jitter, std=std, k=k).BW_image()
numpy.savetxt('test_BW_Grating.txt', testgrating, fmt='%d')
testgrating_loaded = numpy.loadtxt('test_BW_Grating.txt', dtype=int)

from parameters import stimuli_pars
grating_totest = BW_Grating(ori_deg=stimuli_pars.ref_ori,jitter=stimuli_pars.jitter_val,stimuli_pars=stimuli_pars).BW_image()
print((testgrating_loaded == grating_totest).all())