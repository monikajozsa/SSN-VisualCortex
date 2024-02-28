
import os
import sys
import matplotlib.pyplot as plt
import numpy

# Get the current directory of your_file.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

#numpy.random.seed(0)

from util import create_grating_training, BW_Grating, create_grating_pretraining
from parameters import (
    stimuli_pars,
)


def generate_random_pair(min_value, max_value, min_distance):
    num1 = numpy.random.uniform(min_value, max_value)
    random_distance = numpy.random.uniform(min_distance, max_value - min_value)
    num2 = num1 + random_distance
    if num2 > max_value:
        num2 -= (max_value - min_value)

    return num1, num2

inner_radius_pair = generate_random_pair(1.5, 2.8, 0.2)
k_pair = generate_random_pair(1, 2.5, 0.5)
ori_pair = generate_random_pair(0, 180, 5)

test_stimuli = create_grating_pretraining(stimuli_pars, 5)
'''
print((inner_radius_pair[0]-inner_radius_pair[1])/2)
print((k_pair[0]-k_pair[1])/1.5)
print((ori_pair[0]-ori_pair[1])/180)

# max radius is limited by numpy.divide(self.grating_size), self.pixel_per_degree), which is 3.5 in our case
stimuli_pars.inner_radius=inner_radius_pair[0]
stimuli_pars.outer_radius=inner_radius_pair[0]+0.5
stimuli_pars.k=k_pair[0]
stimuli_pars.ref_ori=ori_pair[0]

test_stimulus1= BW_Grating(ori_deg = stimuli_pars.ref_ori, jitter=0, stimuli_pars = stimuli_pars).BW_image()

stimuli_pars.inner_radius=inner_radius_pair[1]
stimuli_pars.outer_radius=inner_radius_pair[1]+0.5
stimuli_pars.k=k_pair[1]
stimuli_pars.ref_ori=ori_pair[1]
test_stimulus2= BW_Grating(ori_deg = stimuli_pars.ref_ori, jitter=0, stimuli_pars = stimuli_pars).BW_image()
'''

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))

axes[0].imshow(numpy.asarray(test_stimuli['grating1'][0]).reshape(129,129))

axes[1].imshow(numpy.asarray(test_stimuli['grating2'][0]).reshape(129,129))
plt.show()