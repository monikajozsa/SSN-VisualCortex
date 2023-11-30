import os
import matplotlib.pyplot as plt
import time, os, json
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import pdb
from functools import partial
import math
import csv
import time
import numpy
from pdb import set_trace
from dataclasses import dataclass

import jax
import jax.numpy as np


def init_set_func(init_set, conn_pars, ssn_pars, middle=False):
    
    
    #ORIGINAL TRAINING!!
    if init_set ==0:
        Js0 = [1.82650658, 0.68194475, 2.06815311, 0.5106321]
        gE, gI = 0.57328625, 0.26144141
        sigEE, sigIE = 0.2, 0.40
        sigEI, sigII = .09, .09
        conn_pars.p_local = [0.4, 0.7]

    if init_set ==1:
        Js0 = [1.82650658, 0.68194475, 2.06815311, 0.5106321]
        gE, gI = 0.37328625*1.5, 0.26144141*1.5
        sigEE, sigIE = 0.2, 0.40
        sigEI, sigII = .09, .09
        conn_pars.p_local = [0.4, 0.7]

    if init_set==2:
        Js0 = [1.72881688, 1.29887564, 1.48514091, 0.76417991]
        gE, gI = 0.5821754, 0.22660373
        sigEE, sigIE = 0.225, 0.242
        sigEI, sigII = .09, .09
        conn_pars.p_local = [0.0, 0.0]
    
    if init_set ==3:
        Js0 = [1.82650658, 0.68194475, 2.06815311, 0.5106321]
        gE, gI = 1,1
        sigEE, sigIE = 0.2, 0.40
        sigEI, sigII = .09, .09
        conn_pars.p_local = [0.4, 0.7]
        
    if init_set=='A':
        Js0 = [2.5, 1.3, 2.4, 1.0]
        gE, gI =  0.4, 0.4
        print(gE, gI)
        sigEE, sigIE = 0.2, 0.40
        sigEI, sigII = .09, .09
        conn_pars.p_local = [0.4, 0.7]
        
    if init_set=='C':
        Js0 = [2.5, 1.3, 4.7, 2.2]
        gE, gI =0.3, 0.25
        sigEE, sigIE = 0.2, 0.40
        sigEI, sigII = .09, .09
        conn_pars.p_local = [0.4, 0.7]
        
    if middle:
        conn_pars.p_local = [1, 1]
        
    if init_set =='C':
        make_J2x2 = lambda Jee, Jei, Jie, Jii: np.array([[Jee, -Jei], [Jie,  -Jii]])  * ssn_pars.psi
    else:
        make_J2x2 = lambda Jee, Jei, Jie, Jii: np.array([[Jee, -Jei], [Jie,  -Jii]]) * np.pi * ssn_pars.psi
        
    J_2x2 = make_J2x2(*Js0)
    s_2x2 = np.array([[sigEE, sigEI],[sigIE, sigII]])
    
    return J_2x2, s_2x2, gE, gI, conn_pars
'''
# Input parameters
@dataclass(frozen=True)
class GridPars():
    gridsize_Nx: int = 9 
    # grid-points across each edge - gives rise to dx = 0.8 mm 
    gridsize_deg: float = 2 * 1.6  
    # edge length in degrees - visual field 
    magnif_factor: float = 2.0
    # converts deg to mm (mm/deg) 
    hyper_col: float = 0.4  
    # ? are our grid points represent columns? (mm) 
    sigma_RF: float = 0.4  
    # deg (visual angle), comes in make_grating_input, which Clara does not use as she uses Gabor filters 

grid_pars = GridPars()


@dataclass
class FilterPars():
    sigma_g = numpy.array(0.39 * 0.5 / 1.04) #
    conv_factor = numpy.array(2)
    k: float = np.pi/(6 * 0.5) # Clara approximated it by 1; Ke used 1 too
    edge_deg: float = grid_pars.gridsize_deg
    degree_per_pixel = numpy.array(0.05) 
    # convert degree to number of pixels (129 x 129), this could be calculated from earlier params 
filter_pars = FilterPars()
   

@dataclass
class StimuliPars():
    inner_radius: float = 2.5 # inner radius of the stimulus
    outer_radius: float = 3.0 # outer radius of the stimulus: together with inner_radius, they define how the edge of the stimulus fades away to the gray background
    grating_contrast: float = 0.8 # from Current Biology 2020 Ke's paper
    std: float = 0.0 # no noise at the moment but this is a Gaussian white noise added to the stimulus
    jitter_val: float = 5.0 # uniform jitter between [-5, 5] to make the training stimulus vary
    k: float = filter_pars.k # It would be great to get rid of this because FILTER_PARS HAS IT but then it is used when it is passed to new_two_stage_training at BW_Grating
    edge_deg: float = filter_pars.edge_deg # same as for k
    degree_per_pixel = filter_pars.degree_per_pixel # same as for k
    ref_ori: float = 55.0
    offset: float = 4.0
stimuli_pars = StimuliPars()



@dataclass(frozen=True)
class SsnPars():
    n: float = 2.0 # power law parameter
    k: float = 0.04 # power law parameter
    tauE: float = 20.0  # time constant for excitatory neurons in ms
    tauI: float = 10.0  # time constant for inhibitory neurons in ms~
    psi: float = 0.774 # when we make J2x2 to normalize
    A = None # normalization param for Gabors to get 100% contrast, see find_A
    A2 = None # normalization param for Gabors to get 100% contrast, see find_A
    phases: int = 2 # or 4
ssn_pars = SsnPars()
'''



# Input parameters
class grid_pars():
    gridsize_Nx: int = 9 
    ''' grid-points across each edge - gives rise to dx = 0.8 mm '''
    gridsize_deg: float = 2 * 1.6  
    ''' edge length in degrees - visual field '''
    magnif_factor: float = 2.0
    ''' converts deg to mm (mm/deg) '''
    hyper_col: float = 0.4  
    ''' ? are our grid points represent columns? (mm) '''

    
    
    

class filter_pars():
    sigma_g = numpy.array(0.27)
    '''Standard deviation of Gaussian in Gabor '''
    conv_factor = numpy.array(2)
    ''' Convert from degrees to mm'''
    k: float = 1 # Clara approximated it by 1; Ke used 1 too
    '''Spatial frequency of Gabor filter'''
    edge_deg: float = grid_pars.gridsize_deg
    '''Axis of Gabor filter goes from -edge_deg, to +edge_deg'''
    degree_per_pixel = numpy.array(0.05) 
    ''' Converts from degrees to number of pixels ''' # convert degree to number of pixels (129 x 129), this could be calculated from earlier params '''

    
    
    
    

@dataclass
class StimuliPars(): #the attributes are changed within SSN_classes for a local instance
    inner_radius: float = 2.5 # inner radius of the stimulus
    outer_radius: float = 3.0 # outer radius of the stimulus: together with inner_radius, they define how the edge of the stimulus fades away to the gray background
    grating_contrast: float = 0.8 # from Current Biology 2020 Ke's paper
    std: float = 0.0 # no noise at the moment but this is a Gaussian white noise added to the stimulus
    jitter_val: float = 5.0 # uniform jitter between [-5, 5] to make the training stimulus vary
    k: float = filter_pars.k # It would be great to get rid of this because FILTER_PARS HAS IT but then it is used when it is passed to new_two_stage_training at BW_Grating
    edge_deg: float = filter_pars.edge_deg # same as for k
    degree_per_pixel = filter_pars.degree_per_pixel # same as for k
    ref_ori: float = 55.0
    offset: float = 4.0
stimuli_pars = StimuliPars()


class ssn_pars():
    n = 2.0 # power law parameter
    k = 0.04 # power law parameter
    tauE = 20.0  # time constant for excitatory neurons in ms
    tauI = 10.0  # time constant for inhibitory neurons in ms~
    psi = 0.774 # when we make J2x2 to normalize
    A = None # normalization param for Gabors to get 100% contrast, see find_A
    A2 = None # normalization param for Gabors to get 100% contrast, see find_A
    phases = 4# or 4
    


class conn_pars_m():
    PERIODIC: bool = False
    p_local = None


class conn_pars_s():
    PERIODIC: bool = False
    p_local = None

    
# Training parameters
class conv_pars():
    dt: float = 1.0
    '''Step size during convergence '''
    xtol: float = 1e-03
    '''Convergence tolerance  '''
    Tmax: float = 250.0
    '''Maximum number of steps to be taken during convergence'''
    Rmax_E = None
    '''Maximum firing rate for E neurons - rates above this are penalised'''
    Rmax_I = None
    '''Maximum firing rate for I neurons - rates above this are penalised '''


@dataclass
class TrainingPars():
    eta = 10e-4
    batch_size = 50
    noise_type = "poisson"
    sig_noise = 2.0 if noise_type != "no_noise" else 0.0
    epochs = 5  # was 5
    num_epochs_to_save = 3
    first_stage_acc = 0.7
    '''Paremeters of sigmoid layer are trained in the first stage until this accuracy is reached '''
training_pars = TrainingPars()


class loss_pars():
    lambda_dx = 1
    ''' Constant for loss with respect to convergence of Euler function'''
    lambda_r_max = 1
    ''' Constant for loss with respect to maximum rates in the network'''
    lambda_w = 1
    ''' Constant for L2 regularizer of sigmoid layer weights'''
    lambda_b = 1
    ''' Constant for L2 regulazier of sigmoid layer bias '''



