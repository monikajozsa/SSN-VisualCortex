import numpy
from dataclasses import dataclass

import jax.numpy as np


# Input parameters
@dataclass(unsafe_hash=True)
class GridPars:
    gridsize_Nx: int = 9
    """ size of the grid is gridsize_Nx x gridsize_Nx """
    gridsize_deg: float = 2 * 1.6
    """ edge length in degrees - visual field *** is it in degree? it is multiplied by 2, which is also the magnification factor"""
    magnif_factor: float = 2.0
    """ converts deg to mm (mm/deg) """
    hyper_col: float = 0.4
    """ parameter to generate orientation map """


grid_pars = GridPars()


@dataclass(unsafe_hash=True)
class FilterPars:
    sigma_g = numpy.array(0.27)  # std of the Gaussian of the Gabor filter
    conv_factor = numpy.array(2) # same as magnification factor
    k: float = 1  # scaling parameters for the spacial frequency of the Gabor filter
    edge_deg: float = grid_pars.gridsize_deg
    degree_per_pixel = numpy.array(0.05)
    # convert degree to number of pixels (129 x 129), note that this could be calculated from earlier parameters


filter_pars = FilterPars()


@dataclass
class StimuliPars:  # the attributes are changed within SSN_classes for a local instance
    inner_radius: float = 2.5  # inner radius of the stimulus
    outer_radius: float = 3.0  # outer radius of the stimulus: together with inner_radius, they define how the edge of the stimulus fades away to the gray background
    grating_contrast: float = 0.8  # contrast of darkest and lightest point in the grid - see Ke's Current Biology paper from 2020
    std: float = 0.0  # no noise at the moment but this is a Gaussian white noise added to the stimulus (after the Gabor filter? so to the grating)
    jitter_val: float = 5.0  # jitter is taken from a uniform distribution [-jitter_val, jitter_val]
    k: float = filter_pars.k
    edge_deg: float = filter_pars.edge_deg  # same as for k
    degree_per_pixel = filter_pars.degree_per_pixel  # same as for k
    ref_ori: float = 55.0 # reference orientation of the stimulus in degree
    offset: float = 4.0 # difference between reference and task orientation in degree (task ori is eith ref_ori + offset or ref_or - offset)


stimuli_pars = StimuliPars()


# Sigmoid parameters
@dataclass(unsafe_hash=True)
class SigPars:
    N_neurons: int = 25  # 
    w_sig = numpy.random.normal(scale = 0.25, size=(N_neurons,)) / np.sqrt(N_neurons) # weights between the superficial and the sigmoid layer
    b_sig: float = 0.0 # bias added to the sigmoid layer

sig_pars = SigPars()


@dataclass(unsafe_hash=True)
class SSNPars:
    n = 2.0  # power law parameter
    k = 0.04  # power law parameter
    tauE = 20.0  # time constant for excitatory neurons in ms
    tauI = 10.0  # time constant for inhibitory neurons in ms~
    A = None  # multiply Gabor filters by this before the training normalization param for Gabors to get 100% contrast, see find_A
    A2 = None  # different normalization for the accross phases normalization param for Gabors to get 100% contrast, see find_A
    phases = 4  # number of inh. and exc. neurons (with different Gabor filt.) per grid point in middle layer (2 or 4)


ssn_pars = SSNPars()
    

@dataclass(unsafe_hash=True)
class ConnParsM:
    PERIODIC: bool = False
    p_local = [1.0, 1.0]


conn_pars_m = ConnParsM()


@dataclass(unsafe_hash=True)
class ConnParsS:
    PERIODIC: bool = False
    p_local = [0.4, 0.7]


conn_pars_s = ConnParsS()


@dataclass(unsafe_hash=True)
class SsnLayerPars:
    sigma_oris = np.asarray([90.0, 90.0]) # degree
    kappa_pre = np.asarray([0.0, 0.0])
    kappa_post = np.asarray([0.0, 0.0])
    f_E: float = np.log(1.11)  # Feedforwards connections
    f_I: float = np.log(0.7)
    c_E: float = 5.0  # Excitatory constant for extra synaptic GABA
    c_I: float = 5.0  # Inhibitory constant for extra synaptic GABA
    psi: float = 0.774
    J_2x2_s = (
        np.array([[1.82650658, -0.68194475], [2.06815311, -0.5106321]]) * np.pi * psi
    )
    s_2x2_s = np.array([[0.2, 0.09], [0.4, 0.09]])
    gE_s = 0.37328625 * 1.5 # multiplied the Gabor filter - not used for superficial
    gI_s = 0.26144141 * 1.5 # 
    J_2x2_m = np.array([[2.5, -1.3], [4.7, -2.2]]) * psi
    gE_m = 0.3 #
    gI_m = 0.25 #
    gE = [gE_m, gE_s]
    gI = [gI_m, gI_s]


ssn_layer_pars = SsnLayerPars()


# Training parameters
@dataclass(unsafe_hash=True)
class ConvPars:
    dt: float = 1.0
    '''Step size during convergence '''
    xtol: float = 1e-04
    '''Convergence tolerance  '''
    Tmax: float = 250.0
    '''Maximum number of steps to be taken during convergence'''
    Rmax_E = None
    '''Maximum firing rate for E neurons - rates above this are penalised'''
    Rmax_I = None
    '''Maximum firing rate for I neurons - rates above this are penalised '''


conv_pars = ConvPars()


@dataclass
class TrainingPars:
    eta = 10e-4  #
    batch_size = 50  # was 50
    noise_type = "poisson"
    sig_noise = 2.0 if noise_type != "no_noise" else 0.0
    epochs = 5  # was 5
    num_epochs_to_save = 3
    first_stage_acc = 0.7 #not used yet but will be as I merge to Clara's current code


training_pars = TrainingPars()


@dataclass(unsafe_hash=True)
class LossPars:
    lambda_dx = 1
    lambda_r_max = 1
    lambda_w = 1
    lambda_b = 1


loss_pars = LossPars()
