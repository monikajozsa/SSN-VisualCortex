import numpy
from dataclasses import dataclass
import jax.numpy as np


# Pretraining parameters
@dataclass
class PreTrainPars:
    is_on = True
    ''' flag for turning pretraining on or off '''
    ref_ori_int = [15, 165]
    ''' interval where the reference orientation is randomly chosen from '''
    ori_dist_int = [10, 20]
    ''' interval where the absolute orientation difference between reference and target is randomly chosen from '''
    acc_th = 0.749
    ''' accuracy threshold to calculate corresponding offset (training task) - used for early stopping of pretraining '''
    acc_check_freq = 3
    ''' frequency (in SGD step) of accuracy check for the training task - used for early stopping of pretraining '''
    min_acc_check_ind = 10
    ''' minimum SGD step where accuracy check happens for the training task '''
    offset_threshold = 6
    ''' threshold for offset where training task achieves accuracy threshold (acc_th)  - used for early stopping of pretraining '''
    batch_size = 50
    ''' number of trials per SGD step during pretraining '''
    SGD_steps = 50
    ''' maximum number of SGD steps during pretraining '''

pretrain_pars = PreTrainPars()


# Training parameters
@dataclass
class TrainingPars:
    eta = 2*10e-4 
    '''learning rate - the maximum rate of parameter change in one SGD step'''
    batch_size = 50
    '''number of trials per SGD step'''
    noise_type = "poisson"
    '''there is an additive Gaussian noise to the model output (rates) that is related to parameters num_readout_noise and dt'''
    SGD_steps = 100
    '''number of SGD step'''
    validation_freq = 50  
    '''frequency of validation loss and accuracy calculation'''
    first_stage_acc_th = 0.55
    '''accuracy threshold for early stopping criterium for the first stage of training'''

training_pars = TrainingPars()


# Convergence parameters
@dataclass
class ConvPars:
    dt: float = 1.0
    '''Step size during convergence of SSN'''
    xtol: float = 1e-04
    '''Convergence tolerance of SSN'''
    Tmax: float = 250.0
    '''Maximum number of steps to be taken during convergence of SSN'''
    Rmax_E = 40
    '''Maximum firing rate for E neurons - rates above this are penalised'''
    Rmax_I = 80
    '''Maximum firing rate for I neurons - rates above this are penalised '''

conv_pars = ConvPars()


# Loss parameters
@dataclass
class LossPars:
    lambda_dx = 1
    ''' Constant for loss with respect to convergence of Euler function'''
    lambda_r_max = 1
    ''' Constant for loss with respect to maximum rates in the network'''
    lambda_w = 1
    ''' Constant for L2 regularizer of sigmoid layer weights'''
    lambda_b = 1
    ''' Constant for L2 regulazier of sigmoid layer bias '''

loss_pars = LossPars()


def xy_distance(gridsize_Nx,gridsize_deg):
    ''' This function calculates distances between grid points of a grid with given sizes. It is used in GridPars class.'''
    Nn = gridsize_Nx**2
    gridsize_mm = gridsize_deg * 2
    Lx = Ly = gridsize_mm
    Nx = Ny = gridsize_Nx

    # Simplified meshgrid creation
    xs = numpy.linspace(0, Lx, Nx)
    ys = numpy.linspace(0, Ly, Ny)
    [x_map, y_map] = numpy.meshgrid(xs - xs[len(xs) // 2], ys - ys[len(ys) // 2])
    y_map = -y_map # without this y_map decreases going upwards

    # Flatten and tile in one step
    x_vec = numpy.tile(x_map.ravel(), 2)
    y_vec = numpy.tile(y_map.ravel(), 2)

    # Reshape x_vec and y_vec
    xs = x_vec.reshape(2, Nn, 1)
    ys = y_vec.reshape(2, Nn, 1)

    # Calculate distance using broadcasting
    xy_dist = numpy.sqrt(numpy.square(xs[0] - xs[0].T) + numpy.square(ys[0] - ys[0].T))

    return np.array(xy_dist), np.array(x_map), np.array(y_map)


# Input parameters
@dataclass
class GridPars:
    gridsize_Nx: int = 9
    ''' size of the grid is gridsize_Nx x gridsize_Nx '''
    gridsize_deg: float = 2 * 1.6
    ''' edge length in degrees - visual field'''
    magnif_factor: float = 2.0
    ''' converts deg to mm (mm/deg) '''
    gridsize_mm = gridsize_deg * magnif_factor
    hyper_col: float = 0.4
    ''' parameter to generate orientation map '''
    xy_dist, x_map, y_map = xy_distance(gridsize_Nx,gridsize_deg)
    ''' distances between grid points '''

grid_pars = GridPars()


# Gabor filter parameters
@dataclass
class FilterPars:
    sigma_g: float = 0.27
    ''' std of the Gaussian of the Gabor filters '''
    conv_factor: float = 2.0
    ''' converts deg to mm (mm/deg), same as magnification factor '''
    k: float = 1.0
    ''' scaling parameter for the spacial frequency of the Gabor filter '''
    edge_deg: float = grid_pars.gridsize_deg
    ''' edge length in degrees - visual field, same as grid_pars.gridsize_deg '''
    degree_per_pixel: float = 0.05
    ''' convert degree to number of pixels (129 x 129), note that this is not an independent parameter and could be calculated from other parameters '''
    gE_m: float = 0.3
    ''' scaling parameter between stimulus and excitatory units in middle layer '''
    gI_m: float = 0.25 
    ''' scaling parameter between stimulus and inhibitory units in middle layer '''
filter_pars = FilterPars()


# Stimulus parameters
@dataclass
class StimuliPars:
    inner_radius: float = 2.5
    ''' inner radius of the stimulus '''
    outer_radius: float = 3.0 
    ''' outer radius of the stimulus, inner_radius and outer_radius define how the edge of the stimulus fades away to the gray background '''
    grating_contrast: float = 0.8 
    ''' contrast of darkest and lightest point in the grid - see Current Biology paper from 2020 '''
    std: float = 200.0
    ''' Gaussian white noise added to the stimulus '''
    jitter_val: float = 5.0
    ''' constant that defines and interval [-jitter_val, jitter_val] from where jitter (same applied for reference an target stimuli orientation) is randomly taken '''
    k: float = filter_pars.k
    ''' scaling parameter for the spacial frequency of the Gabor filter '''
    edge_deg: float = filter_pars.edge_deg  
    ''' edge length in degrees - visual field, same as grid_pars.gridsize_deg '''
    degree_per_pixel = filter_pars.degree_per_pixel  
    ''' convert degree to number of pixels (129 x 129), note that this is not an independent parameter and could be calculated from other parameters '''
    ref_ori: float = 55.0 
    ''' reference orientation of the stimulus in degree '''
    offset: float = 4.0 
    ''' difference between reference and task orientation in degree (task ori is either ref_ori + offset or ref_or - offset) '''

stimuli_pars = StimuliPars()


# Sigmoid parameters
@dataclass
class ReadoutPars:
    readout_grid_size = np.array([grid_pars.gridsize_Nx, 5])
    ''' size of the center grid where readout units come from (first number is for the pretraining, second is for the training) '''
    # Define middle grid indices
    middle_grid_ind = []
    mid_grid_ind0 = int((readout_grid_size[0]-readout_grid_size[1])/2)
    mid_grid_ind1 = int(readout_grid_size[0]) - mid_grid_ind0
    for i in range(mid_grid_ind0,mid_grid_ind1):  
        row_start = i * readout_grid_size[0] 
        middle_grid_ind.extend(range(row_start + mid_grid_ind0, row_start + mid_grid_ind1))
    middle_grid_ind = np.array(middle_grid_ind)
    ''' indices of the middle grid when grid is flattened '''
    # Define w_sig - its size depends on whether pretraining is on
    if pretrain_pars.is_on:
        w_sig = np.zeros(readout_grid_size[0]**2)
        ''' readout weights (between the superficial and the sigmoid layer) - initialized with logistic regression'''
    else:
        w_sig = np.array(numpy.random.normal(scale = 0.25, size=(readout_grid_size[1]**2,)) / readout_grid_size[1])
        ''' readout weights (between the superficial and the sigmoid layer) '''
    b_sig: float = 0.0 
    ''' bias added to the sigmoid layer '''
    num_readout_noise = 125
    ''' defines readout noise level, see generate_noise function for its effect '''

readout_pars = ReadoutPars()


# general SSN parameters
@dataclass
class SSNPars:
    n = 2.0  
    ''' power law parameter '''
    k = 0.04  
    ''' power law parameter '''
    tauE = 20.0 
    '''  time constant for excitatory neurons in ms '''
    tauI = 10.0
    ''' time constant for inhibitory neurons in ms '''
    phases = 4 
    ''' number of inh. and exc. neurons (with different Gabor filt.) per grid point in middle layer (has to be an even integer) '''

ssn_pars = SSNPars()


# SSN layer parameters
@dataclass
class SsnLayerPars:
    sigma_oris = np.asarray([90.0, 90.0])
    ''' range of weights in terms of preferred orientation difference (in degree) '''
    kappa_pre = np.asarray([0.0, 0.0])
    ''' shaping parameter for superficial layer connections - out of use when set to 0 '''
    kappa_post = np.asarray([0.0, 0.0])
    ''' shaping parameter for superficial layer connections - out of use when set to 0 '''
    f_E = 1.11 
    ''' Scaling constant for feedforwards connections to excitatory units in sup layer '''
    f_I = 0.7
    ''' Scaling constant for feedforwards connections to inhibitory units in sup layer '''
    c_E = 5.0 
    ''' baseline excitatory input (constant added to the output of excitatory neurons at both middle and superficial layers) '''
    c_I = 5.0 
    ''' baseline inhibitory input (constant added to the output of inhibitory neurons at both middle and superficial layers) '''
    J_2x2_s = (np.array([[1.82650658, -0.68194475], [2.06815311, -0.5106321]]) * np.pi * 0.774)
    ''' relative strength of weights of different pre/post cell-type in middle layer '''
    J_2x2_m = np.array([[2.5, -1.3], [4.7, -2.2]]) * 0.774
    ''' relative strength of weights of different pre/post cell-type in superficial layer '''
    s_2x2_s = np.array([[0.2, 0.09], [0.4, 0.09]])
    ''' ranges of weights between different pre/post cell-type '''
    p_local_s = [0.4, 0.7]
    ''' relative strength of local parts of E projections in superficial layer '''
    p_local_m = [1.0, 1.0]
    ''' relative strength of local parts of E projections in middle layer '''

ssn_layer_pars = SsnLayerPars()

class MVPA_pars:
    gridsize_Nx = 27
    readout_grid_size = 9