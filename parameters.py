import numpy
from dataclasses import dataclass
import jax.numpy as np


# Pretraining parameters
@dataclass
class PreTrainPars:
    is_on: bool = True
    ''' flag for turning pretraining on or off '''
    ref_ori_int = [15, 165]
    ''' interval where the reference orientation is randomly chosen from '''
    ori_dist_int = [5, 20]
    ''' interval where the absolute orientation difference between reference and target is randomly chosen from '''
    acc_th: float = 0.749
    ''' accuracy threshold to calculate corresponding offset (training task) '''
    acc_check_freq: int = 3
    ''' frequency (in SGD step) of accuracy check for the training task '''
    min_acc_check_ind: int = 1
    ''' minimum SGD step where accuracy check happens for the training task '''
    min_stop_ind: int = 50
    ''' minimum SGD step where pretraining can stop '''
    offset_threshold = [3,10]
    ''' threshold for offset where training task achieves accuracy threshold (acc_th)  - used for early stopping of pretraining '''
    batch_size: int = 100
    ''' number of trials per SGD step during pretraining '''
    SGD_steps: int = 1000
    ''' maximum number of SGD steps during pretraining '''

pretrain_pars = PreTrainPars()


# Training parameters
@dataclass
class TrainingPars:
    eta: float = 1e-3
    ''' learning rate - the maximum rate of parameter change in one SGD step; note that this initial values are irrelevant when we randomize the parameters '''
    batch_size: int = 50
    ''' number of trials per SGD step '''
    SGD_steps: int = 1000
    ''' number of SGD step '''
    validation_freq: int = 10
    ''' frequency of validation loss and accuracy calculation '''
    first_stage_acc_th: float = 0.51
    ''' accuracy threshold for early stopping criterium for the first stage of training '''

training_pars = TrainingPars()


# Convergence parameters
@dataclass
class ConvPars:
    dt: float = 1.0
    ''' step size during convergence of SSN '''
    xtol: float = 1e-04
    ''' convergence tolerance of SSN '''
    Tmax: float = 250.0
    ''' maximum number of steps to be taken during convergence of SSN '''

conv_pars = ConvPars()


# Loss parameters
@dataclass
class LossPars:
    lambda_dx: float = 1
    ''' constant for loss with respect to convergence of Euler function '''
    lambda_w: float = 1
    ''' constant for L2 regularizer of sigmoid layer weights '''
    lambda_b: float = 1
    ''' constant for L2 regulazier of sigmoid layer bias '''
    lambda_r_max: float = 1
    ''' constant for loss with respect to maximum rates in the network '''
    lambda_r_mean: float = 0.01
    ''' constant for loss with respect to maximum rates in the network '''
    Rmax_E: float = 40
    ''' maximum firing rate for E neurons - rates above this are penalised '''
    Rmax_I: float = 80
    ''' maximum firing rate for I neurons - rates above this are penalised '''
    Rmean_E = [25,25]
    ''' mean firing rate for E neurons for the middle and superficial layers - rates deviating from this are penalised. These values may be overwritten after pretraining. '''
    Rmean_I = [50,50]
    ''' mean firing rate for I neurons for the middle and superficial layers - rates deviating from this are penalised. These values may be overwritten after pretraining. '''

loss_pars = LossPars()


def xy_distance(gridsize_Nx,gridsize_mm):
    ''' This function calculates distances between grid points of a grid with given sizes. It is used in GridPars class.'''
    Nn = gridsize_Nx**2
    Lx = Ly = gridsize_mm
    Nx = Ny = gridsize_Nx

    # Simplified meshgrid creation
    xs = numpy.linspace(0, Lx, Nx) # 
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
    ''' size of the 2D grid is gridsize_Nx x gridsize_Nx '''
    gridsize_deg: float = 2 * 1.6
    ''' edge length in degrees of visual angle '''
    c: float = 2.0
    ''' converts deg to mm (mm/deg) '''
    magnif_factor: float = 2.0
    ''' converts deg to mm (mm/deg) '''
    gridsize_mm = gridsize_deg * magnif_factor
    ''' edge length in mm - cortical space '''
    hyper_col: float = 0.4
    ''' size of hypercolumn, parameter to generate orientation map '''
    xy_dist, x_map, y_map = xy_distance(gridsize_Nx,gridsize_mm)
    ''' distances between grid points '''

grid_pars = GridPars()


# Gabor filter parameters
@dataclass
class FilterPars:
    sigma_g: float = 0.27
    ''' std of the Gaussian of the Gabor filters '''
    magnif_factor: float = grid_pars.magnif_factor
    ''' converts deg to mm (mm/deg), same as magnification factor '''
    k: float = 1.0
    ''' scaling parameter for the spacial frequency of the Gabor filter '''
    gridsize_deg: float = grid_pars.gridsize_deg
    ''' edge length in degrees - visual field, same as grid_pars.gridsize_deg '''
    degree_per_pixel: float = 0.05
    ''' convert degree to number of pixels (129 x 129), note that this is not an independent parameter and could be calculated from other parameters '''
    gE_m: float = 0.3
    ''' scaling parameter between stimulus and excitatory units in middle layer; note that this initial values are irrelevant when we randomize the parameters '''
    gI_m: float = 0.25 
    ''' scaling parameter between stimulus and inhibitory units in middle layer; note that this initial values are irrelevant when we randomize the parameters '''
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
    std: float = 100.0
    ''' Gaussian white noise added to the stimulus '''
    jitter_val: float = 5.0
    ''' constant that defines and interval [-jitter_val, jitter_val] from where jitter (same applied for reference an target stimuli orientation) is randomly taken '''
    k: float = filter_pars.k
    ''' scaling parameter for the spacial frequency of the Gabor filter '''
    gridsize_deg: float = filter_pars.gridsize_deg  
    ''' edge length in degrees - visual field, same as grid_pars.gridsize_deg '''
    degree_per_pixel = filter_pars.degree_per_pixel  
    ''' convert degree to number of pixels (129 x 129), note that this is not an independent parameter and could be calculated from other parameters '''
    magnif_factor: float = grid_pars.magnif_factor
    ''' converts deg to mm (mm/deg), same as magnification factor '''
    ref_ori: float = 55.0 
    ''' reference orientation of the stimulus in degree '''
    offset: float = 4.0 
    ''' difference between reference and task orientation in degree (task ori is either ref_ori + offset or ref_or - offset) '''

stimuli_pars = StimuliPars()


# Sigmoid parameters
@dataclass
class ReadoutPars:
    sup_mid_readout_contrib = [1.0, 0.0]
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
        ''' readout weights (between the superficial and the sigmoid layer) - initialized with logistic regression '''
    else:
        w_sig = np.array(numpy.random.normal(scale = 0.25, size=(readout_grid_size[1]**2,)) / readout_grid_size[1])
        ''' readout weights (between the superficial and the sigmoid layer) '''
    b_sig: float = 0.0 
    ''' bias added to the sigmoid layer '''
    num_readout_noise: int = 125
    ''' defines the additive Gaussian readout noise var (meaning is number of neighbouring cells), see generate_noise function '''

readout_pars = ReadoutPars()


# general SSN parameters
@dataclass
class SSNPars:
    n: float = 2.0  
    ''' power law parameter '''
    k: float = 0.04  
    ''' power law parameter '''
    tauE: float = 20.0 
    '''  time constant for excitatory neurons in ms '''
    tauI: float = 10.0
    ''' time constant for inhibitory neurons in ms '''
    phases: int = 4 
    ''' number of inh. and exc. neurons (with different Gabor filt.) per grid point in middle layer (has to be an even integer) '''
    sigma_oris = np.asarray([90.0, 90.0])
    ''' range of weights in terms of preferred orientation difference (in degree) '''
    s_2x2_s = np.array([[0.2, 0.09], [0.4, 0.09]])
    ''' ranges of weights between different pre/post cell-type '''
    p_local_s = [0.4, 0.7]
    ''' relative strength of local parts of E projections in superficial layer '''
    p_local_m = [1.0, 1.0]
    ''' relative strength of local parts of E projections in middle layer '''
    c_E: float = 5.0 
    ''' baseline excitatory input (constant added to the output of excitatory neurons at both middle and superficial layers) '''
    c_I: float = 5.0 
    ''' baseline inhibitory input (constant added to the output of inhibitory neurons at both middle and superficial layers) '''
    
ssn_pars = SSNPars()


# Trained SSN parameters - f and c parameters can be moved between TrainedSSNPars and SSNPars deoending on whether we want to train (and perturb) them or not
@dataclass
class TrainedSSNPars:
    # Note that these initial values are irrelevant when we randomize the parameters
    f_E: float = 1.11 
    ''' scaling constant for feedforwards connections to excitatory units in sup layer '''
    f_I: float = 0.7
    ''' scaling constant for feedforwards connections to inhibitory units in sup layer '''
    J_2x2_s = np.array([[1.83, -0.68], [2.07, -0.51]]) * np.pi * 0.774
    ''' relative strength of weights of different pre/post cell-type in middle layer '''
    J_2x2_m = np.array([[2.5, -1.3], [4.7, -2.2]]) * 0.774
    ''' relative strength of weights of different pre/post cell-type in superficial layer '''
    
trained_pars = TrainedSSNPars()


class RandomizePars:
    J_range = [np.array([4, 4.8]),np.array([1.2,2]), np.array([4.6, 5.4]),np.array([0.8,1.6])]
    ''' range of the perturbed Jm and Js parameters '''
    c_range = np.array([4.5, 5.5])
    ''' range of the perturbed c parameters '''
    f_range = np.array([0.6, 1.2])
    ''' range of the perturbed f parameters '''
    g_range = np.array([0.2, 0.4])
    ''' range of the perturbed g parameters '''
    eta_range = np.array([2*10e-4, 5*10e-4])

randomize_pars = RandomizePars()


class MVPA_pars:
    gridsize_Nx: int = 9
    ''' size of the extended grid that is filtered '''
    size_conv_factor: float = (gridsize_Nx -1)/ (grid_pars.gridsize_Nx - 1)
    ''' adjusted conversion factor to keep the role of the middle grid the same'''
    readout_grid_size: int = 5
    ''' size of the readout grid '''
    middle_grid_ind = []
    mid_grid_ind0 = int((gridsize_Nx-readout_grid_size)/2)
    mid_grid_ind1 = int(gridsize_Nx) - mid_grid_ind0
    for i in range(mid_grid_ind0,mid_grid_ind1):  
        row_start = i * gridsize_Nx
        middle_grid_ind.extend(range(row_start + mid_grid_ind0, row_start + mid_grid_ind1))
    middle_grid_ind = np.array(middle_grid_ind)
    ''' indices of the middle grid when grid is flattened '''
    noise_std: float = 1.0
    ''' std of the noise added to the readout layer '''

mvpa_pars = MVPA_pars()