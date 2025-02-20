import numpy
from dataclasses import dataclass
import jax.numpy as jnp


# Pretraining parameters
@dataclass
class PretrainingPars:
    is_on: bool = True
    """ flag for turning pretraining on or off """
    ref_ori_int = [15, 165]
    """ interval where the reference orientation is randomly chosen from """
    ori_dist_int = [5, 20]
    """ interval where the absolute orientation difference between reference and target is randomly chosen from """
    acc_th: float = 0.749
    """ accuracy threshold to calculate corresponding offset threshold (fine discrimination task) """
    acc_check_freq: int = 2
    """ frequency (in SGD step) of accuracy check for the training task """
    min_acc_check_ind: int = 1
    """ minimum SGD step where accuracy check happens for the training task """
    offset_threshold = [5,25]
    """ interval for offset threshold where training task achieves accuracy threshold (acc_th) - used for early stopping of pretraining """
    batch_size: int = 100
    """ number of trials per SGD step during pretraining """
    SGD_steps: int = 1000
    """ maximum number of SGD steps during pretraining """
    min_stop_ind_stage_0: int = 50
    """ minimum SGD step where pretraining can stop """
    stage_1_acc_th: float = 0.65
    """ accuracy threshold for early stopping criterium for stage 1 """
    shuffle_labels: bool = True
    """ flag for shuffling the labels of the pretraining data """

pretraining_pars = PretrainingPars()


# Training parameters
@dataclass
class TrainingPars:
    pretraining_task: bool = False
    """ flag for training for the pretraining (general) task or the training (fine) discrimination task during stages 1 and 2 """
    shuffle_labels: bool = False
    """ flag for shuffling the labels of the training data """
    opt_readout_before_training: bool = False
    """ flag for optimizing the readout weights before the training """
    eta: float = 0.0
    """ learning rate - the maximum rate of parameter change in one SGD step; note that this initial values are irrelevant when we randomize the parameters """
    batch_size: int = 50
    """ number of trials per SGD step """
    validation_freq: int = 10
    """ frequency of validation loss and accuracy calculation """
    SGD_steps: int = 1500
    """ number of SGD step """
    min_stop_ind: int = 1000
    """ minimum SGD step where training can stop """


# Convergence parameters
@dataclass
class ConvPars:
    dt: float = 1.0
    """ step size during convergence of SSN """
    xtol: float = 1e-04
    """ convergence tolerance of SSN """
    Tmax: float = 300.0 
    """ maximum number of steps to be taken during convergence of SSN """


# Loss parameters
@dataclass
class LossPars:
    lambda_task: float = 0.5
    """ constant for binary entropy loss (reset by set_loss_pars_constants_for_training to 1 before  stages 1 and 2) """
    lambda_dx: float = 1
    """ constant for loss with respect to convergence of Euler function """
    lambda_w: float = 1
    """ constant for L2 regularizer of sigmoid layer weights """
    lambda_b: float = 1
    """ constant for L2 regulazier of sigmoid layer bias """
    lambda_r_max: float = 1
    """ constant for loss with respect to maximum rates in the network """
    lambda_r_mean: float = 0.01
    """ constant for loss with respect to maximum rates in the network (reset by set_loss_pars_constants_for_training to 0.25 before stages 1 and 2) """
    Rmax_E: float = 40
    """ maximum firing rate for E neurons - rates above this are penalised """
    Rmax_I: float = 80
    """ maximum firing rate for I neurons - rates above this are penalised """
    Rmean_E = [25,25]
    """ target mean firing rate for E neurons for the middle and superficial layers - rates deviating from this are penalised (reset by set_loss_pars_constants_for_training before stages 1 and 2) """
    Rmean_I = [50,50]
    """ target mean firing rate for I neurons for the middle and superficial layers - rates deviating from this are penalised (reset by set_loss_pars_constants_for_training before stages 1 and 2) """


def xy_distance(gridsize_Nx,gridsize_mm):
    """ This function calculates distances between grid points of a grid with given sizes. It is used in GridPars class."""
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

    return jnp.array(xy_dist), jnp.array(x_map), jnp.array(y_map)


# Input parameters
@dataclass
class GridPars:
    gridsize_Nx: int = 9
    """ size of the 2D grid is gridsize_Nx x gridsize_Nx """
    gridsize_deg: float = 2 * 1.6
    """ edge length in degrees of visual angle """
    c: float = 2.0
    """ converts deg to mm (mm/deg) """
    magnif_factor: float = 2.0
    """ converts deg to mm (mm/deg) """
    gridsize_mm = gridsize_deg * magnif_factor
    """ edge length in mm - cortical space """
    hyper_col: float = 0.4
    """ size of hypercolumn, parameter to generate orientation map """
    xy_dist, x_map, y_map = xy_distance(gridsize_Nx,gridsize_mm)
    """ distances between grid points """

grid_pars = GridPars()


# Gabor filter parameters
@dataclass
class FilterPars:
    sigma_g: float = 0.27
    """ std of the Gaussian of the Gabor filters """
    magnif_factor: float = grid_pars.magnif_factor
    """ converts deg to mm (mm/deg), same as magnification factor """
    k: float = 1.0
    """ scaling parameter for the spacial frequency of the Gabor filter """
    gridsize_deg: float = grid_pars.gridsize_deg
    """ edge length in degrees - visual field, same as grid_pars.gridsize_deg """
    degree_per_pixel: float = 0.05
    """ convert degree to number of pixels (129 x 129), note that this is not an independent parameter and could be calculated from other parameters """
    gE_m: float = 0.0
    """ scaling parameter between stimulus and excitatory units in middle layer; note that this initial values are irrelevant when we randomize the parameters """
    gI_m: float = 0.0 
    """ scaling parameter between stimulus and inhibitory units in middle layer; note that this initial values are irrelevant when we randomize the parameters """

filter_pars = FilterPars()


# Stimulus parameters
@dataclass
class StimuliPars:
    inner_radius: float = 2.5
    """ inner radius of the stimulus """
    outer_radius: float = 3.0 
    """ outer radius of the stimulus, inner_radius and outer_radius define how the edge of the stimulus fades away to the gray background """
    grating_contrast: float = 0.8 
    """ contrast of darkest and lightest point in the grid - see Current Biology paper from 2020 """
    std: float = 100.0
    """ Gaussian white noise added to the stimulus """
    jitter_val: float = 5.0
    """ constant that defines and interval [-jitter_val, jitter_val] from where jitter (same applied for reference an target stimuli orientation) is randomly taken """
    k: float = filter_pars.k
    """ scaling parameter for the spacial frequency of the Gabor filter """
    gridsize_deg: float = filter_pars.gridsize_deg  
    """ edge length in degrees - visual field, same as grid_pars.gridsize_deg """
    degree_per_pixel = filter_pars.degree_per_pixel  
    """ convert degree to number of pixels (129 x 129), note that this is not an independent parameter and could be calculated from other parameters """
    magnif_factor: float = grid_pars.magnif_factor
    """ converts deg to mm (mm/deg), same as magnification factor """
    ref_ori: float = 55.0 
    """ reference orientation of the stimulus in degree """
    offset: float = 4.0 
    """ difference between reference and task orientation in degree (task ori is either ref_ori + offset or ref_or - offset) """
    max_train_offset: float = 10.0
    """ maximum offset for the training task """

stimuli_pars = StimuliPars()


# Sigmoid parameters
@dataclass
class ReadoutPars:
    sup_mid_readout_contrib = [1.0, 0.0]
    """ contribution of the superficial and middle layer to the readout """
    readout_grid_size = jnp.array([grid_pars.gridsize_Nx, 5])
    """ size of the center grid where readout units come from (first number is for the pretraining, second is for the training) """
    # Define middle grid indices
    middle_grid_ind = []
    mid_grid_ind0 = int((readout_grid_size[0]-readout_grid_size[1])/2)
    mid_grid_ind1 = int(readout_grid_size[0]) - mid_grid_ind0
    for i in range(mid_grid_ind0,mid_grid_ind1):  
        row_start = i * readout_grid_size[0] 
        middle_grid_ind.extend(range(row_start + mid_grid_ind0, row_start + mid_grid_ind1))
    middle_grid_ind = jnp.array(middle_grid_ind)
    """ indices of the middle grid when grid is flattened """
    # Define w_sig - its size depends on whether pretraining is on
    if pretraining_pars.is_on:
        w_sig = jnp.zeros(readout_grid_size[0]**2)
        """ readout weights (between the superficial and the sigmoid layer) - initialized with logistic regression """
    else:
        w_sig = jnp.array(numpy.random.normal(scale = 0.25, size=(readout_grid_size[1]**2,)) / readout_grid_size[1])
        """ readout weights (between the superficial and the sigmoid layer) """
    b_sig: float = 0.0 
    """ bias added to the sigmoid layer """
    num_readout_noise: int = 125
    """ defines the additive Gaussian readout noise var (meaning is number of neighbouring cells), see generate_noise function """


# general SSN parameters
@dataclass
class SSNPars:
    couple_c_ms = True
    """ if True then c_m=c_s """
    n: float = 2.0  
    """ power law parameter """
    k: float = 0.04  
    """ power law parameter """
    tauE: float = 20.0 
    """  time constant for excitatory neurons in ms """
    tauI: float = 10.0
    """ time constant for inhibitory neurons in ms """
    phases: int = 4 
    """ number of inh. and exc. neurons (with different Gabor filt.) per grid point in middle layer (has to be an even integer) """
    sigma_oris = jnp.asarray([90.0, 90.0])
    """ range of weights in terms of preferred orientation difference (in degree) """
    s_2x2_s = jnp.array([[0.2, 0.09], [0.4, 0.09]])
    """ ranges of weights between different pre/post cell-type """
    p_local_s = [0.4, 0.7]
    """ relative strength of local parts of E projections in superficial layer """
    beta = stimuli_pars.ref_ori
    """ shaping tuning curves depending on the reference orientation of the stimulus """
    kappa_range = 90
    """ scaling the effect of all kappa parameters through: tanh_kappa*dist_from_single_ori**2/(2*(kappa_range**2)) """


# Ranges for the randomization of parameters (at initialization)
class RandomizePars:
    J_range = [jnp.array([4, 4.8]),jnp.array([1.2,2]), jnp.array([4.6, 5.4]),jnp.array([0.8,1.6])] # [jnp.array([4, 5.5]),jnp.array([0.7,2]), jnp.array([4, 5.5]),jnp.array([0.7,2])]
    """ range of the initial Jm and Js trainable parameters """
    c_range = jnp.array([4.5, 5.5]) # jnp.array([4.5, 5.5])
    """ range of the initial c trainable parameters """
    f_range = jnp.array([0.6, 1.2]) # jnp.array([0.6, 1.2])
    """ range of the initial f trainable parameters """
    g_range = jnp.array([0.15, 0.45]) # jnp.array([0.15, 0.45])
    """ range of the gE and gI parameters """
    eta_range = jnp.array([3e-3, 5e-3]) # jnp.array([3e-3, 5e-3])
    """ range of the learning rate parameter """

########## Do not change the values in the next two classes as that has no effect on the results - vales are set through random initialization and training ##########
# Trained SSN parameters - parameters can be moved between TrainedSSNPars and SSNPars depending on whether we want to train them or not
@dataclass
class TrainedSSNPars:
    kappa_Jmid = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    """ shaping parameter for middle layer horizontal connections to achieve orientation selectivity - initialized from pretraining """
    kappa_f = jnp.array([0.0, 0.0])
    """ shaping parameter for feedforward connections to achieve orientation selectivity - initialized from pretraining """
    cE_m: float = 0.0
    """ baseline excitatory input (constant added to the output of excitatory neurons at middle layer) - initialized from pretraining """
    cI_m: float = 0.0
    """ baseline inhibitory input (constant added to the output of inhibitory neurons at middle layer) - initialized from pretraining """
    cE_s: float = 0.0
    """ baseline excitatory input (constant added to the output of excitatory neurons at superficial layer) - initialized from pretraining"""
    cI_s: float = 0.0
    """ baseline inhibitory input (constant added to the output of inhibitory neurons at superficial layer) - initialized from pretraining """
    f_E: float = 0.0
    """ scaling constant for feedforwards connections to excitatory units in sup layer - initialized from pretraining """
    f_I: float = 0.0
    """ scaling constant for feedforwards connections to inhibitory units in sup layer - initialized from pretraining """
    J_EE_m: float = 0.0
    """ relative strength of weights of E/E pre/post cell-type in middle layer - initialized from pretraining """
    J_IE_m: float = 0.0
    """ relative strength of weights of E/I pre/post cell-type in middle layer - initialized from pretraining """
    J_EI_m: float = 0.0
    """ relative strength of weights of I/E pre/post cell-type in middle layer - initialized from pretraining """
    J_II_m: float = 0.0
    """ relative strength of weights of I/I pre/post cell-type in middle layer - initialized from pretraining """
    J_EE_s: float = 0.0
    """ relative strength of weights of E/E pre/post cell-type in superficial layer - initialized from pretraining """
    J_IE_s: float = 0.0
    """ relative strength of weights of E/I pre/post cell-type in superficial layer - initialized from pretraining """
    J_EI_s: float = 0.0
    """ relative strength of weights of I/E pre/post cell-type in superficial layer - initialized from pretraining """
    J_II_s: float = 0.0
    """ relative strength of weights of I/I pre/post cell-type in superficial layer - initialized from pretraining """
    kappa_Jsup = jnp.array([[[0.0, 0.0], [0.0, 0.0]],[[0.0, 0.0], [0.0, 0.0]]])
    """ shaping parameter for superficial layer horizontal connections to achieve orientation selectivity;
    kappa_Jsup[0] and kappa_Jsup[1] are for pre- and post- synaptic connections, respectively  - initialized from pretraining"""
    
@dataclass
class PretrainedSSNPars:
    cE_m: float = 0.0 
    """ baseline excitatory input (constant added to the output of excitatory neurons at middle layer) - randomly initialized """
    cI_m: float = 0.0 
    """ baseline inhibitory input (constant added to the output of inhibitory neurons at middle layer) - randomly initialized """
    cE_s: float = 0.0 
    """ baseline excitatory input (constant added to the output of excitatory neurons at superficial layer) - randomly initialized """
    cI_s: float = 0.0 
    """ baseline inhibitory input (constant added to the output of inhibitory neurons at superficial layer) - randomly initialized """
    f_E: float = 0.0 
    """ scaling constant for feedforwards connections to excitatory units in sup layer - randomly initialized """
    f_I: float = 0.0
    """ scaling constant for feedforwards connections to inhibitory units in sup layer - randomly initialized """
    J_EE_m: float = 0.0
    """ relative strength of weights of E/E pre/post cell-type in middle layer - randomly initialized """
    J_IE_m: float = 0.0
    """ relative strength of weights of E/I pre/post cell-type in middle layer - randomly initialized """
    J_EI_m: float = 0.0
    """ relative strength of weights of I/E pre/post cell-type in middle layer - randomly initialized """
    J_II_m: float = 0.0
    """ relative strength of weights of I/I pre/post cell-type in middle layer - randomly initialized """
    J_EE_s: float = 0.0
    """ relative strength of weights of E/E pre/post cell-type in superficial layer - randomly initialized """
    J_IE_s: float = 0.0
    """ relative strength of weights of E/I pre/post cell-type in superficial layer - randomly initialized """
    J_EI_s: float = 0.0
    """ relative strength of weights of I/E pre/post cell-type in superficial layer - randomly initialized """
    J_II_s: float = 0.0
    """ relative strength of weights of I/I pre/post cell-type in superficial layer - randomly initialized """