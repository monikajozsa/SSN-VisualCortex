import numpy
from dataclasses import dataclass

import jax.numpy as np


# Input parameters
@dataclass(unsafe_hash=True)
class GridPars:
    gridsize_Nx: int = 9
    """ grid-points across each edge - gives rise to dx = 0.8 mm """
    gridsize_deg: float = 2 * 1.6
    """ edge length in degrees - visual field """
    magnif_factor: float = 2.0
    """ converts deg to mm (mm/deg) """
    hyper_col: float = 0.4
    """ ? are our grid points represent columns? (mm) """
    sigma_RF: float = 0.4
    """ deg (visual angle), comes in make_grating_input, which Clara does not use as she uses Gabor filters """


grid_pars = GridPars()


@dataclass(unsafe_hash=True)
class FilterPars:
    sigma_g = numpy.array(0.39 * 0.5 / 1.04)  #
    conv_factor = numpy.array(2)
    k: float = np.pi / (6 * 0.5)  # Clara approximated it by 1; Ke used 1 too
    edge_deg: float = grid_pars.gridsize_deg
    degree_per_pixel = numpy.array(0.05)
    # convert degree to number of pixels (129 x 129), this could be calculated from earlier params '''


filter_pars = FilterPars()


@dataclass
class StimuliPars:  # the attributes are changed within SSN_classes for a local instance
    inner_radius: float = 2.5  # inner radius of the stimulus
    outer_radius: float = 3.0  # outer radius of the stimulus: together with inner_radius, they define how the edge of the stimulus fades away to the gray background
    grating_contrast: float = 0.8  # from Current Biology 2020 Ke's paper
    std: float = 0.0  # no noise at the moment but this is a Gaussian white noise added to the stimulus
    jitter_val: float = (
        5.0  # uniform jitter between [-5, 5] to make the training stimulus vary
    )
    k: float = (
        filter_pars.k
    )  # It would be great to get rid of this because FILTER_PARS HAS IT but then it is used when it is passed to new_two_stage_training at BW_Grating
    edge_deg: float = filter_pars.edge_deg  # same as for k
    degree_per_pixel = filter_pars.degree_per_pixel  # same as for k
    ref_ori: float = 55.0
    offset: float = 4.0


stimuli_pars = StimuliPars()


# Network parameters
@dataclass(unsafe_hash=True)
class SigPars:
    N_neurons: int = 25  # Error if you change it in training_supp, line r_ref = r_ref + noise_ref*np.sqrt(jax.nn.softplus(r_ref)) ***
    w_sig = numpy.random.normal(size=(N_neurons,)) / np.sqrt(N_neurons)
    b_sig: float = 0.0


sig_pars = SigPars()


@dataclass(unsafe_hash=True)
class SSNPars:
    n = 2.0  # power law parameter
    k = 0.04  # power law parameter
    tauE = 20.0  # time constant for excitatory neurons in ms
    tauI = 10.0  # time constant for inhibitory neurons in ms~
    A = None  # normalization param for Gabors to get 100% contrast, see find_A
    A2 = None  # normalization param for Gabors to get 100% contrast, see find_A
    phases = 2  # or 4


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
    sigma_oris = np.asarray([90.0, 90.0])
    kappa_pre = np.asarray([0.0, 0.0])
    kappa_post = np.asarray([0.0, 0.0])
    f_E: float = 0.69  # Feedforwards connections
    f_I: float = 0.0
    c_E: float = 5.0  # Excitatory constant for extra synaptic GABA
    c_I: float = 5.0  # Inhibitory constant for extra synaptic GABA
    psi: float = 0.774
    J_2x2_s = (
        np.array([[1.82650658, -0.68194475], [2.06815311, -0.5106321]]) * np.pi * psi
    )
    s_2x2_s = np.array([[0.2, 0.09], [0.4, 0.09]])
    gE_s = 0.37328625 * 1.5
    gI_s = 0.26144141 * 1.5
    J_2x2_m = np.array([[2.5, -1.3], [4.7, -2.2]]) * psi
    gE_m = 0.3
    gI_m = 0.25
    gE = [gE_m, gE_s]
    gI = [gI_m, gI_s]


ssn_layer_pars = SsnLayerPars()


# Training parameters
@dataclass(unsafe_hash=True)
class ConvPars:
    dt: float = 1.0
    xtol: float = 1e-04
    Tmax: float = 250.0
    verbose: bool = False
    silent: bool = True
    Rmax_E = None
    Rmax_I = None


conv_pars = ConvPars()


@dataclass
class TrainingPars:
    eta = 10e-3  # was 0e-4
    batch_size = 100  # was 50
    noise_type = "poisson"
    sig_noise = 2.0 if noise_type != "no_noise" else 0.0
    epochs = 20  # was 5
    num_epochs_to_save = 2


training_pars = TrainingPars()


@dataclass(unsafe_hash=True)
class LossPars:
    lambda_dx = 1
    lambda_r_max = 1
    lambda_w = 1
    lambda_b = 1


loss_pars = LossPars()
