import jax
import math
from PIL import Image
import jax.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy

from parameters import StimuliPars


def Euler2fixedpt(
    dxdt,
    x_initial,
    Tmax,
    dt,
    xtol=1e-5,
    xmin=1e-0,
    Tmin=200,
    PLOT=True,
    save=None,
    inds=None,
    verbose=True,
    print_dt=False,
):
    """
    Finds the fixed point of the D-dim ODE set dx/dt = dxdt(x), using the
    Euler update with sufficiently large dt (to gain in computational time).
    Checks for convergence to stop the updates early.

    IN:
    dxdt = a function handle giving the right hand side function of dynamical system
    x_initial = initial condition for state variables (a column vector)
    Tmax = maximum time to which it would run the Euler (same units as dt, e.g. ms)
    dt = time step of Euler
    xtol = tolerance in relative change in x for determining convergence
    xmin = for x(i)<xmin, it checks convergenece based on absolute change, which must be smaller than xtol*xmin
        Note that one can effectively make the convergence-check purely based on absolute,
        as opposed to relative, change in x, by setting xmin to some very large
        value and inputting a value for 'xtol' equal to xtol_desired/xmin.
    PLOT: if True, plot the convergence of some component
    inds: indices of x (state-vector) to plot

    OUT:
    xvec = found fixed point solution
    CONVG = True if determined converged, False if not
    """

    if PLOT == True:
        if inds is None:
            N = x_initial.shape[0]  # x_initial.size
            inds = [int(N / 4), int(3 * N / 4)]

        xplot = x_initial[np.array(inds)][:, None]
        xplot_all = np.sum(x_initial)
        xplot_max = []
        xplot_max.append(x_initial.max())

    Nmax = np.round(Tmax / dt).astype(int)
    Nmin = np.round(Tmin / dt) if Tmax > Tmin else (Nmax / 2)
    xvec = x_initial
    CONVG = False

    for n in range(Nmax):
        dx = dxdt(xvec) * dt
        xvec = xvec + dx

        if PLOT:
            xplot = np.hstack((xplot, xvec[np.asarray(inds)][:, None]))
            xplot_all = np.hstack((xplot_all, np.sum(xvec)))
            xplot_max.append(xvec.max())

        if n > Nmin:
            if np.abs(dx / np.maximum(xmin, np.abs(xvec))).max() < xtol:  # y
                if verbose:
                    print(
                        "      converged to fixed point at iter={},      as max(abs(dx./max(xvec,{}))) < {} ".format(
                            n, xmin, xtol
                        )
                    )
                CONVG = True
                break

    if not CONVG and verbose:
        print(
            "\n Warning 1: reached Tmax={}, before convergence to fixed point.".format(
                Tmax
            )
        )
        print(
            "       max(abs(dx./max(abs(xvec), {}))) = {},   xtol={}.\n".format(
                xmin, np.abs(dx / np.maximum(xmin, np.abs(xvec))).max(), xtol
            )
        )

    if PLOT == True:
        print("plotting")

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

        axes[0].plot(np.arange(n + 2) * dt, xplot.T, "o-", label=inds)
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Neural responses")
        axes[0].legend()

        axes[1].plot(np.arange(n + 2) * dt, xplot_all)
        axes[1].set_ylabel("Sum of response")
        axes[1].set_xlabel("Steps")
        axes[1].set_ylim([0, 1.2 * np.max(np.asarray(xplot_all[-100:]))])
        axes[1].set_title(
            "Final sum: " + str(np.sum(xvec)) + ", converged " + str(CONVG)
        )

        axes[2].plot(np.arange(n + 2) * dt, np.asarray(xplot_max))
        axes[2].set_ylabel("Maximum response")
        axes[2].set_title(
            "final maximum: " + str(xvec.max()) + "at index " + str(np.argmax(xvec))
        )
        axes[2].set_xlabel("Steps")
        axes[2].set_ylim([0, 1.2 * np.max(np.asarray(xplot_max[-100:]))])

        if save:
            fig.savefig(save + ".png")

        fig.show()
        plt.close()

    print(xvec.max(), np.argmax(xvec))
    return xvec, CONVG


def Euler2fixedpt_fullTmax(
    dxdt, x_initial, Tmax, dt, xtol=1e-5, xmin=1e-0, Tmin=200, PLOT=False, save=None
):
    """
    Finds the fixed point of the D-dim ODE set dx/dt = dxdt(x), using the
    Euler update with sufficiently large dt (to gain in computational time).
    Checks for convergence to stop the updates early.

    IN:
    dxdt = a function handle giving the right hand side function of dynamical system
    x_initial = initial condition for state variables (a column vector)
    Tmax = maximum time to which it would run the Euler (same units as dt, e.g. ms)
    dt = time step of Euler
    xtol = tolerance in relative change in x for determining convergence
    xmin = for x(i)<xmin, it checks convergenece based on absolute change, which must be smaller than xtol*xmin
        Note that one can effectively make the convergence-check purely based on absolute,
        as opposed to relative, change in x, by setting xmin to some very large
        value and inputting a value for 'xtol' equal to xtol_desired/xmin.
    PLOT: if True, plot the convergence of some component
    inds: indices of x (state-vector) to plot

    OUT:
    xvec = found fixed point solution
    CONVG = True if determined converged, False if not
    avg_dx = ...
    """

    Nmax = int(Tmax / dt)
    xvec = x_initial
    CONVG = False
    y = np.zeros(((Nmax)))

    if PLOT:
        xplot_all = np.zeros(((Nmax + 1)))
        xplot_all = xplot_all.at[0].set(np.sum(xvec))

        def loop(n, carry):
            xvec, y, xplot_all = carry
            dx = dxdt(xvec) * dt
            xvec = xvec + dx
            y = y.at[n].set(np.abs(dx / np.maximum(xmin, np.abs(xvec))).max())
            xplot_all = xplot_all.at[n + 1].set(np.sum(xvec))
            return (xvec, y, xplot_all)

        xvec, y, xplot_all = jax.lax.fori_loop(0, Nmax, loop, (xvec, y, xplot_all))

    else:

        def loop(n, carry):
            xvec, y = carry
            dx = dxdt(xvec) * dt
            xvec = xvec + dx
            y = y.at[n].set(np.abs(dx / np.maximum(xmin, np.abs(xvec))).max())
            return (xvec, y)

        xvec, y = jax.lax.fori_loop(0, Nmax, loop, (xvec, y))

    avg_dx = y[int(Nmax / 2) : int(Nmax)].mean() / xtol

    CONVG = False  ## NEEDS UPDATING *** MJ comment: why and how?

    if PLOT:
        import matplotlib.pyplot as plt

        plt.figure(244459)
        plt.plot(np.arange(Nmax + 1) * dt, xplot_all)
        plt.title("Converged to sum of " + str(np.sum(xvec)))

        if save:
            plt.savefig(save + ".png")
        plt.show()
        plt.close()

    return xvec, CONVG, avg_dx


# this is copied from scipy.linalg, to make compatible with jax.numpy
def toeplitz(c, r=None):
    """
    Construct a Toeplitz matrix.
    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row.  If r is not given, ``r == conjugate(c)`` is
    assumed.
    Parameters
    ----------
    c : array_like
        First column of the matrix.  Whatever the actual shape of `c`, it
        will be converted to a 1-D array.
    r : array_like
        First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
        in this case, if c[0] is real, the result is a Hermitian matrix.
        r[0] is ignored; the first row of the returned matrix is
        ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
        converted to a 1-D array.
    Returns
    -------
    A : (len(c), len(r)) ndarray
        The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.
    See also
    --------
    circulant : circulant matrix
    hankel : Hankel matrix
    Notes
    -----
    The behavior when `c` or `r` is a scalar, or when `c` is complex and
    `r` is None, was changed in version 0.8.0.  The behavior in previous
    versions was undocumented and is no longer supported.
    Examples
    --------
    >>> from scipy.linalg import toeplitz
    >>> toeplitz([1,2,3], [1,4,5,6])
    array([[1, 4, 5, 6],
           [2, 1, 4, 5],
           [3, 2, 1, 4]])
    >>> toeplitz([1.0, 2+3j, 4-1j])
    array([[ 1.+0.j,  2.-3.j,  4.+1.j],
           [ 2.+3.j,  1.+0.j,  2.-3.j],
           [ 4.-1.j,  2.+3.j,  1.+0.j]])
    """
    c = np.asarray(c).ravel()
    if r is None:
        r = c.conjugate()
    else:
        r = np.asarray(r).ravel()
    # Form a 1D array of values to be used in the matrix, containing a reversed
    # copy of r[1:], followed by c.
    vals = np.concatenate((r[-1:0:-1], c))
    a, b = np.ogrid[0 : len(c), len(r) - 1 : -1 : -1]
    indx = a + b
    # `indx` is a 2D array of indices into the 1D array `vals`, arranged so
    # that `vals[indx]` is the Toeplitz matrix.
    return vals[indx]


def sigmoid(x, epsilon=0.01):
    """
    Introduction of epsilon stops asymptote from reaching 1 (avoids NaN)
    """
    return (1 - 2 * epsilon) * sig(x) + epsilon


def sig(x):
    return 1 / (1 + np.exp(-x))


def take_log(J_2x2):
    signs = np.array([[1, -1], [1, -1]])
    logJ_2x2 = np.log(J_2x2 * signs)

    return logJ_2x2


def exponentiate(opt_pars):
    signs = np.array([[1, -1], [1, -1]])
    J_2x2 = np.exp(opt_pars["logJ_2x2"]) * signs
    s_2x2 = np.exp(opt_pars["logs_2x2"])

    return J_2x2, s_2x2


def sep_exponentiate(J_s):
    signs = np.array([[1, -1], [1, -1]])
    new_J = np.exp(J_s) * signs

    return new_J


#### CREATE GABOR FILTERS ####
class GaborFilter:
    def __init__(
        self,
        x_i,
        y_i,
        k,
        sigma_g,
        theta,
        edge_deg,
        degree_per_pixel,
        phase=0,
        conv_factor=None,
    ):
        """
        Gabor filter class.
        Called from SSN_mid_local.create_gabor_filters() and SSN_mid.create_gabor_filters() whose outputs are gabor_filters and A (attributes of SSN_mid and SSN_mid_local)
        Inputs:
            x_i, y_i: centre of the filter
            k: preferred spatial frequency in cycles/degrees (radians)
            sigma_g: variance of Gaussian function
            theta: orientation map
            edge_deg: extent of the filter in degrees
            degree_per_pixel: resolution in degrees per pixel
            phase: phase of the Gabor filter (default is 0)
            conv_factor: conversion factor from degrees to mm
        """

        # convert to mm from degrees
        if conv_factor:
            self.conv_factor = conv_factor
            self.x_i = x_i / conv_factor
            self.y_i = y_i / conv_factor
        else:
            self.x_i = x_i
            self.y_i = y_i
        self.k = k
        self.theta = theta * (np.pi / 180)
        self.phase = phase
        self.sigma_g = sigma_g
        self.edge_deg = edge_deg
        self.degree_per_pixel = degree_per_pixel
        self.N_pixels = int(edge_deg * 2 / degree_per_pixel) + 1

        # create image axis
        x_axis = np.linspace(-edge_deg, edge_deg, self.N_pixels, endpoint=True)
        y_axis = np.linspace(-edge_deg, edge_deg, self.N_pixels, endpoint=True)

        # construct filter as an attribute
        self.filter = self.create_filter(x_axis, y_axis)

    def create_filter(self, x_axis, y_axis):
        """
        Create Gabor filters in vectorised form.
        """
        # Reshape the center coordinates into column vectors; repeat and reshape the center coordinates to allow calculating diff_x and diff_y
        x_axis = np.reshape(x_axis, (self.N_pixels, 1))
        x_i = np.repeat(self.x_i, self.N_pixels)
        x_i = np.reshape(x_i, (self.N_pixels, 1))
        diff_x = x_axis.T - x_i

        y_axis = np.reshape(y_axis, (self.N_pixels, 1))
        y_i = np.repeat(self.y_i, self.N_pixels)
        y_i = np.reshape(y_i, (self.N_pixels, 1))
        diff_y = y_axis - y_i.T

        # Calculate the spatial component of the Gabor filter
        spatial = np.cos(
            self.k
            * np.pi
            * 2
            * (diff_x * np.cos(self.theta) + diff_y * np.sin(self.theta))
            + self.phase
        )
        # Calculate the Gaussian component of the Gabor filter
        gaussian = np.exp(-0.5 * (diff_x**2 + diff_y**2) / self.sigma_g**2)

        return gaussian * spatial[::-1]  # same convention as stimuli


### FINDING CONSTANT FOR GABOR FILTERS ###
def find_A(
    k,
    sigma_g,
    edge_deg,
    degree_per_pixel,
    indices,
    phase=0,
    return_all=False,
):
    """
    Find constant to multiply Gabor filters.
    Input:
        gabor_pars: Filter parameters - centre already specified in function
        stimuli_pars: Stimuli parameters (high constrast and spanning all visual field)
        indices: List of orientatins in degrees to calculate filter and corresponding stimuli
    Output:
        A: value of constant so that contrast = 100
    """
    all_A = []
    all_gabors = []
    all_test_stimuli = []

    for ori in indices:
        # generate Gabor filter and stimuli at orientation
        gabor = GaborFilter(
            theta=ori,
            x_i=0,
            y_i=0,
            edge_deg=edge_deg,
            k=k,
            sigma_g=sigma_g,
            degree_per_pixel=degree_per_pixel,
            phase=phase,
        )
        # create local_stimui_pars to pass it to BW_Gratings
        local_stimuli_pars = StimuliPars()
        local_stimuli_pars.edge_deg = edge_deg
        local_stimuli_pars.k = k
        local_stimuli_pars.outer_radius = edge_deg * 2
        local_stimuli_pars.inner_radius = edge_deg * 2
        local_stimuli_pars.degree_per_pixel = degree_per_pixel
        local_stimuli_pars.grating_contrast = 0.99
        local_stimuli_pars.jitter_val = 0

        test_grating = BW_Grating(
            ori_deg=ori,
            stimuli_pars=local_stimuli_pars,
            jitter=local_stimuli_pars.jitter_val,
            phase=phase,
        )
        test_stimuli = test_grating.BW_image()

        mean_removed_filter = gabor.filter - gabor.filter.mean()
        # multiply filter and stimuli
        output_gabor = mean_removed_filter.ravel() @ test_stimuli.ravel()

        all_gabors.append(gabor.filter)
        all_test_stimuli.append(test_stimuli)

        # calculate value of A
        A_value = 100 / (output_gabor)

        # create list of A
        all_A.append(A_value)

    # find average value of A
    all_A = np.array(all_A)
    A = all_A.mean()

    all_gabors = np.array(all_gabors)
    all_test_stimuli = np.array(all_test_stimuli)

    if return_all == True:
        output = A, all_gabors, all_test_stimuli
    else:
        output = A

    return output


class BW_Grating:
    """ """

    def __init__(
        self,
        ori_deg,
        stimuli_pars,
        jitter=0,
        phase=0,
        crop_f=None,
    ):
        self.ori_deg = ori_deg
        self.jitter = jitter
        self.outer_radius = stimuli_pars.outer_radius  # in degrees
        self.inner_radius = stimuli_pars.inner_radius  # in degrees
        self.grating_contrast = stimuli_pars.grating_contrast
        self.std = stimuli_pars.std
        degree_per_pixel = stimuli_pars.degree_per_pixel
        pixel_per_degree = 1 / degree_per_pixel
        self.pixel_per_degree = pixel_per_degree
        edge_deg = stimuli_pars.edge_deg
        size = int(edge_deg * 2 * pixel_per_degree) + 1
        self.size = size
        k = stimuli_pars.k
        spatial_frequency = k * degree_per_pixel  # 0.05235987755982988
        self.phase = phase
        self.crop_f = crop_f
        self.smooth_sd = self.pixel_per_degree / 6
        self.spatial_freq = spatial_frequency or (1 / self.pixel_per_degree)
        self.grating_size = round(self.outer_radius * self.pixel_per_degree)
        self.angle = ((self.ori_deg + self.jitter) - 90) / 180 * numpy.pi

    def BW_image(self):
        _BLACK = 0
        _WHITE = 255
        _GRAY = round((_WHITE + _BLACK) / 2)

        # Generate a 2D grid of coordinates
        x, y = numpy.mgrid[
            -self.grating_size : self.grating_size + 1.0,
            -self.grating_size : self.grating_size + 1.0,
        ]

        # Calculate the distance from the center for each pixel
        edge_control_dist = numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2))
        edge_control = numpy.divide(edge_control_dist, self.pixel_per_degree)

        # Create a matrix (alpha_channel) that is 255 (white) within the inner_radius and exponentially fades to 0 as the radius increases
        overrado = numpy.nonzero(edge_control > self.inner_radius)
        d = self.grating_size * 2 + 1
        annulus = numpy.ones((d, d))

        annulus[overrado] *= numpy.exp(
            -1
            * ((edge_control[overrado] - self.inner_radius) * self.pixel_per_degree)
            ** 2
            / (2 * (self.smooth_sd**2))
        )
        alpha_channel = annulus * _WHITE

        # Generate the grating pattern, which is a centered and tilted sinusoidal matrix
        spatial_component = (
            2
            * math.pi
            * self.spatial_freq
            * (y * numpy.sin(self.angle) + x * numpy.cos(self.angle))
        )
        gabor_sti = _GRAY * (
            1 + self.grating_contrast * numpy.cos(spatial_component + self.phase)
        )

        # Set pixels outside the grating size to gray
        gabor_sti[edge_control_dist > self.grating_size] = _GRAY

        # Add Gaussian white noise to the grating
        noise = numpy.random.normal(loc=0, scale=self.std, size=gabor_sti.shape)
        noisy_gabor_sti = gabor_sti + noise

        # Expand the grating to have three colors andconcatenate it with alpha_channel
        gabor_sti_final = numpy.repeat(noisy_gabor_sti[:, :, numpy.newaxis], 3, axis=-1)
        gabor_sti_final_with_alpha = numpy.concatenate(
            (gabor_sti_final, alpha_channel[:, :, numpy.newaxis]), axis=-1
        )
        gabor_sti_final_with_alpha_image = Image.fromarray(
            gabor_sti_final_with_alpha.astype(numpy.uint8)
        )

        # Create a background image filled with gray
        background = numpy.full((self.size, self.size, 3), _GRAY, dtype=numpy.uint8)
        final_image = Image.fromarray(background)

        # Paste the grating into the final image: paste the grating into a bounding box and apply the alpha channel as a mask
        center_x, center_y = self.size // 2, self.size // 2
        bounding_box = (center_x - self.grating_size, center_y - self.grating_size)
        final_image.paste(
            gabor_sti_final_with_alpha_image,
            box=bounding_box,
            mask=gabor_sti_final_with_alpha_image,
        )

        # Sum the image over color channels
        final_image_np = numpy.array(final_image, dtype=numpy.float16)
        image = numpy.sum(final_image_np, axis=2)

        # Crop the image if crop_f is specified
        if self.crop_f:
            image = image[self.crop_f : -self.crop_f, self.crop_f : -self.crop_f]

        return image


def create_grating_pairs(n_trials, stimuli_pars):
    '''
    Create input stimuli gratings. Both the refence and the target are jitted by the same angle. 
    Input:
       stimuli pars
       n_trials - batch size
    
    Output:
        dictionary containing reference target and label 
    
    '''
    
    #initialise empty arrays
    ref_ori = stimuli_pars.ref_ori
    offset = stimuli_pars.offset
    data_dict = {'ref':[], 'target': [], 'label':[]}
    for i in range(n_trials):
        uniform_dist_value = numpy.random.uniform(low = 0, high = 1)
        if  uniform_dist_value < 0.5:
            target_ori = ref_ori - offset
            label = 1
        else:
            target_ori = ref_ori + offset
            label = 0
        jitter_val = stimuli_pars.jitter_val
        jitter = numpy.random.uniform(low = -jitter_val, high = jitter_val)
        #create reference grating
        ref = BW_Grating(ori_deg = ref_ori, jitter=jitter, stimuli_pars = stimuli_pars).BW_image().ravel()

        #create target grating
        target = BW_Grating(ori_deg = target_ori, jitter=jitter, stimuli_pars = stimuli_pars).BW_image().ravel()
        
        data_dict['ref'].append(ref)
        data_dict['target'].append(target)
        data_dict['label'].append(label)
        #data_dict = {'ref':ref, 'target': target, 'label':label}
        
    data_dict['ref'] = np.asarray(data_dict['ref'])
    data_dict['target'] = np.asarray(data_dict['target'])
    data_dict['label'] = np.asarray(data_dict['label'])

    return data_dict


def create_stimuli(stimuli_pars, ref_ori, number=10, jitter_val=5):
    all_stimuli = []

    for i in range(0, number):
        jitter = numpy.random.uniform(-jitter_val, jitter_val, 1)

        # create reference grating
        ref = (
            BW_Grating(ori_deg=ref_ori, stimuli_pars=stimuli_pars, jitter=jitter)
            .BW_image()
            .ravel()
        )
        all_stimuli.append(ref)

    return np.vstack([all_stimuli])


### ANALYSIS RESULTS
def param_ratios(results_file):
    results = pd.read_csv(results_file, header=0)
    res = results.to_numpy()
    Js = res[:, 2:6]
    ss = res[:, 6:10]
    print(results.columns[2:6])
    print("J ratios = ", np.array((Js[-1, :] / Js[0, :] - 1) * 100, dtype=int))
    print(results.columns[6:10])
    print(ss[-1, :] / ss[0, :])
    print("s ratios = ", np.array((ss[-1, :] / ss[0, :] - 1) * 100, dtype=int))


make_J2x2_o = lambda Jee, Jei, Jie, Jii: np.array([[Jee, -Jei], [Jie, -Jii]])


def constant_to_vec(c_E, c_I, ssn, sup=False):
    edge_length = ssn.grid_pars.gridsize_Nx

    matrix_E = np.ones((edge_length, edge_length)) * c_E
    vec_E = np.ravel(matrix_E)

    matrix_I = np.ones((edge_length, edge_length)) * c_I
    vec_I = np.ravel(matrix_I)

    constant_vec = np.hstack((vec_E, vec_I, vec_E, vec_I))

    if sup == False and ssn.phases == 4:
        constant_vec = np.kron(np.asarray([1, 1]), constant_vec)

    if sup:
        constant_vec = np.hstack((vec_E, vec_I))

    return constant_vec


def x_greater_than(x, constant, slope, height):
    return np.maximum(0, (x * slope - (1 - height)))


def x_less_than(x, constant, slope, height):
    return constant * (x**2)


def leaky_relu(x, R_thresh, slope, height=0.15):
    """Customized relu function for regulating the rates"""
    constant = height / (R_thresh**2)
    # jax.lax.cond(cond, func1, func2, args - same for both functions) meaning if cond then apply func1, if not then apply func2 with the given arguments
    y = jax.lax.cond(
        (x < R_thresh), x_less_than, x_greater_than, x, constant, slope, height
    )

    return y


def binary_loss(n, x):
    return -(n * np.log(x) + (1 - n) * np.log(1 - x))
