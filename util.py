import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy
import copy 
import os
import shutil
from datetime import datetime

from util_gabor import BW_Grating

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


def create_grating_pairs(stimuli_pars, batch_size):
    '''
    Create input stimuli gratings. Both the refence and the target are jitted by the same angle. 
    Input:
       stimuli pars
       batch_size - batch size
    
    Output:
        dictionary containing reference target and label 
    '''
    
    #initialise empty arrays
    ref_ori = stimuli_pars.ref_ori
    offset = stimuli_pars.offset
    data_dict = {'ref':[], 'target': [], 'label':[]}

    for _ in range(batch_size):
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
        
    data_dict['ref'] = np.asarray(data_dict['ref'])
    data_dict['target'] = np.asarray(data_dict['target'])
    data_dict['label'] = np.asarray(data_dict['label'])

    return data_dict


def generate_random_pairs(min_value, max_value, min_distance, batch_size=1):
    '''
    Create batch_size number of pairs of numbers between min_value and max_value with minumum distance min_distance.
    '''
    num1 = numpy.random.uniform(min_value, max_value,batch_size)
    random_distance = numpy.random.uniform(min_distance, max_value - min_value,batch_size)
    num2 = num1 + random_distance
    num2[num2 > max_value] = num2[num2 > max_value] - (max_value - min_value)

    return num1, num2


def create_grating_pretraining(stimuli_pars, batch_size):
    '''
    Create input stimuli gratings for pretraining by randomizing ref_ori, k, inner_radius, outer_radius.
    Output:
        dictionary containing grating1, grating2 and difference between gratings that is calculated from features
    '''
    
    #initialise empty data dictionary - names are not describing the purpose of the variables but this allows for reusing code
    data_dict = {'ref': [], 'target': [], 'label':[]}

    #randomize stimuli features
    #inner_radius1, inner_radius2 = generate_random_pairs(2, 3, 0.2, batch_size)
    #spac_freq1, spac_freq2 = generate_random_pairs(1.5, 2.5, 0.5, batch_size)
    ori1, ori2 = generate_random_pairs(0, 180, 5, batch_size)

    stimuli_pars1 = copy.copy(stimuli_pars)
    #stimuli_pars2 = copy.copy(stimuli_pars)

    for i in range(batch_size):
        #define features of stimulus1 and stimulus 2
        #stimuli_pars1.inner_radius = inner_radius1[i]
        #stimuli_pars1.outer_radius = inner_radius1[i]+5
        #stimuli_pars1.k = spac_freq1[i]
        stimuli_pars1.ref_ori = ori1[i]
        
        #stimuli_pars2.inner_radius = inner_radius1[i] #simplified task - pairs have matching radius and spacial frequesncy
        #stimuli_pars2.outer_radius = inner_radius1[i]+5
        #stimuli_pars2.k = spac_freq1[i]
        #stimuli_pars2.ref_ori = ori2[0]

        #generate stimulus1 and stimulus2
        stim1 = BW_Grating(ori_deg = stimuli_pars1.ref_ori, jitter=0, stimuli_pars = stimuli_pars1).BW_image().ravel()
        #stim2 = BW_Grating(ori_deg = stimuli_pars1.ref_ori, jitter=0, stimuli_pars = stimuli_pars2).BW_image().ravel()
        data_dict['ref'].append(stim1)
        data_dict['target'].append(stim1)
    
    data_dict['ref']=np.asarray(data_dict['ref'])
    data_dict['target']=np.asarray(data_dict['target'])
    
    ## *** simplified task - just orientation difference
    #spac_freq_diff=numpy.abs(spac_freq1-spac_freq2)/1.5**2
    #inner_radius_diff = numpy.abs(inner_radius1-inner_radius2)/1.5**2
    ori_cos=numpy.cos(ori1 * numpy.pi/180)
    #ori_sin=numpy.sqrt(numpy.sin(ori1[0] * numpy.pi/180))
    ## we need to add the sin part as a separate dimension so that the loss is calculated from the difference in cos and sin
    data_dict['label']= ori_cos#*numpy.ones_like(ori1) #+ spac_freq_diff + inner_radius_diff 
   
    return data_dict


make_J2x2_o = lambda Jee, Jei, Jie, Jii: np.array([[Jee, -Jei], [Jie, -Jii]])

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

#binary cross entropy
def binary_loss(n, x):
    return -(n * np.log(x) + (1 - n) * np.log(1 - x))


def save_code():
    '''
    This code is used to save code files to make results replicable.
    1) It copies specific code files into a folder called 'script'
    3) Returns the path to save the results into
    '''
    # Get the current date
    current_date = datetime.now().strftime("%b%d")

    # Create a folder name based on the current date
    folder_name = f"results\{current_date}_v"

    # Find the next available script version
    version = 0
    while os.path.exists(f"{folder_name}{version}"):
        version += 1

    # Create the folder for the results
    final_folder_path = f"{folder_name}{version}"
    os.makedirs(final_folder_path)

    # Create a subfolder for the scripts
    subfolder_script_path = f"{folder_name}{version}\scripts"
    os.makedirs(subfolder_script_path)

    # Get the path to the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))

    # Copy files into the folder
    file_names = ['main.py', 'util_gabor.py', 'pretraining_supp.py', 'parameters.py', 'training.py', 'training_supp.py', 'model.py', 'util.py', 'SSN_classes.py', 'analysis.py', 'visualization.py']
    for file_name in file_names:
        source_path = os.path.join(script_directory, file_name)
        destination_path = os.path.join(subfolder_script_path, file_name)
        shutil.copyfile(source_path, destination_path)

    print(f"Script files copied successfully to: {script_directory}")

    # return path (inclusing filename) to save results into
    results_filename = os.path.join(final_folder_path,f"{current_date}_v{version}_results.csv")

    return results_filename, final_folder_path
