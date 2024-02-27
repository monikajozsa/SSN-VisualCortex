import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy
import os
import shutil
from datetime import datetime

from util_gabor import BW_Grating, BW_image_jit


def test_uniformity(numbers, num_bins=18, alpha=0.25):
    '''
    This function assesses the uniformity of 'numbers' within the range [0, 180] by dividing the range into 'num_bins' 
    equally sized bins and comparing the observed frequencies in these bins against the expected frequencies for a uniform 
    distribution. The test is performed at a significance level 'alpha'.

    Parameters:
    - numbers (list or array-like): The set of numbers to test for uniformity.
    - num_bins (int): The number of bins to use for dividing the range [0, 180]. Default is 10.
    - alpha (float): The significance level for the chi-squared test. Default is 0.1.

    Returns:
    - bool: False if the null hypothesis (that the distribution is uniform) is rejected, True otherwise.
    '''

    n = len(numbers)
    expected_freq = n / num_bins
    observed_freq = [0] * num_bins
    
    for number in numbers:
        if 0 <= number <= 180:  # Ensure the number is within the desired range
            bin_index = int((number / 180) * num_bins)
            observed_freq[bin_index] += 1
    
    chi_squared_stat = sum(((obs - expected_freq) ** 2) / expected_freq for obs in observed_freq)
    
    # Chi-square table for degrees of freedom 1-20 and significance level 0.1, 0.05, 0.025 and 0.01
    sig_levels = numpy.array([0.25, 0.1, 0.05, 0.025, 0.01])
    row_ind = num_bins-1 -1 # degree of freedom is bins -1 and index starts from 0
    col_ind = numpy.argmin(numpy.abs(numpy.ones_like(sig_levels)*alpha-sig_levels))
    
    ChiSquareTable = numpy.array([[1.323,2.706, 3.841, 5.024, 6.635],
                                [2.773,4.605, 5.991, 7.378, 9.210],
                                [4.108, 6.251, 7.815, 9.348, 11.345],
                                [5.385, 7.779, 9.488, 11.143, 13.277],
                                [6.626, 9.236, 11.070, 12.833, 15.086],
                                [7.841, 10.645, 12.592, 14.449, 16.812],
                                [9.037, 12.017, 14.067, 16.013, 18.475],
                                [10.219, 13.362, 15.507, 17.535, 20.090],
                                [11.389, 14.684, 16.919, 19.023, 21.666],
                                [12.549, 15.987, 18.307, 20.483, 23.209],
                                [13.701, 17.275, 19.675, 21.920, 24.725],
                                [14.845, 18.549, 21.026, 23.337, 26.217],
                                [15.984, 19.812, 22.362, 24.736, 27.688],
                                [17.117, 21.064, 23.685, 26.119, 29.141],
                                [18.245, 22.307, 24.996, 27.488, 30.578],
                                [19.369, 23.542, 26.296, 28.845, 32.000],
                                [20.489, 24.769, 27.587, 30.191, 33.409],
                                [21.605, 25.989, 28.869, 31.526, 34.805],
                                [22.718, 27.204, 30.144, 32.852, 36.191],
                                [23.828, 28.412, 31.410, 34.170, 37.566]])
    
    chi_squared_critical = ChiSquareTable[row_ind,col_ind]

    if chi_squared_stat <= chi_squared_critical and all(numpy.array(observed_freq) > expected_freq/3) and all(numpy.array(observed_freq) < expected_freq*3):
        #Fail to reject the null hypothesis: The distribution may be uniform.
        return True
    else:
        #Reject the null hypothesis: The distribution is not uniform.        
        return False


def cosdiff_ring(d_x, L):
    '''
    Calculate the cosine-based distance.
    Parameters:
    d_x: The difference in the angular position.
    L: The total angle.
    '''
    # Calculate the cosine of the scaled angular difference
    cos_angle = np.cos(d_x * 2 * np.pi / L)

    # Calculate scaled distance
    distance = np.sqrt( (1 - cos_angle) * 2) * L / (2 * np.pi)

    return distance


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
    verbose=True
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
    and r as its first row.  If r is not given, ``r == conjugate(c)``.
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
    new_J = np.exp(np.array(J_s, dtype = float)) * signs

    return new_J


def create_grating_pairs(stimuli_pars, batch_size, jit_inp_all= None):
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
        
        # Create reference and target gratings (if jit_inp_all is given then use the jit-compatible version of BW_image)
        if jit_inp_all is None:
            ref = BW_Grating(ori_deg = ref_ori, jitter=jitter, stimuli_pars = stimuli_pars).BW_image().ravel()
            target = BW_Grating(ori_deg = target_ori, jitter=jitter, stimuli_pars = stimuli_pars).BW_image().ravel()
        else:
            x = jit_inp_all[5]
            y = jit_inp_all[6]
            alpha_channel = jit_inp_all[7]
            mask_jax = jit_inp_all[8]
            background = jit_inp_all[9]
            roi =jit_inp_all[10]
            ref = BW_image_jit(jit_inp_all[0:5],x,y,alpha_channel,mask_jax, background, roi, ref_ori, jitter, 1)
            target = BW_image_jit(jit_inp_all[0:5],x,y,alpha_channel,mask_jax, background, roi, target_ori, jitter, 1)
        
        data_dict['ref'].append(ref)
        data_dict['target'].append(target)
        data_dict['label'].append(label)
        
    data_dict['ref'] = np.asarray(data_dict['ref'])
    data_dict['target'] = np.asarray(data_dict['target'])
    data_dict['label'] = np.asarray(data_dict['label'])

    return data_dict


def generate_random_pairs(min_value, max_value, min_distance, max_distance=None, batch_size=1, tot_angle=180, numRnd_ori1=1):
    '''
    Create batch_size number of pairs of numbers between min_value and max_value with minimum distance min_distance and maximum distance max_distance.
    If tot_angle is provided, values wrap around between 0 and tot_angle.
    '''
    if max_distance==None:
        max_distance = max_value - min_value

    # Generate the first numbers
    rnd_numbers = numpy.random.uniform(min_value, max_value, numRnd_ori1) #numpy.random.randint(low=min_value, high=max_value, size=numRnd_ori1, dtype=int)
    num1 = numpy.repeat(rnd_numbers, int(batch_size/numRnd_ori1))

    # Generate a random distance within specified range
    random_distance = numpy.random.choice([-1, 1], batch_size) * numpy.random.uniform(min_distance,max_distance ,batch_size) #numpy.random.randint(low=min_distance,high=max_distance, size=batch_size, dtype=int)

    # Generate the second numbers with correction if they are out of the specified range
    num2 = num1 - random_distance #order and sign are important!

    # Create a mask where flip_numbers equals 1
    swap_numbers = numpy.random.choice([0, 1], batch_size) 
    mask = swap_numbers == 1

    # Swap values where mask is True
    # We'll use a temporary array to hold the values of num1 where the mask is True
    temp_num1 = np.copy(num1[mask])
    num1[mask] = num2[mask]
    num2[mask] = temp_num1
    random_distance[mask] = -random_distance[mask]
    
    # Apply wrap-around logic
    #num2[num2 > tot_angle] = num2[num2 > tot_angle] - tot_angle
    #num2[num2 < 0] = num2[num2 < 0] + tot_angle
    
    return np.array(num1), np.array(num2), random_distance


def create_grating_pretraining(pretrain_pars, batch_size, jit_inp_all, numRnd_ori1=1):
    '''
    Create input stimuli gratings for pretraining by randomizing ref_ori for both reference and target (with random difference between them)
    Output:
        dictionary containing grating1, grating2 and difference between gratings that is calculated from features
    '''
    
    # Initialise empty data dictionary - names are not describing the purpose of the variables but this allows for reusing code
    data_dict = {'ref': [], 'target': [], 'label':[]}

    # Randomize orientations for stimulus 1 and stimulus 2
    L_ring = 180
    min_ori_dist = pretrain_pars.min_ori_dist
    max_ori_dist = pretrain_pars.max_ori_dist
    ori1, ori2, ori_diff = generate_random_pairs(min_value=30, max_value=150, min_distance=min_ori_dist, max_distance=max_ori_dist, batch_size=batch_size, tot_angle=L_ring, numRnd_ori1=numRnd_ori1)

    x = jit_inp_all[5]
    y = jit_inp_all[6]
    alpha_channel = jit_inp_all[7]
    mask_jax = jit_inp_all[8]
    background = jit_inp_all[9]
    roi =jit_inp_all[10]
    for i in range(batch_size):
        # Generate stimulus1 and stimulus2 with no jitter and no noise (seed needs to be randomized if we add noise!)
        stim1 = BW_image_jit(jit_inp_all[0:5], x, y, alpha_channel, mask_jax, background, roi, ori1[i], jitter=0, seed=1)
        stim2 = BW_image_jit(jit_inp_all[0:5], x, y, alpha_channel, mask_jax, background, roi, ori2[i], jitter=0, seed=1)

        data_dict['ref'].append(stim1)
        data_dict['target'].append(stim2)
    
    data_dict['ref']=np.asarray(data_dict['ref'])
    data_dict['target']=np.asarray(data_dict['target'])

    # Define label as the normalized signed difference in angle using cosdiff_ring
    label = np.zeros_like(ori_diff)
    data_dict['label'] = label.at[ori_diff > 0].set(1)

    #data_dict['label'] = ori_diff #np.sign(ori_diff) * cosdiff_ring(np.abs(ori_diff), L_ring) / cosdiff_ring(max_ori_dist - min_ori_dist, L_ring)
   
    return data_dict


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


def save_code():
    '''
    This code is used to save code files to make results replicable.
    1) It copies specific code files into a folder called 'script'
    3) Returns the path to save the results into
    '''
    # Get the current date
    current_date = datetime.now().strftime("%b%d")

    # Create a folder name based on the current date
    folder_name = f"results/{current_date}_v"

    # Find the next available script version
    version = 0
    while os.path.exists(f"{folder_name}{version}"):
        version += 1

    # Create the folder for the results
    final_folder_path = f"{folder_name}{version}"
    os.makedirs(final_folder_path)

    # Create a subfolder for the scripts
    subfolder_script_path = f"{folder_name}{version}/scripts"
    os.makedirs(subfolder_script_path)

    # Get the path to the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))

    # Copy files into the folder
    file_names = ['main.py', 'util_gabor.py', 'pretraining_supp.py', 'parameters.py', 'training.py', 'model.py', 'util.py', 'SSN_classes.py', 'analysis.py', 'visualization.py']
    for file_name in file_names:
        source_path = os.path.join(script_directory, file_name)
        destination_path = os.path.join(subfolder_script_path, file_name)
        shutil.copyfile(source_path, destination_path)

    print(f"Script files copied successfully to: {subfolder_script_path}")

    # return path (inclusing filename) to save results into
    results_filename = os.path.join(final_folder_path,f"{current_date}_v{version}_results.csv")

    return results_filename, final_folder_path
