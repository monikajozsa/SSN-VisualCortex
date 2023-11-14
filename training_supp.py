import matplotlib.pyplot as plt
import jax
from jax import random
import jax.numpy as np
from jax import vmap
from torch.utils.data import DataLoader
import numpy
from SSN_classes import SSN2DTopoV1_ONOFF_local, SSN2DTopoV1
from util import create_gratings

# basic functions - consider moving them to util
def our_max(x, beta=1):
    max_val = np.log(np.sum(np.exp(x * beta))) / beta
    return max_val


def sigmoid(x, epsilon=0.01):
    """
    Introduction of epsilon stops asymptote from reaching 1 (avoids NaN)
    """
    return (1 - 2 * epsilon) * sig(x) + epsilon


def sig(x):
    return 1 / (1 + np.exp(-x))


def f_sigmoid(x, a=0.75):
    return (1.25 - a) + 2 * a * sig(x)


def exponentiate(opt_pars):
    signs = np.array([[1, -1], [1, -1]])
    J_2x2 = np.exp(opt_pars["logJ_2x2"]) * signs
    s_2x2 = np.exp(opt_pars["logs_2x2"])

    return J_2x2, s_2x2


def sep_exponentiate(J_s):
    signs = np.array([[1, -1], [1, -1]])
    new_J = np.exp(J_s) * signs

    return new_J


def leaky_relu(x, R_thresh, slope, height=0.15):
    constant = height / (R_thresh**2)

    y = jax.lax.cond(
        (x < R_thresh), x_less_than, x_greater_than, x, constant, slope, height
    )

    return y


def x_greater_than(x, constant, slope, height):
    return np.maximum(0, (x * slope - (1 - height)))


def x_less_than(x, constant, slope, height):
    return constant * (x**2)


def binary_loss(n, x):
    return -(n * np.log(x) + (1 - n) * np.log(1 - x))


def create_data(stimuli_pars, n_trials=100):
    """
    Create data for given jitter and noise value for testing (not dataloader)
    """
    data = create_gratings(stimuli_pars=stimuli_pars, n_trials=n_trials)
    train_data = next(iter(DataLoader(data, batch_size=len(data), shuffle=False)))
    train_data["ref"] = train_data["ref"].numpy()
    train_data["target"] = train_data["target"].numpy()
    train_data["label"] = train_data["label"].numpy()

    return train_data


def obtain_fixed_point(
    ssn, ssn_input, conv_pars, PLOT=False, save=None, inds=None, print_dt=False):
    r_init = np.zeros(ssn_input.shape[0])
    dt = conv_pars.dt
    xtol = conv_pars.xtol
    Tmax = conv_pars.Tmax
    verbose = conv_pars.verbose
    silent = conv_pars.silent

    # Find fixed point
    if PLOT == True:
        fp, avg_dx = ssn.fixed_point_r_plot(
            ssn_input,
            r_init=r_init,
            dt=dt,
            xtol=xtol,
            Tmax=Tmax,
            verbose=verbose,
            silent=silent,
            PLOT=PLOT,
            save=save,
            inds=inds,
            print_dt=print_dt,
        )
    else:
        fp, _, avg_dx = ssn.fixed_point_r(
            ssn_input,
            r_init=r_init,
            dt=dt,
            xtol=xtol,
            Tmax=Tmax,
            verbose=verbose,
            silent=silent,
            PLOT=PLOT,
            save=save,
        )

    avg_dx = np.maximum(0, (avg_dx - 1))

    return fp, avg_dx


def obtain_fixed_point_centre_E(
    ssn,
    ssn_input,
    conv_pars,
    Rmax_E=40,
    Rmax_I=80,
    inhibition=False,
    PLOT=False,
    save=None,
    inds=None,
    return_fp=False,):
    # Obtain fixed point
    fp, avg_dx = obtain_fixed_point(
        ssn=ssn,
        ssn_input=ssn_input,
        conv_pars=conv_pars,
        PLOT=PLOT,
        save=save,
        inds=inds,
    )

    # Apply bounding box to data
    r_box = (ssn.apply_bounding_box(fp, size=3.2)).ravel()

    # Obtain inhibitory response
    if inhibition == True:
        r_box_i = ssn.apply_bounding_box(fp, size=3.2, select="I_ON").ravel()
        r_box = [r_box, r_box_i]

    max_E = np.max(fp[: ssn.Ne])
    max_I = np.max(fp[ssn.Ne : -1])

    # r_max = np.maximum(0, (max_E/Rmax_E - 1)) + np.maximum(0, (max_I/Rmax_I - 1))
    r_max = leaky_relu(max_E, R_thresh=Rmax_E, slope=1 / Rmax_E) + leaky_relu(
        max_I, R_thresh=Rmax_I, slope=1 / Rmax_I
    )

    if return_fp == True:
        return r_box, r_max, avg_dx, fp, max_E, max_I
    else:
        return r_box, r_max, avg_dx


def test_accuracy(
    ssn_layer_pars,
    readout_pars,
    constant_ssn_pars,
    stimuli_pars,
    offset,
    ref_ori,
    sig_noise,
    save=None,
    number_trials=5,
    batch_size=5,):
    """
    Given network parameters, function generates random trials of data and calculates the accuracy per batch.
    Input:
        network parameters, number of trials and batch size of each trial
    Output:
        histogram of accuracies

    """

    all_accs = []

    for i in range(number_trials):
        testing_data = create_data(stimuli_pars, number=batch_size)

        constant_ssn_pars = generate_noise(
            constant_ssn_pars,
            sig_noise=sig_noise,
            batch_size=batch_size,
            length=readout_pars["w_sig"].shape[0],
        )

        _, _, pred_label, _, _, _ = model(
            ssn_layer_pars=ssn_layer_pars,
            readout_pars=readout_pars,
            constant_ssn_pars=constant_ssn_pars,
            data=testing_data,
            debug_flag=True,
        )

        true_accuracy = np.sum(testing_data["label"] == pred_label) / len(
            testing_data["label"]
        )
        all_accs.append(true_accuracy)

    plt.hist(all_accs)
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")

    if save:
        plt.savefig(save + ".png")

    plt.show()
    plt.close()


def generate_noise(constant_ssn_pars, sig_noise, batch_size, length, noise_type="poisson"):
    constant_ssn_pars["key"], _ = random.split(constant_ssn_pars["key"])
    constant_ssn_pars["noise_ref"] = sig_noise * jax.random.normal(
        constant_ssn_pars["key"], shape=(batch_size, length)
    )
    constant_ssn_pars["key"], _ = random.split(constant_ssn_pars["key"])
    constant_ssn_pars["noise_target"] = sig_noise * jax.random.normal(
        constant_ssn_pars["key"], shape=(batch_size, length)
    )
    return constant_ssn_pars


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


def middle_layer_fixed_point(
    ssn,
    ssn_input,
    conv_pars,
    Rmax_E=40,
    Rmax_I=80,
    inhibition=False,
    PLOT=False,
    save=None,
    inds=None,
    return_fp=False,
    print_dt=False,
):
    fp, avg_dx = obtain_fixed_point(
        ssn=ssn,
        ssn_input=ssn_input,
        conv_pars=conv_pars,
        PLOT=PLOT,
        save=save,
        inds=inds,
        print_dt=print_dt,
    )

    # Add responses from E and I neurons
    fp_E_on = ssn.select_type(fp, map_number=1)
    fp_E_off = ssn.select_type(fp, map_number=(ssn.phases + 1))

    layer_output = fp_E_on + fp_E_off

    # Find maximum rate
    max_E = np.max(np.asarray([fp_E_on, fp_E_off]))
    max_I = np.maximum(
        np.max(fp[3 * int(ssn.Ne / 2) : -1]), np.max(fp[int(ssn.Ne / 2) : ssn.Ne])
    )

    if ssn.phases == 4:
        fp_E_on_pi2 = ssn.select_type(fp, map_number=3)
        fp_E_off_pi2 = ssn.select_type(fp, map_number=7)

        # Changes
        layer_output = layer_output + fp_E_on_pi2 + fp_E_off_pi2
        max_E = np.max(np.asarray([fp_E_on, fp_E_off, fp_E_on_pi2, fp_E_off_pi2]))
        max_I = np.max(
            np.asarray([fp[int(x) : int(x) + 80] for x in numpy.linspace(81, 567, 4)])
        )

    # Loss for high rates
    r_max = leaky_relu(max_E, R_thresh=Rmax_E, slope=1 / Rmax_E) + leaky_relu(
        max_I, R_thresh=Rmax_I, slope=1 / Rmax_I
    )

    # layer_output = layer_output/ssn.phases
    if return_fp == True:
        return layer_output, r_max, avg_dx, fp, max_E, max_I
    else:
        return layer_output, r_max, avg_dx


def _new_model(logJ_2x2, logs_2x2, c_E, c_I, f_E, f_I, w_sig, b_sig, sigma_oris, kappa_pre, kappa_post, ssn_mid_ori_map, ssn_sup_ori_map, train_data, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE_m, gI_m, gE_s, gI_s, filter_pars, conv_pars, loss_pars, noise_ref, noise_target, noise_type ='poisson', train_ori = 55, debug_flag=False):
    
    '''
    SSN two-layer model. SSN layers are regenerated every run. Static arguments in jit specify parameters that stay constant throughout training. Static parameters cant be dictionaries
    Inputs:
        individual parameters - having taken logs of differentiable parameters
        noise_type: select different noise models
        debug_flag: to be used in pdb mode allowing debugging inside function
    Outputs:
        losses to take gradient with respect to
        sig_input, x: I/O values for sigmoid layer
    '''
    
    J_2x2_m = sep_exponentiate(logJ_2x2[0])
    J_2x2_s = sep_exponentiate(logJ_2x2[1])
    s_2x2_s = np.exp(logs_2x2)
    sigma_oris_s = np.exp(sigma_oris)
    
    _f_E = np.exp(f_E)
    _f_I = np.exp(f_I)
   
    kappa_pre = np.tanh(kappa_pre)
    kappa_post = np.tanh(kappa_post)

    #Initialise network
    ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE_m, gI=gI_m, ori_map = ssn_mid_ori_map)
    ssn_sup=SSN2DTopoV1(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_s, filter_pars=filter_pars, J_2x2=J_2x2_s, s_2x2=s_2x2_s, gE = gE_s, gI=gI_s, sigma_oris = sigma_oris_s, ori_map = ssn_sup_ori_map, train_ori = train_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)
    
    #Create vector using extrasynaptic constants
    constant_vector = constant_to_vec(c_E = c_E, c_I = c_I, ssn= ssn_mid)
    constant_vector_sup = constant_to_vec(c_E = c_E, c_I = c_I, ssn = ssn_sup, sup=True)
    
    #Apply Gabor filters to stimuli
    output_ref=np.matmul(ssn_mid.gabor_filters, train_data['ref'])
    output_target=np.matmul(ssn_mid.gabor_filters, train_data['target'])
    
    #Rectify output
    SSN_input_ref = np.maximum(0, output_ref) + constant_vector
    SSN_input_target = np.maximum(0, output_target) + constant_vector

    #Find fixed point for middle layer
    r_ref_mid, r_max_ref_mid, avg_dx_ref_mid, _, max_E_mid, max_I_mid = middle_layer_fixed_point(ssn_mid, SSN_input_ref, conv_pars, return_fp = True)
    r_target_mid, r_max_target_mid, avg_dx_target_mid = middle_layer_fixed_point(ssn_mid, SSN_input_target, conv_pars)
    
    #Input to superficial layer
    sup_input_ref = np.hstack([r_ref_mid*_f_E, r_ref_mid*_f_I]) + constant_vector_sup
    sup_input_target = np.hstack([r_target_mid*_f_E, r_target_mid*_f_I]) + constant_vector_sup 
    
    #Find fixed point for superficial layer
    r_ref, r_max_ref_sup, avg_dx_ref_sup, _, max_E_sup, max_I_sup= obtain_fixed_point_centre_E(ssn_sup, sup_input_ref, conv_pars, return_fp= True)
    r_target, r_max_target_sup, avg_dx_target_sup= obtain_fixed_point_centre_E(ssn_sup, sup_input_target, conv_pars)


    #Add noise
    if noise_type =='additive':
        r_ref = r_ref + noise_ref
        r_target = r_target + noise_target
        
    elif noise_type == 'multiplicative':
        r_ref = r_ref*(1 + noise_ref)
        r_target = r_target*(1 + noise_target)
        
    elif noise_type =='poisson':
        r_ref = r_ref + noise_ref*np.sqrt(jax.nn.softplus(r_ref))
        r_target = r_target + noise_target*np.sqrt(jax.nn.softplus(r_target))

    delta_x = r_ref - r_target
    sig_input = np.dot(w_sig, (delta_x)) + b_sig
    
    #Apply sigmoid function - combine ref and target
    x = sigmoid(sig_input)
    
    #Calculate losses
    loss_binary=binary_loss(train_data['label'], x)
    loss_avg_dx = loss_pars.lambda_dx*(avg_dx_ref_mid + avg_dx_target_mid + avg_dx_ref_sup + avg_dx_target_sup )/4
    loss_r_max =  loss_pars.lambda_r_max*(r_max_ref_mid + r_max_target_mid + r_max_ref_sup + r_max_target_sup )/4
    loss_w = loss_pars.lambda_w*(np.linalg.norm(w_sig)**2)
    loss_b = loss_pars.lambda_b*(b_sig**2)
    
    #Combine all losses
    loss = loss_binary + loss_w + loss_b +  loss_avg_dx + loss_r_max
    all_losses = np.vstack((loss_binary, loss_avg_dx, loss_r_max, loss_w, loss_b, loss))
    pred_label = np.round(x) 
    
    return loss, all_losses, pred_label, sig_input, x,  [max_E_mid, max_I_mid, max_E_sup, max_I_sup]


jitted_model = jax.jit(_new_model, static_argnums = [ 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27,  28, 29])#, device = jax.devices()[1])


#Vmap implementation of model function
vmap_model_jit = vmap(jitted_model, in_axes = ([None, None], None, None, None, None, None, None
                            , None, None, None, None, None, None, {'ref':0, 'target':0, 'label':0}, None, None, None, None, None, None, None, None, None, None, None, 0, 0, None, None, None) )

vmap_model = vmap(_new_model, in_axes = ([None, None], None, None, None, None, None, None
                            , None, None, None, None, None, None, {'ref':0, 'target':0, 'label':0}, None, None, None, None, None, None, None, None, None, None, None, 0, 0, None, None, None) )



def model(ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag=False):
    
    '''
    Wrapper function for model.
    Inputs: 
        parameters assembled  into dictionaries
    Output:
        output of model using unwrapped parameters
    '''
    
    #Obtain variables from dictionaries
    logJ_2x2 = ssn_layer_pars['logJ_2x2']
    c_E = constant_ssn_pars['c_E']
    c_I = constant_ssn_pars['c_I']
    f_E = constant_ssn_pars['f_E']
    f_I = constant_ssn_pars['f_I']
    
    sigma_oris = constant_ssn_pars['sigma_oris']
    kappa_pre = ssn_layer_pars['kappa_pre']
    kappa_post = ssn_layer_pars['kappa_post']
        
    w_sig = readout_pars['w_sig']
    b_sig = readout_pars['b_sig']
    
    ssn_mid_ori_map = constant_ssn_pars['ssn_mid_ori_map']
    logs_2x2 = constant_ssn_pars['logs_2x2']
    ssn_pars = constant_ssn_pars['ssn_pars']
    grid_pars = constant_ssn_pars['grid_pars']
    conn_pars_m = constant_ssn_pars['conn_pars_m']
    conn_pars_s =constant_ssn_pars['conn_pars_s']
    gE_m =constant_ssn_pars['gE'][0]
    gE_s =constant_ssn_pars['gE'][1]
    gI_m = constant_ssn_pars['gI'][0]
    gI_s = constant_ssn_pars['gI'][1]
    filter_pars = constant_ssn_pars['filter_pars']
    conv_pars = constant_ssn_pars['conv_pars']
    loss_pars = constant_ssn_pars['loss_pars']
    noise_ref = constant_ssn_pars['noise_ref']
    noise_target = constant_ssn_pars['noise_target']
    noise_type = constant_ssn_pars['noise_type']
    train_ori = constant_ssn_pars['train_ori']
    
    return vmap_model_jit(logJ_2x2, logs_2x2, c_E, c_I, f_E, f_I, w_sig, b_sig, sigma_oris, kappa_pre, kappa_post, ssn_mid_ori_map, ssn_mid_ori_map, data, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, gE_m, gI_m, gE_s, gI_s, filter_pars, conv_pars, loss_pars, noise_ref, noise_target, noise_type, train_ori, debug_flag)
    

def loss(ssn_layer_pars, readout_pars, constant_ssn_pars, data, debug_flag=False):
    """
    Function to take gradient with respect to. Output returned as two variables (jax grad takes gradient with respect to first output)
    Inputs:
        parameters assembled into dictionaries
    Ouputs:
        total loss to take gradient with respect to
    """

    total_loss, all_losses, pred_label, sig_input, x, max_rates = model(
        ssn_layer_pars=ssn_layer_pars,
        readout_pars=readout_pars,
        constant_ssn_pars=constant_ssn_pars,
        data=data,
        debug_flag=debug_flag,
    )
    loss = np.mean(total_loss)
    all_losses = np.mean(all_losses, axis=0)
    max_rates = [item.max() for item in max_rates]
    true_accuracy = np.sum(data["label"] == pred_label) / len(data["label"])

    return loss, [all_losses, true_accuracy, sig_input, x, max_rates]


def save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch):
    """
    Assemble trained parameters and epoch information into single dictionary for saving
    Inputs:
        dictionaries containing trained parameters
        other epoch parameters (accuracy, epoch number)
    Outputs:
        single dictionary concatenating all information to be saved
    """

    save_params = {}
    save_params = dict(epoch=epoch, val_accuracy=true_acc)

    J_2x2_m = sep_exponentiate(ssn_layer_pars["logJ_2x2"][0])
    Jm = dict(
        J_EE_m=J_2x2_m[0, 0],
        J_EI_m=J_2x2_m[0, 1],
        J_IE_m=J_2x2_m[1, 0],
        J_II_m=J_2x2_m[1, 1],
    )

    J_2x2_s = sep_exponentiate(ssn_layer_pars["logJ_2x2"][1])
    Js = dict(
        J_EE_s=J_2x2_s[0, 0],
        J_EI_s=J_2x2_s[0, 1],
        J_IE_s=J_2x2_s[1, 0],
        J_II_s=J_2x2_s[1, 1],
    )

    save_params.update(Jm)
    save_params.update(Js)

    if "c_E" in ssn_layer_pars.keys():
        save_params["c_E"] = ssn_layer_pars["c_E"]
        save_params["c_I"] = ssn_layer_pars["c_I"]

    if "sigma_oris" in ssn_layer_pars.keys():
        if len(ssn_layer_pars["sigma_oris"]) == 1:
            # save_params[key] = np.exp(ssn_layer_pars[key]) # 
            print("MJ error message: save_params[key] = np.exp(ssn_layer_pars[key]) would not work as key is undefined!")
        elif np.shape(ssn_layer_pars["sigma_oris"]) == (2, 2):
            save_params["sigma_orisEE"] = np.exp(ssn_layer_pars["sigma_oris"][0, 0])
            save_params["sigma_orisEI"] = np.exp(ssn_layer_pars["sigma_oris"][0, 1])
        else:
            sigma_oris = dict(
                sigma_orisE=np.exp(ssn_layer_pars["sigma_oris"][0]),
                sigma_orisI=np.exp(ssn_layer_pars["sigma_oris"][1]),
            )
            save_params.update(sigma_oris)

    if "kappa_pre" in ssn_layer_pars.keys():
        if np.shape(ssn_layer_pars["kappa_pre"]) == (2, 2):
            save_params["kappa_preEE"] = np.tanh(ssn_layer_pars["kappa_pre"][0, 0])
            save_params["kappa_preEI"] = np.tanh(ssn_layer_pars["kappa_pre"][0, 1])
            save_params["kappa_postEE"] = np.tanh(ssn_layer_pars["kappa_post"][0, 0])
            save_params["kappa_postEI"] = np.tanh(ssn_layer_pars["kappa_post"][0, 1])

        else:
            save_params["kappa_preE"] = np.tanh(ssn_layer_pars["kappa_pre"][0])
            save_params["kappa_preI"] = np.tanh(ssn_layer_pars["kappa_pre"][1])
            save_params["kappa_postE"] = np.tanh(ssn_layer_pars["kappa_post"][0])
            save_params["kappa_postI"] = np.tanh(ssn_layer_pars["kappa_post"][1])

    if "f_E" in ssn_layer_pars.keys():
        save_params["f_E"] = np.exp(
            ssn_layer_pars["f_E"]
        )  
        save_params["f_I"] = np.exp(ssn_layer_pars["f_I"])

    # Add readout parameters
    save_params.update(readout_pars)

    return save_params


def evaluate_model_response(
    ssn_mid, ssn_sup, c_E, c_I, f_E, f_I, conv_pars, train_data):
    # Create vector using extrasynaptic constants
    constant_vector = constant_to_vec(c_E=c_E, c_I=c_I, ssn=ssn_mid)
    constant_vector_sup = constant_to_vec(c_E=c_E, c_I=c_I, ssn=ssn_sup, sup=True)

    # Apply Gabor filters to stimuli
    output_mid = np.matmul(ssn_mid.gabor_filters, train_data)

    # Rectify output
    SSN_input = np.maximum(0, output_mid) + constant_vector

    # Find fixed point for middle layer
    r_ref_mid, _, _, fp, _, _ = middle_layer_fixed_point(
        ssn_mid, SSN_input, conv_pars, return_fp=True
    )

    # Input to superficial layer
    sup_input_ref = np.hstack([r_ref_mid * f_E, r_ref_mid * f_I]) + constant_vector_sup

    # Find fixed point for superficial layer
    r_ref, _, _, fp_sup, _, _ = obtain_fixed_point_centre_E(
        ssn_sup, sup_input_ref, conv_pars, return_fp=True
    )

    # Evaluate total E and I input to neurons
    W_E_mid = ssn_mid.W.at[:, 1 : ssn_mid.phases * 2 : 2].set(0)
    W_I_mid = ssn_mid.W.at[:, 0 : ssn_mid.phases * 2 - 1 : 2].set(0)
    W_E_sup = ssn_sup.W.at[:, 81:].set(0)
    W_I_sup = ssn_sup.W.at[:, :81].set(0)

    fp = np.reshape(fp, (-1, ssn_mid.Nc))
    E_mid_input = np.ravel(W_E_mid @ fp) + SSN_input
    E_sup_input = W_E_sup @ fp_sup + sup_input_ref

    I_mid_input = np.ravel(W_I_mid @ fp)
    I_sup_input = W_I_sup @ fp_sup

    return r_ref, E_mid_input, E_sup_input, I_mid_input, I_sup_input


vmap_evaluate_response = vmap(
    evaluate_model_response, in_axes=(None, None, None, None, None, None, None, 0)
)


def vmap_eval2(
    opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars, conv_pars
):
    """
    For a given value of the weights, calculate the loss for all the stimuli.
    Output:
        losses: size(n_stimuli)
        Accuracy: scalar
    """

    eval_vmap = vmap(
        model,
        in_axes=(
            {
                "b_sig": None,
                "logJ_2x2": None,
                "logs_2x2": None,
                "w_sig": None,
                "c_E": None,
                "c_I": None,
            },
            None,
            None,
            {"PERIODIC": None, "p_local": [None, None], "sigma_oris": None},
            {"ref": 0, "target": 0, "label": 0},
            {
                "conv_factor": None,
                "degree_per_pixel": None,
                "edge_deg": None,
                "k": None,
                "sigma_g": None,
            },
            {"Tmax": None, "dt": None, "silent": None, "verbose": None, "xtol": None},
        ),
    )
    losses, pred_labels = eval_vmap(
        opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars, conv_pars
    )

    accuracy = np.sum(test_data["label"] == pred_labels) / len(test_data["label"])

    return losses, accuracy


def vmap_eval3(
    opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars, conv_pars
):
    """
    Iterates through all values of 'w' to give the losses at each stimuli and weight, and the accuracy at each weight
    Output:
        losses: size(n_weights, n_stimuli )
        accuracy: size( n_weights)
    """

    eval_vmap = vmap(
        vmap_eval2,
        in_axes=(
            {
                "b_sig": None,
                "logJ_2x2": None,
                "logs_2x2": None,
                "w_sig": 0,
                "c_E": None,
                "c_I": None,
            },
            None,
            None,
            {"PERIODIC": None, "p_local": [None, None], "sigma_oris": None},
            {"ref": None, "target": None, "label": None},
            {
                "conv_factor": None,
                "degree_per_pixel": None,
                "edge_deg": None,
                "k": None,
                "sigma_g": None,
            },
            {"Tmax": None, "dt": None, "silent": None, "verbose": None, "xtol": None},
        ),
    )
    losses, accuracies = eval_vmap(
        opt_pars, ssn_pars, grid_pars, conn_pars, test_data, filter_pars, conv_pars
    )
    print(losses.shape)
    print(accuracies.shape)
    return losses, accuracies


def test_accuracies(
    opt_pars,
    ssn_pars,
    grid_pars,
    conn_pars,
    filter_pars,
    conv_pars,
    stimuli_pars,
    trials=5,
    p=0.9,
    printing=True,
):
    key = random.PRNGKey(7)
    N_neurons = 25
    accuracies = []
    key, _ = random.split(key)
    opt_pars["w_sig"] = random.normal(key, shape=(trials, N_neurons)) / np.sqrt(
        N_neurons
    )

    train_data = create_data(stimuli_pars)

    print(opt_pars["w_sig"].shape)
    val_loss, accuracies = vmap_eval3(
        opt_pars, ssn_pars, grid_pars, conn_pars, train_data, filter_pars, conv_pars
    )

    # calcualate how many accuracies are above 90
    higher_90 = np.sum(accuracies[accuracies > p]) / len(accuracies)

    if printing:
        print(
            "grating contrast = {}, jitter = {}, noise std={}, acc (% >90 ) = {}".format(
                stimuli_pars.grating_contrast,
                stimuli_pars.jitter_val,
                stimuli_pars.std,
                higher_90,
            )
        )
    return higher_90, accuracies