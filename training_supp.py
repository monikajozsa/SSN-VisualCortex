import numpy
import jax
import jax.numpy as np
from jax import vmap
from SSN_classes import SSN_mid, SSN_sup

from util import sep_exponentiate, binary_loss, sigmoid
from model import evaluate_model_response


def generate_noise(sig_noise,  batch_size, length):
    '''
    Creates vectors of neural noise. Function creates N vectors, where N = batch_size, each vector of length = length. 
    '''
    return sig_noise*numpy.random.randn(batch_size, length)


def ori_discrimination(ssn_layer_pars_dict, readout_pars_dict, constant_pars, train_data, noise_ref, noise_target): 
    
    '''
    Orientation discrimanation task ran using SSN two-layer model.The reference and target are run through the two layer model individually. 
    Inputs:
        individual parameters - having taken logs of differentiable parameters
        noise_type: select different noise models
    Outputs:
        losses to take gradient with respect to
        sig_input, x: I/O values for sigmoid layer
    '''
    pretraining = constant_pars.pretraining
    log_J_2x2_m = ssn_layer_pars_dict['log_J_2x2_m']
    log_J_2x2_s = ssn_layer_pars_dict['log_J_2x2_s']
    c_E = ssn_layer_pars_dict['c_E']
    c_I = ssn_layer_pars_dict['c_I']
    f_E = np.exp(ssn_layer_pars_dict['f_E'])
    f_I = np.exp(ssn_layer_pars_dict['f_I'])
    kappa_pre = np.tanh(ssn_layer_pars_dict['kappa_pre'])
    kappa_post = np.tanh(ssn_layer_pars_dict['kappa_post'])
    w_sig = readout_pars_dict['w_sig']
    b_sig = readout_pars_dict['b_sig']
    gI = constant_pars.ssn_layer_pars.gI_m
    gE = constant_pars.ssn_layer_pars.gE_m
    s_2x2 = constant_pars.ssn_layer_pars.s_2x2_s
    sigma_oris = constant_pars.ssn_layer_pars.sigma_oris
    ref_ori = constant_pars.stimuli_pars.ref_ori
    
    J_2x2_m = sep_exponentiate(log_J_2x2_m)
    J_2x2_s = sep_exponentiate(log_J_2x2_s)   

    loss_pars = constant_pars.loss_pars
    conv_pars = constant_pars.conv_pars
    # Create middle and superficial SSN layers *** this is something that would be great to get tid of - to call the ssn classes from inside the training
    ssn_mid=SSN_mid(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_m, filter_pars=constant_pars.filter_pars, J_2x2=J_2x2_m, gE = gE, gI=gI, ori_map = constant_pars.ssn_ori_map)
    ssn_sup=SSN_sup(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_s, J_2x2=J_2x2_s, s_2x2=s_2x2, sigma_oris = sigma_oris, ori_map = constant_pars.ssn_ori_map, train_ori = ref_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)
    
    #Run reference and targetthrough two layer model
    r_ref, [r_max_ref_mid, r_max_ref_sup], [avg_dx_ref_mid, avg_dx_ref_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'], conv_pars, c_E, c_I, f_E, f_I)
    r_target, [r_max_target_mid, r_max_target_sup], [avg_dx_target_mid, avg_dx_target_sup], _, _= evaluate_model_response(ssn_mid, ssn_sup, train_data['target'], conv_pars, c_E, c_I, f_E, f_I)
    
    #Select the middle grid
    N_readout=int(np.sqrt(len(noise_ref)))
    N_grid=constant_pars.grid_pars.gridsize_Nx
    start=int((N_grid-N_readout)/2)
    r_ref_2D=np.reshape(r_ref,(N_grid,N_grid))
    r_ref_box = jax.lax.dynamic_slice(r_ref_2D, (start, start), (N_readout,N_readout)).ravel()
    r_target_2D=np.reshape(r_target,(N_grid,N_grid))
    r_target_box = jax.lax.dynamic_slice(r_target_2D, (start, start), (N_readout,N_readout)).ravel()

    #Add noise
    r_ref_box = r_ref_box + noise_ref*np.sqrt(jax.nn.softplus(r_ref_box))
    r_target_box = r_target_box + noise_target*np.sqrt(jax.nn.softplus(r_target_box))
    
    #Multiply (reference - target) by sigmoid layer weights and add bias
    sig_input = np.dot(w_sig, (r_ref_box - r_target_box)) + b_sig 
    
    #Apply sigmoid function
    sig_output = sigmoid(sig_input)
    
    #Calculate losses
    if pretraining:
        loss_output = np.mean(np.abs(train_data['label']-sig_output))
        pred_label = None
    else:
        loss_output = binary_loss(train_data['label'], sig_output)
        pred_label = np.round(sig_output) 
    loss_avg_dx = loss_pars.lambda_dx*(avg_dx_ref_mid + avg_dx_target_mid + avg_dx_ref_sup + avg_dx_target_sup )/4
    loss_r_max =  loss_pars.lambda_r_max*(r_max_ref_mid + r_max_target_mid + r_max_ref_sup + r_max_target_sup )/4
    loss_w = loss_pars.lambda_w*(np.linalg.norm(w_sig)**2)
    loss_b = loss_pars.lambda_b*(b_sig**2)
    
    #Combine all losses    
    loss = loss_output + loss_w + loss_b +  loss_avg_dx + loss_r_max
    all_losses = np.vstack((loss_output, loss_avg_dx, loss_r_max, loss_w, loss_b, loss))
    
    return loss, all_losses, pred_label, sig_input, sig_output,  [max_E_mid, max_I_mid, max_E_sup, max_I_sup]

#Parallelize orientation discrimination task
vmap_ori_discrimination = vmap(ori_discrimination, in_axes = ({'log_J_2x2_m': None, 'log_J_2x2_s':None, 'c_E':None, 'c_I':None, 'f_E':None, 'f_I':None, 'kappa_pre':None, 'kappa_post':None}, {'w_sig':None, 'b_sig':None}, None, {'ref':0, 'target':0, 'label':0}, 0, 0) )
jit_ori_discrimination = jax.jit(vmap_ori_discrimination, static_argnums = [2])


def training_loss(ssn_layer_pars_dict, readout_pars_dict, constant_pars, train_data, noise_ref, noise_target, jit_on=True):
    
    '''
    Run orientation discrimination task on given batch of data. Returns losses averaged over the trials within the batch. Function over which the gradient is taken.
    '''
    
    #Run orientation discrimination task
    if jit_on:
        total_loss, all_losses, pred_label, sig_input, sig_output, max_rates = jit_ori_discrimination(ssn_layer_pars_dict, readout_pars_dict, constant_pars, train_data, noise_ref, noise_target)
    else:
        total_loss, all_losses, pred_label, sig_input, sig_output, max_rates = vmap_ori_discrimination(ssn_layer_pars_dict, readout_pars_dict, constant_pars, train_data, noise_ref, noise_target)
    
    #Total loss to take gradient with respect to 
    loss= np.mean(total_loss)
    
    #Find mean of different losses
    all_losses = np.mean(all_losses, axis = 0)
    
    #Find maximum rates across trials
    max_rates = [item.max() for item in max_rates]
    
    #Calculate accuracy 
    if constant_pars.pretraining:
        true_accuracy = np.sum(np.abs(train_data['label']-sig_output))/len(train_data['label'])
    else:
        true_accuracy = np.sum(train_data['label'] == pred_label)/len(train_data['label'])  
    
    return loss, [all_losses, true_accuracy, sig_input, sig_output, max_rates]

