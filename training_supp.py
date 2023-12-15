import jax
import jax.numpy as np
from jax import vmap

from SSN_classes import SSN_mid, SSN_sup
from util import sep_exponentiate, binary_loss, sigmoid
from model import evaluate_model_response


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
    if 'kappa_pre' in ssn_layer_pars_dict:
        kappa_pre = np.tanh(ssn_layer_pars_dict['kappa_pre'])
        kappa_post = np.tanh(ssn_layer_pars_dict['kappa_post'])
    else:
        kappa_pre = constant_pars.ssn_layer_pars.kappa_pre
        kappa_post = constant_pars.ssn_layer_pars.kappa_post
    w_sig = readout_pars_dict['w_sig']
    b_sig = readout_pars_dict['b_sig']
    p_local_s = constant_pars.ssn_layer_pars.p_local_s
    s_2x2 = constant_pars.ssn_layer_pars.s_2x2_s
    sigma_oris = constant_pars.ssn_layer_pars.sigma_oris
    ref_ori = constant_pars.stimuli_pars.ref_ori
    
    J_2x2_m = sep_exponentiate(log_J_2x2_m)
    J_2x2_s = sep_exponentiate(log_J_2x2_s)   

    loss_pars = constant_pars.loss_pars
    conv_pars = constant_pars.conv_pars
    # Create middle and superficial SSN layers *** this is something that would be great to change - to call the ssn classes from inside the training
    ssn_mid=SSN_mid(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, J_2x2=J_2x2_m)
    ssn_sup=SSN_sup(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, J_2x2=J_2x2_s, p_local=p_local_s, oris=constant_pars.oris, s_2x2=s_2x2, sigma_oris = sigma_oris, ori_dist = constant_pars.ori_dist, train_ori = ref_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)
    
    #Run reference and targetthrough two layer model
    r_ref, [r_max_ref_mid, r_max_ref_sup], [avg_dx_ref_mid, avg_dx_ref_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'], conv_pars, c_E, c_I, f_E, f_I, constant_pars.gabor_filters)
    
    #Select the middle grid
    N_readout=constant_pars.readout_grid_size
    N_grid=constant_pars.grid_pars.gridsize_Nx
    start=((N_grid-N_readout)/2)
    start=int(start)
    r_ref_2D=np.reshape(r_ref,(N_grid,N_grid))
    r_ref_box = jax.lax.dynamic_slice(r_ref_2D, (start, start), (N_readout,N_readout)).ravel()
    
    #Add noise
    r_ref_box = r_ref_box + noise_ref*np.sqrt(jax.nn.softplus(r_ref_box))
    
    #repeat for target if not pretraining
    if not pretraining:
        r_target, [r_max_target_mid, r_max_target_sup], [avg_dx_target_mid, avg_dx_target_sup], _, _= evaluate_model_response(ssn_mid, ssn_sup, train_data['target'], conv_pars, c_E, c_I, f_E, f_I, constant_pars.gabor_filters)
        r_target_2D=np.reshape(r_target,(N_grid,N_grid))
        r_target_box = jax.lax.dynamic_slice(r_target_2D, (start, start), (N_readout,N_readout)).ravel()
        r_target_box = r_target_box + noise_target*np.sqrt(jax.nn.softplus(r_target_box))
    
    #Calculate readout loss
    if pretraining:
        sig_input = np.dot(w_sig, r_ref_box) + b_sig  
        sig_output = sig_input
        loss_readout = np.mean(np.abs(sig_output-train_data['label']))
        pred_label = None
        loss_avg_dx = loss_pars.lambda_dx*(avg_dx_ref_mid  + avg_dx_ref_sup )/4
        loss_r_max =  loss_pars.lambda_r_max*(r_max_ref_mid + r_max_ref_sup )/4
    else:
        #Multiply (reference - target) by sigmoid layer weights, add bias and apply sigmoid funciton
        sig_input = np.dot(w_sig, (r_ref_box - r_target_box)) + b_sig     
        sig_output = sigmoid(sig_input)
        #calculate readout loss and the predicted label
        loss_readout = binary_loss(train_data['label'], sig_output)
        pred_label = np.round(sig_output) 
        loss_avg_dx = loss_pars.lambda_dx*(avg_dx_ref_mid + avg_dx_target_mid + avg_dx_ref_sup + avg_dx_target_sup )/4
        loss_r_max =  loss_pars.lambda_r_max*(r_max_ref_mid + r_max_target_mid + r_max_ref_sup + r_max_target_sup )/4
    loss_w = loss_pars.lambda_w*(np.linalg.norm(w_sig)**2)
    loss_b = loss_pars.lambda_b*(b_sig**2)
        
    #Combine all losses    
    loss = loss_readout + loss_w + loss_b +  loss_avg_dx + loss_r_max
    all_losses = np.vstack((loss_readout, loss_avg_dx, loss_r_max, loss_w, loss_b, loss))
    
    return loss, all_losses, pred_label, sig_input, sig_output,  [max_E_mid, max_I_mid, max_E_sup, max_I_sup]


def training_loss(ssn_layer_pars_dict, readout_pars_dict, constant_pars, train_data, noise_ref, noise_target, jit_on=True):
    
    '''
    Run orientation discrimination task on given batch of data. Returns losses averaged over the trials within the batch. Function over which the gradient is taken.
    '''
    #Parallelize orientation discrimination task
    vmap_ori_discrimination = vmap(ori_discrimination, in_axes = (None, None, None, {'ref':0, 'target':0, 'label':0}, 0, 0) )
    jit_ori_discrimination = jax.jit(vmap_ori_discrimination, static_argnums = [2])


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
        sig_output_clipped = np.clip(sig_output, -1 + 1e-6, 1 - 1e-6)
        label_clipped = np.clip(train_data['label'], -1 + 1e-6, 1 - 1e-6)
        true_accuracy = (np.pi - np.mean(np.abs(np.arccos(sig_output_clipped) - np.arccos(label_clipped))))/np.pi
    else:
        true_accuracy = np.sum(train_data['label'] == pred_label)/len(train_data['label'])  
    
    return loss, [all_losses, true_accuracy, sig_input, sig_output, max_rates]
