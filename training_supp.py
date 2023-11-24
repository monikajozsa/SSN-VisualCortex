import numpy
import jax
import jax.numpy as np
from jax import vmap
from SSN_classes import SSN_mid_local, SSN_sup

from util import sep_exponentiate, constant_to_vec, binary_loss, sigmoid
from model import two_layer_model


rng_noise = numpy.random.default_rng(10)
def generate_noise(sig_noise,  batch_size, length):
    '''
    Creates vectors of neural noise. Function creates N vectors, where N = batch_size, each vector of length = length. 
    '''
    return  rng_noise.normal(size = (batch_size, length))*sig_noise #sig_noise*numpy.random.randn(batch_size, length)


def ori_discrimination(ssn_layer_pars, readout_pars, constant_pars, conv_pars, loss_pars, train_data, noise_ref, noise_target):
    
    '''
    Orientation discrimanation task ran using SSN two-layer model.The reference and target are run through the two layer model individually. 
    Inputs:
        individual parameters - having taken logs of differentiable parameters
        noise_type: select different noise models
    Outputs:
        losses to take gradient with respect to
        sig_input, x: I/O values for sigmoid layer
    '''
    
    logJ_2x2_m = ssn_layer_pars['J_2x2_m']
    logJ_2x2_s = ssn_layer_pars['J_2x2_s']
    c_E = ssn_layer_pars['c_E']
    c_I = ssn_layer_pars['c_I']
    f_E = np.exp(ssn_layer_pars['f_E'])
    f_I = np.exp(ssn_layer_pars['f_I'])
    kappa_pre = np.tanh(ssn_layer_pars['kappa_pre'])
    kappa_post = np.tanh(ssn_layer_pars['kappa_post'])
    
    w_sig = readout_pars['w_sig']
    b_sig = readout_pars['b_sig']
    
    J_2x2_m = sep_exponentiate(logJ_2x2_m)
    J_2x2_s = sep_exponentiate(logJ_2x2_s)
    ssn_mid=SSN_mid_local(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_m, filter_pars=constant_pars.filter_pars, J_2x2=J_2x2_m, gE = constant_pars.gE[0], gI=constant_pars.gI[0], ori_map = constant_pars.ssn_ori_map)
    ssn_sup=SSN_sup(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_s, J_2x2=J_2x2_s, s_2x2=constant_pars.s_2x2, sigma_oris = constant_pars.sigma_oris, ori_map = constant_pars.ssn_ori_map, train_ori = constant_pars.ref_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)
    
    #Create vector of extrasynaptic constants
    constant_vector_mid = constant_to_vec(c_E = c_E, c_I = c_I, ssn= ssn_mid)
    constant_vector_sup = constant_to_vec(c_E = c_E, c_I = c_I, ssn = ssn_sup, sup=True)
    
    #Run reference through two layer model
    r_ref, [r_max_ref_mid, r_max_ref_sup], [avg_dx_ref_mid, avg_dx_ref_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = two_layer_model(ssn_mid, ssn_sup, train_data['ref'], conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I)
    
    #Run target through two layer model
    r_target, [r_max_target_mid, r_max_target_sup], [avg_dx_target_mid, avg_dx_target_sup], _, _= two_layer_model(ssn_mid, ssn_sup, train_data['target'], conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I)
    
    noise_type = constant_pars.noise_type
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
    
    #Find difference between reference and target
    delta_x = r_ref - r_target
    
    #Multiply delta by sigmoid lyer weights and add bias
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


#Parallelize orientation discrimination task
vmap_ori_discrimination = vmap(ori_discrimination, in_axes = ({'J_2x2_m': None, 'J_2x2_s':None, 'c_E':None, 'c_I':None, 'f_E':None, 'f_I':None, 'kappa_pre':None, 'kappa_post':None}, {'w_sig':None, 'b_sig':None}, None, None, None, {'ref':0, 'target':0, 'label':0}, 0, 0) )
jit_ori_discrimination = jax.jit(vmap_ori_discrimination, static_argnums = [2, 3, 4])


def training_loss(ssn_layer_pars, readout_pars, constant_pars, conv_pars, loss_pars, train_data, noise_ref, noise_target):
    
    '''
    Run orientation discrimination task on given batch of data. Returns losses averaged over the trials within the batch. Function over which the gradient is taken.
    '''
    
    #Run orientation discrimination task
    total_loss, all_losses, pred_label, sig_input, x, max_rates = jit_ori_discrimination(ssn_layer_pars, readout_pars, constant_pars, conv_pars, loss_pars, train_data, noise_ref, noise_target)
    
    #Total loss to take gradient with respect to 
    loss= np.mean(total_loss)
    
    #Find mean of different losses
    all_losses = np.mean(all_losses, axis = 0)
    
    #Find maximum rates across trials
    max_rates = [item.max() for item in max_rates]
    
    #Calculate accuracy 
    true_accuracy = np.sum(train_data['label'] == pred_label)/len(train_data['label'])  
    
    return loss, [all_losses, true_accuracy, sig_input, x, max_rates]


def save_params_dict_two_stage(ssn_layer_pars, readout_pars, true_acc, epoch ):
    
    '''
    Assemble trained parameters and epoch information into single dictionary for saving
    Inputs:
        dictionaries containing trained parameters
        other epoch parameters (accuracy, epoch number)
    Outputs:
        single dictionary concatenating all information to be saved
    '''
    
    
    save_params = {}
    save_params= dict(epoch = epoch, val_accuracy= true_acc)
    
    
    J_2x2_m = sep_exponentiate(ssn_layer_pars['J_2x2_m'])
    Jm = dict(J_EE_m= J_2x2_m[0,0], J_EI_m = J_2x2_m[0,1], 
                              J_IE_m = J_2x2_m[1,0], J_II_m = J_2x2_m[1,1])
            
    J_2x2_s = sep_exponentiate(ssn_layer_pars['J_2x2_s'])
    Js = dict(J_EE_s= J_2x2_s[0,0], J_EI_s = J_2x2_s[0,1], 
                              J_IE_s = J_2x2_s[1,0], J_II_s = J_2x2_s[1,1])
            
    save_params.update(Jm)
    save_params.update(Js)
    
    if 'c_E' in ssn_layer_pars.keys():
        save_params['c_E'] = ssn_layer_pars['c_E']
        save_params['c_I'] = ssn_layer_pars['c_I']

   
    if 'sigma_oris' in ssn_layer_pars.keys():

        if np.shape(ssn_layer_pars['sigma_oris'])==(2,2):
            save_params['sigma_orisEE'] = np.exp(ssn_layer_pars['sigma_oris'][0,0])
            save_params['sigma_orisEI'] = np.exp(ssn_layer_pars['sigma_oris'][0,1])
        else:
            sigma_oris = dict(sigma_orisE = np.exp(ssn_layer_pars['sigma_oris'][0]), sigma_orisI = np.exp(ssn_layer_pars['sigma_oris'][1]))
            save_params.update(sigma_oris)
      
    if 'kappa_pre' in ssn_layer_pars.keys():
        if np.shape(ssn_layer_pars['kappa_pre']) == (2,2):
            save_params['kappa_preEE'] = np.tanh(ssn_layer_pars['kappa_pre'][0,0])
            save_params['kappa_preEI'] = np.tanh(ssn_layer_pars['kappa_pre'][0,1])
            save_params['kappa_postEE'] = np.tanh(ssn_layer_pars['kappa_post'][0,0])
            save_params['kappa_postEI'] = np.tanh(ssn_layer_pars['kappa_post'][0,1])


        else:
            save_params['kappa_preE'] = np.tanh(ssn_layer_pars['kappa_pre'][0])
            save_params['kappa_preI'] = np.tanh(ssn_layer_pars['kappa_pre'][1])
            save_params['kappa_postE'] = np.tanh(ssn_layer_pars['kappa_post'][0])
            save_params['kappa_postI'] = np.tanh(ssn_layer_pars['kappa_post'][1])
    
    if 'f_E' in ssn_layer_pars.keys():

        save_params['f_E'] = np.exp(ssn_layer_pars['f_E'])#*f_sigmoid(ssn_layer_pars['f_E'])
        save_params['f_I'] = np.exp(ssn_layer_pars['f_I'])
        
    #Add readout parameters
    save_params.update(readout_pars)

    return save_params
