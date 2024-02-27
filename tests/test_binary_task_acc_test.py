import numpy
import copy
import time

numpy.random.seed(0)

from util_gabor import create_gabor_filters_util, BW_image_jax_supp, make_orimap
from util import cosdiff_ring, test_uniformity
from pretraining_supp import  load_parameters
from parameters import (
    grid_pars,
    filter_pars,
    stimuli_pars,
    readout_pars,
    ssn_pars,
    ssn_layer_pars,
    conv_pars,
    training_pars,
    loss_pars,
    pretrain_pars # Setting pretraining to be true (pretrain_pars.is_on=True) should happen in parameters.py because w_sig depends on it
)

# Checking that pretrain_pars.is_on is on
if not pretrain_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to True in parameters.py to run training with pretraining!')

########## Initialize orientation map and gabor filters ############

ref_ori_saved = float(stimuli_pars.ref_ori)
offset_saved = float(stimuli_pars.offset)

class ConstantPars:
    def __init__(self, grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, ssn_ori_map, oris, ori_dist, gabor_filters, 
                 readout_pars):
        self.grid_pars = grid_pars
        self.stimuli_pars = stimuli_pars
        self.filter_pars = filter_pars
        self.ssn_ori_map = ssn_ori_map
        self.oris = oris
        self.ori_dist = ori_dist
        self.ssn_pars = ssn_pars
        self.ssn_layer_pars = ssn_layer_pars
        self.conv_pars = conv_pars
        self.loss_pars = loss_pars
        self.training_pars = training_pars
        self.gabor_filters = gabor_filters
        self.readout_grid_size = readout_pars.readout_grid_size
        self.middle_grid_ind = readout_pars.middle_grid_ind
        self.pretrain_pars = pretrain_pars
        self.BW_image_jax_inp = BW_image_jax_supp(stimuli_pars)

def def_constant_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars):
    """Define constant_pars with a randomly generated orientation map."""
    X = grid_pars.x_map
    Y = grid_pars.y_map
    is_uniform = False
    map_gen_ind = 0
    while not is_uniform:
        ssn_ori_map = make_orimap(X, Y, hyper_col=None, nn=30, deterministic=False)
        ssn_ori_map_flat = ssn_ori_map.ravel()
        is_uniform = test_uniformity(ssn_ori_map_flat[readout_pars.middle_grid_ind], num_bins=10, alpha=0.25)
        map_gen_ind = map_gen_ind+1
        if map_gen_ind>20:
            print('############## After 20 attemptsm the randomly generated maps did not pass the uniformity test ##############')
            break
    
    gabor_filters, _, _ = create_gabor_filters_util(ssn_ori_map, ssn_pars.phases, filter_pars, grid_pars, ssn_layer_pars.gE_m, ssn_layer_pars.gI_m)
    oris = ssn_ori_map.ravel()[:, None]
    ori_dist = cosdiff_ring(oris - oris.T, 180)

    # Collect parameters that are not trained into a single class
    constant_pars = ConstantPars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, ssn_ori_map, oris, ori_dist, gabor_filters, 
                 readout_pars)

    return constant_pars

from training import mean_training_task_acc_test, offset_at_baseline_acc

pretrain_pars.is_on=False
constant_pars = def_constant_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                loss_pars, training_pars, pretrain_pars, readout_pars)
ssn_layer_pars_pretrain = copy.deepcopy(ssn_layer_pars)
readout_pars_pretrain = copy.deepcopy(readout_pars)

# Perturb them by percent % and collect them into two dictionaries for the two stages of the pretraining
readout_pars_dict, ssn_layer_pars_dict, _ = load_parameters('results/Feb23_v1/results_0.csv', iloc_ind = 100)
start_time = time.time()
acc_mean1, acc_all1, loss_all1 = mean_training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, constant_pars, False, [2,4,6,9,12,15], sample_size =1)
print(time.time()-start_time)
constant_pars.training_pars.batch_size = [200,100]
start_time = time.time()
acc_mean2, acc_all2, loss_all2 = mean_training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, constant_pars, False, [2,4,6,9,12,15], sample_size = 1)
print(time.time()-start_time)
constant_pars.training_pars.batch_size = [300,200]
start_time = time.time()
acc_mean3, acc_all3, loss_all3 = mean_training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, constant_pars, False, [2,4,6,9,12,15], sample_size = 1)
print(time.time()-start_time)
start_time = time.time()
acc_mean4, acc_all4, loss_all4 = mean_training_task_acc_test(ssn_layer_pars_dict, readout_pars_dict, constant_pars, False, [2,3,4,5,6,9,12,15], sample_size = 1)
print(time.time()-start_time)

#print(sum(numpy.std(acc_all1[i,:], ddof=1) for i in range(6)))
#print(sum(numpy.std(acc_all2[i,:], ddof=1) for i in range(6)))
#print(sum(numpy.std(acc_all3[i,:], ddof=1) for i in range(6)))
#sample_size=10, time: 58, 68, 88
#0.285103511233052
#0.2264426728755637
#0.16070200507613622

print(acc_mean1)
print(acc_mean2)
print(acc_mean3)
print(acc_mean4)
#sample_size=10
#[0.558      0.62200004 0.77       0.78000003 0.824      0.82600003]
#[0.612      0.62700003 0.70500004 0.793      0.815      0.85400003]
#[0.57199997 0.671      0.7185     0.7955     0.8285     0.85400003]
#sample_size=10
#[0.55200005 0.664      0.704      0.792      0.85199994 0.88799995]
#[0.59400004 0.64       0.726      0.78000003 0.85       0.822     ]
#[0.583      0.651      0.698      0.786      0.8360001  0.857     ]

offset1 = offset_at_baseline_acc(acc_mean1, offset_vec=[2,4,6,9,12,15])
offset2 = offset_at_baseline_acc(acc_mean2, offset_vec=[2,4,6,9,12,15])
offset3 = offset_at_baseline_acc(acc_mean3, offset_vec=[2,4,6,9,12,15])
offset4 = offset_at_baseline_acc(acc_mean4, offset_vec=[2,3,4,5,6,9,12,15])

print(offset1)
print(offset2)
print(offset3)
print(offset4)

print('decision is that batch size will be 300 but sample size is 1 (no need for calling the mean_binary_task_acc_test function)')