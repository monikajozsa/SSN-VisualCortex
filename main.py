import numpy
import copy
import time

numpy.random.seed(0)

from util_gabor import create_gabor_filters_util, BW_image_jax_supp, make_orimap
from util import save_code, cosdiff_ring, test_uniformity
from training import train_ori_discr
from analysis import tuning_curves
from pretraining_supp import randomize_params, load_parameters
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

# Defining the number of random initializations for pretraining + training
N_training = 5

# Save scripts
results_filename, final_folder_path = save_code()

starting_time_in_main= time.time()
numFailedRuns = 0
i=0
while i < N_training and numFailedRuns < 20:
    constant_pars = def_constant_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars)
    constant_pars.stimuli_pars.offset=offset_saved
    constant_pars.stimuli_pars.ref_ori=ref_ori_saved # this changes during training because of the staircase
    constant_pars.pretrain_pars.is_on=True

    results_filename = f"{final_folder_path}/results_{i}.csv"
    results_filename_train_only = f"{final_folder_path}/results_train_only{i}.csv"
    tuning_curves_prepre = f"{final_folder_path}/tc_prepre_{i}.csv"
    tuning_curves_prepost = f"{final_folder_path}/tc_prepost_{i}.csv"
    tuning_curves_post = f"{final_folder_path}/tc_post_{i}.csv"

    ##### PRETRAINING: GENERAL ORIENTAION DISCRIMINATION #####
    # Get baseline parameters to-be-trained
    ssn_layer_pars_pretrain = copy.deepcopy(ssn_layer_pars)
    readout_pars_pretrain = copy.deepcopy(readout_pars)

    # Perturb them by percent % and collect them into two dictionaries for the two stages of the pretraining
    trained_pars_stage1, trained_pars_stage2 = randomize_params(readout_pars_pretrain, ssn_layer_pars_pretrain, constant_pars, percent=0.1)

    # Pretrain parameters
    training_output_df, first_stage_final_step = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            constant_pars,
            results_filename=results_filename,
            jit_on=True
        )
    if training_output_df is None:
        print('######### Stopped run {} because of NaN values  - num failed runs = {} #########'.format(i, numFailedRuns))
        numFailedRuns = numFailedRuns + 1
        continue

    constant_pars.pretrain_pars.is_on=False
    constant_pars.first_stage_final_step = first_stage_final_step
    
    trained_pars_stage1, trained_pars_stage2 = load_parameters(results_filename, iloc_ind = numpy.min([10,training_pars.SGD_steps[0]]))
    responses_sup_prepre, responses_mid_prepre = tuning_curves(constant_pars, trained_pars_stage2, tuning_curves_prepre)#pretty slow

    ##### FINE DISCRIMINATION #####
    
    trained_pars_stage1, trained_pars_stage2 = load_parameters(results_filename, iloc_ind = first_stage_final_step-1)
    responses_sup_prepost, responses_mid_prepost = tuning_curves(constant_pars, trained_pars_stage2, tuning_curves_prepost)

    training_output_df, _ = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            constant_pars,
            results_filename=results_filename,
            jit_on=True
        )
    
    last_row = training_output_df.iloc[-1]
    J_m_keys = ['logJ_m_EE','logJ_m_EI','logJ_m_IE','logJ_m_II'] 
    J_s_keys = ['logJ_s_EE','logJ_s_EI','logJ_s_IE','logJ_s_II']
    J_m_values = last_row[J_m_keys].values.reshape(2, 2)
    J_s_values = last_row[J_s_keys].values.reshape(2, 2)

    pars_stage2 = dict(
        log_J_2x2_m = J_m_values,
        log_J_2x2_s = J_s_values,
        c_E=last_row['c_E'],
        c_I=last_row['c_I'],
        f_E=last_row['f_E'],
        f_I=last_row['f_I'],
    )
    responses_sup_post, responses_mid_post = tuning_curves(constant_pars, pars_stage2, tuning_curves_post)
    i = i + 1
    print('runtime of {} pretraining + training run(s)'.format(i), time.time()-starting_time_in_main)
    print('number of failed runs = ', numFailedRuns)

    # Running training only with the same initialization and orimap
    trained_pars_stage1, trained_pars_stage2 = load_parameters(results_filename, iloc_ind = 0)
    training_output_df, _ = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            constant_pars,
            results_filename=results_filename_train_only,
            jit_on=True
        )

######### PLOT RESULTS ############

from visualization import plot_results_from_csvs, barplots_from_csvs

#final_folder_path='results/Feb22_v9'
#N_training=1
plot_results_from_csvs(final_folder_path, N_training)

boxplot_file_name = 'boxplot_pretraining'
barplots_from_csvs(final_folder_path, boxplot_file_name)
