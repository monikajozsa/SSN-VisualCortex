import numpy
import time
'''
numpy.random.seed(0)

from visualization import tuning_curve
from util_gabor import init_untrained_pars
from util import save_code
from training import train_ori_discr
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
if pretrain_pars.is_on:
    raise ValueError('Set pretrain_pars.is_on to False in parameters.py to run training without pretraining!')

########## Initialize orientation map and gabor filters ############

# Save out initial offset and reference orientation
ref_ori_saved = float(stimuli_pars.ref_ori)
offset_saved = float(stimuli_pars.offset)

# Save scripts into scripts folder and create figures and train_only folders
results_filename, final_folder_path = save_code()
#loaded_orimap =  np.load(os.path.join(os.getcwd(), 'ssn_map_uniform_good.npy'))

# Run N_training number of pretraining + training
tc_ori_list = numpy.arange(0,180,2)
N_training = 1
starting_time_in_main= time.time()
numFailedRuns = 0
i=0
while i < N_training and numFailedRuns < 20:

    # Set offset and reference orientation to their initial values
    stimuli_pars.offset=offset_saved
    stimuli_pars.ref_ori=ref_ori_saved

    # Create file names
    results_filename_train_only = f"{final_folder_path}/train_only/results_train_only{i}.csv"
    tc_pre_train_only_filename = f"{final_folder_path}/train_only/tc_train_only_pre_{i}.csv"
    tc_post_train_only_filename = f"{final_folder_path}/train_only/tc_train_only_post_{i}.csv"
    orimap_filename = f"{final_folder_path}/orimap_{i}.npy"

    # Initialize untrained parameters (calculate gabor filters, orientation map related variables)
    untrained_pars = init_untrained_pars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                 loss_pars, training_pars, pretrain_pars, readout_pars, orimap_filename)

    # Perturb readout_pars and ssn_layer_pars by percent % and collect them into two dictionaries for the two stages of the pretraining
    # Note that orimap is regenerated if conditions do not hold!
    trained_pars_stage1, trained_pars_stage2, untrained_pars = randomize_params(readout_pars, ssn_layer_pars, untrained_pars, percent=0.1, orimap_filename=orimap_filename)
    
    # Run training
    training_output_df, _ = train_ori_discr(
            trained_pars_stage1,
            trained_pars_stage2,
            untrained_pars,
            results_filename=results_filename_train_only,
            jit_on=True
        )
    
    # Calculate and save tuning curves
    trained_pars_stage1, trained_pars_stage2, _ = load_parameters(results_filename_train_only, iloc_ind = 0)
    _, _ = tuning_curve(untrained_pars, trained_pars_stage2, tc_pre_train_only_filename, ori_vec=tc_ori_list)
    trained_pars_stage1, trained_pars_stage2, _ = load_parameters(results_filename_train_only, iloc_ind = -1)
    _, _ = tuning_curve(untrained_pars, trained_pars_stage2, tc_post_train_only_filename, ori_vec=tc_ori_list)
    
    i = i + 1
    print('runtime of {} training run(s)'.format(i), time.time()-starting_time_in_main)
    print('number of failed runs = ', numFailedRuns)

######### PLOT RESULTS ############
'''
from visualization import plot_results_from_csvs, boxplots_from_csvs, plot_tuning_curves, plot_tc_features
from Mahal_distances import Mahalanobis_dist
tc_cells=numpy.array([10,40,100,130,650,690,740,760])

## Figure file names
final_folder_path='results/Mar15_v5'
N_training=1
tc_ori_list = numpy.arange(0,180,2)
boxplot_file_name_train_only = 'boxplot_train_only'
final_folder_path_train_only = final_folder_path + '/train_only'
folder_to_save = final_folder_path + '/figures'
mahal_file_name_train_only = 'Mahal_dist_train_only'
## Plotting functions
#plot_results_from_csvs(final_folder_path_train_only, N_training, folder_to_save=folder_to_save)
#boxplots_from_csvs(final_folder_path_train_only,folder_to_save, boxplot_file_name_train_only)
#Mahalanobis_dist(N_training, final_folder_path_train_only, folder_to_save, mahal_file_name_train_only)
#plot_tc_features(final_folder_path_train_only, N_training, tc_ori_list, train_only_str='train_only_')
plot_tuning_curves(final_folder_path_train_only,tc_cells,N_training,folder_to_save,train_only_str='train_only_')
