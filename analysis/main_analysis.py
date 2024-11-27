import os
import time
import numpy
import pandas as pd

from analysis.analysis_functions import save_tc_features, MVPA_anova, make_exclude_run_csv, csv_to_numpy, main_MVPA
from analysis.visualization import plot_tuning_curves, plot_corr_triangles, plot_tc_features, plot_param_offset_correlations, boxplots_from_csvs, plot_MVPA_or_Mahal_scores, plot_MVPA_or_Mahal_scores_match_Kes_fig

def main_analysis(folder_path, num_runs, conf_names):
    ''' Main function for analysis of the training results. 
    1) Creates excluded_runs_all.csv
    2) plot boxplots, param_offset_correlations, tuning_curves and tuning curve features
    3) run MVPA and plot MVPA scores, Mahalanobis scores, anova results and correlation triangles
    '''
    
    ######### ######### ######### ######### ######### #########
    ######## Asses run indices that should be excluded ########
    ######### ######### ######### ######### ######### #########
    if not os.path.exists(os.path.join(folder_path, 'excluded_runs_all.csv')):
        excluded_run_inds = []
        for i, conf in enumerate(conf_names):
            config_folder = os.path.join(folder_path, conf)
            if conf.endswith('baseline'):
                offset_condition = True
            else:
                offset_condition = False
            excluded_run_inds_config = make_exclude_run_csv(config_folder, num_runs, offset_condition)
            excluded_run_inds.append(excluded_run_inds_config)
        # squeeze the list and keep only unique indices
        if len(excluded_run_inds) > 0:
            excluded_run_inds_unique = numpy.unique(numpy.concatenate(excluded_run_inds))
        else:
            excluded_run_inds_unique = []

        # Save the excluded runs in a csv file
        excluded_runs_df = pd.DataFrame(excluded_run_inds_unique)
        excluded_runs_df.to_csv(os.path.join(folder_path, 'excluded_runs_all.csv'), index=False, header=False)
        excluded_runs_summary_df = pd.DataFrame(excluded_run_inds, index=conf_names)
        excluded_runs_summary_df.to_csv(os.path.join(folder_path, 'excluded_runs_details.csv'))
        print('Excluded runs saved in excluded_runs_all.csv')
    

    ######### ######### ######### #########
    ######## Plots on included runs #######
    ######### ######### ######### #########

    # Replot boxplots and plot tuning curves
    # read 'excluded_runs_all.csv'
    file_name = os.path.join(folder_path, 'excluded_runs_all.csv')
    if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
        excluded_runs_df = pd.read_csv(file_name, header=None)
        excluded_runs = excluded_runs_df.values.flatten()
    else:
        excluded_runs = []
    
    for i, conf in enumerate(conf_names):
        config_folder = os.path.join(folder_path, conf)
        if i == 0:
            num_time_inds = 3
        else:
            num_time_inds = 2
        boxplots_from_csvs(config_folder, num_time_inds = num_time_inds, excluded_runs=excluded_runs)
        plot_param_offset_correlations(config_folder, excluded_runs=excluded_runs)
        
        # plot tuning curves and features
        tc_cells=[10,40,100,130,172,202,262,292,334,364,424,454,496,526,586,616,650,690,740,760] 
        # these are indices of representative cells from the different layers and types: every pair is for off center and center from 
        # mEph0(1-2), mIph0(3-4), mEph1(5-6), mIph1(7-8), mEph2(9-10), mIph2(11-12), mEph3(13-14), mIph3(15-16), sE(17-18), sI(19-20)
        
        plot_tuning_curves(config_folder, tc_cells, num_runs, excluded_runs=excluded_runs)
        if i == 0:
            stages = [0,1,2]
        else:
            stages = [1,2]
        #plot_tc_features(config_folder, stages=stages, color_by='pref_ori_range', excluded_runs=excluded_runs)
        plot_tc_features(config_folder, stages=stages, color_by='type', add_cross=True, excluded_runs=excluded_runs)
        plot_tc_features(config_folder, stages=stages, color_by='run_index', excluded_runs=excluded_runs)
        plot_tc_features(config_folder, stages=stages, color_by='pref_ori', excluded_runs=excluded_runs)
        plot_tc_features(config_folder, stages=stages, color_by='phase', excluded_runs=excluded_runs)
        print('\n')
        print(f'Finished plots for {conf_names[i]}')
        print('\n')
        

    ########## ########## ########## ##########
    ########## MVPA on included runs ##########
    ########## ########## ########## ##########
    
    for i, conf in enumerate(conf_names):
        start_time = time.time()
        config_folder = os.path.join(folder_path, conf)
        main_MVPA(config_folder, num_runs=num_runs, num_stages=3, sigma_filter=2, r_noise=True, num_noisy_trials=200, excluded_runs=excluded_runs)
        print('Done with calculating MVPA for configuration ', conf, ' in ', time.time()-start_time, ' seconds')
    
    for conf in conf_names:
        start_time = time.time()
        config_folder = os.path.join(folder_path, conf)
        plot_MVPA_or_Mahal_scores(config_folder, 'MVPA_scores')
        plot_MVPA_or_Mahal_scores_match_Kes_fig(config_folder, 'MVPA_scores')
        plot_MVPA_or_Mahal_scores(config_folder, 'Mahal_scores')
        MVPA_anova(config_folder)
        plot_corr_triangles(config_folder, excluded_runs=excluded_runs)
        print('Done with plotting MVPA results for configuration ', conf, ' in ', time.time()-start_time, ' seconds')
