import time
import jax.numpy as np
import numpy
import matplotlib.pyplot as plt
#import pingouin as pg
from scipy.stats import ttest_1samp
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from analysis import  filtered_model_response, filtered_model_response_task, select_response


######### MVPA scores:  #########

def MVPA_score(folder, num_training, num_SGD_inds=2, sigma_filter=5, plot_flag=False):
    ''' Calculate MVPA scores for before pretraining, after pretraining and after training'''
    ori_list = numpy.asarray([55, 125, 0])
    num_layers=2 # number of layers
    num_noisy_trials=200 
    
    # parameters of the SVM classifier (MVPA analysis)
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))

    # Initialize the MVPA scores matrix
    MVPA_scores = numpy.zeros((num_training,num_layers,num_SGD_inds, len(ori_list)-1))

    # Iterate over the different parameter initializations (runs or trainings)
    start_time = time.time()
    for run_ind in range(num_training):
        # Calculate num_noisy_trials filtered model response for each oris in ori list and for each parameter set (that come from file_name at num_SGD_inds rows)
        response_all, SGD_step_inds = filtered_model_response(folder, run_ind, ori_list= ori_list, num_noisy_trials = num_noisy_trials, num_SGD_inds=num_SGD_inds, r_noise=True, sigma_filter=sigma_filter)
        print(f'Time taken for filtered model response task for run {run_ind}:', time.time()-start_time)   

        # Iterate over the layers and SGD steps
        for layer in range(num_layers):
            for SGD_ind in range(num_SGD_inds):
                # Select the responses for the trained and untrained orientations
                response_0, _ = select_response(response_all, SGD_step_inds[SGD_ind], layer, ori_list[0])
                response_1, _ = select_response(response_all, SGD_step_inds[SGD_ind], layer, ori_list[1])
                response_2, _ = select_response(response_all, SGD_step_inds[SGD_ind], layer, ori_list[2])
                
                # MVPA for distinguishing trained orientation and control orientation
                # Combine the responses for the 0 and 2 oris (along axis-0)
                response_0_2 = numpy.concatenate((response_0, response_2))
                response_0_2 = response_0_2.reshape(response_0_2.shape[0],-1)
                label_0_2 = numpy.concatenate((numpy.zeros(len(response_0)), numpy.ones(len(response_2))))
                # make test-train split
                X_train, X_test, y_train, y_test = train_test_split(response_0_2, label_0_2, test_size=0.5, random_state=42)
                MVPA_scores[run_ind,layer,SGD_ind, 0] = clf.fit(X_train, y_train).score(X_test, y_test)
                
                # MVPA for distinguishing untrained orientation and control orientation
                # Combine the responses for the 1 and 2 oris
                response_1_2 = numpy.concatenate((response_1, response_2))
                response_1_2 = response_1_2.reshape(response_1_2.shape[0],-1)
                label_1_2 = numpy.concatenate((numpy.zeros(len(response_1)), numpy.ones(len(response_2))))
                
                # make test-train split
                X_train, X_test, y_train, y_test = train_test_split(response_1_2, label_1_2, test_size=0.2, random_state=42)
                MVPA_scores[run_ind,layer,SGD_ind, 1] = clf.fit(X_train, y_train).score(X_test, y_test)
                if plot_flag & (run_ind<2):
                    # plot 10 trials from response_0,response_1, response_2 on a 10 x 3 subplot
                    fig, axs = plt.subplots(4, 3, figsize=(15, 30))
                    for i in range(4):
                        axs[i, 0].imshow(response_0[i])
                        axs[i, 1].imshow(response_1[i])
                        axs[i, 2].imshow(response_2[i])
                    # save figure
                    axs[0, 0].set_title(f'MVPA: {MVPA_scores[run_ind,layer,SGD_ind, 0]}')
                    axs[0, 1].set_title(f'MVPA: {MVPA_scores[run_ind,layer,SGD_ind, 1]}')
                    plt.savefig(f'{folder}/response_{run_ind}_{layer}_{sigma_filter}.png')
            

    # t-test for significant change in scores before and after training
    MVPA_t_test = numpy.zeros((num_training,num_layers,len(ori_list)-1,2))
    for run_ind in range(num_training):
        for layer in range(num_layers):
            for ori_ind in range(len(ori_list)-1):
                t_stat, p_val = ttest_1samp(MVPA_scores[:,layer,1, ori_ind]-MVPA_scores[:,layer,-1, ori_ind], 0)
                MVPA_t_test[run_ind,layer,ori_ind,0] = t_stat
                MVPA_t_test[run_ind,layer,ori_ind,1] = p_val

    return MVPA_scores, MVPA_t_test


def task_score(folder, num_training, num_SGD_inds=2, sigma_filter=5):
    ''' Calculate MVPA scores for before pretraining, after pretraining and after training'''
    ori_list = numpy.asarray([55, 125, 0])
    num_layers=2 # number of layers
    num_noisy_trials=200 
    
    # parameters of the SVM classifier
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))

    # Initialize the scores matrices
    SVM_scores = numpy.zeros((num_training,num_layers,num_SGD_inds, len(ori_list)))
    LogReg_scores = numpy.zeros((num_training,num_layers,num_SGD_inds, len(ori_list)))

    # Iterate over the different parameter initializations (runs or trainings)
    start_time = time.time()
    for run_ind in range(num_training):
        
        # Calculate num_noisy_trials filtered model response for each oris in ori list and for each parameter set (that come from file_name at num_SGD_inds rows)
        response_all, SGD_step_inds = filtered_model_response_task(folder, run_ind, ori_list= ori_list, num_noisy_trials = num_noisy_trials, num_SGD_inds=num_SGD_inds, r_noise=True, sigma_filter=sigma_filter)
        print(f'Time taken for filtered model response task for run {run_ind}:', time.time()-start_time)   

        # Iterate over the layers and SGD steps
        for layer in range(num_layers):
            for SGD_ind in range(num_SGD_inds):            
                for ori_ind in range(len(ori_list)):
                    # Define filter to select the responses corresponding to SGD_ind, layer (sup-0 mid-1) and ori
                    response, label = select_response(response_all, SGD_step_inds[SGD_ind], layer, ori_list[ori_ind])
                
                    # MVPA analysis
                    X_train, X_test, y_train, y_test = train_test_split(response, label, test_size=0.5, random_state=42)
                    log_reg = LogisticRegression(max_iter=100)
                    log_reg.fit(X_train, y_train)
                    y_pred = log_reg.predict(X_test)
                    LogReg_scores[run_ind,layer,SGD_ind, ori_ind] =  accuracy_score(y_test, y_pred)
                    SVM_scores[run_ind,layer,SGD_ind, ori_ind] = clf.fit(X_train, y_train).score(X_test, y_test) # this model gave me very bad accuracies, so I used logistic regression instead
    # After calculating scores for all runs, calculate t-tests for significant change in scores before and after training
    SVM_t_test = numpy.zeros((num_training,num_layers,len(ori_list),2))
    LogReg_t_test = numpy.zeros((num_training,num_layers,len(ori_list),2))
    for run_ind in range(num_training):
        for layer in range(num_layers):
            for ori_ind in range(len(ori_list)-1):
                t_stat, p_val = ttest_1samp(SVM_scores[:,layer,1, ori_ind]-SVM_scores[:,layer,-1, ori_ind], 0)
                SVM_t_test[run_ind,layer,ori_ind,0] = t_stat
                SVM_t_test[run_ind,layer,ori_ind,1] = p_val
                t_stat, p_val = ttest_1samp(LogReg_scores[:,layer,1, ori_ind]-LogReg_scores[:,layer,-1, ori_ind], 0)
                LogReg_t_test[run_ind,layer,ori_ind,0] = t_stat
                LogReg_t_test[run_ind,layer,ori_ind,1] = p_val
                
    return SVM_scores, LogReg_scores, SVM_t_test, LogReg_t_test


def Scores_from_csv(final_folder_path, num_training, folder_to_save, num_SGD_inds=2, sigma_filter=5, task_score_flag=False, plot_flag=False):
    ''' Calculate MVPA scores for before pretraining, after pretraining and after training - score should increase for trained ori more than for other two oris especially in superficial layer'''
    plt.close()
    MVPA_scores, MVPA_t_test = MVPA_score(final_folder_path,num_training, num_SGD_inds, sigma_filter=sigma_filter,plot_flag=plot_flag)
    
    # save the output into folder_to_save as npy files
    numpy.save(f"{folder_to_save}/{'MVPA_scores'}.npy", MVPA_scores) 
    numpy.save(f"{folder_to_save}/{'MVPA_t_test'}.npy", MVPA_t_test)
    
    print('Before and after for 55~0, sup layer:',[np.mean(MVPA_scores[:,0,0,0]),np.mean(MVPA_scores[:,0,-1,0])])
    print('Before and after for 55~0, mid layer:',[np.mean(MVPA_scores[:,1,0,0]),np.mean(MVPA_scores[:,1,-1,0])])
    print('Before and after for 125~0, sup layer:',[np.mean(MVPA_scores[:,0,0,1]),np.mean(MVPA_scores[:,0,-1,1])])
    print('Before and after for 125~0, mid layer:',[np.mean(MVPA_scores[:,1,0,1]),np.mean(MVPA_scores[:,1,-1,1])])

    if task_score_flag:
        SVM_scores, LogReg_scores, SVM_t_test, LogReg_t_test = task_score(final_folder_path,num_training, num_SGD_inds, sigma_filter=sigma_filter)
        numpy.save(f"{folder_to_save}/{'SVM_scores'}.npy", SVM_scores)
        numpy.save(f"{folder_to_save}/{'SVM_t_test'}.npy", SVM_t_test)
        numpy.save(f"{folder_to_save}/{'Logreg_scores'}.npy", LogReg_scores)
        numpy.save(f"{folder_to_save}/{'Logreg_t_test'}.npy", LogReg_t_test)
        print('Before and after for 55, sup layer:',[np.mean(SVM_scores[:,0,0,0]),np.mean(SVM_scores[:,0,-1,0])])
        print('Before and after for 55, mid layer:',[np.mean(SVM_scores[:,1,0,0]),np.mean(SVM_scores[:,1,-1,0])])
        print('Before and after for 125, sup layer:',[np.mean(SVM_scores[:,0,0,1]),np.mean(SVM_scores[:,0,-1,1])])
        print('Before and after for 125, mid layer:',[np.mean(SVM_scores[:,1,0,1]),np.mean(SVM_scores[:,1,-1,1])])
        print('Before and after for 0, sup layer:',[np.mean(SVM_scores[:,0,0,2]),np.mean(SVM_scores[:,0,-1,2])])
        print('Before and after for 0, mid layer:',[np.mean(SVM_scores[:,1,0,2]),np.mean(SVM_scores[:,1,-1,2])])

        print('Before and after for 55, sup layer:',[np.mean(LogReg_scores[:,0,0,0]),np.mean(LogReg_scores[:,0,-1,0])])
        print('Before and after for 55, mid layer:',[np.mean(LogReg_scores[:,1,0,0]),np.mean(LogReg_scores[:,1,-1,0])])
        print('Before and after for 125, sup layer:',[np.mean(LogReg_scores[:,0,0,1]),np.mean(LogReg_scores[:,0,-1,1])])
        print('Before and after for 125, mid layer:',[np.mean(LogReg_scores[:,1,0,1]),np.mean(LogReg_scores[:,1,-1,1])])
        print('Before and after for 0, sup layer:',[np.mean(LogReg_scores[:,0,0,2]),np.mean(LogReg_scores[:,0,-1,2])])
        print('Before and after for 0, mid layer:',[np.mean(LogReg_scores[:,1,0,2]),np.mean(LogReg_scores[:,1,-1,2])])

        return MVPA_scores, MVPA_t_test, SVM_scores, LogReg_scores, SVM_t_test, LogReg_t_test
    else:
        return MVPA_scores, MVPA_t_test

