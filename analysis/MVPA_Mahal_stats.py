import numpy
import pandas as pd
from scipy.stats import ttest_1samp
import os
from scipy.stats import ttest_ind
import pandas as pd

final_folder_path=os.path.join('results','Apr10_v1', 'sigmafilt_2')
num_layers = 2
num_SGD_inds = 3
num_runs = 50
num_ori = 3
num_training = 50
ori_list = [55, 125, 0]

# Load the data
df_mahal = pd.read_csv(os.path.join(final_folder_path, 'df_mahal.csv'))
df_LMI = pd.read_csv(os.path.join(final_folder_path, 'df_LMI.csv'))
LMI_across = df_LMI['LMI_across'].values.reshape((num_runs,num_layers,num_SGD_inds-1))
LMI_within = df_LMI['LMI_within'].values.reshape((num_runs,num_layers,num_SGD_inds-1))
LMI_ratio = df_LMI['LMI_ratio'].values.reshape((num_runs,num_layers,num_SGD_inds-1))
MVPA_scores = numpy.load(os.path.join(final_folder_path, 'MVPA_scores.npy'))

################# Mahalanobis distance #################
mahal_pre = df_mahal[(df_mahal['layer']==0) & (df_mahal['SGD_ind']==1)]
mahal_post = df_mahal[(df_mahal['layer']==0) & (df_mahal['SGD_ind']==2)]
t_statistic, p_value1 = ttest_ind(mahal_post['ori55_across'], mahal_pre['ori55_across'])
print('Significance of Mahalanobis distance change pre-post training, sup layer (p-val)',p_value1)
mahal_pre = df_mahal[(df_mahal['layer']==1) & (df_mahal['SGD_ind']==1)]
mahal_post = df_mahal[(df_mahal['layer']==1) & (df_mahal['SGD_ind']==2)]
t_statistic, p_value1 = ttest_ind(mahal_post['ori55_across'], mahal_pre['ori55_across'])
print('Significance of Mahalanobis distance change pre-post training, mid layer (p-val)',p_value1)

# create a 1x2 subplot for the two layers and in each subplot, draw 6 boxplots for the 6 conditions (before and after training for each of the 3 orientations)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# iterate over the two layers
layer_label = ['sup', 'mid']
for layer in range(num_layers):
    # create a list of the 6 conditions
    data = [df_mahal[(df_mahal['layer']==layer) & (df_mahal['SGD_ind']==1)]['ori55_across'], df_mahal[(df_mahal['layer']==layer) & (df_mahal['SGD_ind']==2)]['ori55_across'],
            df_mahal[(df_mahal['layer']==layer) & (df_mahal['SGD_ind']==1)]['ori125_across'], df_mahal[(df_mahal['layer']==layer) & (df_mahal['SGD_ind']==2)]['ori125_across'],
            df_mahal[(df_mahal['layer']==layer) & (df_mahal['SGD_ind']==1)]['ori55_within'], df_mahal[(df_mahal['layer']==layer) & (df_mahal['SGD_ind']==2)]['ori55_within'],
            df_mahal[(df_mahal['layer']==layer) & (df_mahal['SGD_ind']==1)]['ori125_within'], df_mahal[(df_mahal['layer']==layer) & (df_mahal['SGD_ind']==2)]['ori125_within']]
    # draw the boxplot
    ax[layer].boxplot(data, positions=[1, 3, 5, 7, 9, 11, 13, 15])
    ax[layer].set_xticklabels(['55 pre', '55 post', '125 pre', '125 post', '55 within', '55 within', '125 within', '125 within'])
    ax[layer].set_title(f'Layer {layer_label[layer]}')
    ax[layer].set_ylabel('Mahalanobis distance')
    # draw lines to connect the pre and post training for each sample
    for i in range(num_runs):
        #gray lines
        ax[layer].plot([1, 3], [data[0].values[i], data[1].values[i]], color='gray', alpha=0.5, linewidth=0.5)
        ax[layer].plot([5, 7], [data[2].values[i], data[3].values[i]], color='gray', alpha=0.5, linewidth=0.5)
        ax[layer].plot([9, 11], [data[4].values[i], data[5].values[i]], color='gray', alpha=0.5, linewidth=0.5)
        ax[layer].plot([13, 15], [data[6].values[i], data[7].values[i]], color='gray', alpha=0.5, linewidth=0.5)
plt.savefig(final_folder_path+'/Mahalanobis_boxplot.png')
plt.close()

################# LMI #################
LMI_ttests = numpy.zeros((num_layers,3))# 3 is for across, within and ratio
LMI_ttest_p = numpy.zeros((num_layers,3))
SGD_ind=1 # we are only interested in training and not pretraining
for layer in range(num_layers):
    LMI_ttests[layer,0], LMI_ttest_p[layer,0] = ttest_1samp(LMI_across[:,layer,SGD_ind],0) # compare it to mean 0
    LMI_ttests[layer,1], LMI_ttest_p[layer,1] = ttest_1samp(LMI_within[:,layer,SGD_ind],0)
    LMI_ttests[layer,2], LMI_ttest_p[layer,2] = ttest_1samp(LMI_ratio[:,layer,SGD_ind],0)

print('Significance of LMI change pre-post training, sup layer Across-Within-Ratio (p-val)',LMI_ttest_p[0])
print('Significance of LMI change pre-post training, mid layer Across-Within-Ratio (p-val)',LMI_ttest_p[1])

################# MVPA #################
MVPA_t_test = numpy.zeros((num_layers,len(ori_list)-1,2))
for layer in range(num_layers):
    for ori_ind in range(len(ori_list)-1):
        t_stat, p_val = ttest_1samp(MVPA_scores[:,layer,1, ori_ind]-MVPA_scores[:,layer,-1, ori_ind], 0)
        MVPA_t_test[layer,ori_ind,0] = t_stat
        MVPA_t_test[layer,ori_ind,1] = p_val

print('Significance of MVPA change pre-post training, sup layer (p-val)',MVPA_t_test[0,:,1])
print('Significance of MVPA change pre-post training, mid layer (p-val)',MVPA_t_test[1,:,1])

# create a 1x2 subplot for the two layers and in each subplot, draw 4 boxplots for the 4 conditions
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# iterate over the two layers
layer_label = ['sup', 'mid']
for layer in range(num_layers):
    # create a list of the 4 conditions
    data = [MVPA_scores[:,layer,1, 0], MVPA_scores[:,layer,2, 0], MVPA_scores[:,layer,1, 1], MVPA_scores[:,layer,2, 1]]
    # draw the boxplot
    ax[layer].boxplot(data, positions=[1, 2, 3, 4])
    ax[layer].set_xticklabels(['55 pre', '55 post', '125 pre', '125 post'])
    ax[layer].set_title(f'Layer {layer_label[layer]}')
    ax[layer].set_ylabel('MVPA score')
    # draw lines to connect the pre and post training for each sample
    for i in range(num_runs):
        #gray lines
        ax[layer].plot([1, 2], [data[0][i], data[1][i]], color='gray', alpha=0.5, linewidth=0.5)
        ax[layer].plot([3, 4], [data[2][i], data[3][i]], color='gray', alpha=0.5, linewidth=0.5)
plt.savefig(final_folder_path+'/MVPA_boxplot.png')
plt.close()

'''
################# ANOVA #################
import pingouin as pg

## reorganize the data such that Factor1 (layer) and Factor2 (control, untrained, trained) are the first dim (6) in the data and samples are the second dim (100)
MVPA_ANOVA_sup = numpy.zeros(((num_SGD_inds-1)*(num_ori-1)*num_runs))
MVPA_ANOVA_mid = numpy.zeros(((num_SGD_inds-1)*(num_ori-1)*num_runs))

# Perform ANOVA - THIS PART OF THE CODE ONLY WORKS FROM TERMINAL FOR MJ - NOT FROM VSCODE BECAUSE OF DEBUGPY PACKAGE THAT VSCODE USES - MIGHT BE FIXABLE THROUGH SOME JSON FILE CONFIGURATION
i=0
for SGD_ind in range(num_SGD_inds-1):
    for run_ind in range(num_runs):
        for ori_ind in range(num_ori-1):
            MVPA_ANOVA_sup[i] = MVPA_scores[run_ind,0,SGD_ind+1, ori_ind]
            MVPA_ANOVA_mid[i] = MVPA_scores[run_ind,1,SGD_ind+1, ori_ind]
            i+=1
factor1_levels = numpy.repeat(['pre', 'post'], repeats=num_runs*2)
factor2_levels = numpy.tile(numpy.arange(1, 3), reps=num_runs*2)
indices = numpy.column_stack((factor1_levels, factor2_levels))
MVPA_ANOVA_sup_df = pd.DataFrame({
    'Subject': numpy.tile(numpy.repeat(numpy.arange(1, num_runs+1),2), 2),
    'Layer': factor1_levels,
    'Ori': factor2_levels,
    'dependent_variable': MVPA_ANOVA_sup
})
MVPA_ANOVA_sup_df['Layer'] = MVPA_ANOVA_sup_df['Layer'].astype('category')
MVPA_ANOVA_sup_df['Ori'] = MVPA_ANOVA_sup_df['Ori'].astype('category')
MVPA_ANOVA_mid_df = pd.DataFrame({
    'Subject': numpy.tile(numpy.repeat(numpy.arange(1, num_runs+1),2), 2),
    'Layer': factor1_levels,
    'Ori': factor2_levels,
    'dependent_variable': MVPA_ANOVA_mid
})
MVPA_ANOVA_mid_df['Layer'] = MVPA_ANOVA_mid_df['Layer'].astype('category')
MVPA_ANOVA_mid_df['Ori'] = MVPA_ANOVA_mid_df['Ori'].astype('category')

anova_sup = pg.rm_anova(dv='dependent_variable', within=['Layer', 'Ori'], subject='Subject', data=MVPA_ANOVA_sup_df, correction=True, effsize="np2", detailed=True)
F_stat_sup = anova_sup['F'][0]
p_val_sup = anova_sup['p-unc'][0]
anova_mid = pg.rm_anova(dv='dependent_variable', within=['Layer', 'Ori'], subject='Subject', data=MVPA_ANOVA_mid_df, correction=True, effsize="np2", detailed=True)
F_stat_mid = anova_mid['F'][0]
p_val_mid = anova_mid['p-unc'][0]

print(f'Significance of MVPA learning x orientation, sup layer (p-val) {p_val_sup}')
print(f'Significance of MVPA learning x orientation, mid layer (p-val) {p_val_mid}')
'''