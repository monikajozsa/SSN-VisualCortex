import os
import visualization
import pandas as pd
import matplotlib.pyplot as plt

# Set the working directory
os.chdir('C:/Users/jozsa/Desktop/Postdoc 2023-24/ABL-MJ/')
results_filename = os.path.join('results/','perturb_all_results.csv')

#Load data from csv
df = pd.read_csv(results_filename, header=0)

# Create a subplot with 2 rows and 4 columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(25, 10))
for i in range(5):
    filter_run=df['run']==i
    axes[0, 0].plot(df[filter_run]['epoch'], df[filter_run]['J_m_EE'], label='J_m_EE')
    axes[0, 1].plot(df[filter_run]['epoch'], df[filter_run]['J_m_EI'], label='J_m_EI')
    axes[1, 0].plot(df[filter_run]['epoch'], df[filter_run]['J_m_IE'], label='J_m_IE')
    axes[1, 1].plot(df[filter_run]['epoch'], df[filter_run]['J_m_II'], label='J_m_II')
fig.suptitle('J mid')
fig.savefig("VaryingAll_Jm.png")


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(25, 10))
for i in range(5):
    filter_run=df['run']==i
    axes[0, 0].plot(df[filter_run]['epoch'], df[filter_run]['J_s_EE'], label='J_s_EE')
    axes[0, 1].plot(df[filter_run]['epoch'], df[filter_run]['J_s_EI'], label='J_s_EI')
    axes[1, 0].plot(df[filter_run]['epoch'], df[filter_run]['J_s_IE'], label='J_s_IE')
    axes[1, 1].plot(df[filter_run]['epoch'], df[filter_run]['J_s_II'], label='J_s_II')
fig.suptitle('J sup')
fig.show()
fig.savefig("VaryingAll_Js.png")


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(25, 10))
for i in range(5):
    filter_run=df['run']==i
    axes[0, 0].plot(df[filter_run]['epoch'], df[filter_run]['w_sig_1'], label='w_sig_1')
    axes[0, 1].plot(df[filter_run]['epoch'], df[filter_run]['w_sig_2'], label='w_sig_2')
    axes[1, 0].plot(df[filter_run]['epoch'], df[filter_run]['w_sig_3'], label='w_sig_3')
    axes[1, 1].plot(df[filter_run]['epoch'], df[filter_run]['w_sig_4'], label='w_sig_4')
fig.suptitle('w sig')
fig.show()
fig.savefig("VaryingAll_w.png")


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
for i in range(5):
    filter_run=df['run']==i
    axes[0].plot(df[filter_run]['epoch'], df[filter_run]['c_E'], label='c_E')
    axes[1].plot(df[filter_run]['epoch'], df[filter_run]['c_I'], label='c_I')
fig.suptitle('c')
fig.show()
fig.savefig("VaryingAll_c.png")


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
for i in range(5):
    filter_run=df['run']==i
    axes[0].plot(df[filter_run]['epoch'], df[filter_run]['f_E'], label='f_E')
    axes[1].plot(df[filter_run]['epoch'], df[filter_run]['f_I'], label='f_I')
fig.suptitle('f')
fig.show()
fig.savefig("VaryingAll_f.png")

