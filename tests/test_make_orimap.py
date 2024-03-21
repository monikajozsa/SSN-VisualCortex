import numpy
import sys
import matplotlib.pyplot as plt
folder_path = "C:/Users/jozsa/Desktop/Postdoc 2023-24/ABL-MJ/"

if folder_path not in sys.path:
    sys.path.append(folder_path)

import os
import numpy

numpy.random.seed(0)

from parameters import pretrain_pars
# Setting pretraining to be true
pretrain_pars.is_on=True

########## Initialize orientation map and gabor filters ############

ssn_ori_map_loaded = numpy.load(os.path.join(os.getcwd(), "ssn_map_uniform_good.npy"))

from util_gabor import make_orimap
from parameters import grid_pars, ssn_pars

x_map = grid_pars.x_map
y_map = grid_pars.y_map

orimap_rnd = make_orimap(x_map, y_map, hyper_col=None, nn=30, deterministic=False)
orimap_det = make_orimap(x_map, y_map, hyper_col=None, nn=30, deterministic=True)

fig, axes = plt.subplots(nrows=1, ncols=3)
vmin = min(ssn_ori_map_loaded.min(), orimap_rnd.min(), orimap_det.min())
vmax = max(ssn_ori_map_loaded.max(), orimap_rnd.max(), orimap_det.max())

fig, axes = plt.subplots(nrows=1, ncols=3)
axes[0].imshow(ssn_ori_map_loaded, cmap='viridis', aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)
axes[0].set_title('Loaded orimap')
axes[1].imshow(orimap_rnd, cmap='viridis', aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)
axes[1].set_title('Randomly generated orimap')
im = axes[2].imshow(orimap_det, cmap='viridis', aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)
axes[2].set_title('Regular orimap')

fig.colorbar(im, ax=axes, orientation='vertical')

fig.show()

