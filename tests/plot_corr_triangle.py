import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from visualization import plot_corr_triangle


time_start = time.time()
# Generate sample data
np.random.seed(0)
data = pd.DataFrame({
    'MVPA': np.random.rand(50),
    'dJm_ratio': np.random.rand(50),
    'd_offset': np.random.rand(50)
})

plot_corr_triangle(data)

print('Time taken: ', time.time()-time_start)