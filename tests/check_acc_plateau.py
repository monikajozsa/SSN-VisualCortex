import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def has_plateaued(accuracy, window=50, threshold=0.01, slope_threshold=0.1):
    """
    Check if accuracy has plateaued by checking the change in mean in the last two windows and the slope of the last window.

    Parameters:
    accuracy (list): A list of accuracy values over time.
    window (int): The number of recent points to check for plateau.
    threshold (float): The maximum allowable change in average accuracy to consider as plateau.
    slope_threshold (float): The maximum allowable slope to consider as plateau.

    Returns:
    bool: True if the accuracy has plateaued, False otherwise.
    """
    if len(accuracy) < 2 * window:
        return (0,0)

    recent_window = accuracy[-window:]
    previous_window = accuracy[-2 * window:-window]

    recent_avg = sum(recent_window) / window
    previous_avg = sum(previous_window) / window

    x = np.arange(window)*(max(accuracy)-min(accuracy))/window
    y = np.array(accuracy[-window:])
    slope, _ = np.polyfit(x, y, 1)

    mean_cond = int(abs(recent_avg - previous_avg) < threshold)
    slope_cond = int(abs(slope) < slope_threshold)
    print(mean_cond)
    print(slope_cond)
    return (mean_cond, slope_cond)


# Read in the file 'Jul16_v0/results.csv' and create a numpy array from the column 'acc' as acc_vec
data = pd.read_csv('results/Jul16_v0/results.csv')
acc_vec = data['acc'].to_numpy()

# Smooth the data using a moving average
smooth_window_size = 10
smoothed_acc =np.convolve(acc_vec, np.ones(smooth_window_size)/smooth_window_size, mode='valid')

# Apply has_plateaued function to acc_vec in a for loop with a sliding window of 50
window_size = 100
plateau_vec = np.zeros((len(acc_vec) - window_size + 1,2))

for i in range(len(acc_vec) - 2*window_size + 1):
    acc_vec_i = smoothed_acc[i:i + 2*window_size]
    a = has_plateaued(acc_vec_i, window=window_size, threshold=0.02, slope_threshold=0.3)
    plateau_vec[i,0] = a[0]
    plateau_vec[i,1] = a[1]

# Plot acc_vec and plateau_vec on the same plot (note that plateau_vec is 0 or 1)
plt.figure(figsize=(12, 6))
plt.plot(acc_vec, label='Accuracy')
plt.plot(smoothed_acc, label='Smoothed accuracy', linestyle = '--', c= 'yellow', linewidth=2)
plt.scatter(range(window_size - 1, len(acc_vec)), (plateau_vec[:,0]+0.5)/2, label='Plateau Indicator mean', c='red')
plt.scatter(range(window_size - 1, len(acc_vec)), (plateau_vec[:,1]+0.7)/2, label='Plateau Indicator slope', c='green')
plt.xlabel('Time')
plt.ylabel('Accuracy / Plateau Indicator')
plt.title('Accuracy and Plateau Indicator over Time')
plt.legend()
plt.show()

print('done')