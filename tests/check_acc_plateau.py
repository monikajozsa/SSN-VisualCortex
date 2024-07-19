import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def has_plateaued_old(accuracy, window=50, threshold=0.01, slope_threshold=0.1):
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

from scipy.optimize import curve_fit
from scipy.stats import ttest_ind
def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def has_plateaued(loss, slope_threshold=0.001):
    """
    Check if the loss has plateaued by fitting an exponential decay curve and checking the derivative at the end.

    Parameters:
    loss (list): A list of loss values over time.
    slope_threshold (float): The maximum allowable derivative value to consider as plateau.

    Returns:
    bool: True if the loss has plateaued, False otherwise.
    """
    if len(loss) < 2:
        return False

    # Fit an exponential decay function to the loss vector
    x = np.arange(len(loss))
    popt, _ = curve_fit(exponential_decay, x, loss, maxfev=10000)

    # Calculate the derivative of the exponential decay function
    a, b, c = popt
    
    fitted_curve = exponential_decay(x, a, b, c)

    # plot the fitted curve and the loss vector
    plt.plot(x, loss, label='Loss')
    plt.plot(x, fitted_curve, label='Fitted Curve', c='red')
    plt.show()
    
    # Check if the derivatives at the end of the loss vector is small and if the mean of the last 10 values is significantly different from the previous 10 values
    end_derivative = -a * b * np.exp(-b * x[-1])
    _, p_value = ttest_ind(loss[-10:-1], loss[-20:-10], equal_var=False)
    plt.close
    return abs(end_derivative) < slope_threshold and p_value > 0.05

# Read in the file 'Jul16_v0/results.csv' and create a numpy array from the column 'acc' as acc_vec
data = pd.read_csv('results/Jul16_v0_NoPretrain/results.csv')
loss_vec = data['loss_all'].to_numpy()
loss_vec = loss_vec[0:1000]
# Smooth the data using a moving average
#smooth_window_size = 10
#smoothed_acc =np.convolve(acc_vec, np.ones(smooth_window_size)/smooth_window_size, mode='valid')

# Apply has_plateaued function to acc_vec in a for loop with a sliding window of 50
window_size = 100
plateau_vec = np.zeros((len(loss_vec) - window_size + 1))

for i in range(0,len(loss_vec) - window_size, 50):
    loss_vec_i = loss_vec[0:i +window_size]
    a = has_plateaued(loss_vec_i)
    plateau_vec[i] = int(a)

# Plot acc_vec and plateau_vec on the same plot (note that plateau_vec is 0 or 1)
plt.figure(figsize=(12, 6))
plt.plot(loss_vec, label='Accuracy')
plt.scatter(range(window_size - 1, len(loss_vec)), (plateau_vec+0.5)/2, label='Plateau Indicator', c='red')
plt.xlabel('Time')
plt.ylabel('Accuracy / Plateau Indicator')
plt.title('Accuracy and Plateau Indicator over Time')
plt.legend()
plt.show()

print('done')