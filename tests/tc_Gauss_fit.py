##### THIS CODE IS UNDER DEVELOPMENT - WILL SERVE AS A BASES FOR SMOOTHING TUNING CURVES AND CALCULATING MORE ACCURATE TUNING CURVE FEATURES #####

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# (1) check what happens with the middle layer tuning curves when I adjust the phase of the input stimuli to match the phase of the Gabors
# (2) center the x_data, y_data to have the peak in the middle - it's because of the circularity of the tuning curves
# (3) automate the initial_guess values
# (4) cut the Gaussians so that it only applies for the bumps

# Define a Gaussian function
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

# Define a function that is a sum of N Gaussians
def sum_of_gaussians(x, *params):
    y = np.zeros_like(x)
    num_gaussians = len(params) // 3
    for i in range(num_gaussians):
        amp = params[i * 3]
        cen = params[i * 3 + 1]
        wid = params[i * 3 + 2]
        y += gaussian(x, amp, cen, wid)
    return y

# Generate some sample data
x_data = np.linspace(0, 90, 90)
# data from Apr10_v1 run 20 tc_postpre.csv
y_data = np.array([1.65,1.74,1.70,1.58,1.43,1.42,1.42,1.42,1.54,1.81,2.19,2.65,3.19,3.76,4.35,4.93,5.49,6.03,6.56,7.09,7.63,8.22,8.87,9.60,10.4,11.3,12.4,13.4,14.5,15.5,16.3,16.8,16.7,15.9,14.2,11.5,7.90,3.80,1.42,1.42,1.42,1.42,1.42,1.42,1.42,1.42,1.42,1.42,3.57,12.1,19.9,24.5,24.7,20.2,12.1,3.15,1.42,1.42,1.42,1.42,1.42,1.42,1.42,7.71,13.3,15.3,13.4,8.59,3.01,1.42,1.42,1.42,1.42,1.42,1.42,2.01,4.50,6.18,6.41,5.34,3.59,1.94,1.42,1.42,1.42,1.42,1.42,1.42,1.42,1.42])

# Initial guess for parameters: [amp1, cen1, wid1, amp2, cen2, wid2, ...]
initial_guess = [17,30, 4, 26,50, 2, 15,70, 2,7, 80,2]

# Fit the sum of Gaussians to the data
popt, pcov = curve_fit(sum_of_gaussians, x_data, y_data, p0=initial_guess)

# Generate the fitted curve
y_fit = sum_of_gaussians(x_data, *popt)

# Plot the data and the fit
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'b-', label='data')
plt.plot(x_data, y_fit, 'r-', label='fit')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

plt.savefig('Gauss_fit_test2')
print(popt)
# If you want to fit the curves in the provided image, you'd replace y_data with your actual data arrays
