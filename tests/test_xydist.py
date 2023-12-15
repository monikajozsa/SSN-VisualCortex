import numpy as np
gridsize_Nx = 9
Ni = Ne = gridsize_Nx**2
gridsize_deg=2 * 1.6
gridsize_mm = gridsize_deg * 2
Lx = Ly = gridsize_mm
Nx = Ny = gridsize_Nx

# Simplified meshgrid creation
xs = np.linspace(0, Lx, Nx)
ys = np.linspace(0, Ly, Ny)
[X, Y] = np.meshgrid(xs - xs[len(xs) // 2], ys - ys[len(ys) // 2]) 

# Flatten and tile in one step
x_vec = np.tile(X.ravel(), 2)
y_vec = np.tile(Y.ravel(), 2)

# Reshape x_vec and y_vec
xs = x_vec.reshape(2, Ne, 1)
ys = y_vec.reshape(2, Ne, 1)

# Calculate distance using broadcasting
xy_dist = np.sqrt(np.square(xs[0] - xs[0].T) + np.square(ys[0] - ys[0].T))

xs2 = np.linspace(0, Lx, Nx)
ys2 = np.linspace(0, Ly, Ny)
[X2, Y2] = np.meshgrid(	xs2 - xs2[Nx // 2], ys2 - ys2[Ny // 2]) 
Y2 = -Y2 
x_vec2 = np.tile(X2.ravel(), (2,))
y_vec2 = np.tile(Y2.ravel(), (2,))
#Prevent kink in function
absdiff_x2 = absdiff_y2 = lambda d_x2: np.abs(d_x2)
xs2 = np.reshape(x_vec2, (2, Ne, 1)) # (cell-type, grid-location, None)
ys2 = np.reshape(y_vec2, (2, Ne, 1)) # (cell-type, grid-location, None)
# to generalize the next two lines, can replace 0's with a and b in range(2) (pre and post-synaptic cell-type indices)
xy_dist2 = np.sqrt(absdiff_x2(xs2[0] - xs2[0].T)**2 + absdiff_y2(ys2[0] - ys2[0].T)**2)

all(xy_dist==xy_dist2)