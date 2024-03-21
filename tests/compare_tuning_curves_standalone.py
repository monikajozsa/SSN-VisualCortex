# MAIN SCRIPT STARTS AT ABOUT LINE 1700
import time
import os
import math
from PIL import Image
import jax
import numpy
import jax.numpy as np
from jax import random, vmap
import pandas as pd
import matplotlib.pyplot as plt

from parameters import StimuliPars, ssn_pars, grid_pars, filter_pars, stimuli_pars, conv_pars, ssn_layer_pars, loss_pars, training_pars, pretrain_pars, readout_pars
from visualization import tuning_curve
from util_gabor import create_gabor_filters_ori_map, BW_image_jit, UntrainedPars, cosdiff_ring
from util import sep_exponentiate, load_parameters
from model import middle_layer_fixed_point as mlfp_mj
from model import obtain_fixed_point as ofp_mj
from SSN_classes import SSN_mid

make_J2x2_o = lambda Jee, Jei, Jie, Jii: np.array([[Jee, -Jei], [Jie,  -Jii]])

class conn_pars_m():
    PERIODIC: bool = False
    p_local = None

class conn_pars_s():
    PERIODIC: bool = False
    p_local = None

def Euler2fixedpt(dxdt, x_initial, Tmax, dt, xtol=1e-5, xmin=1e-0, Tmin=200, PLOT=True, save= None, inds=None, verbose=True, silent=False, print_dt = False):
    """
    Finds the fixed point of the D-dim ODE set dx/dt = dxdt(x), using the
    Euler update with sufficiently large dt (to gain in computational time).
    Checks for convergence to stop the updates early.

    IN:
    dxdt = a function handle giving the right hand side function of dynamical system
    x_initial = initial condition for state variables (a column vector)
    Tmax = maximum time to which it would run the Euler (same units as dt, e.g. ms)
    dt = time step of Euler
    xtol = tolerance in relative change in x for determining convergence
    xmin = for x(i)<xmin, it checks convergenece based on absolute change, which must be smaller than xtol*xmin
        Note that one can effectively make the convergence-check purely based on absolute,
        as opposed to relative, change in x, by setting xmin to some very large
        value and inputting a value for 'xtol' equal to xtol_desired/xmin.
    PLOT: if True, plot the convergence of some component
    inds: indices of x (state-vector) to plot

    OUT:
    xvec = found fixed point solution
    CONVG = True if determined converged, False if not
    """

    Nmax = np.round(Tmax/dt).astype(int)
    Nmin = np.round(Tmin/dt) if Tmax > Tmin else (Nmax/2)
    xvec = x_initial 
    CONVG = False
    
    for n in range(Nmax):
        
        dx = dxdt(xvec) * dt
        
        xvec = xvec + dx
        
        if n > Nmin:
            if np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max() < xtol: # y
                if verbose:
                    print("      converged to fixed point at iter={},      as max(abs(dx./max(xvec,{}))) < {} ".format(n, xmin, xtol))
                CONVG = True
                break

    if not CONVG and not silent: # n == Nmax:
        print("\n Warning 1: reached Tmax={}, before convergence to fixed point.".format(Tmax))
        print("       max(abs(dx./max(abs(xvec), {}))) = {},   xtol={}.\n".format(xmin, np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max(), xtol))


    return xvec, CONVG

class _SSN_Base(object):
    def __init__(self, n, k, Ne, Ni, tau_vec=None, W=None):
        self.n = n
        self.k = k
        self.Ne = Ne
        self.Ni = Ni
        self.N = self.Ne + self.Ni

        ## JAX CHANGES ##
        self.EI=[b"E"]*(self.Ne) + [b"I"]*(self.N - self.Ne)
        self.condition= np.array([bool(self.EI[x]==b"E") for x in range(len(self.EI))])
        
        if tau_vec is not None:
            self.tau_vec = tau_vec # rate time-consants of neurons. shape: (N,)
        # elif  not hasattr(self, "tau_vec"):
        #     self.tau_vec = np.random.rand(N) * 20 # in ms
        if W is not None:
            self.W = W # connectivity matrix. shape: (N, N)
        # elif  not hasattr(self, "W"):
        #     W = np.random.rand(N,N) / np.sqrt(self.N)
        #     sign_vec = np.hstack(np.ones(self.Ne), -np.ones(self.Ni))
        #     self.W = W * sign_vec[None, :] # to respect Dale

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k)

    @property
    def dim(self):
        return self.N

    @property
    def tau_x_vec(self):
        """ time constants for the generalized state-vector, x """
        return self.tau_vec


    def powlaw(self, u):
        return  self.k * np.maximum(0,u)**self.n

    def drdt(self, r2, inp_vec):
        out = ( -r2 + self.powlaw(self.W @ r2 + inp_vec) ) / self.tau_vec
        return out

    def drdt_multi(self, r, inp_vec, print_dt = False):
        """
        Compared to self.drdt allows for inp_vec and r to be
        matrices with arbitrary shape[1]
        """
        return (( -r + self.powlaw(self.W @ r + inp_vec) ).T / self.tau_vec ).T

    def dxdt(self, x, inp_vec):
        """
        allowing for descendent SSN types whose state-vector, x, is different
        than the rate-vector, r.
        """
        return self.drdt(x, inp_vec)

    def gains_from_v(self, v):
        return self.n * self.k * np.maximum(0,v)**(self.n-1)

    def gains_from_r(self, r):
        return self.n * self.k**(1/self.n) * r**(1-1/self.n)

    def DCjacobian(self, r):
        """
        DC Jacobian (i.e. zero-frequency linear response) for
        linearization around rate vector r
        """
        Phi = self.gains_from_r(r)
        return -np.eye(self.N) + Phi[:, None] * self.W

    def jacobian(self, DCjacob=None, r=None):
        """
        dynamic Jacobian for linearization around rate vector r
        """
        if DCjacob is None:
            assert r is not None
            DCjacob = self.DCjacobian(r)
        return DCjacob / self.tau_x_vec[:, None] # equivalent to np.diag(tau_x_vec) * DCjacob

    def jacobian_eigvals(self, DCjacob=None, r=None):
        Jacob = self.jacobian(DCjacob=DCjacob, r=r)
        return np.linalg.eigvals(Jacob)

    def inv_G(self, omega, DCjacob, r=None):
        """
        inverse Green's function at angular frequency omega,
        for linearization around rate vector r
        """
        if DCjacob is None:
            assert r is not None
            DCjacob = self.DCjacobian(r)
        return -1j*omega * np.diag(self.tau_x_vec) - DCjacob

   ######## USE IN FIXED POINT FUNCTION #################
    
    def fixed_point_r(self, inp_vec, r_init=None, Tmax=500, dt=1, xtol=1e-5, PLOT=False, save=None):
        
        if r_init is None:
            r_init = np.zeros(inp_vec.shape) # np.zeros((self.N,))
        drdt = lambda r : self.drdt(r, inp_vec)
        if inp_vec.ndim > 1:
            drdt = lambda r : self.drdt_multi(r, inp_vec)

        r_fp, avg_dx = self.Euler2fixedpt_fullTmax(drdt, r_init, Tmax, dt, xtol=xtol, PLOT=PLOT, save=save)

        return r_fp, avg_dx

    def fixed_point_r_mj(self, inp_vec, r_init=None, Tmax=500, dt=1, xtol=1e-5, xmin=1e-0):
        
        if r_init is None:
            r_init = np.zeros(inp_vec.shape) # np.zeros((self.N,))
        drdt = lambda r : self.drdt(r, inp_vec)

        Nmax = int(Tmax/dt)
        r_fp = r_init 
        y = np.zeros(((Nmax)))  
		
        def loop(n, carry):
            r_fp, y = carry
            dr = drdt(r_fp) * dt
            r_fp = r_fp + dr
            y = y.at[n].set(np.abs( dr /np.maximum(xmin, np.abs(r_fp)) ).max())
            return (r_fp, y)

        r_fp, y = jax.lax.fori_loop(0, Nmax, loop, (r_fp, y))
        
        avg_dx = y[int(Nmax/2):int(Nmax)].mean()/xtol
    
        return r_fp, avg_dx

    
    #@partial(jax.jit, static_argnums=(0, 1, 3, 4, 5, 6, 7, 8), device = jax.devices()[1])
    def Euler2fixedpt_fullTmax(self, dxdt, x_initial, Tmax, dt, xtol=1e-5, xmin=1e-0, Tmin=200, PLOT= False, save=None):
        
        Nmax = int(Tmax/dt)
        xvec = x_initial 
        CONVG = False
        y = np.zeros(((Nmax)))        
        
        
        if PLOT:
                
                xplot_all = np.zeros(((Nmax+1)))
                xplot_all = xplot_all.at[0].set(np.sum(xvec))
                


                def loop(n, carry):
                    xvec, y, xplot_all = carry
                    dx = dxdt(xvec) * dt
                    xvec = xvec + dx
                    y = y.at[n].set(np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max())
                    xplot_all = xplot_all.at[n+1].set(np.sum(xvec))
                    return (xvec, y, xplot_all)

                xvec, y, xplot_all = jax.lax.fori_loop(0, Nmax, loop, (xvec, y, xplot_all))
            
            
        else:
                
                def loop(n, carry):
                    xvec, y = carry
                    dx = dxdt(xvec) * dt
                    xvec = xvec + dx
                    y = y.at[n].set(np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max())
                    return (xvec, y)

                xvec, y = jax.lax.fori_loop(0, Nmax, loop, (xvec, y))
        
        avg_dx = y[int(Nmax/2):int(Nmax)].mean()/xtol
        
        #CONVG = np.abs( dx /np.maximum(xmin, np.abs(xvec)) ).max() < xtol
        
        if PLOT:
            import matplotlib.pyplot as plt
            plt.figure(244459)
            plt.plot(np.arange(Nmax+1)*dt, xplot_all) #Nmax+2
            plt.title('Converged to sum of '+str(np.sum(xvec)))
            
            if save:
                    plt.savefig(save+'.png')
            plt.show()
            plt.close()
        
        
        return xvec, avg_dx


##############################################################        
class GaborFilter:
    
    def __init__(self, x_i, y_i, k, sigma_g, theta, edge_deg, degree_per_pixel, phase=0, conv_factor=None):
        
        '''
        Gabor filter class.
        Inputs:
            x_i, y_i: centre of filter
            k: preferred spatial frequency in cycles/degrees (radians)
            sigma_g: variance of Gaussian function
            theta: preferred oritnation 
            conv_factor: conversion factor from degrees to mm
        '''
        
        #convert to mm from degrees
        if conv_factor:
            self.conv_factor = conv_factor
            self.x_i=x_i/conv_factor
            self.y_i=y_i/conv_factor
        else:
            self.x_i=x_i
            self.y_i=y_i
        self.k=k 
        self.theta=theta*(np.pi/180) 
        self.phase=phase 
        self.sigma_g=sigma_g
        self.edge_deg = edge_deg
        self.degree_per_pixel = degree_per_pixel
        self.N_pixels=int(edge_deg*2/degree_per_pixel) +1 
        
        
        #create image axis
        x_axis=np.linspace(-edge_deg, edge_deg, self.N_pixels, endpoint=True)  
        y_axis=np.linspace(-edge_deg, edge_deg, self.N_pixels, endpoint=True)
        
        #construct filter as attribute
        self.filter = self.create_filter(x_axis, y_axis)

    
    def create_filter(self, x_axis, y_axis):
        '''
        Create Gabor filters in vectorised form. 
        '''
        x_axis=np.reshape(x_axis, (self.N_pixels, 1))
        #self.theta=np.pi/2 - self.theta
       
        x_i=np.repeat(self.x_i, self.N_pixels)
        x_i=np.reshape(x_i, (self.N_pixels, 1))
        diff_x= (x_axis.T - x_i)

        y_axis=np.reshape(y_axis, (self.N_pixels, 1))

        y_i=np.repeat(self.y_i, self.N_pixels)
        y_i=np.reshape(y_i, (self.N_pixels, 1))
        diff_y=((y_axis - y_i.T))
        
        spatial=np.cos(self.k*np.pi*2*(diff_x*np.cos(self.theta) + diff_y*np.sin(self.theta)) + self.phase) 
        gaussian= np.exp(-0.5 *( diff_x**2 + diff_y**2)/self.sigma_g**2)
        
        return gaussian*spatial[::-1] #same convention as stimuli
    
class BW_Grating:
    """ """

    def __init__(
        self,
        ori_deg,
        stimuli_pars,
        jitter=0,
        phase=0,
        crop_f=None,
    ):
        self.ori_deg = ori_deg
        self.jitter = jitter
        self.outer_radius = stimuli_pars.outer_radius  # in degrees
        self.inner_radius = stimuli_pars.inner_radius  # in degrees
        self.grating_contrast = stimuli_pars.grating_contrast
        self.std = stimuli_pars.std
        degree_per_pixel = stimuli_pars.degree_per_pixel
        pixel_per_degree = 1 / degree_per_pixel
        self.pixel_per_degree = pixel_per_degree
        edge_deg = stimuli_pars.edge_deg
        size = int(edge_deg * 2 * pixel_per_degree) + 1
        self.size = size
        k = stimuli_pars.k
        spatial_frequency = k * degree_per_pixel  # 0.05235987755982988
        self.phase = phase
        self.crop_f = crop_f
        self.smooth_sd = self.pixel_per_degree / 6
        self.spatial_freq = spatial_frequency or (1 / self.pixel_per_degree)
        self.grating_size = round(self.outer_radius * self.pixel_per_degree)
        self.angle = ((self.ori_deg + self.jitter) - 90) / 180 * numpy.pi
        
    def BW_image(self):
        _BLACK = 0
        _WHITE = 255
        _GRAY = round((_WHITE + _BLACK) / 2)

        # Generate a 2D grid of coordinates
        x, y = numpy.mgrid[
            -self.grating_size : self.grating_size + 1.0,
            -self.grating_size : self.grating_size + 1.0,
        ]

        # Calculate the distance from the center for each pixel
        edge_control_dist = numpy.sqrt(numpy.power(x, 2) + numpy.power(y, 2))
        edge_control = numpy.divide(edge_control_dist, self.pixel_per_degree)

        # Create a matrix (alpha_channel) that is 255 (white) within the inner_radius and exponentially fades to 0 as the radius increases
        overrado = numpy.nonzero(edge_control > self.inner_radius)
        d = self.grating_size * 2 + 1
        annulus = numpy.ones((d, d))

        annulus[overrado] *= numpy.exp(
            -1
            * ((edge_control[overrado] - self.inner_radius) * self.pixel_per_degree)
            ** 2
            / (2 * (self.smooth_sd**2))
        )
        alpha_channel = annulus * _WHITE

        # Generate the grating pattern, which is a centered and tilted sinusoidal matrix
        spatial_component = (
            2
            * math.pi
            * self.spatial_freq
            * (y * numpy.sin(self.angle) + x * numpy.cos(self.angle))
        )
        gabor_sti = _GRAY * (
            1 + self.grating_contrast * numpy.cos(spatial_component + self.phase)
        )

        # Set pixels outside the grating size to gray
        gabor_sti[edge_control_dist > self.grating_size] = _GRAY
        
        # Add Gaussian white noise to the grating
        #noise = numpy.random.normal(loc=0, scale=self.std, size=gabor_sti.shape)
        #noisy_gabor_sti = gabor_sti + noise

        # Expand the grating to have three colors andconcatenate it with alpha_channel
        gabor_sti_final = numpy.repeat(gabor_sti[:, :, numpy.newaxis], 3, axis=-1)
        gabor_sti_final_with_alpha = numpy.concatenate(
            (gabor_sti_final, alpha_channel[:, :, numpy.newaxis]), axis=-1
        )
        gabor_sti_final_with_alpha_image = Image.fromarray(
            gabor_sti_final_with_alpha.astype(numpy.uint8)
        )

        # Create a background image filled with gray
        background = numpy.full((self.size, self.size, 3), _GRAY, dtype=numpy.uint8)
        final_image = Image.fromarray(background)

        # Paste the grating into the final image: paste the grating into a bounding box and apply the alpha channel as a mask
        center_x, center_y = self.size // 2, self.size // 2
        bounding_box = (center_x - self.grating_size, center_y - self.grating_size)
        final_image.paste(
            gabor_sti_final_with_alpha_image,
            box=bounding_box,
            mask=gabor_sti_final_with_alpha_image,
        )

        # Sum the image over color channels
        final_image_np = numpy.array(final_image, dtype=numpy.float16)
        image = numpy.sum(final_image_np, axis=2)

        # Crop the image if crop_f is specified
        if self.crop_f:
            image = image[self.crop_f : -self.crop_f, self.crop_f : -self.crop_f]
    
        noise = numpy.random.normal(loc=0, scale=self.std, size=image.shape)

        image_noisy = image + noise
        
        return image_noisy
    
### FINDING CONSTANT FOR GABOR FILTERS ###
def find_A(
    k,
    sigma_g,
    edge_deg,
    degree_per_pixel,
    indices,
    phase=0,
    return_all=False,
):
    """
    Find constant to multiply Gabor filters.
    Input:
        gabor_pars: Filter parameters - centre already specified in function
        stimuli_pars: Stimuli parameters (high constrast and spanning all visual field)
        indices: List of orientatins in degrees to calculate filter and corresponding stimuli
    Output:
        A: value of constant so that contrast = 100
    """
    all_A = []
    all_stimuli_mean =[]
    all_stimuli_max = []
    all_stimuli_min = []

    for ori in indices:
        # generate Gabor filter and stimuli at orientation
        gabor = GaborFilter(
            theta=ori,
            x_i=0,
            y_i=0,
            edge_deg=edge_deg,
            k=k,
            sigma_g=sigma_g,
            degree_per_pixel=degree_per_pixel,
            phase=phase,
        )
        # create local_stimui_pars to pass it to the BW_Gratings
        local_stimuli_pars = StimuliPars()
        local_stimuli_pars.edge_deg=edge_deg
        local_stimuli_pars.k=k
        local_stimuli_pars.outer_radius=edge_deg * 2
        local_stimuli_pars.inner_radius=edge_deg * 2
        local_stimuli_pars.degree_per_pixel=degree_per_pixel
        local_stimuli_pars.grating_contrast=0.99
        local_stimuli_pars.jitter = 0
        local_stimuli_pars.std = 0
        
        #Create test grating
        test_grating = BW_Grating(
            ori_deg=ori,
            jitter = local_stimuli_pars.jitter,    
            stimuli_pars=local_stimuli_pars,
            phase=phase,
        )
        test_stimuli = test_grating.BW_image()
        mean_removed_filter = gabor.filter - gabor.filter.mean()
        
        all_stimuli_mean.append(test_stimuli.mean())
        all_stimuli_min.append(test_stimuli.min())
        all_stimuli_max.append(test_stimuli.max())
        
        # multiply filter and stimuli
        output_gabor = mean_removed_filter.ravel() @ test_stimuli.ravel()

        # calculate value of A
        A_value = 100 / (output_gabor)

        # create list of A
        all_A.append(A_value)

    # find average value of A
    all_A = np.array(all_A)
    A = all_A.mean()
            
    return A

class SSN2DTopoV1_ONOFF(_SSN_Base):
    _Lring = 180

    def __init__(self, ssn_pars, grid_pars,  conn_pars, filter_pars, J_2x2, gE, gI, sigma_oris =None, s_2x2 = None, ori_map = None, number_phases = 2, **kwargs):
        self.Nc = grid_pars.gridsize_Nx**2 #number of columns
        Ni = Ne = 2 * self.Nc 
        
        n=ssn_pars.n
        k=ssn_pars.k
        tauE= ssn_pars.tauE
        self.tauE = tauE
        self.tauI = tauI
        tauI=ssn_pars.tauI
        self.phases = ssn_pars.phases
        tau_vec = np.hstack([tauE * np.ones(self.Nc), tauI * np.ones(self.Nc)])
        tau_vec = np.kron(np.array([1,1]), tau_vec )
        
  
        super(SSN2DTopoV1_ONOFF, self).__init__(n=n, k=k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, **kwargs)
        
        self.grid_pars = grid_pars
        self.conn_pars = conn_pars
        self._make_retinmap()
        
        if ori_map==None:
            self.ori_map = self._make_orimap()
        else:
            self.input_ori_map(ori_map)
        
        self.gE, self.gI = gE, gI
        

       
        self.edge_deg = filter_pars.edge_deg
        self.sigma_g = filter_pars.sigma_g
        self.k = filter_pars.k
        self.conv_factor =  filter_pars.conv_factor
        self.degree_per_pixel = filter_pars.degree_per_pixel
        
        self.A=ssn_pars.A

        #Create Gabor filters
        self.gabor_filters, self.A = self.create_gabor_filters()
        
            
        
        #if conn_pars is not None: # conn_pars = None allows for ssn-object initialization without a W
            
        self.make_W(J_2x2, s_2x2, sigma_oris)

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k,
                    tauE=self.tau_vec[0], tauI=self.tau_vec[self.Ne])
    @property
    def maps_vec(self):
        return np.vstack([self.x_vec, self.y_vec, self.ori_vec]).T

    @property
    def center_inds(self):
        """ indices of center-E and center-I neurons """
        return np.where((self.x_vec==0) & (self.y_vec==0))[0]

    @property
    def x_vec_degs(self):
        return self.x_vec / self.grid_pars.magnif_factor

    @property
    def y_vec_degs(self):
        return self.y_vec / self.grid_pars.magnif_factor

    def xys2inds(self, xys=[[0,0]], units="degree"):
        """
        indices of E and I neurons at location (x,y) (by default in degrees).
        In:
            xys: array-like list of xy coordinates.
            units: specifies unit for xys. By default, "degree" of visual angle.
        Out:
            inds: shape = (2, len(xys)), inds[0] = vector-indices of E neurons
                                         inds[1] = vector-indices of I neurons
        """
        inds = []
        for xy in xys:
            if units == "degree": # convert to mm
                xy = self.grid_pars.magnif_factor * np.asarray(xy)
            distsq = (self.x_vec - xy[0])**2 + (self.y_vec - xy[1])**2
            inds.append([np.argmin(distsq[:self.Ne]), self.Ne + np.argmin(distsq[self.Ne:])])
        return np.asarray(inds).T

    def xys2Emapinds(self, xys=[[0,0]], units="degree"):
        """
        (i,j) of E neurons at location (x,y) (by default in degrees).
        In:
            xys: array-like list of xy coordinates.
            units: specifies unit for xys. By default, "degree" of visual angle.
        Out:
            map_inds: shape = (2, len(xys)), inds[0] = row_indices of E neurons in map
                                         inds[1] = column-indices of E neurons in map
        """
        vecind2mapind = lambda i: np.array([i % self.grid_pars.gridsize_Nx,
                                            i // self.grid_pars.gridsize_Nx])
        return vecind2mapind(self.xys2inds(xys)[0])

    def vec2map(self, vec):
        assert vec.ndim == 1
        Nx = self.grid_pars.gridsize_Nx
        if len(vec) == self.Nc:
            map = np.reshape(vec, (Nx, Nx))
        elif len(vec) == self.Ne:
            map = (np.reshape(vec[:self.Nc], (Nx, Nx)),
                   np.reshape(vec[self.Nc:], (Nx, Nx)))
        elif len(vec) == self.N:
            map = (np.reshape(vec[:self.Nc], (Nx, Nx)),
                   np.reshape(vec[self.Nc:self.Nc*2], (Nx, Nx)),
                   np.reshape(vec[self.Nc*2:self.Nc*3], (Nx, Nx)),
                   np.reshape(vec[self.Nc*3:], (Nx, Nx)))
            
       
        return map

    def _make_maps(self, grid_pars=None):
        if grid_pars is None:
            grid_pars = self.grid_pars
        else:
            self.grid_pars = grid_pars

        self._make_retinmap()
        self._make_orimap()

        return self.x_map, self.y_map, self.ori_map
    
    def input_ori_map(self, ori_map):
        self.ori_map= ori_map
        self.ori_vec = np.tile(self.ori_map.ravel(), (self.phases*2,))
        self._make_distances()
        self._make_retinmap()

    def _make_retinmap(self, grid_pars=None):
        """
        make square grid of locations with X and Y retinotopic maps
        """
        if grid_pars is None:
            grid_pars = self.grid_pars
        else:
            self.grid_pars = grid_pars
        if not hasattr(grid_pars, "gridsize_mm"):
            self.grid_pars.gridsize_mm = grid_pars.gridsize_deg * grid_pars.magnif_factor
        Lx = Ly = self.grid_pars.gridsize_mm
        Nx = Ny = grid_pars.gridsize_Nx
        dx = dy = Lx/(Nx - 1)
        self.grid_pars.dx = dx # in mm
        self.grid_pars.dy = dy # in mm

        xs = np.linspace(0, Lx, Nx)
        ys = np.linspace(0, Ly, Ny)
        [X, Y] = np.meshgrid(xs - xs[len(xs)//2], ys - ys[len(ys)//2]) # doing it this way, as opposed to using np.linspace(-Lx/2, Lx/2, Nx) (for which this fails for even Nx), guarantees that there is always a pixel with x or y == 0
        Y = -Y # without this Y decreases going upwards

        self.x_map = X
        self.y_map = Y
        
        self.x_vec = np.tile(X.ravel(), (self.phases*2,))
        self.y_vec = np.tile(Y.ravel(), (self.phases*2,))
      
        
        return self.x_map, self.y_map

    def _make_orimap(self, hyper_col=None, nn=30, X=None, Y=None):
        '''
        Makes the orientation map for the grid, by superposition of plane-waves.
        hyper_col = hyper column length for the network in retinotopic degrees
        nn = (30 by default) # of planewaves used to construct the map

        Outputs/side-effects:
        OMap = self.ori_map = orientation preference for each cell in the network
        self.ori_vec = vectorized OMap
        '''
        if hyper_col is None:
             hyper_col = self.grid_pars.hyper_col
        else:
             self.grid_pars.hyper_col = hyper_col
        X = self.x_map if X is None else X
        Y = self.y_map if Y is None else Y

        z = np.zeros_like(X)
        #key = random.PRNGKey(87)
        #numpy.random.seed(6)
        for j in range(nn):
            kj = np.array([np.cos(j * np.pi/nn), np.sin(j * np.pi/nn)]) * 2*np.pi/(hyper_col)
            
            ## JAX CHANGES ##
            #key, subkey = random.split(key)
            #sj = 2 *random.randint(key=key, shape=[1,1], minval=0, maxval=2)-1 #random number that's either + or -1.
            #key, subkey = random.split(key)
            #phij = random.uniform(key, shape=[1,1], minval=0, maxval=1)*2*np.pi
            
            #NUMPY RANDOM
            sj = 2 * numpy.random.randint(0, 2)-1 #random number that's either + or -1.
            phij = numpy.random.rand()*2*np.pi

            tmp = (X*kj[0] + Y*kj[1]) * sj + phij
            z = z + np.exp(1j * tmp)


        # ori map with preferred orientations in the range (0, _Lring] (i.e. (0, 180] by default)
        self.ori_map = (np.angle(z) + np.pi) * SSN2DTopoV1_ONOFF._Lring/(2*np.pi)
        self.ori_vec = np.tile(self.ori_map.ravel(), (4,))

        return self.ori_map

    def _make_distances(self):
        Lx = Ly = self.grid_pars.gridsize_mm
        absdiff_ring = lambda d_x, L: np.minimum(np.abs(d_x), L - np.abs(d_x))
        cosdiff_ring = lambda d_x, L: np.sqrt(2 * (1 - np.cos(d_x * 2 * np.pi/L))) * L / 2/ np.pi
        PERIODIC = self.conn_pars.PERIODIC
        if PERIODIC:
            absdiff_x = absdiff_y = lambda d_x: absdiff_ring(d_x, Lx + self.grid_pars.dx)
        else:
            absdiff_x = absdiff_y = lambda d_x: np.abs(d_x)
        
        xs = np.reshape(self.x_vec, (self.phases*2, self.Nc, 1)) # (cell-type, grid-location, None)
        ys = np.reshape(self.y_vec, (self.phases*2, self.Nc, 1)) # (cell-type, grid-location, None)
        oris = np.reshape(self.ori_vec, (self.phases*2, self.Nc, 1)) # (cell-type, grid-location, None)
        
        # to generalize the next two lines, can replace 0's with a and b in range(2) (pre and post-synaptic cell-type indices)
        xy_dist = np.sqrt(absdiff_x(xs[0] - xs[0].T)**2 + absdiff_y(ys[0] - ys[0].T)**2)
        ori_dist = cosdiff_ring(oris[0] - oris[0].T, SSN2DTopoV1_ONOFF._Lring)
        self.xy_dist = xy_dist
        self.ori_dist = ori_dist

        return xy_dist, ori_dist     
    
    def create_gabor_filters(self):
        
        #Create array of filters
        e_filters=[] 
        if self.phases==4:
            e_filters_pi2 = []

        #Iterate over SSN map
        for i in range(self.ori_map.shape[0]):
            for j in range(self.ori_map.shape[1]):
                gabor=GaborFilter(x_i=self.x_map[i,j], y_i=self.y_map[i,j], edge_deg=self.edge_deg, k=self.k_filt, sigma_g=self.sigma_g, theta=self.ori_map[i,j], conv_factor=self.conv_factor, degree_per_pixel=self.degree_per_pixel)
                
                e_filters.append(gabor.filter.ravel())
                
                if self.phases==4:
                    gabor_2 = GaborFilter(x_i=self.x_map[i,j], y_i=self.y_map[i,j], edge_deg=self.edge_deg, k=self.k_filt, sigma_g=self.sigma_g, theta=self.ori_map[i,j], conv_factor=self.conv_factor, degree_per_pixel=self.degree_per_pixel, phase = np.pi/2)
                    e_filters_pi2.append(gabor_2.filter.ravel())
                
                        
        #i_constant= gI / gE
        e_filters_o =np.array(e_filters)
        e_filters = self.gE * e_filters_o
        i_filters = self.gI * e_filters_o

        #create filters with phase equal to pi
        e_off_filters = - e_filters
        i_off_filters = - i_filters
        

        if self.phases ==4:
            e_filters_o_pi2 =np.array(e_filters_pi2)

            e_filters_pi2 = self.gE * e_filters_o_pi2
            i_filters_pi2 = self.gI * e_filters_o_pi2

            #create filters with phase equal to -pi/2
            e_off_filters_pi2 = - e_filters_pi2
            i_off_filters_pi2 = - i_filters_pi2
            SSN_filters=np.vstack([e_filters, i_filters, e_filters_pi2, i_filters_pi2,  e_off_filters, i_off_filters, e_off_filters_pi2, i_off_filters_pi2])
        
        
        else:
            SSN_filters=np.vstack([e_filters, i_filters, e_off_filters, i_off_filters])
            
        
        if self.A == None:
            A= find_A(return_all =False, k=self.k_filt, sigma_g=self.sigma_g, edge_deg=self.edge_deg,  degree_per_pixel=self.degree_per_pixel, indices=np.sort(self.ori_map.ravel()))
            self.A = A
            
        
        #Normalise Gabor filters
        SSN_filters = SSN_filters*self.A
        
        if self.phases ==4:
            if self.A2 ==None:
                A2 = find_A(return_all =False, k=self.k_filt, sigma_g=self.sigma_g, edge_deg=self.edge_deg,  degree_per_pixel=self.degree_per_pixel, indices=np.sort(self.ori_map.ravel()), phase = np.pi/2)
                self.A2 = A2
                
                
            SSN_filters=np.vstack([e_filters*self.A, i_filters*self.A, e_filters_pi2*self.A2, i_filters_pi2*self.A2,  e_off_filters*self.A, i_off_filters*self.A, e_off_filters_pi2*self.A2, i_off_filters_pi2*self.A2])
        
        #remove mean so that input to constant grating is 0
        SSN_filters = SSN_filters - np.mean(SSN_filters, axis=1)[:, None]
        self.gabor_filters = SSN_filters

        return SSN_filters, self.A
    
    def select_type(self, vec, map_number):
        out = vec[(map_number-1)*self.Nc:map_number*self.Nc]
        return out
        

    def apply_bounding_box(self, vec, size = 3.2, select=1):
        
        Nx = self.grid_pars.gridsize_Nx

        map_vec = self.select_type(vec, map_number = select).reshape(Nx,Nx)

        size = int(size / (self.grid_pars.dx)) +1

        start = int((self.grid_pars.gridsize_Nx - size) / 2)   
        
        map_vec = jax.lax.dynamic_slice(map_vec, (start, start), (size, size))
        #map_vec = map_vec[start:start+size, start:start+size]

        return map_vec

class SSN2DTopoV1_ONOFF_local(SSN2DTopoV1_ONOFF):

    def __init__(self, ssn_pars, grid_pars,  conn_pars, filter_pars, J_2x2, gE, gI, ori_map = None, **kwargs):
        self.phases = ssn_pars.phases
        self.Nc = grid_pars.gridsize_Nx**2 #number of columns
        Ni = Ne = self.phases * self.Nc 
        n=ssn_pars.n
        
        self.k=ssn_pars.k
        tauE= ssn_pars.tauE
        tauI=ssn_pars.tauI
        
        tau_vec = np.hstack([tauE * np.ones(self.Nc), tauI * np.ones(self.Nc)])
        tau_vec = np.kron(np.ones((1, self.phases)), tau_vec ).squeeze()
        self.tauE = tauE
        self.tauI = tauI
        #tau_vec = np.kron(np.array([1,1]), tau_vec )
  
        super(SSN2DTopoV1_ONOFF, self).__init__(n=n, k=self.k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, **kwargs)
        
        self.grid_pars = grid_pars
        self.conn_pars = conn_pars
        
        self._make_retinmap()
        if ori_map==None:
            self.ori_map = self._make_orimap()
        else:
            self.input_ori_map(ori_map)
            
        self.gE, self.gI = gE, gI
        
        #Gabor filter parameters
        self.edge_deg = filter_pars.edge_deg
        self.sigma_g = filter_pars.sigma_g
        self.k_filt = filter_pars.k
        self.conv_factor =  filter_pars.conv_factor
        self.degree_per_pixel = filter_pars.degree_per_pixel
        
        if hasattr(ssn_pars, 'A'): # mj deleted A and A2 from ssn_pars as there were out of use
            self.A=ssn_pars.A
            if ssn_pars.phases==4:
                self.A2 = ssn_pars.A2
        else:
            self.A=None
            self.A2=None
                    
        self.gabor_filters, self.A = self.create_gabor_filters()
        
        self.make_local_W(J_2x2)
        

    def drdt(self, r, inp_vec):
        r1 = np.reshape(r, (-1, self.Nc))
        out = ( -r + self.powlaw(np.ravel(self.W @ r1) + inp_vec) ) / self.tau_vec
        return out
    
    def make_local_W(self, J_2x2):
        #self.W = np.kron(np.ones((2,2)), np.asarray(J_2x2))
        self.W = np.kron(np.eye(self.phases), np.asarray(J_2x2))

# =========================== 2D topographic models ============================
class SSN2DTopoV1(_SSN_Base):
    _Lring = 180

    def __init__(self, ssn_pars, grid_pars, conn_pars, J_2x2, sigma_oris =None, s_2x2 = None, ori_map=None, train_ori = None, kappa_post = None, kappa_pre = None, **kwargs):
        Ni = Ne = grid_pars.gridsize_Nx**2
        n=ssn_pars.n
        self.k=ssn_pars.k
        tauE= ssn_pars.tauE
        tauI=ssn_pars.tauI
        self.tauE = tauE
        self.tauI = tauI
        tau_vec = np.hstack([tauE * np.ones(Ne), tauI * np.ones(Ni)])

        super(SSN2DTopoV1, self).__init__(n=n, k=self.k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, **kwargs)

        self.grid_pars = grid_pars
        self.conn_pars = conn_pars
        self.train_ori = train_ori
        self._make_retinmap()
        
        if ori_map==None:
            self.ori_map = self._make_orimap()
        else:
            self.input_ori_map(ori_map)

            
        
        self.s_2x2 = s_2x2
        self.sigma_oris = sigma_oris
   
        #self.edge_deg = filter_pars.edge_deg
        #self.sigma_g = filter_pars.sigma_g
        #self.k_filt = filter_pars.k
        #self.conv_factor =  filter_pars.conv_factor
        #self.degree_per_pixel = filter_pars.degree_per_pixel
        
        if hasattr(ssn_pars,'A'):
            self.A=ssn_pars.A
        else:
            self.A = None
     
        if kappa_pre==None:
            kappa_pre = np.asarray([ 0.0, 0.0])
            kappa_post = kappa_pre
   
        self.W = self.make_W(J_2x2, kappa_pre, kappa_post)

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k,
                    tauE=self.tau_vec[0], tauI=self.tau_vec[self.Ne])
    @property
    def maps_vec(self):
        return np.vstack([self.x_vec, self.y_vec, self.ori_vec]).T

    @property
    def center_inds(self):
        """ indices of center-E and center-I neurons """
        return np.where((self.x_vec==0) & (self.y_vec==0))[0]

    @property
    def x_vec_degs(self):
        return self.x_vec / self.grid_pars.magnif_factor

    @property
    def y_vec_degs(self):
        return self.y_vec / self.grid_pars.magnif_factor

    def xys2inds(self, xys=[[0,0]], units="degree"):
        """
        indices of E and I neurons at location (x,y) (by default in degrees).
        In:
            xys: array-like list of xy coordinates.
            units: specifies unit for xys. By default, "degree" of visual angle.
        Out:
            inds: shape = (2, len(xys)), inds[0] = vector-indices of E neurons
                                         inds[1] = vector-indices of I neurons
        """
        inds = []
        for xy in xys:
            if units == "degree": # convert to mm
                xy = self.grid_pars.magnif_factor * np.asarray(xy)
            distsq = (self.x_vec - xy[0])**2 + (self.y_vec - xy[1])**2
            inds.append([np.argmin(distsq[:self.Ne]), self.Ne + np.argmin(distsq[self.Ne:])])
        return np.asarray(inds).T

    def xys2Emapinds(self, xys=[[0,0]], units="degree"):
        """
        (i,j) of E neurons at location (x,y) (by default in degrees).
        In:
            xys: array-like list of xy coordinates.
            units: specifies unit for xys. By default, "degree" of visual angle.
        Out:
            map_inds: shape = (2, len(xys)), inds[0] = row_indices of E neurons in map
                                         inds[1] = column-indices of E neurons in map
        """
        vecind2mapind = lambda i: np.array([i % self.grid_pars.gridsize_Nx,
                                            i // self.grid_pars.gridsize_Nx])
        return vecind2mapind(self.xys2inds(xys)[0])

    def vec2map(self, vec):
        assert vec.ndim == 1
        Nx = self.grid_pars.gridsize_Nx
        if len(vec) == self.Ne:
            map = np.reshape(vec, (Nx, Nx))
        elif len(vec) == self.N:
            map = (np.reshape(vec[:self.Ne], (Nx, Nx)),
                   np.reshape(vec[self.Ne:], (Nx, Nx)))
        return map

    def _make_maps(self, grid_pars=None):
        if grid_pars is None:
            grid_pars = self.grid_pars
        else:
            self.grid_pars = grid_pars

        self._make_retinmap()
        self._make_orimap()

        return self.x_map, self.y_map, self.ori_map

    def _make_retinmap(self, grid_pars=None):
        """
        make square grid of locations with X and Y retinotopic maps
        """
        if grid_pars is None:
            grid_pars = self.grid_pars
        else:
            self.grid_pars = grid_pars
        if not hasattr(grid_pars, "gridsize_mm"):
            self.grid_pars.gridsize_mm = grid_pars.gridsize_deg * grid_pars.magnif_factor
        Lx = Ly = self.grid_pars.gridsize_mm
        Nx = Ny = grid_pars.gridsize_Nx
        dx = dy = Lx/(Nx - 1)
        self.grid_pars.dx = dx # in mm
        self.grid_pars.dy = dy # in mm

        xs = np.linspace(0, Lx, Nx)
        ys = np.linspace(0, Ly, Ny)
        [X, Y] = np.meshgrid(xs - xs[len(xs)//2], ys - ys[len(ys)//2]) # doing it this way, as opposed to using np.linspace(-Lx/2, Lx/2, Nx) (for which this fails for even Nx), guarantees that there is always a pixel with x or y == 0
        Y = -Y # without this Y decreases going upwards

        self.x_map = X
        self.y_map = Y
        self.x_vec = np.tile(X.ravel(), (2,))
        self.y_vec = np.tile(Y.ravel(), (2,))
        return self.x_map, self.y_map

    def _make_orimap(self, hyper_col=None, nn=30, X=None, Y=None):
        '''
        Makes the orientation map for the grid, by superposition of plane-waves.
        hyper_col = hyper column length for the network in retinotopic degrees
        nn = (30 by default) # of planewaves used to construct the map
        Outputs/side-effects:
        OMap = self.ori_map = orientation preference for each cell in the network
        self.ori_vec = vectorized OMap
        '''
        if hyper_col is None:
             hyper_col = self.grid_pars.hyper_col
        else:
             self.grid_pars.hyper_col = hyper_col
        X = self.x_map if X is None else X
        Y = self.y_map if Y is None else Y

        z = np.zeros_like(X)

        for j in range(nn):
            kj = np.array([np.cos(j * np.pi/nn), np.sin(j * np.pi/nn)]) * 2*np.pi/(hyper_col)

            #NUMPY RANDOM
            sj = 2 * numpy.random.randint(0, 2)-1 #random number that's either + or -1.
            phij = numpy.random.rand()*2*np.pi
            
            tmp = (X*kj[0] + Y*kj[1]) * sj + phij
            z = z + np.exp(1j * tmp)

        # ori map with preferred orientations in the range (0, _Lring] (i.e. (0, 180] by default)
        self.ori_map = (np.angle(z) + np.pi) * SSN2DTopoV1._Lring/(2*np.pi)

        self.ori_vec = np.tile(self.ori_map.ravel(), (2,))
        return self.ori_map
    
    def input_ori_map(self, ori_map):
        self.ori_map= ori_map
        self.ori_vec = np.tile(self.ori_map.ravel(), (2,))
        self._make_distances()
        self._make_retinmap()

    def _make_distances(self):
        PERIODIC = self.conn_pars.PERIODIC
        Lx = Ly = self.grid_pars.gridsize_mm
        absdiff_ring = lambda d_x, L: np.minimum(np.abs(d_x), L - np.abs(d_x))
        #Prevent kink in function
        cosdiff_ring = lambda d_x, L: np.sqrt(2 * (1 - np.cos(d_x * 2 * np.pi/L))) * L / 2/ np.pi
        if PERIODIC:
            absdiff_x = absdiff_y = lambda d_x: absdiff_ring(d_x, Lx + self.grid_pars.dx)
        else:
            absdiff_x = absdiff_y = lambda d_x: np.abs(d_x)
        xs = np.reshape(self.x_vec, (2, self.Ne, 1)) # (cell-type, grid-location, None)
        ys = np.reshape(self.y_vec, (2, self.Ne, 1)) # (cell-type, grid-location, None)
        oris = np.reshape(self.ori_vec, (2, self.Ne, 1)) # (cell-type, grid-location, None)
        
        # to generalize the next two lines, can replace 0's with a and b in range(2) (pre and post-synaptic cell-type indices)
        xy_dist = np.sqrt(absdiff_x(xs[0] - xs[0].T)**2 + absdiff_y(ys[0] - ys[0].T)**2)
        ori_dist = cosdiff_ring(oris[0] - oris[0].T, SSN2DTopoV1._Lring)
        if self.train_ori!=None:
            trained_ori_dist = cosdiff_ring(oris[0] - self.train_ori, SSN2DTopoV1._Lring) #NEW - calculate distance to trained orientation
            self.trained_ori_dist = trained_ori_dist.squeeze()
            
        self.xy_dist = xy_dist
        self.ori_dist = ori_dist
        

        return xy_dist, ori_dist  
  
    
    def make_W(self, J_2x2, kappa_pre, kappa_post, Jnoise=0,
                Jnoise_GAUSSIAN=True, MinSyn=1e-4, CellWiseNormalized=False,
                                                        PERIODIC=True): #, prngKey=0):
            """
            make the full recurrent connectivity matrix W
            In:
             J_2x2 = total strength of weights of different pre/post cell-type
             s_2x2 = ranges of weights between different pre/post cell-type
             p_local = relative strength of local parts of E projections
             sigma_oris = range of wights in terms of preferred orientation difference
            Output/side-effects:
            self.W
            """
            s_2x2 = self.s_2x2
            sigma_oris = self.sigma_oris
            PERIODIC = self.conn_pars.PERIODIC
            p_local = self.conn_pars.p_local

            if hasattr(self, "xy_dist") and hasattr(self, "ori_dist"):
                xy_dist = self.xy_dist
                ori_dist = self.ori_dist
                trained_ori_dist = self.trained_ori_dist
            else:
                xy_dist, ori_dist = self._make_distances()
            
            #Reshape sigma_oris
            if np.shape(sigma_oris) == (1,): sigma_oris = sigma_oris * np.ones((2,2))
            elif np.shape(sigma_oris) == (2,): sigma_oris = np.ones((2,1)) * np.array(sigma_oris)
            
            #Reshape kappa pre
            if np.shape(kappa_pre) == (1,): kappa_pre = kappa_pre * np.ones((2,2))
            elif np.shape(kappa_pre) == (2,): kappa_pre = np.ones((2,1)) * np.array(kappa_pre) 
            
            #Reshape kappa post
            if np.shape(kappa_post) == (1,): kappa_post = kappa_post * np.ones((2,2))
            elif np.shape(kappa_post) == (2,): kappa_post = np.ones((2,1)) * np.array(kappa_post) 
            

            if np.isscalar(s_2x2):
                s_2x2 = s_2x2 * np.ones((2,2))
            else:
                assert s_2x2.shape == (2,2)

            if np.isscalar(p_local) or len(p_local) == 1:
                p_local = np.asarray(p_local) * np.ones(2)

            #Create empty matrix
            Wblks = [[1,1],[1,1]]
            
            # loop over post- (a) and pre-synaptic (b) cell-types
            for a in range(2):
                for b in range(2):
                    '''
                    if b == 0: # E projections
                        W = np.exp(-xy_dist/s_2x2[a,b] -ori_dist**2/(2*sigma_oris[a,b]**2) -trained_ori_dist[:, None]**2/(2*sigma_post[a]**2) -trained_ori_dist[None,:]**2/(2*sigma_pre[b]**2) )

                    elif b == 1: # I projections
                        W = np.exp(-xy_dist**2/(2*s_2x2[a,b]**2) -ori_dist**2/(2*sigma_oris[a,b]**2) -trained_ori_dist[:, None]**2/(2*sigma_post[a]**2) -trained_ori_dist[None,:]**2/(2*sigma_pre[b]**2 )) 
                    ''' 
                    
                    if b == 0: # E projections
                        W = np.exp(-xy_dist/s_2x2[a,b] -ori_dist**2/(2*sigma_oris[a,b]**2) - kappa_post[a,b]*trained_ori_dist[:, None]**2/2 /45**2  -kappa_pre[a,b]*trained_ori_dist[None,:]**2/2/45**2 )
                        
                    elif b == 1: # I projections 
                        W = np.exp(-xy_dist**2/(2*s_2x2[a,b]**2) -ori_dist**2/(2*sigma_oris[a,b]**2) -kappa_post[a,b] * trained_ori_dist[:, None]**2/2/45**2  -kappa_pre[a,b]*trained_ori_dist[None,:]**2/2/45**2)


                    if Jnoise > 0: # add some noise
                        if Jnoise_GAUSSIAN:
                            ##JAX CHANGES##
                            #jitter = np.random.standard_normal(W.shape)
                            key = random.PRNGKey(87)
                            key, subkey=random.split(key)
                            jitter = random.normal(key, W.shape)
                        else:
                            ##JAX CHANGES##
                           #jitter = 2* np.random.random(W.shape) - 1
                            key = random.PRNGKey(87)
                            key, subkey=random.split(key)
                            jitter = 2* random.uniform(key, W.shape) - 1
                        W = (1 + Jnoise * jitter) * W

                    # sparsify (set small weights to zero)
                    W = np.where(W < MinSyn, 0, W) # what's the point of this if not using sparse matrices

                    # row-wise normalize
                    tW = np.sum(W, axis=1)
                    if not CellWiseNormalized:
                        tW = np.mean(tW)
                        W =  W / tW
                    else:
                        W = W / tW[:, None]

                    # for E projections, add the local part
                    # NOTE: alterntaively could do this before adding noise & normalizing
                    if b == 0:
                        W = p_local[a] * np.eye(*W.shape) + (1-p_local[a]) * W

                    Wblks[a][b] = J_2x2[a, b] * W
            
            
            W = np.block(Wblks)
            return W
            

    
    def select_type(self, vec, select='E'):
    
        assert vec.ndim == 1
        maps = self.vec2map(vec)

        if select=='E':
            output = maps[0]

        if select =='I':
            output=maps [1]

        return output
    
    
    def apply_bounding_box(self, vec, size = 3.2, select='E'):

        map_vec = self.select_type(vec, select)

        size = int(size / (self.grid_pars.dx)) +1

        start = int((self.grid_pars.gridsize_Nx - size) / 2)   
        
        map_vec = jax.lax.dynamic_slice(map_vec, (start, start), (size, size))

        return map_vec

def constant_to_vec(c_E, c_I, ssn, sup = False):
    
    edge_length = ssn.grid_pars.gridsize_Nx

    matrix_E = np.ones((edge_length, edge_length)) * c_E
    vec_E = np.ravel(matrix_E)
    
    matrix_I = np.ones((edge_length, edge_length))* c_I
    vec_I = np.ravel(matrix_I)
    
    constant_vec = np.hstack((vec_E, vec_I, vec_E, vec_I))
  
    if sup==False and ssn.phases ==4:
        constant_vec = np.kron(np.asarray([1,1]), constant_vec)

    if sup:
        constant_vec = np.hstack((vec_E, vec_I))
        
    return constant_vec

def create_grating_single(stimuli_pars, n_trials = 10):

    all_stimuli = []
    jitter_val = stimuli_pars.jitter_val
    ref_ori = stimuli_pars.ref_ori

    for i in range(0, n_trials):
        jitter = numpy.random.uniform(-jitter_val, jitter_val, 1)

        #create reference grating
        ref = BW_Grating(ori_deg = ref_ori, jitter=jitter, stimuli_pars = stimuli_pars).BW_image().ravel()
        all_stimuli.append(ref)
    
    return np.vstack([all_stimuli])

def middle_layer_fixed_point(ssn, ssn_input, conv_pars, inhibition = False, PLOT=False, save=None, inds=None, return_fp = False, print_dt = False):
    
    fp, avg_dx = obtain_fixed_point(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars, PLOT = PLOT, save = save, inds = inds, print_dt = print_dt)
    fp_mj, avg_dx_mj = ofp_mj(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars)

    #Add responses from E and I neurons
    fp_E_on = ssn.select_type(fp, map_number = 1)
    fp_E_off = ssn.select_type(fp, map_number = (ssn.phases+1))

    layer_output = fp_E_on + fp_E_off
    
    #Find maximum rate
    max_E =  np.max(np.asarray([fp_E_on, fp_E_off]))
    max_I = np.maximum(np.max(fp[3*int(ssn.Ne/2):-1]), np.max(fp[int(ssn.Ne/2):ssn.Ne]))
   
    if ssn.phases==4:
        fp_E_on_pi2 = ssn.select_type(fp, map_number = 3)
        fp_E_off_pi2 = ssn.select_type(fp, map_number = 7)

        #Changes
        layer_output = layer_output + fp_E_on_pi2 + fp_E_off_pi2    
        max_E =  np.max(np.asarray([fp_E_on, fp_E_off, fp_E_on_pi2, fp_E_off_pi2]))
        max_I = np.max(np.asarray([fp[int(x):int(x)+80] for x in numpy.linspace(81, 567, 4)]))
        mean_E = np.mean(np.asarray([fp_E_on, fp_E_off, fp_E_on_pi2, fp_E_off_pi2]))
        mean_I = np.mean(np.asarray([fp[int(x):int(x)+80] for x in numpy.linspace(81, 567, 4)]))
     
   
    #Loss for high rates
    # r_max = homeo_loss(mean_E, max_E, R_mean_const = 6.1, R_max_const = 50) + homeo_loss(mean_I, max_I, R_mean_const = 10.3, R_max_const = 100)
    r_max = np.maximum(0, (max_E/conv_pars.Rmax_E - 1)) + np.maximum(0, (max_I/conv_pars.Rmax_I - 1))
    #r_max = leaky_relu(r = max_E, R_thresh = conv_pars.Rmax_E, slope_2 = 1/conv_pars.Rmax_E) + leaky_relu(r = max_I, R_thresh = conv_pars.Rmax_I, slope_2 = 1/conv_pars.Rmax_I)
    
    #layer_output = layer_output/ssn.phases
    if return_fp ==True:
            return layer_output, r_max, avg_dx, fp, max_E, max_I
    else:
        return layer_output, r_max, avg_dx

def obtain_fixed_point(ssn, ssn_input, conv_pars, PLOT=False, save=None, inds=None, print_dt = False):
    
    r_init = np.zeros(ssn_input.shape[0])
    dt = conv_pars.dt
    xtol = conv_pars.xtol
    Tmax = conv_pars.Tmax
    
    #Find fixed point
    fp, avg_dx = ssn.fixed_point_r(ssn_input, r_init=r_init, dt=dt, xtol=xtol, Tmax=Tmax, PLOT=PLOT, save=save)
    #fp_mj, _ =  ssn.fixed_point_r_mj(ssn_input, r_init=r_init, dt=dt, xtol=xtol, Tmax=Tmax) # *** tested and it is ok

    avg_dx = np.maximum(0, (avg_dx -1))
    return fp, avg_dx

def obtain_fixed_point_centre_E(ssn, ssn_input, conv_pars, inhibition = False, PLOT=False, save=None, inds=None, return_fp = False):
    
    #Obtain fixed point
    fp, avg_dx = obtain_fixed_point(ssn=ssn, ssn_input = ssn_input, conv_pars = conv_pars, PLOT = PLOT, save = save, inds = inds)

    #Apply bounding box to data
    r_box = (ssn.apply_bounding_box(fp, size=3.2)).ravel()
    
    #Obtain inhibitory response 
    if inhibition ==True:
        r_box_i = ssn.apply_bounding_box(fp, size=3.2, select='I_ON').ravel()
        r_box = [r_box, r_box_i]
 

    max_E = np.max(fp[:ssn.Ne])
    max_I = np.max(fp[ssn.Ne:-1])
    mean_E = np.mean(fp[:ssn.Ne])
    mean_I= np.mean(fp[ssn.Ne:-1])
    
    #Loss for high rates
    r_max = np.maximum(0, (max_E/conv_pars.Rmax_E - 1)) + np.maximum(0, (max_I/conv_pars.Rmax_I - 1))
    #r_max = leaky_relu(r = max_E, R_thresh = conv_pars.Rmax_E, slope_2 = 1/conv_pars.Rmax_E) + leaky_relu(max_I, R_thresh = conv_pars.Rmax_I, slope_2 = 1/conv_pars.Rmax_I)
    #r_max = homeo_loss(mean_E, max_E, R_mean_const = 12, R_max_const = 50) + homeo_loss(mean_I, max_I, R_mean_const = 41.5, R_max_const = 100)
    
    if return_fp ==True:
        return r_box, r_max, avg_dx, fp, max_E, max_I
    else:
        return r_box, r_max, avg_dx
    
def two_layer_model(ssn_m, ssn_s, stimuli, conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I):
    
    '''
    Run individual stimulus through two layer model. 
    
    Inputs:
     ssn_mid, ssn_sup: middle and superficial layer classes
     stimuli: stimuli to pass through network
     conv_pars: convergence parameters for ssn 
     constant_vector_mid, constant_vector_sup: extra synaptic constants for middle and superficial layer
     f_E, f_I: feedforward connections between layers
    
    Outputs:
     r_sup - fixed point of centre neurons (5x5) in superficial layer
     loss related terms (wrt to middle and superficial layer) :
         - r_max_": loss minimising maximum rates
         - avg_dx_": loss minimising number of steps taken during convergence 
     max_(E/I)_(mid/sup): maximum rate for each type of neuron in each layer 
     
    '''
    
    #Find input of middle layer
    stimuli_gabor=np.matmul(ssn_m.gabor_filters, stimuli)
 
    #Rectify input
    SSN_mid_input = np.maximum(0, stimuli_gabor) + constant_vector_mid
    
    #Calculate steady state response of middle layer
    r_mid, r_max_mid, avg_dx_mid, fp_mid, max_E_mid, max_I_mid = middle_layer_fixed_point(ssn_m, SSN_mid_input, conv_pars, return_fp = True) #***
    r_mid_mj, r_max_mid_mj, avg_dx_mid_mj, fp_mid_mj, max_E_mid_mj, max_I_mid_mj = mlfp_mj(ssn_m, SSN_mid_input, conv_pars, return_fp = True) #***
    
    #Concatenate input to superficial layer
    sup_input_ref = np.hstack([r_mid*f_E, r_mid*f_I]) + constant_vector_sup
    
    #Calculate steady state response of superficial layer
    r_sup, r_max_sup, avg_dx_sup, fp_sup, max_E_sup, max_I_sup= obtain_fixed_point_centre_E(ssn_s, sup_input_ref, conv_pars, return_fp= True)
    
    return r_sup, [r_max_mid, r_max_sup], [avg_dx_mid, avg_dx_sup], [max_E_mid, max_I_mid, max_E_sup, max_I_sup], [fp_mid, fp_sup]

vmap_two_layer_model =  vmap(two_layer_model, in_axes = (None, None, 0, None, None, None, None, None))

def surround_suppression(ssn_mid, ssn_sup, tuning_pars, conv_pars, radius_list, constant_vector_mid, constant_vector_sup, f_E, f_I, ref_ori, title= None):    
    
    '''
    Produce matrix response for given two layer ssn network given a list of varying stimuli radii
    '''
    
    all_responses_sup = []
    all_responses_mid = []
    
    tuning_pars.ref_ori = ref_ori
   
    print(ref_ori) #create stimuli in the function just input radii)
    for radii in radius_list:
        
        tuning_pars.outer_radius = radii
        tuning_pars.inner_radius = radii*(2.5/3)
        
        stimuli = create_grating_single(n_trials = 1, stimuli_pars = tuning_pars)#n_trials = 50
    
        _, _, _, _, [fp_mid, fp_sup] = vmap_two_layer_model(ssn_mid, ssn_sup, stimuli, conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I)
        #_, _, _, _, [fp_mid, fp_sup] = two_layer_model(ssn_mid, ssn_sup, stimuli[0,:], conv_pars, constant_vector_mid, constant_vector_sup, f_E, f_I)
         
        #Take average over noisy trials
        all_responses_sup.append(fp_sup.mean(axis = 0))
        all_responses_mid.append(fp_mid.mean(axis = 0))
        #print('Mean population response {} (max in population {}), centre neurons {}'.format(fp_sup.mean(), fp_sup.max(), fp_sup.mean()))
    
    
    return np.vstack(all_responses_sup), np.vstack(all_responses_mid), stimuli

def response_matrix(J_2x2_m, J_2x2_s, kappa_pre, kappa_post, c_E, c_I, f_E, f_I, constant_pars, tuning_pars, radius_list, ori_list, trained_ori):
    '''
    Construct a response matrix of sizze n_orientations x n_neurons x n_radii
    '''
    #Initialize ssn
    ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_m, filter_pars=constant_pars.filter_pars, J_2x2=J_2x2_m, gE = constant_pars.gE[0], gI=constant_pars.gI[0], ori_map = constant_pars.ssn_ori_map)
    #ssn_mid.gabor_filters
    ssn_sup=SSN2DTopoV1(ssn_pars=constant_pars.ssn_pars, grid_pars=constant_pars.grid_pars, conn_pars=constant_pars.conn_pars_s, J_2x2=J_2x2_s, s_2x2=constant_pars.s_2x2, sigma_oris = constant_pars.sigma_oris, ori_map = constant_pars.ssn_ori_map, train_ori = trained_ori, kappa_post = kappa_post, kappa_pre = kappa_pre)

    responses_sup = []
    responses_mid = []
    conv_pars = constant_pars.conv_pars
    constant_vector_mid = constant_to_vec(c_E = c_E, c_I = c_I, ssn= ssn_mid)
    constant_vector_sup = constant_to_vec(c_E = c_E, c_I = c_I, ssn = ssn_sup, sup=True)
    
    for i in range(len(ori_list)):
        
        #Find responses at different stimuli radii
        x_response_sup, x_response_mid, stimuli = surround_suppression(ssn_mid, ssn_sup, tuning_pars, conv_pars, radius_list, constant_vector_mid, constant_vector_sup, f_E, f_I, ref_ori = ori_list[i])

        responses_sup.append(x_response_sup)
        responses_mid.append(x_response_mid)
        
    return np.stack(responses_sup, axis = 2), np.stack(responses_mid, axis = 2), stimuli

def load_param_from_csv(results_filename, epoch, stage=0):
    
    '''
    Load parameters from csv file given file name and desired epoch.
    '''
    
    all_results = pd.read_csv(results_filename, header = 0)
    if epoch == -1:
        epoch_params = all_results.tail(1)
    else:
        epoch_params = all_results.loc[all_results['SGD_steps'] == epoch]
        epoch_params = epoch_params.loc[epoch_params['stage'] == stage]
    
    params = []
    
    J_m = [np.abs(epoch_params[i].values[0]) for i in ['J_m_EE', 'J_m_EI', 'J_m_IE', 'J_m_II']]
    J_s = [np.abs(epoch_params[i].values[0]) for i in ['J_s_EE', 'J_s_EI', 'J_s_IE', 'J_s_II']]

    J_2x2_m = make_J2x2_o(*J_m)
    J_2x2_s = make_J2x2_o(*J_s)
    params.append(J_2x2_m)
    params.append(J_2x2_s)

    
    if 'c_E' in epoch_params.columns:
        c_E = epoch_params['c_E'].values[0]
        c_I = epoch_params['c_I'].values[0]
        params.append(c_E)
        params.append(c_I)
    
    if 'sigma_orisE' in epoch_params.columns:
        sigma_oris = np.asarray([epoch_params['sigma_orisE'].values[0], epoch_params['sigma_orisI'].values[0]])
        params.append(sigma_oris)
    
    if 'f_E' in epoch_params.columns:
        f_E = epoch_params['f_E'].values[0]
        f_I = epoch_params['f_I'].values[0]
        params.append(f_E)
        params.append(f_I)
    
    if 'kappa_preE' in epoch_params.columns:
        kappa_pre = np.asarray([epoch_params['kappa_preE'].values[0], epoch_params['kappa_preI'].values[0]])
        kappa_post = np.asarray([epoch_params['kappa_postE'].values[0], epoch_params['kappa_postI'].values[0]])
        params.append(kappa_pre)
        params.append(kappa_post)
        
    return params

########################################################################################################################
############################################### STARTING THE MAIN SCRIPT ###############################################
########################################################################################################################
start_time = time.time()
# Define source file location and SGD_step to check the tuning curves on
results_dir= os.path.join(os.getcwd(), 'results/Mar21_v5')
results_filename = os.path.join(results_dir, 'results_0.csv')
epoch = 0
ori_list = numpy.arange(0,180,6)
tc_cells=[10,40,100,130,650,690,740,760]

# Setting up parameters for tuning curve calculation
# stimuli parameters
trained_ori = stimuli_pars.ref_ori
tuning_pars = StimuliPars()
tuning_pars.jitter_val = 0
tuning_pars.std = 0.0

# parameters not trained
gE = [filter_pars.gE_m, 0.0]
gI = [filter_pars.gI_m, 0.0]
s_2x2=ssn_layer_pars.s_2x2_s
conn_pars_m.p_local = ssn_layer_pars.p_local_m
conn_pars_s.p_local = ssn_layer_pars.p_local_s

# Superficial layer W parameters
sigma_oris = ssn_layer_pars.sigma_oris #np.asarray([90.0, 90.0])
kappa_pre = ssn_layer_pars.kappa_pre #np.asarray([ 0.0, 0.0])
kappa_post = ssn_layer_pars.kappa_post #np.asarray([ 0.0, 0.0])

[J_2x2_m, J_2x2_s, c_E, c_I, f_E, f_I] = load_param_from_csv(results_filename = results_filename, epoch = epoch)

ssn_ori_map_loaded = np.load(os.path.join(results_dir, 'orimap_0.npy'))
class constant_pars:
    ssn_pars =ssn_pars
    s_2x2 = s_2x2
    sigma_oris = sigma_oris
    grid_pars = grid_pars
    conn_pars_m = conn_pars_m
    conn_pars_s = conn_pars_s
    gE = gE
    gI = gI
    filter_pars = filter_pars
    noise_type = 'poisson'
    ssn_ori_map = ssn_ori_map_loaded
    ref_ori = stimuli_pars.ref_ori
    conv_pars = conv_pars

#List of orientations and stimuli  radii
radius_list = np.asarray([stimuli_pars.outer_radius]) # used for outer_radius = radii and inner_radius = radii*(2.5/3)

# File handling
home_dir = os.getcwd()
saving_dir =os.path.join(results_dir, 'response_matrices')
run_dir = os.path.join(saving_dir, 'response_epoch'+str(epoch))

ssn_mid=SSN2DTopoV1_ONOFF_local(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=J_2x2_m, gE = gE[0], gI=gI[0], ori_map = ssn_ori_map_loaded)
gabor_filters_cp=ssn_mid.gabor_filters
A_cp=ssn_mid.A
A2_cp=ssn_mid.A2
response_sup_clara, response_mid_clara, stimuli = response_matrix(J_2x2_m, J_2x2_s, kappa_pre, kappa_post, c_E, c_I, f_E, f_I, constant_pars, tuning_pars, radius_list, ori_list, trained_ori = trained_ori)
# Note that trained_ori = trained_ori does not matter because it only enters as a multiplicative of kappas that are zeros

################################
########## MJ version ##########
################################

gabor_filters_mj, A_mj, A2_mj = create_gabor_filters_ori_map(ssn_ori_map_loaded, ssn_pars.phases, filter_pars, grid_pars)
oris = ssn_ori_map_loaded.ravel()[:, None]
ori_dist = cosdiff_ring(oris - oris.T, 180)
untrained_pars = UntrainedPars(grid_pars, stimuli_pars, filter_pars, ssn_pars, ssn_layer_pars, conv_pars, 
                loss_pars, training_pars, pretrain_pars, ssn_ori_map_loaded, oris, ori_dist, gabor_filters_mj, 
                readout_pars)

_, trained_pars_stage2, _ = load_parameters(results_filename, iloc_ind = 0)

## this should be unnecessary as the trained pars are only used through trained_pars_stage2
#untrained_pars.ssn_layer_pars.f_E=np.exp(trained_pars_stage2['log_f_E'])
#untrained_pars.ssn_layer_pars.f_I=np.exp(trained_pars_stage2['log_f_E'])
#untrained_pars.ssn_layer_pars.c_E=trained_pars_stage2['c_E']
#untrained_pars.ssn_layer_pars.c_I=trained_pars_stage2['c_I']
#untrained_pars.ssn_layer_pars.J_2x2_s=sep_exponentiate(trained_pars_stage2['log_J_2x2_s'])
#untrained_pars.ssn_layer_pars.J_2x2_m=sep_exponentiate(trained_pars_stage2['log_J_2x2_m'])

response_sup_monika, response_mid_monika = tuning_curve(untrained_pars, trained_pars_stage2, 'test_tc_Monika', ori_vec=ori_list)

gabor_diff=numpy.zeros(648)
for i in range(648):
    gabor_diff[i]=np.mean(np.abs((gabor_filters_mj[i,:]-gabor_filters_cp[i,:])))

# Printing metrics on mismatch (1-2% difference is accaptable)

print('Relative error in A and A2', [np.abs((A_cp-A_mj)/A_cp), np.abs((A2_cp-A2_mj)/A2_cp)])
print('Maximum relative error in mid and sup layer responses=', float(np.max((response_mid_clara[0,:,0]-response_mid_monika[0,:])/response_mid_clara[0,:,0])), float(np.max((response_sup_clara[0,:,0]-response_sup_monika[0,:])/response_sup_clara[0,:,0])))
print('Maximum difference in gabor filters=',np.max(gabor_diff))
print(time.time()-start_time)
# Plotting tuning curves
num_tc_cells=len(tc_cells)
num_mid_cells=response_mid_monika.shape[1]
fig, axes = plt.subplots(nrows=2, ncols=num_tc_cells, figsize=(5*num_tc_cells, 5*2))

for i in range(num_tc_cells):
    axes[1,i].axis('off')  # Hide axis for the print statements
    if tc_cells[i]<num_mid_cells:
        axes[0,i].plot(ori_list,response_mid_monika[:,tc_cells[i]-1], label=f'cell {tc_cells[i]} MJ', lw=4)
        axes[0,i].plot(ori_list,response_mid_clara[0,tc_cells[i]-1,:], label=f'cell {tc_cells[i]} CP')
    else:
        axes[0,i].plot(ori_list,response_sup_monika[:,tc_cells[i]-num_mid_cells-1], label=f'cell {tc_cells[i]} MJ', lw=4)
        axes[0,i].plot(ori_list,response_sup_clara[0,tc_cells[i]-num_mid_cells-1,:], label=f'cell {tc_cells[i]} CP')
    axes[0,i].legend(loc='upper left', fontsize=20)
# Plot print statements
axes[1,0].text(0, 0.25, f'Relative error in A and A2: {np.abs((A_cp - A_mj) / A_cp)}, {np.abs((A2_cp - A2_mj) / A2_cp)}', fontsize=20)
axes[1,0].text(0, 0.5, f'Max relative error in mid and sup layer responses: {float(np.max((response_mid_clara[0, :, 0] - response_mid_monika[0, :]) / response_mid_clara[0, :, 0]))}, {float(np.max((response_sup_clara[0, :, 0] - response_sup_monika[0, :]) / response_sup_clara[0, :, 0]))}', fontsize=20)
axes[1,0].text(0, 0.75, f'Max difference in gabor filters: {np.max(gabor_diff)}', fontsize=20)

plt.savefig('Tuning_curve_comparision')

'''
## Additional code, comparing middle_layer_fixed_point on the same stimulus
# Generate stimulus
x = untrained_pars.BW_image_jax_inp[5]
y = untrained_pars.BW_image_jax_inp[6]
alpha_channel = untrained_pars.BW_image_jax_inp[7]
mask = untrained_pars.BW_image_jax_inp[8]
background = untrained_pars.BW_image_jax_inp[9]

stimuli = BW_image_jit(untrained_pars.BW_image_jax_inp[0:5], x, y, alpha_channel, mask, background, np.array([0]), np.zeros(1))
stimuli_gabor=np.transpose(np.matmul(gabor_filters_cp, np.transpose(stimuli)))
constant_vector_mid = constant_to_vec(c_E = c_E, c_I = c_I, ssn= ssn_mid)
constant_vector_mid = np.expand_dims(constant_vector_mid, axis=0)
SSN_mid_input = np.maximum(0, stimuli_gabor) + constant_vector_mid
    
_, _, _, fp_mid_clara, _, _ = middle_layer_fixed_point(ssn_mid, SSN_mid_input[0,:], conv_pars, return_fp = True)

# getting the middle layer fixed point from MJ code
ssn_mid_mj=SSN_mid(ssn_pars=untrained_pars.ssn_pars, grid_pars=untrained_pars.grid_pars, J_2x2=J_2x2_m)
stimuli_gabor=np.transpose(np.matmul(gabor_filters_mj, np.transpose(stimuli)))
SSN_mid_input = np.maximum(0, stimuli_gabor) + constant_vector_mid
constant_vector_mid = constant_to_vec(c_E = c_E, c_I = c_I, ssn= ssn_mid_mj)
SSN_mid_input = np.maximum(0, stimuli_gabor) + constant_vector_mid
_, _, _, fp_mid_v2, _, _ = mlfp_mj(ssn_mid_mj, SSN_mid_input[0,:], conv_pars, return_fp = True)

print('Relative error in the output of middle layer fixed point=',np.max((fp_mid_v2-fp_mid_clara)/fp_mid_v2))
'''