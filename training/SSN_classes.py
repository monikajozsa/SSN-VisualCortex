import jax
import jax.numpy as jnp
from jax import lax
from jax import vmap

class _SSN_Base(object):
    """ Base class for SSN models """
    def __init__(self, n: int, k: float, Ne: int, Ni: int, tau_vec=None, W=None):
        self.n = n
        self.k = k
        self.Ne = Ne
        self.Ni = Ni
        self.N = self.Ne + self.Ni

        if tau_vec is not None:
            self.tau_vec = tau_vec  # rate time-consants of neurons. shape: (N,)
        if W is not None:
            self.W = W  # connectivity matrix. shape: (N, N)

    @property
    def neuron_params(self):
        """ Returns the two key parameters n, k, and tau_vec of the SSN class """
        return dict(n=self.n, k=self.k, tau_vec=self.tau_vec)

    @property
    def dim(self):
        """ Returns the number of neurons in the SSN class """
        return self.N

    ######## FIXED POINT FUNCTIONS ########
    def powlaw(self, u):
        """ Power-law nonlinearity """
        return self.k * jnp.maximum(0, u) ** self.n

    def drdt(self, r, inp_vec):
        """ Differential equation for the rate vector r """
        return (-r + self.powlaw(self.W @ r + inp_vec)) / self.tau_vec

    def fixed_point_r(self, inp_vec, r_init=None, Tmax=500, dt=1, xtol=1e-5, xmin=1e-0):
        """ Find the fixed point of the rate vector r """
        if r_init is None:
            r_init = jnp.zeros(inp_vec.shape)
        if len(jnp.shape(self.W)) == 3: # middle layer
            drdt = lambda r : self.drdt(self.W, r, inp_vec)
        else: # superficial layer
            drdt = lambda r : self.drdt(r, inp_vec)

        Nmax = int(Tmax/dt)
        r_fp = r_init 
        y = jnp.zeros(((Nmax)))  
		
        def loop(n, carry):
            r_fp, y = carry
            dr = drdt(r_fp) * dt
            r_fp = r_fp + dr
            y = y.at[n].set(jnp.abs( dr /jnp.maximum(xmin, jnp.abs(r_fp)) ).max())
            return (r_fp, y)

        r_fp, y = jax.lax.fori_loop(0, Nmax, loop, (r_fp, y))
        
        avg_dx = y[int(Nmax/2):int(Nmax)].mean()/xtol
    
        return r_fp, avg_dx


class SSN_sup(_SSN_Base):
    """ Class for the superficial SSN layer. """
    def __init__(self, ssn_pars, grid_pars, J_2x2, dist_from_single_ori, ori_dist, kappa_Jsup= jnp.array([[[0.0, 0.0], [0.0, 0.0]],[[0.0, 0.0], [0.0, 0.0]]]), **kwargs):
        Ni = Ne = grid_pars.gridsize_Nx**2
        tauE = ssn_pars.tauE
        tauI = ssn_pars.tauI
        tau_vec = jnp.hstack([tauE * jnp.ones(Ne), tauI * jnp.ones(Ni)])
        self.couple_c_ms = ssn_pars.couple_c_ms

        super(SSN_sup, self).__init__(n=ssn_pars.n, k=ssn_pars.k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, **kwargs)

        # Define the parameters that are necessary for calculating the connectivity matrix W
        self.grid_pars = grid_pars
        self.p_local = ssn_pars.p_local_s # relative strength of local and horizontal parts of E projections
        self.s_2x2 = ssn_pars.s_2x2_s # ranges of weights between different pre/post cell-type
        self.sigma_oris = ssn_pars.sigma_oris # range of weights in terms of preferred orientation difference
   
        xy_dist = grid_pars.xy_dist

        self.W = self.make_W(J_2x2, xy_dist, ori_dist, dist_from_single_ori, kappa_Jsup, ssn_pars.kappa_range)
    
        
    def make_W(self, J_2x2, xy_dist, ori_dist, dist_from_single_ori, kappa, kappa_range, MinSyn=1e-4, CellWiseNormalized=False):
        """
        Make the full recurrent connectivity matrix W
        Input:
            J_2x2 = total strength of weights of different pre/post cell-type
            xy_dist = distance matrix of spatial distance between grid points
            ori_dist = distance matrix of preferred orientation difference
            dist_from_single_ori = distance of each orientation in the map from the beta orientation
            kappa = strength of dist_from_single_ori contribution in the horizontal connections
        Output:
        self.W
        """
        new_normalization = False
        # Unpack parameters  
        p_local = self.p_local
        tanh_kappa_pre = jnp.tanh(kappa[0])
        tanh_kappa_post = jnp.tanh(kappa[1])
        sigma_oris = self.sigma_oris * jnp.ones((2,2))
        if jnp.isscalar(self.s_2x2):
            s_2x2 = self.s_2x2 * jnp.ones((2,2))
        else:
            s_2x2 = self.s_2x2
            assert s_2x2.shape == (2,2)

        # Create matrix for possible connections between E and I cells
        Wblks = [[1,1],[1,1]]

        # Loop over post- (a) and pre-synaptic (b) cell-types
        for a in range(2): # post-synaptic cell type
            for b in range(2): # pre-synaptic cell type
                # term determining the dependence of connectivity on the distance from trained orientation 
                kappa_contrib = tanh_kappa_pre[a][b]*dist_from_single_ori**2/(2*(kappa_range**2)) + tanh_kappa_post[a][b]*dist_from_single_ori.T**2/(2*(kappa_range**2))
                # dependence of connectivity on the distance from trained orientation and distance between grid point orientations (ori_dist)
                if new_normalization:
                    ori_dist_contrib = ori_dist**2/(2*sigma_oris[a,b]**2)
                else:
                    ori_dist_contrib = ori_dist**2/(2*sigma_oris[a,b]**2) + kappa_contrib
                
                if b == 0: # E projections
                    W = jnp.exp(-xy_dist/s_2x2[a,b] - ori_dist_contrib)
                elif b == 1: # I projections 
                    W = jnp.exp(-xy_dist**2/(2*s_2x2[a,b]**2) - ori_dist_contrib)

                # sparsify (set small weights to zero)
                W = jnp.where(W < MinSyn, 0, W)

                # row-wise normalize
                tW = jnp.sum(W, axis=1)
                if CellWiseNormalized:
                    W = W / tW[:, None]
                else:
                    tW = jnp.mean(tW)
                    W =  W / tW                    

                # for E projections, add the local part
                # NOTE: alterntaively could do this before normalizing
                if b == 0:
                    W = p_local[a] * jnp.eye(*W.shape) + (1-p_local[a]) * W

                if new_normalization:
                    Wblks[a][b] = J_2x2[a, b] * W * jnp.exp(-kappa_contrib)
                else:
                    Wblks[a][b] = J_2x2[a, b] * W

        W_out = jnp.block(Wblks)
        return W_out


    def select_type(self, vec, select='E'):
        """ Select the E or I part of the vector vec """
        assert vec.ndim == 1
        Nx = self.grid_pars.gridsize_Nx
        # reshape vector into matrix form
        if len(vec) == self.Ne:
            maps = jnp.reshape(vec, (Nx, Nx))
        elif len(vec) == self.N:
            maps = (jnp.reshape(vec[:self.Ne], (Nx, Nx)),
                   jnp.reshape(vec[self.Ne:], (Nx, Nx)))

        if select=='E': # second half of vec
            return maps[0]
        if select =='I': # first half of vec
            return maps[1]


class SSN_mid(_SSN_Base):
    """ Class for the middle SSN layer. """
    def __init__(
        self,
        ssn_pars,
        grid_pars,
        J_2x2,
        dist_from_single_ori,
        kappa = jnp.array([[0.0, 0.0], [0.0, 0.0]]),
        **kwargs
    ):
        self.phases = ssn_pars.phases
        self.grid_pars = grid_pars
        self.Nc = grid_pars.gridsize_Nx**2 # number of cells per phase

        Ni = Ne = self.phases * self.Nc
        tau_vec = jnp.hstack([ssn_pars.tauE * jnp.ones(self.Nc),  ssn_pars.tauI * jnp.ones(self.Nc)])
        tau_vec = jnp.kron(jnp.ones((1, self.phases)), tau_vec).squeeze()

        super(SSN_mid, self).__init__(n=ssn_pars.n, k=ssn_pars.k, Ne=Ne, Ni=Ni, tau_vec=tau_vec, **kwargs)

        self.W = self.make_W(J_2x2, dist_from_single_ori[:,0], kappa, ssn_pars.kappa_range)
    

    def make_W(self, J_2x2, distance_from_single_ori, kappa, kappa_range, MinSyn=1e-4):
        """ Compute the 2x2x81 block with the exponential scaling per grid point """
        tanh_kappa = jnp.tanh(kappa)
        
        W_type_grid_block = jnp.exp(tanh_kappa[:, :, None] * distance_from_single_ori[None, None, :]**2/(2*kappa_range**2))
        
        # sparsify (set small weights to zero)
        W_type_grid_block = jnp.where(W_type_grid_block < MinSyn, 0, W_type_grid_block)
        
        # connection-type-wise normalize
        W_out = J_2x2[:, :, None] * W_type_grid_block / jnp.expand_dims(jnp.mean(W_type_grid_block, axis=2), axis=2)
        
        return W_out

    def drdt(self, W, r, inp_vec):
        """ Differential equation for the rate vector r """
        r1 = jnp.reshape(r, (-1, 2, self.Nc)) # phase x type x grid-point
        I =  W[None,:,:,:] * r1[:,None,:,:] # (4 x 2 x 2 x 81)  x recurrent input        
        I = jnp.sum(I, axis=2) # n_phases x 2 x 81 sum over pre-synaptic type
        I = I.ravel() # 648
        out = (-r + self.powlaw(I + inp_vec)) / self.tau_vec
        
        return out
    
    '''
    def drdt_ones(self, r, inp_vec):
        """ Differential equation for the rate vector r """
        r1 = jnp.reshape(r, (-1, 2, self.Nc)) # phase x type x grid-point
        r1 = jnp.sum(r1, axis=0) # sum over phases - change from eye to ones
        I =  self.W * r1[None, :, :] # 2 x 2 x 81m recurrent input        
        I = jnp.sum(I, axis=1) # 2x81 sum over pre-synaptic type
        # TODO: tile I num_phase times along a new first axis
        out = (-r + self.powlaw(I + inp_vec)) / self.tau_vec
        
        return out
    '''

    def select_type(self, vec, map_numbers):
        """ Select the E or I part of the vector vec """
        # Calculate start and end indices for each map_number (corresponding to a phase)
        start_indices = (map_numbers - 1) * self.Nc
        
        out = []
        for start in start_indices:
            slice = lax.dynamic_slice(vec, (start,), (self.Nc,))
            out.append(slice)

        return jnp.array(out)