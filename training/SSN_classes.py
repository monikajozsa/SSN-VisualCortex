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
        if len(jnp.shape(self.W)) == 3: # middle layer - vmapped over grid points
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
    def __init__(self, ssn_pars, grid_pars, J_2x2, dist_from_single_ori, ori_dist, kappa_Jsup= jnp.array([[0.0, 0.0], [0.0, 0.0]]), **kwargs):
        Ni = Ne = grid_pars.gridsize_Nx**2
        tauE = ssn_pars.tauE
        tauI = ssn_pars.tauI
        tau_vec = jnp.hstack([tauE * jnp.ones(Ne), tauI * jnp.ones(Ni)])

        super(SSN_sup, self).__init__(n=ssn_pars.n, k=ssn_pars.k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, **kwargs)

        # Define the parameters that are necessary for calculating the connectivity matrix W
        self.grid_pars = grid_pars
        self.p_local = ssn_pars.p_local_s # relative strength of local and horizontal parts of E projections
        self.s_2x2 = ssn_pars.s_2x2_s # ranges of weights between different pre/post cell-type
        self.sigma_oris = ssn_pars.sigma_oris # range of weights in terms of preferred orientation difference
   
        xy_dist = grid_pars.xy_dist

        self.W = self.make_W(J_2x2, xy_dist, ori_dist, dist_from_single_ori, kappa_Jsup)
    
        
    def make_W(self, J_2x2, xy_dist, ori_dist, dist_from_single_ori, kappa, MinSyn=1e-4, CellWiseNormalized=False):
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
        # Unpack parameters  
        p_local = self.p_local
        tanh_kappa = jnp.tanh(kappa)
        tanh_kappa_pre = [[tanh_kappa[0][0], 0], [tanh_kappa[0][1], 0]]
        tanh_kappa_post = [[tanh_kappa[1][0], 0], [tanh_kappa[1][1], 0]]
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
                #ori_dist_contrib = ori_dist**2/(sigma_oris[a,b]**2) + tanh_kappa_pre[a][b]*dist_from_single_ori**2/(2*(45**2)) + tanh_kappa_post[a][b]*dist_from_single_ori.T**2/(2*(45**2))
                ori_dist_contrib = ori_dist**2/(sigma_oris[a,b]**2)
                if b == 0: # E projections
                    W = jnp.exp(-xy_dist/s_2x2[a,b] - ori_dist_contrib)
                elif b == 1: # I projections 
                    W = jnp.exp(-xy_dist**2/(s_2x2[a,b]**2) - ori_dist_contrib)

                # sparsify (set small weights to zero)
                W = jnp.where(W < MinSyn, 0, W)

                # row-wise normalize
                tW = jnp.sum(W, axis=1)
                if not CellWiseNormalized:
                    tW = jnp.mean(tW)
                    W =  W / tW
                else:
                    W = W / tW[:, None]

                # for E projections, add the local part
                # NOTE: alterntaively could do this before normalizing
                if b == 0:
                    W = p_local[a] * jnp.eye(*W.shape) + (1-p_local[a]) * W

                #Wblks[a][b] = J_2x2[a, b] * W
                Wblks[a][b] = J_2x2[a, b] * W * jnp.exp(tanh_kappa_pre[a][b]*dist_from_single_ori**2/(2*(45**2)) + tanh_kappa_post[a][b]*dist_from_single_ori.T**2/(2*(45**2)))

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

        if select=='E': # second half
            return maps[0]
        if select =='I': # first half
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
        tau_vec = jnp.tile(jnp.array([ssn_pars.tauE,  ssn_pars.tauI]), self.phases)

        super(SSN_mid, self).__init__(n=ssn_pars.n, k=ssn_pars.k, Ne=Ne, Ni=Ni, tau_vec=tau_vec, **kwargs)

        self.make_W(J_2x2, jnp.tanh(kappa), dist_from_single_ori[:,0])
    
    def drdt_per_grid_point(self, W, r, inp_vec):
        """ Differential equation for the rate vector r """
        out = (-r + self.powlaw(jnp.ravel(W @ r) + inp_vec)) / self.tau_vec  # W is 8x8, r is 8, inp_vec is 8 and tau_vec is 8
        return out

    def drdt(self, W, r, inp_vec):
        """ Compute dr/dt for all grid points in parallel. """
        r_reshaped = jnp.reshape(r, (-1, self.Nc))
        inp_vec_reshaped = jnp.reshape(inp_vec, (-1, self.Nc))
        r_next = vmap(self.drdt_per_grid_point, in_axes=(2, 1, 1))(W, r_reshaped, inp_vec_reshaped) # r_next shape is (81, 8) - default vmap feature
        r_next_transposed = jnp.transpose(r_next)  # Shape becomes (8, 81) again
        r_next_flat = jnp.ravel(r_next_transposed)

        return r_next_flat

    def make_W(self, J_2x2, tanh_kappa, distance_from_single_ori):
        """Create the recurrent connectivity matrix W - a block diagonal matrix with J_2x2 as the block matrix."""
        num_phases = self.phases

        # Compute the 8x8x81 block with the exponential scaling per grid point
        W_type_grid_block = J_2x2[:, :, None] * jnp.exp(tanh_kappa[:, :, None] * distance_from_single_ori[None, None, :]) 
        
        num_type,_,grid_size_2D = jnp.shape(W_type_grid_block)
        W = jnp.zeros((num_type*num_phases, num_type*num_phases, grid_size_2D))
        for i in range(num_phases):
            W = W.at[i*num_type:(i+1)*num_type, i*num_type:(i+1)*num_type, :].set(W_type_grid_block)
        # The next line is Wmid with mixed phase connections. Tile W_type_grid_block for num_phases x num_phases in the first two dimensions
        # W = jnp.tile(W_type_grid_block, (num_phases, num_phases, 1)) / num_phases # Shape: (phases*num_types, phases*num_types, Nc)

        # Save the connectivity matrix to the instance
        self.W = W

    def select_type(self, vec, map_numbers):
        """ Select the E or I part of the vector vec """
        # Calculate start and end indices for each map_number (corresponding to a phase)
        start_indices = (map_numbers - 1) * self.Nc
        
        out = []
        for start in start_indices:
            slice = lax.dynamic_slice(vec, (start,), (self.Nc,))
            out.append(slice)

        return jnp.array(out)