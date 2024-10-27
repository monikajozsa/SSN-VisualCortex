import jax
import jax.numpy as np
from jax import lax

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
        return self.k * np.maximum(0, u) ** self.n

    def drdt(self, r, inp_vec):
        """ Differential equation for the rate vector r """
        return (-r + self.powlaw(self.W @ r + inp_vec)) / self.tau_vec

    def fixed_point_r(self, inp_vec, r_init=None, Tmax=500, dt=1, xtol=1e-5, xmin=1e-0):
        """ Find the fixed point of the rate vector r """
        if r_init is None:
            r_init = np.zeros(inp_vec.shape)
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


class SSN_sup(_SSN_Base):
    """ Class for the superficial SSN layer. """
    def __init__(self, ssn_pars, grid_pars, J_2x2, dist_from_single_ori, ori_dist, kappa= np.array([[0.0, 0.0], [0.0, 0.0]]), **kwargs):
        Ni = Ne = grid_pars.gridsize_Nx**2
        tauE = ssn_pars.tauE
        tauI = ssn_pars.tauI
        tau_vec = np.hstack([tauE * np.ones(Ne), tauI * np.ones(Ni)])

        super(SSN_sup, self).__init__(n=ssn_pars.n, k=ssn_pars.k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, **kwargs)

        # Define the parameters that are necessary for calculating the connectivity matrix W
        self.grid_pars = grid_pars
        self.p_local = ssn_pars.p_local_s # relative strength of local and horizontal parts of E projections
        self.s_2x2 = ssn_pars.s_2x2_s # ranges of weights between different pre/post cell-type
        self.sigma_oris = ssn_pars.sigma_oris # range of weights in terms of preferred orientation difference
   
        xy_dist = grid_pars.xy_dist

        self.W = self.make_W(J_2x2, xy_dist, ori_dist, dist_from_single_ori, kappa)
    
        
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
        tanh_kappa = 4*np.tanh(10*kappa)
        tanh_kappa_pre = [[tanh_kappa[0][0], 0], [tanh_kappa[0][1], 0]]
        tanh_kappa_post = [[tanh_kappa[1][0], 0], [tanh_kappa[1][1], 0]]
        sigma_oris = self.sigma_oris * np.ones((2,2))
        if np.isscalar(self.s_2x2):
            s_2x2 = self.s_2x2 * np.ones((2,2))
        else:
            s_2x2 = self.s_2x2
            assert s_2x2.shape == (2,2)

        # Create matrix for possible connections between E and I cells
        Wblks = [[1,1],[1,1]]

        # Loop over post- (a) and pre-synaptic (b) cell-types
        for a in range(2): # post-synaptic cell type
            for b in range(2): # pre-synaptic cell type  
                horizontal_conn_from_oris = ori_dist**2/(sigma_oris[a,b]**2) + tanh_kappa_pre[a][b]*dist_from_single_ori**2/(2*(45**2)) + tanh_kappa_post[a][b]*dist_from_single_ori.T**2/(2*(45**2))            
                if b == 0: # E projections
                    W = np.exp(-xy_dist/s_2x2[a,b] - horizontal_conn_from_oris)
                elif b == 1: # I projections 
                    W = np.exp(-xy_dist**2/(s_2x2[a,b]**2) - horizontal_conn_from_oris)

                # sparsify (set small weights to zero)
                W = np.where(W < MinSyn, 0, W)

                # row-wise normalize
                tW = np.sum(W, axis=1)
                if not CellWiseNormalized:
                    tW = np.mean(tW)
                    W =  W / tW
                else:
                    W = W / tW[:, None]

                # for E projections, add the local part
                # NOTE: alterntaively could do this before normalizing
                if b == 0:
                    W = p_local[a] * np.eye(*W.shape) + (1-p_local[a]) * W

                Wblks[a][b] = J_2x2[a, b] * W          

        W_out = np.block(Wblks)
        return W_out


    def select_type(self, vec, select='E'):
        """ Select the E or I part of the vector vec """
        assert vec.ndim == 1
        Nx = self.grid_pars.gridsize_Nx
        # reshape vector into matrix form
        if len(vec) == self.Ne:
            maps = np.reshape(vec, (Nx, Nx))
        elif len(vec) == self.N:
            maps = (np.reshape(vec[:self.Ne], (Nx, Nx)),
                   np.reshape(vec[self.Ne:], (Nx, Nx)))

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
        **kwargs
    ):
        self.phases = ssn_pars.phases
        self.grid_pars = grid_pars
        self.Nc = grid_pars.gridsize_Nx**2 # number of cells per phase

        Ni = Ne = self.phases * self.Nc
        tau_vec = np.hstack([ssn_pars.tauE * np.ones(self.Nc),  ssn_pars.tauI * np.ones(self.Nc)])
        tau_vec = np.kron(np.ones((1, self.phases)), tau_vec).squeeze()

        super(SSN_mid, self).__init__(n=ssn_pars.n, k=ssn_pars.k, Ne=Ne, Ni=Ni, tau_vec=tau_vec, **kwargs)

        self.make_W(J_2x2)

    def drdt(self, r, inp_vec):
        """ Differential equation for the rate vector r """
        r1 = np.reshape(r, (-1, self.Nc))
        out = (-r + self.powlaw(np.ravel(self.W @ r1) + inp_vec)) / self.tau_vec
        return out

    def make_W(self, J_2x2):
        """Create the recurrent connectivity matrix W - a block diagonal matrix with J_2x2 as the block matrix."""
        self.W = np.kron(np.eye(self.phases), np.asarray(J_2x2))

    def select_type(self, vec, map_numbers):
        """ Select the E or I part of the vector vec """
        # Calculate start and end indices for each map_number (corresponding to a phase)
        start_indices = (map_numbers - 1) * self.Nc
        
        out = []
        for start in start_indices:
            slice = lax.dynamic_slice(vec, (start,), (self.Nc,))
            out.append(slice)

        return np.array(out)