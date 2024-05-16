import jax
import jax.numpy as np
from jax import lax

class _SSN_Base(object):
    def __init__(self, n, k, Ne, Ni, tau_vec=None, W=None):
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
        return dict(n=self.n, k=self.k)

    @property
    def dim(self):
        return self.N

    @property
    def tau_x_vec(self):
        """time constants for the generalized state-vector, x"""
        return self.tau_vec

    def powlaw(self, u):
        return self.k * np.maximum(0, u) ** self.n

    def drdt(self, r, inp_vec):
        return (-r + self.powlaw(self.W @ r + inp_vec)) / self.tau_vec

    def gains_from_v(self, v):
        return self.n * self.k * np.maximum(0, v) ** (self.n - 1)

    def gains_from_r(self, r):
        return self.n * self.k ** (1 / self.n) * r ** (1 - 1 / self.n)

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
        return (
            DCjacob / self.tau_x_vec[:, None]
        )  # equivalent to np.diag(tau_x_vec) * DCjacob

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
        return -1j * omega * np.diag(self.tau_x_vec) - DCjacob

    ######## FIXED POINT FUNCTIONS #################

    def fixed_point_r(self, inp_vec, r_init=None, Tmax=500, dt=1, xtol=1e-5, xmin=1e-0):
        
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
    
    def make_noise_cov(self, noise_pars):
        # the script assumes independent noise to E and I, and spatially uniform magnitude of noise
        noise_sigsq = np.hstack(
            (
                noise_pars.stdevE**2 * np.ones(self.Ne),
                noise_pars.stdevI**2 * np.ones(self.Ni),
            )
        )
        spatl_filt = np.array(1)

        return noise_sigsq, spatl_filt


class SSN_sup(_SSN_Base):
    _Lring = 180

    def __init__(self, ssn_pars, grid_pars, J_2x2, p_local, oris, sigma_oris =None, s_2x2 = None, ori_dist=None, train_ori = None, kappa_post = None, kappa_pre = None, **kwargs):
        Ni = Ne = grid_pars.gridsize_Nx**2
        n=ssn_pars.n
        self.k=ssn_pars.k
        tauE= ssn_pars.tauE
        tauI=ssn_pars.tauI
        self.tauE = tauE
        self.tauI = tauI
        tau_vec = np.hstack([tauE * np.ones(Ne), tauI * np.ones(Ni)])

        super(SSN_sup, self).__init__(n=n, k=self.k, Ne=Ne, Ni=Ni,
                                    tau_vec=tau_vec, **kwargs)

        self.grid_pars = grid_pars
        self.p_local = p_local
        self.train_ori = train_ori

        self.s_2x2 = s_2x2
        self.sigma_oris = sigma_oris
     
        if kappa_pre==None:
            kappa_pre = np.asarray([ 0.0, 0.0])
            kappa_post = kappa_pre
   
        xy_dist = grid_pars.xy_dist
        cosdiff_ring = lambda d_x, L: np.sqrt(2 * (1 - np.cos(d_x * 2 * np.pi/L))) * L / 2/ np.pi
        trained_ori_dist = cosdiff_ring(oris - self.train_ori, SSN_sup._Lring)
        self.trained_ori_dist = trained_ori_dist.squeeze()
        self.W = self.make_W(J_2x2, xy_dist, ori_dist, kappa_pre, kappa_post)

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k,
                    tauE=self.tau_vec[0], tauI=self.tau_vec[self.Ne])
    
        
    def make_W(self, J_2x2, xy_dist, ori_dist, kappa_pre, kappa_post, MinSyn=1e-4, CellWiseNormalized=False):
            """
            make the full recurrent connectivity matrix W
            Input:
             J_2x2 = total strength of weights of different pre/post cell-type
             s_2x2 = ranges of weights between different pre/post cell-type
             p_local = relative strength of local parts of E projections
             sigma_oris = range of weights in terms of preferred orientation difference
            Output/side-effects:
            self.W
            """
            s_2x2 = self.s_2x2
            sigma_oris = self.sigma_oris
            p_local = self.p_local
            trained_ori_dist = self.trained_ori_dist
            
            #Reshape sigma_oris, kappa pre and kappa post
            sigma_oris = sigma_oris * np.ones((2,2))
            kappa_pre = kappa_pre * np.ones((2,2))
            kappa_post = kappa_post * np.ones((2,2))
            
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
                    if b == 0: # E projections
                        W = np.exp(-xy_dist/s_2x2[a,b] -ori_dist**2/(2*sigma_oris[a,b]**2) - kappa_post[a,b]*trained_ori_dist[:, None]**2/2 /45**2  -kappa_pre[a,b]*trained_ori_dist[None,:]**2/2/45**2 )                        
                    elif b == 1: # I projections 
                        W = np.exp(-xy_dist**2/(2*s_2x2[a,b]**2) -ori_dist**2/(2*sigma_oris[a,b]**2) -kappa_post[a,b] * trained_ori_dist[:, None]**2/2/45**2  -kappa_pre[a,b]*trained_ori_dist[None,:]**2/2/45**2)

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
            
            W = np.block(Wblks)
            return W
            
    def vec2map(self, vec):
        assert vec.ndim == 1
        Nx = self.grid_pars.gridsize_Nx
        if len(vec) == self.Ne:
            map = np.reshape(vec, (Nx, Nx))
        elif len(vec) == self.N:
            map = (np.reshape(vec[:self.Ne], (Nx, Nx)),
                   np.reshape(vec[self.Ne:], (Nx, Nx)))
        return map


    def select_type(self, vec, select='E'):    
        assert vec.ndim == 1
        maps = self.vec2map(vec)

        if select=='E': # second half
            output = maps[0]
        if select =='I': # first half
            output = maps[1]

        return output
    

class SSN_mid(_SSN_Base):
    _Lring = 180

    def __init__(
        self,
        ssn_pars,
        grid_pars,
        J_2x2,
        **kwargs
    ):
        self.phases = ssn_pars.phases
        self.k = ssn_pars.k
        self.grid_pars = grid_pars
        self.Nc = grid_pars.gridsize_Nx**2 # number of cells per phase

        Ni = Ne = self.phases * self.Nc
        n = ssn_pars.n
        tau_vec = np.hstack([ssn_pars.tauE * np.ones(self.Nc),  ssn_pars.tauI * np.ones(self.Nc)])
        tau_vec = np.kron(np.ones((1, self.phases)), tau_vec).squeeze()

        super(SSN_mid, self).__init__(
            n=n, k=self.k, Ne=Ne, Ni=Ni, tau_vec=tau_vec, **kwargs
        )

        self.make_W(J_2x2)

    def drdt(self, r, inp_vec):
        r1 = np.reshape(r, (-1, self.Nc))
        out = (-r + self.powlaw(np.ravel(self.W @ r1) + inp_vec)) / self.tau_vec
        return out
    
    def drdt_nobatch(self, r, inp_vec):
        r1 = np.reshape(r, (-1, self.Nc))
        out = (-r + self.powlaw(np.ravel(self.W @ r1) + inp_vec)) / self.tau_vec
        return out

    def make_W(self, J_2x2):
        '''Create the recurrent connectivity matrix W - a block diagonal matrix with J_2x2 as the block matrix.'''
        self.W = np.kron(np.eye(self.phases), np.asarray(J_2x2))

    @property
    def neuron_params(self):
        return dict(
            n=self.n, k=self.k, tauE=self.tau_vec[0], tauI=self.tau_vec[self.Ne]
        )
    
    def select_type(self, vec, map_numbers):
        # Calculate start and end indices for each map_number
        start_indices = (map_numbers - 1) * self.Nc
        
        out = []
        for start in start_indices:
            slice = lax.dynamic_slice(vec, (start,), (self.Nc,))
            out.append(slice)

        return np.array(out)