import jax
import jax.numpy as np
from jax import random


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

    def dxdt(self, x, inp_vec):
        """
        allowing for descendent SSN types whose state-vector, x, is different
        than the rate-vector, r.
        """
        return self.drdt(x, inp_vec)

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

    def fixed_point_r(self, inp_vec, r_init=None, Tmax=500, dt=1, xtol=1e-5):
        # Initialize the starting point for iteration.
        if r_init is None:
            r_init = np.zeros(inp_vec.shape)

        # Set the derivative function
        drdt = self.drdt

        # Calculate the maximum number of iterations (Nmax) based on the total time (Tmax) and time step (dt).
        Nmax = int(Tmax / dt)
        
        # Initialize the vector to be iterated over and an array to track the maximum change in each iteration.
        xvec = r_init
        y = np.zeros(Nmax)

        # n: current iteration index, carry: tuple containing the current state and change tracking array.
        def loop(n, carry):
            xvec, y = carry
            # Calculate the change (dx) using drdt and multiply by time step (dt).
            dx = drdt(xvec, inp_vec) * dt
            # Update the state by adding the change to the current state.
            xvec = xvec + dx
            # Update the change tracking array (y) with the maximum relative change at this iteration.
            y = y.at[n].set(np.abs(dx / np.maximum(xtol, np.abs(xvec))).max())
            return xvec, y

        # Run the loop for Nmax iterations using JAX's fori_loop
        r_fp, y = jax.lax.fori_loop(0, Nmax, loop, (xvec, y))

        # Calculate the average of the maximum relative change over the second half of the iterations,
        # normalized by the tolerance (xtol). This measures the convergence rate.
        avg_dx = y[Nmax // 2:].mean() / xtol

        # Return the final state (fixed point) and the average convergence rate.
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
        self.A=ssn_pars.A
     
        if kappa_pre==None:
            kappa_pre = np.asarray([ 0.0, 0.0])
            kappa_post = kappa_pre
   
        xy_dist = grid_pars.xy_dist
        cosdiff_ring = lambda d_x, L: np.sqrt(2 * (1 - np.cos(d_x * 2 * np.pi/L))) * L / 2/ np.pi
        trained_ori_dist = cosdiff_ring(oris - self.train_ori, SSN_sup._Lring) #NEW - calculate distance to trained orientation
        self.trained_ori_dist = trained_ori_dist.squeeze()
        self.W = self.make_W(J_2x2, xy_dist, ori_dist, kappa_pre, kappa_post)

    @property
    def neuron_params(self):
        return dict(n=self.n, k=self.k,
                    tauE=self.tau_vec[0], tauI=self.tau_vec[self.Ne])
    
        
    def make_W(self, J_2x2, xy_dist, ori_dist, kappa_pre, kappa_post, Jnoise=0,
                Jnoise_GAUSSIAN=True, MinSyn=1e-4, CellWiseNormalized=False): #, prngKey=0):
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
                    
                    if b == 0: # E projections
                        W = np.exp(-xy_dist/s_2x2[a,b] -ori_dist**2/(2*sigma_oris[a,b]**2) - kappa_post[a,b]*trained_ori_dist[:, None]**2/2 /45**2  -kappa_pre[a,b]*trained_ori_dist[None,:]**2/2/45**2 )
                        
                    elif b == 1: # I projections 
                        W = np.exp(-xy_dist**2/(2*s_2x2[a,b]**2) -ori_dist**2/(2*sigma_oris[a,b]**2) -kappa_post[a,b] * trained_ori_dist[:, None]**2/2/45**2  -kappa_pre[a,b]*trained_ori_dist[None,:]**2/2/45**2)


                    if Jnoise > 0: # multiply W with 1+ Jnoise*jitter, where Jnoise is a given magnitude and jitter is a random matrix with elements from Gaussian or uniform distribution
                        if Jnoise_GAUSSIAN:
                            ##JAX CHANGES##
                            key = random.PRNGKey(87)
                            key, _ = random.split(key)
                            jitter = random.normal(key, W.shape)
                        else:
                            ##JAX CHANGES##
                            key = random.PRNGKey(87)
                            key, _ = random.split(key)
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

        if select=='E':
            output = maps[0]
        if select =='I':
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
        self.Nc = grid_pars.gridsize_Nx**2  # number of columns

        Ni = Ne = self.phases * self.Nc
        n = ssn_pars.n
        tau_vec = np.hstack([ssn_pars.tauE * np.ones(self.Nc),  ssn_pars.tauI * np.ones(self.Nc)])
        tau_vec = np.kron(np.ones((1, self.phases)), tau_vec).squeeze()

        super(SSN_mid, self).__init__(
            n=n, k=self.k, Ne=Ne, Ni=Ni, tau_vec=tau_vec, **kwargs
        )

        self.A = ssn_pars.A
        if ssn_pars.phases == 4:
            self.A2 = ssn_pars.A2

        self.make_W(J_2x2)

    def drdt(self, r, inp_vec):
        r1 = np.reshape(r, (-1, self.Nc))
        out = (-r + self.powlaw(np.ravel(self.W @ r1) + inp_vec)) / self.tau_vec
        return out

    def make_W(self, J_2x2):
        self.W = np.kron(np.eye(self.phases), np.asarray(J_2x2))


    @property
    def neuron_params(self):
        return dict(
            n=self.n, k=self.k, tauE=self.tau_vec[0], tauI=self.tau_vec[self.Ne]
        )
    
    def select_type(self, vec, map_numbers):
        # Calculate start and end indices for each map_number
        start_indices = (map_numbers - 1) * self.Nc
        end_indices = map_numbers * self.Nc

        # Use a list comprehension to extract slices for each map_number
        out = np.array([vec[start:end] for start, end in zip(start_indices, end_indices)])
        return out