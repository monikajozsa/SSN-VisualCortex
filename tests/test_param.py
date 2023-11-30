from parameters_Clara import *
from parameters_Clara import init_set_func

################### PARAMETER SPECIFICATION #################
#SSN layer parameter initialisation
init_set_m ='C'
init_set_s=1
J_2x2_s, s_2x2, gE_s, gI_s, conn_pars_s  = init_set_func(init_set_s, conn_pars_s, ssn_pars)
J_2x2_m, _, gE_m, gI_m, conn_pars_m  = init_set_func(init_set_m, conn_pars_m, ssn_pars, middle = True)

#Excitatory and inhibitory constants for extra synaptic GABA
c_E = 5.0
c_I = 5.0

#Superficial layer W parameters
sigma_oris = np.asarray([90.0, 90.0])
kappa_pre = np.asarray([ 0.0, 0.0])
kappa_post = np.asarray([ 0.0, 0.0])

#Feedforwards connections
f_E =  np.log(1.11)
f_I = np.log(0.7)

#Constants for Gabor filters
gE = [gE_m, gE_s]
gI = [gI_m, gI_s]
#Sigmoid layer parameters
N_neurons = 25

ssn_layer_pars = dict(J_2x2_m = J_2x2_m, J_2x2_s = J_2x2_s, c_E = c_E, c_I = c_I, f_E=f_E, f_I = f_I, kappa_pre = kappa_pre, kappa_post = kappa_post)
print(ssn_layer_pars['J_2x2_m'])
print(ssn_layer_pars['J_2x2_s'])
print(conn_pars_s.p_local)
print(conn_pars_m.p_local)

from parameters import *
print(ssn_layer_pars.J_2x2_m)
print(ssn_layer_pars.J_2x2_s)
print(conn_pars_s.p_local)
print(conn_pars_m.p_local)