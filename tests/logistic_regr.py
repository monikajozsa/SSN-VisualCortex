import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy

from SSN_classes import SSN_mid, SSN_sup
from parameters import stimuli_pars, ssn_pars, grid_pars, conn_pars_m, conn_pars_s, filter_pars, ssn_layer_pars, conv_pars
from util import create_grating_pairs
from util_gabor import create_gabor_filters_util
from model import evaluate_model_response

def softplus(x):
    return numpy.log(1 + numpy.exp(x))

ssn_ori_map = numpy.load(os.path.join(os.getcwd(), "ssn_map_uniform_good.npy"))
ssn_mid=SSN_mid(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_m, filter_pars=filter_pars, J_2x2=ssn_layer_pars.J_2x2_m, gE = ssn_layer_pars.gE_m, gI=ssn_layer_pars.gI_m, ori_map = ssn_ori_map)
ssn_sup=SSN_sup(ssn_pars=ssn_pars, grid_pars=grid_pars, conn_pars=conn_pars_s, J_2x2=ssn_layer_pars.J_2x2_s, s_2x2=ssn_layer_pars.s_2x2_s, sigma_oris = ssn_layer_pars.sigma_oris, ori_map = ssn_ori_map, train_ori = 55, kappa_post = [0.0,0.0], kappa_pre = [0.0,0.0])
gabor_filters = create_gabor_filters_util(ssn_ori_map, ssn_mid, filter_pars, ssn_pars.phases)

# Allocate dictionary to save to file
data_dict = {
    "d_reftarget": [],
    "label": []
}

# Generate data for logistic regression
for i in range(200):
    train_data = create_grating_pairs(stimuli_pars, 1)
    
    #Run reference and targetthrough two layer model
    r_ref, [r_max_ref_mid, r_max_ref_sup], [avg_dx_ref_mid, avg_dx_ref_sup],[max_E_mid, max_I_mid, max_E_sup, max_I_sup], _ = evaluate_model_response(ssn_mid, ssn_sup, train_data['ref'].ravel(), conv_pars, ssn_layer_pars.c_E, ssn_layer_pars.c_I, ssn_layer_pars.f_E, ssn_layer_pars.f_I, gabor_filters)
    r_target, [r_max_target_mid, r_max_target_sup], [avg_dx_target_mid, avg_dx_target_sup], _, _= evaluate_model_response(ssn_mid, ssn_sup, train_data['target'].ravel(), conv_pars, ssn_layer_pars.c_E, ssn_layer_pars.c_I, ssn_layer_pars.f_E, ssn_layer_pars.f_I, gabor_filters)

    #Select the middle grid
    N_readout=5
    N_grid=grid_pars.gridsize_Nx
    start=((N_grid-N_readout)/2)
    start=int(start)

    r_ref_2D=numpy.reshape(r_ref,(N_grid,N_grid))
    r_ref_box = r_ref_2D[start:start + N_readout,start:start + N_readout].ravel()
    r_target_2D=numpy.reshape(r_target,(N_grid,N_grid))
    r_target_box = r_target_2D[start:start + N_readout,start:start + N_readout].ravel()

    #Add noise
    noise_ref_target=2*numpy.random.randn(2, 25)
    r_ref_box = r_ref_box + noise_ref_target[0]*numpy.sqrt(softplus(r_ref_box))
    r_target_box = r_target_box + noise_ref_target[1]*numpy.sqrt(softplus(r_target_box))

    data_dict['d_reftarget'].append(r_ref_box - r_target_box)
    data_dict['label'].append(train_data['label'])
    if numpy.round(i/10) == i/10:
        print(i)


# Save the dictionary to a file
with open('data_dict.pkl', 'wb') as file:
    pickle.dump(data_dict, file)

# Load the dictionary from the file
with open('data_dict.pkl', 'rb') as file:
    loaded_data_dict = pickle.load(file)

# Prepare data for logistic regression
X = [x for x in loaded_data_dict['d_reftarget']]
y = loaded_data_dict['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Perform logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
w_sig_logreg = log_reg.coef_
print('accuracy=',accuracy)
