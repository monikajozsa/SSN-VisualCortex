def config(config_groups):
    ########## ########## ########## ##########
    ########  Training configurations  ########
    ########## ########## ########## ##########
    ## Define the configurations for training.
    ## Each configuration list contains the following elements:
    ## 1. Training parameters (e.g., 'cE_m', 'cI_m', etc.). Required, no default.
    ## 2. Readout contribution from superficial and middle layers ([superficial, middle]). (default: [1.0, 0.0])
    ## 3. Task type: False = fine discrimination, True = general discrimination. (default: False)
    ## 4. p_local_s: relative strength of local E projections in the superficial layer ([1, 1] = no local part). (default: [0.4, 0.7])
    ## 5. shuffle_labels: whether to shuffle the labels for the training task or not. (default: False)
    ## 6. opt_readout_before_training: whether there is a logistic regression optimization for readout parameters or not before training (default: False)

    # special cases
    conf_baseline = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jsup']]#, 'kappa_Jmid', 'kappa_f']] # training all parameters (baseline case)
    conf_gentask = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f'], [1.0, 0.0], True] # training with general discrimination task (control case)
    conf_no_horiconn = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f'], [1.0, 0.0], False, [1.0, 1.0]] 
    conf_shuffled_labels = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f'], [1.0, 0.0], False, [0.4, 0.7], True]
    conf_special_dict = {'conf_baseline': conf_baseline,
                        #'conf_gentask': conf_gentask,
                        #'conf_no_horiconn': conf_no_horiconn,
                        #'conf_shuffled_labels': conf_shuffled_labels
                        }

    # changed readout configurations - readout is optimized with logistic regression before training
    conf_suponly_readout = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f'], [1.0, 0.0], False, [0.4, 0.7], False, True] # reading out from middle layer (ablation)
    conf_mixed_readout = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f'], [0.5, 0.5], False, [0.4, 0.7], False, True] # training all parameters but reading out from both middle and superficial layers
    conf_midonly_readout = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f'], [0.0, 1.0], False, [0.4, 0.7], False, True] # reading out from middle layer (ablation)
    conf_suponly_no_hori_readout = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f'], [1.0, 0.0], False, [1.0, 1.0], False, True] # reading out from middle layer (ablation)
    conf_readout_dict = {'conf_suponly_readout': conf_suponly_readout,
                        'conf_mixed_readout': conf_mixed_readout,
                        'conf_midonly_readout': conf_midonly_readout,
                        'conf_suponly_no_hori_readout': conf_suponly_no_hori_readout
                        }

    # training with all parameters but a few
    conf_kappa_Jsup_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jmid', 'kappa_f']] # training all parameters but kappa (ablation)
    conf_kappa_Jmid_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jsup', 'kappa_f']] # training all parameters but kappa (ablation)
    conf_kappa_f_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jsup', 'kappa_Jmid']] # training all parameters but kappa (ablation)
    conf_cms_excluded = [['f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f']] # training all but cE_m, cI_m, cE_s, cI_s (ablation)
    conf_JI_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_IE_m', 'J_EE_s', 'J_IE_s', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f']] # training all but JI (ablation)
    conf_JE_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_II_m', 'J_EI_m', 'J_II_s', 'J_EI_s', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f']] # training all but JI (ablation)
    conf_Jm_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f']] # training all but Jm (ablation)
    conf_Js_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f']] # training all but Js (ablation)
    conf_f_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa_Jsup', 'kappa_Jmid', 'kappa_f']] # training all but f_E, f_I (ablation)
    conf_excluded_dict = {'conf_kappa_Jsup_excluded': conf_kappa_Jsup_excluded,
                        'conf_kappa_Jmid_excluded': conf_kappa_Jmid_excluded,
                        'conf_kappa_f_excluded': conf_kappa_f_excluded,
                        'conf_cms_excluded': conf_cms_excluded,
                        'conf_JI_excluded': conf_JI_excluded,
                        'conf_JE_excluded': conf_JE_excluded,
                        'conf_Jm_excluded': conf_Jm_excluded,
                        'conf_Js_excluded': conf_Js_excluded,
                        'conf_f_excluded': conf_f_excluded
                        }

    # training with only a few parameters
    conf_kappa_Jsup_only = [['kappa_Jsup']] # training only kappa_Jsup (ablation)
    conf_kappa_Jmid_only = [['kappa_Jmid']] # training only kappa_Jsup (ablation)
    conf_kappa_f_only = [['kappa_f']] # training only kappa_Jsup (ablation)
    conf_cms_only = [['cE_m', 'cI_m', 'cE_s', 'cI_s']] # training only cE_m, cI_m, cE_s, cI_s (ablation)
    conf_JI_only = [['J_II_m', 'J_EI_m', 'J_II_s', 'J_EI_s']] # training only JI
    conf_JE_only = [['J_EE_m', 'J_IE_m', 'J_EE_s', 'J_IE_s']] # training only JE
    conf_Jm_only = [['J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m']] # training only Jm
    conf_Js_only = [['J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s']] # training only Js
    conf_f_only = [['f_E','f_I']] # training only f
    conf_only_dict = {'conf_kappa_Jsup_only': conf_kappa_Jsup_only,
                    'conf_kappa_Jmid_only': conf_kappa_Jmid_only,
                    'conf_kappa_f_only': conf_kappa_f_only,
                    'conf_cms_only': conf_cms_only,
                    'conf_JI_only': conf_JI_only,
                    'conf_JE_only': conf_JE_only,
                    'conf_Jm_only': conf_Jm_only,
                    'conf_Js_only': conf_Js_only,
                    'conf_f_only': conf_f_only
                    }

    # return the combined dictionary
    conf_dict = {}
    for conf_group in config_groups:
        if conf_group == 'special':
            conf_dict.update(conf_special_dict)
        elif conf_group == 'readout':
            conf_dict.update(conf_readout_dict)
        elif conf_group == 'excluded':
            conf_dict.update(conf_excluded_dict)
        elif conf_group == 'only':
            conf_dict.update(conf_only_dict)
    conf_names = list(conf_dict.keys())
    conf_list = list(conf_dict.values())

    return conf_dict, conf_names, conf_list