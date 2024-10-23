import os
import time
from pylatex import Document, Section, Subsection, Figure, NoEscape, NewPage, Package

#################### Define configurations ####################

# special cases
conf_baseline = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa']] # training all parameters (baseline case)
conf_gentask = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], True] # training with general discrimination task (control case)
conf_no_horiconn = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], False, [1.0, 1.0]] 
conf_shuffled_labels = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], False, [0.4, 0.7], True]
conf_special_dict = {'conf_baseline': conf_baseline,
                    'conf_gentask': conf_gentask,
                    'conf_no_horiconn': conf_no_horiconn,
                    'conf_shuffled_labels': conf_shuffled_labels
                    }

# changed readout configurations - readout is optimized with logistic regression before training
conf_suponly_readout = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], False, [0.4, 0.7], False, True] # reading out from middle layer (ablation)
conf_mixed_readout = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [0.5, 0.5], False, [0.4, 0.7], False, True] # training all parameters but reading out from both middle and superficial layers
conf_midonly_readout = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [0.0, 1.0], False, [0.4, 0.7], False, True] # reading out from middle layer (ablation)
conf_suponly_no_hori_readout = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa'], [1.0, 0.0], False, [1.0, 1.0], False, True] # reading out from middle layer (ablation)
conf_readout_dict = {'conf_suponly_readout': conf_suponly_readout,
                    'conf_mixed_readout': conf_mixed_readout,
                    'conf_midonly_readout': conf_midonly_readout,
                    'conf_suponly_no_hori_readout': conf_suponly_no_hori_readout
                    }

# training with all parameters but a few
conf_kappa_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s']] # training all parameters but kappa (ablation)
conf_cms_excluded = [['f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa']] # training all but cE_m, cI_m, cE_s, cI_s (ablation)
conf_JI_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_IE_m', 'J_EE_s', 'J_IE_s', 'kappa']] # training all but JI (ablation)
conf_JE_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_II_m', 'J_EI_m', 'J_II_s', 'J_EI_s', 'kappa']] # training all but JI (ablation)
conf_Jm_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa']] # training all but Jm (ablation)
conf_Js_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'f_E', 'f_I', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'kappa']] # training all but Js (ablation)
conf_f_excluded = [['cE_m', 'cI_m', 'cE_s', 'cI_s', 'J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m', 'J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s', 'kappa']] # training all but f_E, f_I (ablation)
conf_excluded_dict = {'conf_kappa_excluded': conf_kappa_excluded,
                    'conf_cms_excluded': conf_cms_excluded,
                    'conf_JI_excluded': conf_JI_excluded,
                    'conf_JE_excluded': conf_JE_excluded,
                    'conf_Jm_excluded': conf_Jm_excluded,
                    'conf_Js_excluded': conf_Js_excluded,
                    'conf_f_excluded': conf_f_excluded
                    }

# training with only a few parameters
conf_kappa_only = [['kappa']] # training only kappa (ablation)
conf_cms_only = [['cE_m', 'cI_m', 'cE_s', 'cI_s']] # training only cE_m, cI_m, cE_s, cI_s (ablation)
conf_JI_only = [['J_II_m', 'J_EI_m', 'J_II_s', 'J_EI_s']] # training only JI
conf_JE_only = [['J_EE_m', 'J_IE_m', 'J_EE_s', 'J_IE_s']] # training only JE
conf_Jm_only = [['J_EE_m', 'J_EI_m', 'J_IE_m', 'J_II_m']] # training only Jm
conf_Js_only = [['J_EE_s', 'J_EI_s', 'J_IE_s', 'J_II_s']] # training only Js
conf_f_only = [['f_E','f_I']] # training only f
conf_only_dict = {'conf_kappa_only': conf_kappa_only,
                'conf_cms_only': conf_cms_only,
                'conf_JI_only': conf_JI_only,
                'conf_JE_only': conf_JE_only,
                'conf_Jm_only': conf_Jm_only,
                'conf_Js_only': conf_Js_only,
                'conf_f_only': conf_f_only
                }

# create dictionary of configurations to loop over
conf_dict =  {**conf_special_dict, **conf_excluded_dict, **conf_only_dict, **conf_readout_dict}

conf_names = list(conf_dict.keys())
conf_list = list(conf_dict.values())

def add_multiple_figures_to_page(doc, folder, list_of_fig_names, titles=None, captions=None, scaling_param=None):
    """ Adds multiple figures (and LaTeX captions/titles) to a PyLaTeX document with scaling support. """
    for i, fig_name in enumerate(list_of_fig_names):
        fig_path = os.path.join(folder, fig_name + '.png')
        
        # Determine scaling (default to 0.5\textwidth if no scaling provided)
        scale = scaling_param[i] if scaling_param and i < len(scaling_param) else 0.5
        
        # Add figure
        with doc.create(Figure(position='h!')) as fig:
            fig.add_image(fig_path, width=NoEscape(f'{scale}\\textwidth'))  # Apply scaling here
            # Add title (if available)
            if titles and i < len(titles):
                fig.add_caption(NoEscape(titles[i]))
            
            # Add caption (if available)
            if captions and i < len(captions):
                doc.append(NoEscape(captions[i]))
        doc.append(NoEscape(r'\vspace{1cm}'))  # Adds 1cm of vertical space after each figure
        

def make_pdf_from_figures(folder, list_of_fig_names, list_of_titles, list_of_captions, sup_title, pdf_file_name, scaling_params=None):
    """ Create a PDF file from a list of figures and captions using PyLaTeX, with scaling support """
    # Create a new PyLaTeX document
    doc = Document()

    # Add necessary packages
    doc.packages.append(Package('amsmath'))  # Add amsmath package
    doc.packages.append(Package('geometry', options=['margin=1in']))
    doc.packages.append(Package('caption'))  # To control caption alignment

    # Set captions to be left-aligned
    doc.append(NoEscape(r'\captionsetup{justification=raggedright}'))

    # Add a super title
    doc.preamble.append(NoEscape(r'\title{%s}' % sup_title))
    doc.preamble.append(NoEscape(r'\date{}'))  # Remove date
    doc.append(NoEscape(r'\maketitle'))  # Add the title to the document
    
    # Loop over the list of figures and add them to the document
    for i, fig_names in enumerate(list_of_fig_names):
        # Pass scaling_params[i] to the page function
        add_multiple_figures_to_page(doc, folder, fig_names, list_of_titles[i], list_of_captions[i], scaling_params[i] if scaling_params else None)
        doc.append(NewPage())  # Start a new page after each set of figures
    # Generate the PDF
    pdf_file_path = os.path.join(folder, pdf_file_name)
    doc.generate_pdf(pdf_file_path, clean_tex=False, compiler='pdflatex')

################## MAIN CODE ##################

list_of_fig_names = [
    ['offset_pre_post','boxplot_relative_changes', 'boxplot_relative_changes_ylim'], 
    ['tc_features_train_color_by_phase', 'tc_features_train_color_by_run_index', 'tc_features_train_color_by_pref_ori', 'tc_features_train_color_by_type'], 
    ['tc_slope_train_color_by_type'], 
    ['corr_psychometric_Jraw','corr_psychometric_Jcombined','corr_psychometric_f_c','corr_psychometric_kappa'], 
    ['MVPA_match_paper_fig','MVPA_scores', 'Mahal_scores'],
    ['combined_corr_triangles']
    ]
list_of_titles = [
    ['Discrimination thresholds before and after training', 'Relative and absolute changes in model parameters before and after training', 'Relative and absolute changes in model parameters before and after training with matching y-axis limits accross boxes and configurations'], 
    ['Tuning curve features of the model before vs after training colored by phase', 'Tuning curve features colored by run index', 'Tuning curve features colored by preferred orientation', 'Tuning curve features colored by cell type'], 
    ['Absolute change (post-pre) in tuning curve slope measured at 55 and 125 and their interpolated curves for excitatory and inhibitory cells + p-values from Mann-Whitney U test'], 
    ['Regression plot between relative changes in psychometric threshold (dependent) and J parameters', 'Regression plot between relative changes in psychometric threshold (dependent) and combined J parameters', 'Regression plot between relative changes in psychometric threshold (dependent) and f and c parameters', r'Regression plot between relative changes in psychometric threshold (dependent) and $\kappa$ parameters'], 
    ['MVPA scores arranged as in the experimental paper', 'MVPA scores with case-lines', 'Mahalanobis distances with case-lines'],
    [r'Correlation between relative change in MVPA scores, relative change in $J_E/J_I$ ratio, and psychometric threshold']
    ]
list_of_captions = [''] * len(list_of_fig_names)
scaling_vector = [ [0.7, 1, 1], [1,1,1,1], [1], [1,0.8,0.5,0.5], [0.8,0.9,0.9], [1]]
sup_title = 'Figure list for NN model analysis on Perceptual Learning'

# Define destination folder and create it if it does not exist
destination_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results', 'Aug15_v0', 'summary_figures')
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
'''
# Loop over the different configurations and create a PDF file for each
for i, conf in enumerate(conf_names):
    start_time = time.time()
    source_folder= os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results', 'Aug15_v0', conf, 'figures')
    pdf_file_name = destination_folder + f'/{conf}_figures.pdf'
    make_pdf_from_figures(source_folder, list_of_fig_names, list_of_titles, list_of_captions, sup_title, pdf_file_name, scaling_vector)
    print(f'Created PDF file for {conf} configuration in {time.time()-start_time} seconds.')
'''
# Make a pdf for the 'boxplot_relative_changes_ylim' from all config folders into one pdf by defining a new list_of_fig_names (empty list_of_titles and list_of_captions)
list_of_fig_names_boxplots = []
list_of_titles_boxplots = [[conf_name for conf_name in conf_names]]
# Replace _ with space in the configuration names for the title
list_of_titles_boxplots = [[conf_name.replace('_', ' ') for conf_name in conf_names]]

list_of_captions_boxplots = [['' for i in range(len(conf_names))]]
scaling_vector_boxplots = [[1 for i in range(len(conf_names))]]

source_folder_boxplots = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results', 'Aug15_v0')
for i, conf in enumerate(conf_names):
    list_of_fig_names_boxplots.append(os.path.join(conf, 'figures', 'boxplot_relative_changes_ylim'))
list_of_fig_names_boxplots = [list_of_fig_names_boxplots]
sup_title_boxplots = 'Relative and absolute changes in model parameters with matching y-axis limits accross boxes and configurations'
pdf_file_name_boxplots = destination_folder + '/boxplot_relative_changes_ylim_all_configs.pdf'
make_pdf_from_figures(source_folder_boxplots, list_of_fig_names_boxplots, list_of_titles_boxplots, list_of_captions_boxplots, sup_title_boxplots, pdf_file_name_boxplots, scaling_vector_boxplots)