import os
import shutil
from datetime import datetime
from dataclasses import fields

def save_code(ssn_layer_pars, stim_pars):
    # Get the current date
    current_date = datetime.now().strftime("%b%d")

    # Create a folder name based on the current date
    folder_name = f"results\{current_date}_v"

    # Find the next available script version
    version = 0
    while os.path.exists(f"{folder_name}{version}"):
        version += 1

    # Create the folder for the results
    final_folder_path = f"{folder_name}{version}"
    os.makedirs(final_folder_path)

    # Create a subfolder for the scripts
    subfolder_script_path = f"{folder_name}{version}\scripts"
    os.makedirs(subfolder_script_path)

    # Get the path to the script's directory
    script_directory = os.path.dirname(os.path.realpath(__file__))

    # Copy files into the folder
    file_names = ['parameters.py', 'training.py', 'training_supp.py', 'model.py', 'util.py', 'SSN_classes.py', 'analysis.py', 'visualization.py']
    for file_name in file_names:
        source_path = os.path.join(script_directory, file_name)
        destination_path = os.path.join(subfolder_script_path, file_name)
        shutil.copyfile(source_path, destination_path)

    print(f"Script files copied successfully to: {script_directory}")

    # return path (inclusing filename) to save results into
    results_filename = os.path.join(final_folder_path,f"{current_date}_v{version}_results.csv")
    print(script_directory)

    # if parameters were updated then add a comment to parameters.py about the updates
    param_file_loc = os.path.join(script_directory, subfolder_script_path)
    comment_param_file(ssn_layer_pars,stim_pars,param_file_loc)
    return results_filename, param_file_loc
    
def comment_param_file(ssn_layer_pars, stim_pars, param_file_loc):
    if ssn_layer_pars is not None:
        # Read the existing content of the file
        param_file_name =  os.path.join(param_file_loc, 'parameters.py')
        with open(param_file_name, 'r') as file:
            file_content = file.read()

            # Create a comment based on the fields of the dataclass
            comment_lines = ['# Updated ssn_layer_pars using randomize_params:']
            for field in fields(ssn_layer_pars):
                comment_lines.append(f'# {field.name}: {getattr(ssn_layer_pars, field.name)}')

                # Add the comment to the existing content
                updated_content = file_content + '\n' + '\n'.join(comment_lines)

                # Write the updated content back to the file
                with open(param_file_name, 'w') as file:
                    file.write(updated_content)

    if stim_pars is not None:
        # Read the existing content of the file
        with open(param_file_name, 'r') as file:
            file_content = file.read()

            # Create a comment based on the fields of the dataclass
            comment_lines = ['# Updated stim_pars using randomize_params:']
            for field in fields(stim_pars):
                comment_lines.append(f'# {field.name}: {getattr(stim_pars, field.name)}')

                # Add the comment to the existing content
                updated_content = file_content + '\n' + '\n'.join(comment_lines)

                # Write the updated content back to the file
                with open(param_file_name, 'w') as file:
                    file.write(updated_content)
    
