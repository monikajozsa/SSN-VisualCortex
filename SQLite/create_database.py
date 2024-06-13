import sqlite3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy
import pandas as pd
import time

##### Set up SQL connection #####
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to your SQLite database file
database_path = os.path.join(script_dir, 'Apr10.db')

# Path to your SQL file
sql_file_path = os.path.join(script_dir, 'init_tables.sql')

# Connect to the SQLite database (create if it doesn't exist)
conn = sqlite3.connect(database_path)

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

##### clear the database #####
# Get a list of all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Drop each table
for table_name in tables:
    cursor.execute(f"DROP TABLE IF EXISTS {table_name[0]};")

##### Create the tables #####
# Read the SQL file
with open(sql_file_path, 'r') as sql_file:
    sql_script = sql_file.read()

# Execute the SQL script
cursor.executescript(sql_script)

# Commit the changes
conn.commit()

###### Fill up cell_info table ######
# grid size and phases could be loaded from parameters.py
types=['E','I']
# Middle layer cells
cell_ind=0
for i in range(9):
    for j in range(9):
        for phase_ind in range(4):
            for type_ind in range(2):
                cursor.execute('''INSERT INTO cell_info (grid_row, grid_col, phase, type, cell_id) VALUES (?,?,?,?,?)''',
                               (i,j,phase_ind,types[type_ind],cell_ind))
                cell_ind += 1
# Superficial layer cells
for type_ind in range(2):
    for i in range(9):
        for j in range(9):        
            cursor.execute('''INSERT INTO cell_info (grid_row, grid_col, phase, type, cell_id) VALUES (?,?,?,?,?)''',
                            (i,j,0,types[type_ind],cell_ind))
            cell_ind += 1
conn.commit()

###### Fill up orientation map table ######
# Loop through the CSV files
num_runs = 50
final_folder_path = os.path.join('results','Apr10_v1')
for i in range(num_runs): 
    orimap_filename = os.path.join(final_folder_path, f"orimap_{i}.npy")
    orimap_loaded = numpy.load(orimap_filename)
    
    for row in range(orimap_loaded.shape[0]):
        for col in range(orimap_loaded.shape[1]):
            orientation = float(orimap_loaded[row, col])
            grid_row = row
            grid_col = col
            init_index = i
            
            # Insert the data into the orientation_map table
            cursor.execute('''
                INSERT INTO orientation_map (grid_row, grid_col, orientation, init_index)
                VALUES (?, ?, ?, ?)
            ''', (grid_row, grid_col, orientation, init_index))
conn.commit()

##### fill up the training_results table ######
start_time = time.time()
for run_ind in range(num_runs):
    # Load the resutls.csv file
    csv_file_path = os.path.join(final_folder_path, f"results_{run_ind}.csv")
    df = pd.read_csv(csv_file_path)
    
    # Extract the header row (column names)
    columns = df.columns.tolist()
    
    # Ensure the first two columns are 'SGD_step' and 'stage'
    if columns[0] != 'stage' or columns[1] != 'SGD_steps':
        raise ValueError(f"Unexpected column names in results_{run_ind}: {columns[:2]}")
    
    # Loop over the rows in the DataFrame
    rows_to_insert = []
    for index, row in df.iterrows():
        training_stage = row['stage']
        SGD_step = row['SGD_steps']
        
        # Loop over the remaining columns to get var_name and var_value
        for var_name in columns[2:]:
            var_value = row[var_name]
            row_table = (SGD_step, training_stage, run_ind, var_name, var_value)
            rows_to_insert.append(row_table)
    # Insert the data into the training_results table
    cursor.executemany('''
        INSERT INTO training_results (SGD_step, training_stage, init_index, var_name, var_value)
        VALUES (?, ?, ?, ?, ?)
    ''', rows_to_insert)
    print('finished training_results for run', run_ind, ' in time:', time.time()-start_time)
conn.commit()

##### fill up the tuning_curves table #####
stage_str = ['prepre','postpre','post']
start_time = time.time()
# Loop over runs and stages
for run_ind in range(num_runs):
    # Prepare a list to hold all rows to be inserted
    rows_to_insert = []
    for stage in range(3):
        csv_file_path = os.path.join(final_folder_path, f"tc_{stage_str[stage]}_{run_ind}.csv")
        df = pd.read_csv(csv_file_path, header=None)
        oris = numpy.arange(0, 180, 180/df.shape[0])

        for ori_ind in range(df.shape[0]):
            ori = oris[ori_ind]
            for cell_ind in range(df.shape[1]):
                # Collect data for a single row
                row_table = (cell_ind, ori_ind, stage, run_ind, df.iloc[ori_ind, cell_ind], ori)
                rows_to_insert.append(row_table)

    # Insert all collected rows in a single batch
    cursor.executemany('''INSERT INTO tuning_curves (cell_id, ori_ind, training_stage, init_index, tc_value, orientation)
        VALUES (?, ?, ?, ?, ?, ?)''', rows_to_insert)
    print('finished tuning_curves for run and stage', run_ind, ' in time:', time.time()-start_time)

##### check the tables #####
# orientation map
cursor.execute('''SELECT orientation from orientation_map WHERE grid_row == 0 and grid_col == 0''')
oris = cursor.fetchall()
i=0
for ori in oris[0:min(4,num_runs)]:
    print(f'ori at (0,0) point in run {i}',ori)
    i += 1

# cell_info
cursor.execute('''SELECT * from cell_info WHERE id == 0''')
cell0 = cursor.fetchall()
print(f'cell 0 properties:',cell0)

# training_results
#cursor.execute('''SELECT * from training_results WHERE SGD_step == 0 and init_index == 0 and var_name LIKE 'J%' ''')
#step0_J = cursor.fetchall()
#print(f'J params at step 0:',step0_J)

# tuning curves
cursor.execute('''SELECT * from training_results WHERE cell_id == 0 and init_index == 0 and ori_ind == 0 ''')
tc_example = cursor.fetchall()
print(f'tc value for cell 0, run 0, ori 0:',tc_example)

# Commit the changes and close the connection
#conn.commit()
conn.close()