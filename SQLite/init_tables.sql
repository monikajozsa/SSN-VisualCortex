CREATE TABLE cell_info (
    grid_row INTEGER NOT NULL,
    grid_col INTEGER NOT NULL,
    phase INTEGER NOT NULL,
    cell_type TEXT NOT NULL,
    cell_id INTEGER PRIMARY KEY
);

CREATE TABLE orientation_map (
    grid_row INTEGER NOT NULL,
    grid_col INTEGER NOT NULL,
    orientation REAL NOT NULL,
    init_index INTEGER NOT NULL,
    PRIMARY KEY (grid_row, grid_col, init_index)
);

CREATE TABLE training_results (
    SGD_step INTEGER NOT NULL,
    training_stage INTEGER NOT NULL,
    init_index INTEGER NOT NULL,
    var_name TEXT NOT NULL,
    var_value REAL,
    PRIMARY KEY (SGD_step, training_stage, init_index, var_name)
);

CREATE TABLE tuning_curves (
    cell_id INTEGER NOT NULL,
    ori_ind INTEGER NOT NULL,
    training_stage INTEGER NOT NULL,
    init_index INTEGER NOT NULL,
    tc_value REAL NOT NULL,
    PRIMARY KEY (cell_id, ori_ind, training_stage, init_index)
);

-- CREATE TABLE tuning_curves_2d (
--     cell_id INTEGER NOT NULL,
--     orientation INTEGER NOT NULL,
--     contrast REAL NOT NULL,    
--     training_stage INTEGER NOT NULL,
--     init_index INTEGER NOT NULL,
--     tc_value REAL NOT NULL,
--     PRIMARY KEY (cell_id, orientation, contrast, training_stage, init_index)
-- );