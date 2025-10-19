import os

# Parse Input
def parse_input(data):
    """
    Parse input for the following parameters:
        - number of atoms (N)
        - box dimensions (L)
        - time steps (time_steps)
        - temperature (T)
        - pressure (P)
        - initial positions of atoms
        - initial velocities of atoms
        - atomic masses
    Assumes the following inputs and may be subject to change in future versions:
        - cube box shape
        - periodic boundary conditions
        - lennard-jones potential
        - velocity verlet integration
    """
    # match keywords to expected parameters in line and store in dictionary
    parameter_dict = {}
    for line in data:
        keyword = line.split()[0].lower()
        # if single input parameter - N, L, time_steps, T, P
        if keyword in ['n', 'l', 'time_steps', 't', 'p']:
            parameter_dict[keyword] = line.strip().split()[1]
        # if multiple input parameters - positions, velocities, masses, read lines until hit ###
        elif keyword in ['positions', 'velocities', 'masses']:
            values = []
            for subline in data[data.index(line)+1:]:
                if subline.strip() == "###":
                    break
                values.append(subline.strip())
            parameter_dict[keyword] = values
        else:
            raise ValueError(f"Unexpected keyword '{keyword}' in input data.")
    return parameter_dict

# Read data from an input file (containing number of atoms, grid dimensions, etc.)
def read_input_file(file_path):
    """
    Read data from an input file and parses input file for simulation parameters.
    """
    with open(file_path, 'r') as file:
        data = file.readlines()
        parameter_dict = parse_input(data)
    return parameter_dict

# Write program output to a file (containing final positions of atoms, energies, etc.)
def write_output_file(file_path, data):
    """
    Write data obtained from MD simulation to an output file.
    """
    with open(file_path, 'w') as file:
        file.writelines(data)


# Append logs to a log file (containing simulation progress, errors, etc.)
def append_log_file(file_path, log_data):
    """
    Append log data to a log file with the following information:
        - timestamp
        - simulation step
        - energy values
        - error messages (if any)
    """
    with open(file_path, 'a') as file:
        file.writelines(log_data)