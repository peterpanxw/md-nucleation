import os
import json
from datetime import datetime

def parse_input(data):
    """
    Parse simulation input data for key molecular dynamics parameters.

    Parameters
    ----------
    data : list of str
        Lines from an input file containing simulation parameters.

    Returns
    -------
    dict
        Dictionary mapping parameter keywords to their corresponding values:
        - 'n' : int
            Number of atoms.
        - 'l' : float
            Box length (assumed cubic box).
        - 'time_steps' : int
            Total number of MD time steps.
        - 't' : float
            Temperature in Kelvin.
        - 'p' : float
            Pressure in bar.
        - 'positions' : list of str
            Initial positions of atoms.
        - 'velocities' : list of str
            Initial velocities of atoms.
        - 'masses' : list of str
            Atomic masses.

    Raises
    ------
    ValueError
        If an unexpected keyword is encountered in the input data.

    Examples
    --------
    >>> data = ["N 2", "L 10.0", "time_steps 100", "T 300", "P 1.0"]
    >>> params = parse_input(data)
    >>> params["t"]
    '300'
    """
    parameter_dict = {}
    for line in data:
        keyword = line.split()[0].lower()
        if keyword in ['n', 'l', 'time_steps', 't', 'p']:
            parameter_dict[keyword] = line.strip().split()[1]
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


def read_input_file(file_path):
    """
    Read and parse an input file for molecular dynamics simulation parameters.

    Parameters
    ----------
    file_path : str
        Path to the input file containing simulation parameters.

    Returns
    -------
    dict
        Parsed simulation parameters (see `parse_input`).
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
        parameter_dict = parse_input(data)
    return parameter_dict


def write_output_file(file_path, data):
    """
    Write simulation output data (e.g., energies, coordinates) to a file.

    Parameters
    ----------
    file_path : str
        Path to the output file.
    data : list of str
        List of strings representing simulation results or snapshots.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(data)
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")


def append_log_file(file_path, log_data):
    """
    Append log information (progress or errors) to an existing log file.

    Parameters
    ----------
    file_path : str
        Path to the log file.
    log_data : list of str
        List of log entries (strings) to append to the file.
    """
    with open(file_path, 'a', encoding='utf-8') as file:
        file.writelines(log_data)


def save_checkpoint(file_path, positions, velocities, step):
    """
    Save a simulation checkpoint file for restarting MD runs.

    Parameters
    ----------
    file_path : str
        Path to the checkpoint file (recommended: `.json` extension).
    positions : list of list of float
        Atomic positions at the current step.
    velocities : list of list of float
        Atomic velocities at the current step.
    step : int
        Current simulation step number.
    """
    checkpoint = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "positions": positions,
        "velocities": velocities
    }
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(checkpoint, file, indent=4)
        return True
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False