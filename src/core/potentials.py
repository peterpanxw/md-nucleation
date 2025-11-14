"""
Defines interatomic potentials and force calculations.
"""

import numpy as np


def lennard_jones_potential(r, epsilon, sigma):
    """
    Compute Lennard-Jones potential for a given separation.

    Parameters:
        r: float or np.ndarray
            Inter-particle distance.
        epsilon: float
            Depth of the potential well.
        sigma: float
            Finite distance at which potential is zero.
    
    Returns:
        lj_potential: float or np.ndarray
            Potential energy value.
    """
    pass


def compute_forces(positions, box_length):
    """
    Compute forces on all particles using the Lennard-Jones potential.

    Parameters:
        positions: np.ndarray
            (N x 3) array of particle positions.
        box_length: float or np.ndarray
            Simulation box length for periodic boundaries.
    
    Returns:
        particle_forces: np.ndarray
            (N x 3) array of particle forces.
    """
    pass


def potential_energy(positions, box_length):
    """
    Compute total potential energy of the system.

    Parameters:
        positions: np.ndarray
            (N x 3) array of particle positions.
        box_length: float or np.ndarray
            Simulation box length for periodic boundaries.
    
    Returns:
        potential_energy: float
            Total potential energy.
    """
    pass
