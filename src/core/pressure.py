"""
pressure.py
-----------
Virial pressure calculation for the MD simulation framework.

This module provides functionality to compute the instantaneous
virial pressure of the system in reduced units (k_B = 1).

Dependencies
------------
- initialization.py   (for the simulation object structure)
- potentials.py       (for Lennard–Jones forces via radial derivative)
- temperature.py      (for kinetic energy helper)

where `sim` is typically a VelocityVerletIntegrator or MinimizeEnergy
object that inherits from InitializeSimulation and EnergyComputationsMixin.
"""

from __future__ import annotations
import numpy as np


class PressureCalculator:
    """
    Compute virial pressure for a given MD simulation object.
    
    The simulation object `sim` is expected to have at least:
        sim.atoms_positions : (N, 3) array
        sim.atoms_velocity  : (N, 3) array
        sim.atom_mass       : (N,)   array
        sim.box_size        : length-6 array [Lx, Ly, Lz, 90, 90, 90]
        sim.neighbor_lists  : list of neighbor arrays
        sim.sigma_ij_list   : LJ sigma_ij per atom i
        sim.epsilon_ij_list : LJ epsilon_ij per atom i
        sim.compute_distance(position_i, positions_j, only_norm=False)
    
    All values are assumed to be in the simulation's reduced units,
    so the resulting pressure is also in reduced units.
    """
    
    def __init__(self, sim):
        """
        Parameters
        ----------
        sim : object
            MD simulation object (e.g., VelocityVerletIntegrator)
            providing positions, velocities, neighbor lists, etc.
        """
        self.sim = sim
        
        # Eager sanity checks for PressureCalculator to work
        required_attrs = [
            "atoms_positions",
            "atoms_velocity",
            "atom_mass",
            "box_size",
            "neighbor_lists",
            "sigma_ij_list",
            "epsilon_ij_list",
        ]
        for attr in required_attrs:
            if not hasattr(sim, attr):
                raise AttributeError(
                    f"Simulation object is missing required attribute '{attr}' "
                    f"needed for pressure calculation."
                )
        if not hasattr(sim, "compute_distance"):
            raise AttributeError(
                "Simulation object must provide a 'compute_distance' method "
                "from EnergyComputationsMixin."
            )
        if not hasattr(sim, "instantaneous_temperature"):
            raise AttributeError(
                "Simulation object must provide a 'instantaneous_temperature' method "
                "from Thermostat."
            )
    
    
    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------
    def instantaneous_pressure(self):
        """
        Compute instantaneous virial pressure of the investigated system.
        
        The pressure is evaluated using the standard virial expression
        in reduced MD units (with k_B = 1):
        
            `P = (N T) / V  +  (1 / (3 V)) * W`
        
        where:
            - N is the number of particles
            - T is the instantaneous temperature
            - V is the simulation volume
            - W is the virial term:
                    `W = Σ_{i<j} r_ij · F_ij`
        
        The temperature is obtained via the same kinetic-energy
        expression used in `temperature.py`. The virial term is 
        computed using Lennard-Jones pair forces from `potentials.py`
        through `_pair_virial()`.
        
        Parameters
        ----------
        None
            This method operates on the internal simulation state.

        Returns
        -------
        float
            Instantaneous pressure of the system in reduced units.
        """
        N = len(self.sim.atom_mass)
        V = self._volume()
        
        T = self.sim.instantaneous_temperature()
        W = self._pair_virial()
        
        P = (N * T) / V + W / (3.0 * V)
        return float(P)
    
    
    # -------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------
    def _volume(self):
        """
        Compute simulation box volume from box_size[:3].
        
        Assumes orthorhombic box: (Lx, Ly, Lz, 90, 90, 90).
        """
        Lx, Ly, Lz = self.sim.box_size[:3]
        return float(Lx * Ly * Lz)
    
    
    def _pair_virial(self):
        """
        Compute the pairwise virial contribution to the system pressure.
        
        The virial term is defined as:
        
            `W = Σ_{i<j} ( r_ij · F_ij )`
        
        where:
            - r_ij is the minimum-image displacement vector from particle j to i
            - F_ij is the LJ radial force magnitude force on particle i due to particle j
            - Only pairs i < j are included to avoid double counting
        
        This method uses:
            - Neighbor lists in `self.sim.neighbor_lists`
            - Minimum-image distances from `self.sim.compute_distance`
            - Pairwise force matrix from `self.sim.compute_force(return_vector=False)`
            - Lennard-Jones forces computed inside the force matrix (via potentials)
        
        Parameters
        ----------
        None
            Operates on the internal simulation state.

        Returns
        -------
        float
            The total virial W of the system in reduced MD units.
        
        Notes
        -----
        - `compute_force(return_vector=False)` yields an (N, N, 3) force matrix,
          where F[i, j] is the force on i from j.
        - `compute_distance(pos[i], pos[j], only_norm=False)` returns both the
          scalar distances and the displacement vectors r_ij under periodic
          boundary conditions.
        - The contraction r_ij · F_ij is computed via:
          
            `np.einsum("ij,ij->i", r_vec, F_ij)`
        
          returning one scalar per neighbor.
        """
        pos = self.sim.atoms_positions
        n_total = len(pos)

        # (N, N, 3) full pairwise force matrix
        F = self.sim.compute_force(return_vector=False)

        W = 0.0

        for i in range(n_total - 1):
            neighbors = self.sim.neighbor_lists[i]
            if neighbors is None or len(neighbors) == 0:
                continue
            
            # Convert neighbor list to numpy array for vectorized indexing
            idx_j = np.asarray(neighbors, dtype=int)
            
            # r_ij returned as (pos[i] - pos[j]); vector from j→i
            r_ij, r_vec = self.sim.compute_distance(
                pos[i],
                pos[idx_j],
                only_norm=False
            )

            # Extract F_ij vectors from the force matrix
            F_ij = F[i, idx_j, :]  # shape (k, 3)

            # Contribution r_ij · F_ij for all neighbors of i
            # r_vec and F_ij are both (k, 3)
            W += float(np.sum(np.einsum("ij,ij->i", r_vec, F_ij)))
        
        return W