"""
energy_minimization.py
----------------------
Module providing Lennard–Jones (LJ) potential / force evaluation and a simple
steepest–descent energy minimizer built on top of class `InitializeSimulation`
from `initialization.py`.

This module:
- Computes Lennard–Jones 12–6 pair potentials and their radial forces
- Measures interatomic distances under periodic boundary conditions
- Uses neighbor lists and precomputed cross-coefficients (σ_ij, ε_ij)
- Provides a steepest–descent routine to relax initial configurations and
  remove atomic overlaps
- Ensures atoms remain inside the simulation box through periodic wrapping

Author: Xiwei Pan (Princeton University)
Course: APC 524 – Molecular Dynamics for Nucleation
Date: 2025-11-14
"""

from __future__ import annotations
from typing import Tuple
from initialization import InitializeSimulation
import copy
import numpy as np


def potentials(
    epsilon_ij: np.ndarray,
    sigma_ij: np.ndarray,
    r_ij: np.ndarray,
    derivative: bool = False,
):
    """
    Evaluate Lennard–Jones 12-6 pair potential or its radial derivative.
    
    Parameters
    ----------
    epsilon_ij : ndarray
        Lennard–Jones energy parameters `ε_ij` in SI units.
    sigma_ij : ndarray
        Lennard–Jones size parameters `σ_ij` in SI units.
    r_ij : ndarray
        Pair distances `r_ij` (meters). Must be strictly positive for a
        physically meaningful result.
    derivative : bool, optional
        If False (default), return the potential energy `U(r)`.
        If True, return the radial force magnitude `F(r) = -dU/dr`
        (sign convention: positive ⇒ repulsive).
    
    Returns
    -------
    ndarray
        Same shape as `r_ij`; pair energies or radial force magnitudes.
    
    Notes
    -----
    - `U(r) = 4ε[(σ / r)^12 - (σ / r)^6]`
    
    - `F(r) = 24ε[2(σ / r)^12 - (σ / r)^6] / r`
    """
    epsilon_ij = np.asarray(epsilon_ij, dtype=float)
    sigma_ij = np.asarray(sigma_ij, dtype=float)
    r_ij = np.asarray(r_ij, dtype=float)
    
    # Avoid RuntimeWarnings (such as divide-by-zero) and handle the result later manually using np.where
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_r = np.where(r_ij > 0.0, 1.0 / r_ij, 0.0)  # Safe inverse, zero when r_ij <= 0.0
        s_over_r = sigma_ij * inv_r
        s_over_r6 = s_over_r ** 6
        s_over_r12 = s_over_r6 ** 2

        if not derivative:
            energy = 4.0 * epsilon_ij * (s_over_r12 - s_over_r6)  # Lennard-Jones potential between pairs i and j
            energy = np.where(r_ij > 0.0, energy, 0.0)            # Do not need potential energy if r_ij <= 0.0
            return energy

        # Radial (conservative) force magnitude F(r) = -dU/dr
        force = 24.0 * epsilon_ij * (2.0 * s_over_r12 - s_over_r6) * inv_r
        force = np.where(r_ij > 0.0, force, 0.0)
        return force


# ---------------------------------------------------------------------------
# High-level energy / force utilities (mixin)
# ---------------------------------------------------------------------------
class EnergyComputationsMixin:
    """
    Mixin providing distance, energy and force evaluation for MD systems.
    
    Assumes these attributes exist on the subclass:
    
    - number_atoms : 1D array, number of atoms per species.
    - atoms_positions : (N, 3) array, positions in meters.
    - neighbor_lists : list of arrays, neighbor indices per atom.
    - sigma_ij_list / epsilon_ij_list : precomputed LJ cross params.
    - box_size : length-6 array [Lx, Ly, Lz, 90, 90, 90].
    - box_boundaries : (3, 2) array, lower/upper bounds per dimension.
    """
    
    _TINY = 1e-16  # to avoid division by zero
    
    @property
    def n_atoms(self):
        """Total number of atoms in the system."""
        return int(np.sum(self.number_atoms))
    
    # ---- distance utilities -------------------------------------------------
    def compute_distance(
        self,
        position_i: np.ndarray,
        positions_j: np.ndarray,
        only_norm: bool = True,
    ):
        """
        Compute minimum-image distances between one atom and neighbors.
        
        This function applies the minimum-image convention under periodic
        boundary conditions to compute the distances between a reference
        atom i and a set of neighboring atoms j. Optionally, both the wrapped
        displacement vectors and their norms can be returned.

        
        Parameters
        ----------
        position_i : ndarray, shape (3,)
            Cartesian coordinates of the reference atom i.
        positions_j : ndarray, shape (M, 3)
            Cartesian coordinates of M neighboring atoms j.
        only_norm : bool, optional
            If True (default), return only the norm `|r_ij|`.
            If False, return `(|r_ij|, r_vec_ij)`, the `r_vec_ij` is the
            wrapped displacement vectors, which is used for computing force.
        
        Returns
        -------
        ndarray or (ndarray, ndarray)
            Distances `r_ij` and optionally the wrapped vectors.
        
        Notes
        -----
        - The minimum-image convention is applied using only the box lengths
          `box_size[:3]`. Box angles are assumed to be 90° (orthorhombic box).
        - Periodic boundary conditions are enforced in all three directions.
        - This method is used by both potential and force evaluations.
        """
        position_i = np.asarray(position_i, dtype=float)
        positions_j = np.asarray(positions_j, dtype=float)
        
        box_lengths = np.asarray(self.box_size[:3], dtype=float)
        delta = position_i - positions_j  # Displacement vectors
        
        rij_vec = np.remainder(delta + box_lengths / 2.0, box_lengths) - box_lengths / 2.0  # Minimum-image convention (MIC). delta + L/2 wrapped into [0, L), then shift back to [-L/2, L/2)
        rij = np.linalg.norm(rij_vec, axis=1)  # Nearest-neighbor distances in a periodic system
        
        if only_norm:
            return rij
        return rij, rij_vec
    
    
    def compute_potential(self):
            """
            Compute total Lennard–Jones potential energy for the current state.
            
            This method loops over all atoms, evaluates interactions only with
            neighbors listed in `self.neighbor_lists`, and accumulates the
            pairwise LJ energy using the precomputed cross parameters`sigma_ij_list`
            and `epsilon_ij_list`. Each pair is counted exactly once (i < j).
            
            Returns
            -------
            float
                Total LJ potential energy of the current configuration.
            
            Notes
            -----
            - Neighbor lists must be updated prior to calling this function.
            - Distances are evaluated through `compute_distance(..., only_norm=True)`,
            which applies periodic boundary conditions via the minimum-image
            convention.
            - Uses the global `potentials(epsilon_ij, sigma_ij, r_ij)` function to
            evaluate per-pair LJ energy.
            """
            total_energy = 0.0
            n_total = self.n_atoms                  # Total number of atoms

            for i in range(n_total - 1):            # Loop over all atoms except the last one, to avoid double counting U_ij = U_ji
                neighbors = self.neighbor_lists[i]  # Neighbor indices for atom i, only j > i are included
                if not neighbors:
                    continue
                
                # Compute distances of atom i to its neighbors
                r_ij = self.compute_distance(
                    self.atoms_positions[i],
                    self.atoms_positions[neighbors],
                    only_norm=True,
                )
                
                # Retrieve precomputed LJ cross coefficients for atom i
                sigma_ij = self.sigma_ij_list[i]
                epsilon_ij = self.epsilon_ij_list[i]
                
                # Accumulate the system's total LJ potential energy by repeatedly calling the `potentials` kernel for each atom i
                total_energy += np.sum(potentials(epsilon_ij, sigma_ij, r_ij))
            
            return float(total_energy)
    
    
    def compute_force(self, return_vector: bool = True):
            """
            Compute Lennard–Jones forces on all atoms in the system.
            
            This method evaluates the pairwise LJ forces between atoms that fall
            within the cutoff radius. The computation uses the pre-computed
            cross-interaction coefficients (`sigma_ij_list` and `epsilon_ij_list`)
            as well as the neighbor lists to avoid unnecessary pair evaluations.
            
            Parameters
            ----------
            return_vector : bool, optional
                If True (default), return the net force on each atom as an array of
                shape `(N, 3)`.
                If False, return a full force matrix of shape `(N, N, 3)`, where element
                `[i, j]` contains the force exerted on atom i by atom j.
                This matrix form is used in later calculations such as pressure.
            
            Returns
            -------
            ndarray
                If `return_vector=True`:
                    An array of shape `(N, 3)` containing the total force on each atom.

                If `return_vector=False`:
                    An array of shape `(N, N, 3)` containing pairwise force contributions.
            
            Notes
            -----
            - Only neighbors listed in `self.neighbor_lists` are considered, which reduces
            computational cost.
            - Forces are computed using the derivative of the LJ potential. The helper
            function `potentials(..., derivative=True)` must return the scalar magnitude
            of the force between each pair, which is then converted to vector form.

            Examples
            --------
            >>> forces = sim.compute_force()
            >>> forces.shape
            (5, 3)

            >>> force_matrix = sim.compute_force(return_vector=False)
            >>> force_matrix.shape
            (5, 5, 3)
            """
            n_total = self.n_atoms
            
            if return_vector:
                forces = np.zeros((n_total, 3), dtype=float)
            else:
                forces = np.zeros((n_total, n_total, 3), dtype=float)
            
            for i in range(n_total - 1):  # Compute forces only for i < j, the rest are filled by Newton's third law
                neighbors = self.neighbor_lists[i]
                if not neighbors:
                    continue
                
                r_ij, r_vec = self.compute_distance(
                    self.atoms_positions[i],
                    self.atoms_positions[neighbors],
                    only_norm=False,
                )
                
                sigma_ij = self.sigma_ij_list[i]
                epsilon_ij = self.epsilon_ij_list[i]
                
                # Radial LJ force magnitude (scalar)
                f_r = potentials(epsilon_ij, sigma_ij, r_ij, derivative=True)
                
                safe_r = np.clip(r_ij, self._TINY, None)  # avoid division by zero
                direction = (r_vec.T / safe_r).T          # unit vectors \hat{r}_ij indicating force directions
                f_vec = f_r[:, None] * direction          # force vectors
                
                if return_vector:
                    # Newton's third law
                    forces[i] += np.sum(f_vec, axis=0)
                    forces[neighbors] -= f_vec
                else:
                    forces[i, neighbors, :] += f_vec      # Pairwise forces on atom i due to neighbors j used for later pressure calculation
            
            return forces
        
        
    def wrap_in_box(self):
        """
        Wrap all atoms back into the simulation box using periodic boundaries.
        
        This method enforces periodic boundary conditions (PBC) by ensuring
        that any atom whose coordinate lies outside the simulation box
        (as defined by `self.box_boundaries`) is shifted back into the box
        by exactly one box length along the corresponding dimension.
        The operation is applied independently in x, y, and z directions.
        
        Parameters
        ----------
        None
            The method operates directly on the instance attribute
            `self.atoms_positions` and does not take external arguments.
        
        Returns
        -------
        None
            The atomic coordinates stored in `self.atoms_positions` are
            modified in-place so that all atoms lie within the simulation
            box (domain).
        
        Notes
        -----
        - The method assumes 90-degree-angle boxes defined by `self.box_boundaries`.
        - Called after displacement steps during energy minimization or molecular
        dynamics to maintain valid positions.
        """
        positions = self.atoms_positions
        for dim in range(3):
            low, high = self.box_boundaries[dim]
            length = high - low
            
            # Wrap positions above the upper boundary (by subtracting box length)
            mask_hi = positions[:, dim] > high
            positions[mask_hi, dim] -= length
            
            # Wrap positions below the lower boundary (by adding box length)
            mask_lo = positions[:, dim] < low
            positions[mask_lo, dim] += length
        
        self.atoms_positions = positions


# ---------------------------------------------------------------------------
# Steepest-descent energy minimizer
# ---------------------------------------------------------------------------
class MinimizeEnergy(EnergyComputationsMixin, InitializeSimulation):
    """
    Steepest-descent potential energy minimizer.
    
    Extends InitializeSimulation with energy / force evaluation and a
    steepest-descent minimizer using an adaptive displacement size.
    """
    
    def __init__(self, maximum_steps: int, *args, **kwargs):
        """
        Initialize the steepest-descent energy minimizer.
        
        Parameters
        ----------
        maximum_steps : int
            Maximum number of steepest-descent iterations.
        *args, **kwargs :
            Additional positional and keyword arguments passed directly to
            the parent class initializer. These include the parameters
            required to set up the molecular system (e.g., `number_atoms`,
            `sigma`, `epsilon`, `atom_mass`, `box_dimensions`, `cut_off`).
        
        Attributes
        ----------
        maximum_steps : int
            Maximum number of iterations allowed during minimization.
        displacement : float
            Dimensionless displacement factor used to scale atom movements
            along the force direction. Adaptively increased or decreased
            during optimization.
        Epot : float or None
            Current potential energy of the system. Initialized as None and
            computed on the first iteration.
        MaxF : float or None
            Maximum force magnitude in the system. Used to normalize
            displacement steps. Initialized as None.
        """
        self.maximum_steps = int(maximum_steps)
        # Initialize positions, neighbor lists, LJ params, etc.
        super().__init__(*args, **kwargs)
        
        # Minimization state
        self.displacement = 0.01  # Initial dimensionless displacement factor
        self.Epot: float | None = None
        self.MaxF: float | None = None
    
    
    def run(self):
        """
        Perform steepest-descent energy minimization in place.
        
        Updates:
        - self.atoms_positions
        - self.Epot  (final potential energy)
        - self.MaxF  (final maximum force magnitude)
        """
        # Ensure neighbor lists and cross coefficients are initialized
        self._update_neighbor_lists(force_update=True)
        self._update_cross_coefficients(force_update=True)
        
        for step in range(self.maximum_steps + 1):
            self.step = step  # Current iteration number (coordinate with `self.neighbor` to determine the update frequency)
            
            # Current potential energy & forces
            forces = self.compute_force(return_vector=True)  # Also serves as negative gradient direction (-dU/dr)
            current_E = self.compute_potential()
            max_force = float(np.max(np.abs(forces)))        # Maximum force magnitude for normalizing the update (minimizing) direction
            
            if self.Epot is None:
                self.Epot = current_E
            if self.MaxF is None:
                self.MaxF = max_force
            
            # Store current state before trial move
            init_Epot = self.Epot
            init_MaxF = self.MaxF
            init_positions = copy.deepcopy(self.atoms_positions)
            
            # Trial move along negative gradient to lower energy (gradient descent)
            if init_MaxF > 0.0:
                trial_positions = (
                    self.atoms_positions + forces / init_MaxF * self.displacement
                )
            else:
                # Already mechanically relaxed, stop minimizing
                break
            
            # Apply trial move and wrap outside atoms into box
            self.atoms_positions = trial_positions
            self.wrap_in_box()
            
            # Refresh neighbors / cross coefficients if needed
            self._update_neighbor_lists()
            self._update_cross_coefficients()
            
            # Energy after the trail move
            trial_E = self.compute_potential()
            
            if trial_E < init_Epot:
                # Accept: downhill move
                self.Epot = trial_E
                forces = self.compute_force(return_vector=True)
                self.MaxF = float(np.max(np.abs(forces)))
                self.displacement *= 1.2
            else:
                # Reject: revert and shrink step
                self.atoms_positions = init_positions
                self.Epot = init_Epot
                self.MaxF = init_MaxF
                self.displacement *= 0.2
                self.wrap_in_box()