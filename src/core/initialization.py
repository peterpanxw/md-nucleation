"""
initialize_simulation.py
------------------------
Module defining the InitializeSimulation class used to set up
the molecular dynamics simulation environment for nucleation studies.

This module:
- Nondimensionalizes input parameters
- Defines the simulation box
- Populates it with atoms (randomly or from given positions)
- Builds neighbor lists for efficient force computation
- Precomputes Lennard-Jones cross coefficients

Author: Xiwei Pan (Princeton University)
Course: APC 524 – Molecular Dynamics for Nucleation
Date: 2025-11-08
"""

import pytest
import numpy as np
from pint import UnitRegistry
from MDAnalysis.analysis import distances


class InitializeSimulation:
    """Class responsible for setting up the molecular simulation box and atom configuration."""

    def __init__(
        self,
        ureg,
        number_atoms,
        epsilon,
        sigma,
        atom_mass,
        box_dimensions,
        cut_off,
        initial_positions=None,
        neighbor=1,
    ):
        """
        Initialize the basic MD simulation setup.

        Parameters
        ----------
        ureg : pint.UnitRegistry
            Unit registry for handling unit conversions.
        number_atoms : list[int]
            Number of atoms for each species, e.g., [N1, N2, ...].
        epsilon : list[pint.Quantity]
            Lennard-Jones epsilon values (energy depth), representing the strength of the
            intermolecular interaction.
        sigma : list[pint.Quantity]
            Lennard-Jones sigma values (size parameter).
        atom_mass : list[pint.Quantity]
            Mass of each atom type.
        box_dimensions : list[pint.Quantity]
            Dimensions of the simulation box along three axes [Lx, Ly, Lz].
        cut_off : pint.Quantity
            Cutoff radius for interaction calculations (distance within which the LJ potential is computed).
        initial_positions : np.ndarray, optional
            Predefined atom positions `total_atoms x 3`. If None, positions are randomly assigned.
        neighbor : int, optional
            Steps between neighbor list updates. Default = 1 (update every simulation step).
        """
        self.ureg = ureg
        self.number_atoms = np.array(number_atoms)
        self.epsilon = np.array([eps.to_base_units().magnitude for eps in epsilon])  # Convert to SI base units
        self.sigma = np.array([sig.to_base_units().magnitude for sig in sigma])
        self.atom_mass = np.array([m.to_base_units().magnitude for m in atom_mass])

        # Core simulation parameters
        self.box_dimensions = np.array([L.to_base_units().magnitude for L in box_dimensions])
        self.cut_off = cut_off.to_base_units().magnitude
        self.initial_positions = initial_positions
        self.neighbor = neighbor
        self.step = 0  # Initialize simulation step counter
        
        # For each atom type, repeat its sigma and epsilon values according to how many atoms of that type exist
        self.atoms_sigma = np.repeat(self.sigma, self.number_atoms)
        self.atoms_epsilon = np.repeat(self.epsilon, self.number_atoms)
        self.atoms_mass = np.repeat(self.atom_mass, self.number_atoms)

        # Run setup sequence for initialization
        self._define_box()
        self._populate_box()
        self._update_neighbor_lists()
        self._update_cross_coefficients()


    # ----------------------------------------------------------------------
    # Private setup methods
    # ----------------------------------------------------------------------
    def _define_box(self):
        """
        Define the cubic simulation box centered at origin.
        
        The method constructs a 3D cubic simulation box using the
        dimensions provided in `self.box_dimensions`. The box is
        symmetric about the origin and stored as lower and upper
        boundaries for each Cartesian direction. It also defines
        the box size in MDAnalysis convention: ``[Lx, Ly, Lz, α, β, γ]``,
        where the last three are the box angles in degrees.
        
        Parameters
        ----------
        None
            This method does not take any arguments. It uses the instance
            attribute `self.box_dimensions` set during initialization.
        
        Returns
        -------
        None
            The function updates two attributes of the class:
            - `self.box_boundaries` : ndarray of shape (3, 2)
                Lower and upper limits for x, y, z axes.
            - `self.box_size` : ndarray of shape (6,)
                [Lx, Ly, Lz, 90, 90, 90], following `MDAnalysis` format.
        
        Raises
        ------
        ValueError
            If `box_dimensions` is not a sequence of three positive numbers.
        
        Examples
        --------
        >>> from pint import UnitRegistry
        >>> import numpy as np
        >>> ureg = UnitRegistry()
        >>> eps = [0.2 * ureg.kcal / ureg.mol, 0.4 * ureg.kcal / ureg.mol]
        >>> sig = [3.0 * ureg.angstrom, 4.0 * ureg.angstrom]
        >>> masses = [10.0 * ureg.gram / ureg.mol, 20.0 * ureg.gram / ureg.mol]
        >>> L = 20.0 * ureg.angstrom
        >>> rc = 2.5 * sig[0]
        >>> sim = InitializeSimulation(
        ...     ureg=ureg,
        ...     number_atoms=[2, 3],
        ...     epsilon=eps,
        ...     sigma=sig,
        ...     atom_mass=masses,
        ...     box_dimensions=[L, L, L],
        ...     cut_off=rc,
        ... )
        >>> sim.box_boundaries
        array([[-1.0e-09,  1.0e-09],
            [-1.0e-09,  1.0e-09],
            [-1.0e-09,  1.0e-09]])  # in meters (after unit conversion)
        >>> sim.box_size
        array([2.0e-09, 2.0e-09, 2.0e-09, 90., 90., 90.])
        
        """
        if len(self.box_dimensions) != 3 or np.any(np.array(self.box_dimensions) <= 0):
            raise ValueError("`box_dimensions` must contain three positive values.")
        
        box_boundaries = np.zeros((3, 2))
        for i, L in enumerate(self.box_dimensions):
            box_boundaries[i] = [-L / 2, L / 2]

        self.box_boundaries = box_boundaries
        lengths = np.diff(box_boundaries, axis=1).flatten()
        self.box_size = np.concatenate([lengths, np.array([90.0, 90.0, 90.0])])


    def _populate_box(self):
        """
        Place atoms in the defined simulation box.
        
        This internal method assigns initial coordinates to all atoms
        in the system. If no predefined coordinates are provided
        (``self.initial_positions`` is None), atoms are placed randomly
        within the cubic simulation box defined by ``self.box_boundaries``.
        Otherwise, the provided array of positions is used directly.
        
        Parameters
        ----------
        None
            This method does not take any arguments. It uses the instance
            attributes `self.number_atoms`, `self.box_boundaries`, and
            `self.initial_positions` set during initialization.
        
        Returns
        -------
        None
            Updates the following instance attribute:

            atoms_positions : (N, 3) ndarray
                Cartesian coordinates of all atoms in the system.
                Randomly generated if no initial positions are supplied.
                Units are SI magnitudes after conversion in ``__init__``.
        
        Raises
        ------
        ValueError
            If `initial_positions` is provided but its shape does not match
            the expected ``(total_atoms, 3)``.
        
        Examples
        --------
        >>> from pint import UnitRegistry
        >>> import numpy as np
        >>> ureg = UnitRegistry()
        >>> eps = [0.2 * ureg.kcal / ureg.mol, 0.4 * ureg.kcal / ureg.mol]
        >>> sig = [3.0 * ureg.angstrom, 4.0 * ureg.angstrom]
        >>> masses = [10.0 * ureg.gram / ureg.mol, 20.0 * ureg.gram / ureg.mol]
        >>> L = 20.0 * ureg.angstrom
        >>> rc = 2.5 * sig[0]
        >>> sim = InitializeSimulation(
        ...     ureg=ureg,
        ...     number_atoms=[2, 3],
        ...     epsilon=eps,
        ...     sigma=sig,
        ...     atom_mass=masses,
        ...     box_dimensions=[L, L, L],
        ...     cut_off=rc,
        ... )
        >>> sim.atoms_positions.shape
        (5, 3)
        >>> np.all((sim.atoms_positions[:, 0] >= sim.box_boundaries[0, 0]) &
        ...         (sim.atoms_positions[:, 0] <= sim.box_boundaries[0, 1]))
        True
        
        """
        total_atoms = int(np.sum(self.number_atoms))

        if self.initial_positions is None:
            positions = np.zeros((total_atoms, 3))
            for dim in range(3):
                L_dim = np.diff(self.box_boundaries[dim])[0]
                positions[:, dim] = np.random.random(total_atoms) * L_dim - L_dim / 2
        else:
            positions = np.array(self.initial_positions, dtype=float)
            if positions.shape != (total_atoms, 3):
                raise ValueError(
                    f"Initial positions shape {positions.shape} does not match expected {(total_atoms, 3)}."
                )
        self.atoms_positions = positions


    def _update_neighbor_lists(self, force_update=False):
        """
        Generate neighbor lists for each atom within cutoff radius.
        
        This method identifies, for each atom, the set of neighboring atoms
        located within the Lennard-Jones cutoff radius `cut_off`. The neighbor
        list is used to limit force and energy calculations to nearby pairs,
        improving computational efficiency. The lists are updated every
        `self.neighbor` steps or when `force_update` is True.
        
        Parameters
        ----------
        force_update : bool, optional
            If True, rebuild the neighbor lists regardless of the current
            simulation step. Default is False.
        
        Returns
        -------
        None
            Updates the following instance attribute:

            neighbor_lists : list[list[int]]
                A list of length `total_atoms - 1`. Each element contains the indices
                of atoms that are within the cutoff radius of atom *i*. Only neighbors
                with index `j > i` are stored to avoid redundant pairs.
        
        Notes
        -----
        - The neighbor search uses `MDAnalysis.analysis.distances.contact_matrix`,
          which efficiently computes a Boolean contact matrix given atom positions and
          a cutoff distance.
        - Periodic boundary conditions are applied using the box information
          stored in `self.box_size`
        
        Examples
        --------
        >>> from pint import UnitRegistry
        >>> import numpy as np
        >>> ureg = UnitRegistry()
        >>> eps = [0.2 * ureg.kcal / ureg.mol, 0.4 * ureg.kcal / ureg.mol]
        >>> sig = [3.0 * ureg.angstrom, 4.0 * ureg.angstrom]
        >>> masses = [10.0 * ureg.gram / ureg.mol, 20.0 * ureg.gram / ureg.mol]
        >>> L = 20.0 * ureg.angstrom
        >>> rc = 2.5 * sig[0]
        >>> sim = InitializeSimulation(
        ...     ureg=ureg,
        ...     number_atoms=[2, 3],
        ...     epsilon=eps,
        ...     sigma=sig,
        ...     atom_mass=masses,
        ...     box_dimensions=[L, L, L],
        ...     cut_off=rc,
        ... )
        >>> len(sim.neighbor_lists)
        4
        >>> isinstance(sim.neighbor_lists[0], list)
        True
        
        """
        # Only update neighbor lists at specified intervals or if forced, preventing unnecessary recalculations
        if (self.step % self.neighbor == 0) or force_update:
            matrix = distances.contact_matrix(
                self.atoms_positions,
                cutoff=self.cut_off,
                returntype="numpy",
                box=self.box_size,
            )

            neighbor_lists = []
            for i, row in enumerate(matrix[:-1]):
                neighbors = np.where(row)[0]  # Neighbors becomes an array of all atom indices j that are close enough to atom i
                neighbors = [j for j in neighbors if j > i]
                neighbor_lists.append(neighbors)

            self.neighbor_lists = neighbor_lists


    def _update_cross_coefficients(self, force_update=False):
        """
        Compute Lennard-Jones cross-interaction coefficients for efficiency.
        
        This internal method precomputes the Lennard-Jones (LJ) cross parameters between
        atom pairs listed in the neighbor list. These coefficients are used later in the
        force and potential energy calculations to avoid repeatedly computing them during
        each simulation step. Here, the arithmetic mean mixing rules are applied:
        - ``σ_ij = (σ_i + σ_j) / 2``
        - ``ε_ij = (ε_i + ε_j) / 2``
        
        Parameters
        ----------
        force_update : bool, optional
            If True, recompute cross coefficients regardless of the current
            simulation step. Default is False.
        
        Returns
        -------
        None
            Updates the following instance attributes:

            sigma_ij_list : list[float]
                List of pairwise mixed LJ size parameters `σ_{ij}`.
            epsilon_ij_list : list[float]
                List of pairwise mixed LJ energy parameters `ε_{ij}`.
        
        Raises
        ------
        AttributeError
            If `self.neighbor_lists`, `self.sigma`, or `self.epsilon`
            are not defined before calling this method.
        ValueError
            If the number of atoms or neighbor list structure is inconsistent,
            or if any of the `σ` or `ε` values are non-positive.
        
        Notes
        -----
        - The values of the cross coefficients between atom of type 1 and 2 are
          assumed to follow the arithmetic mean.
        - This operation scales with the number of atom pairs found in
          `self.neighbor_lists`.
        - Precomputing these values improves the efficiency of the subsequent
          force evaluations.
        """
        # Check that required attributes exist
        if not hasattr(self, "neighbor_lists") or self.neighbor_lists is None:
            raise AttributeError("Neighbor lists are not defined. Run `_update_neighbor_lists()` first.")
        if not hasattr(self, "sigma") or not hasattr(self, "epsilon"):
            raise AttributeError("LJ parameters (sigma, epsilon) must be defined before computing cross coefficients.")

        # Validate sigma and epsilon arrays
        if np.any(np.array(self.sigma) <= 0) or np.any(np.array(self.epsilon) <= 0):
            raise ValueError("All sigma and epsilon values must be positive.")
        
        if (self.step % self.neighbor == 0) or force_update:
            sigma_ij_list, epsilon_ij_list = [], []
            total_atoms = int(np.sum(self.number_atoms))
            
            for i in range(total_atoms - 1):
                sigma_i = self.atoms_sigma[i]
                epsilon_i = self.atoms_epsilon[i]
                neighbors = self.neighbor_lists[i]
                
                sigma_j = self.atoms_sigma[neighbors]
                epsilon_j = self.atoms_epsilon[neighbors]
                
                sigma_ij_list.append((sigma_i + sigma_j) / 2)
                epsilon_ij_list.append((epsilon_i + epsilon_j) / 2)
            
            self.sigma_ij_list = sigma_ij_list
            self.epsilon_ij_list = epsilon_ij_list
    
    
    def validate_positions(self):
        """
        Ensure all atoms are located within the defined box boundaries.
        
        This diagnostic utility checks whether every atom's Cartesian coordinates
        fall within the box limits defined in ``self.box_boundaries``.
        It raises a ValueError if any atom lies outside the defined domain.
        
        Parameters
        ----------
        None
            The method does not take external arguments. It operates on the
            instance attributes:
            - `self.atoms_positions` : (N, 3) ndarray  
              Cartesian coordinates of all atoms in the system.
            - `self.box_boundaries` : (3, 2) ndarray  
              Lower and upper boundaries along x, y, and z directions.

        Returns
        -------
        bool
            Returns True if all atoms are inside the defined box boundaries.
        
        Raises
        ------
        ValueError
            If any atom coordinate is outside its respective lower or upper bound.
            The error message includes the position of the offending atom.
        
        """
        for pos in self.atoms_positions:
            for coord, bounds in zip(pos, self.box_boundaries):
                if not (bounds[0] <= coord <= bounds[1]):
                    raise ValueError(f"Atom at {pos} outside box boundaries.")
        return True


# ----------------------------------------------------------------------
# Example usage / pytest-style validation
# ----------------------------------------------------------------------
ureg = UnitRegistry()
L = 20 * ureg.angstrom
sig_1, sig_2 = [3, 4] * ureg.angstrom
eps_1, eps_2 = [0.2, 0.4] * ureg.kcal / ureg.mol
mss_1, mss_2 = [10, 20] * ureg.gram / ureg.mol
rc = 2.5 * sig_1

init = InitializeSimulation(
    ureg=ureg,
    number_atoms=[2, 3],
    epsilon=[eps_1, eps_2],
    sigma=[sig_1, sig_2],
    atom_mass=[mss_1, mss_2],
    box_dimensions=[L, L, L],
    cut_off=rc,
)

def test_box():
        assert init.validate_positions()
        print("All atoms within box boundaries.")

if __name__ == "__main__":
    pytest.main(["-s", __file__])
