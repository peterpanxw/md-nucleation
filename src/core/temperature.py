"""
temperature.py
---------------
Thermostat control module for the MD simulation package.
All temperature-related functionalities are grouped here.

This module provides:
    - instantaneous temperature measurement
    - velocity rescaling thermostat
    - Maxwell–Boltzmann velocity randomization
    - Berendsen thermostat
    - Andersen thermostat
    - Nose–Hoover thermostat (single chain)

The design assumes compatibility with the InitializeSimulation class
(see 'Initialize the simulation' PDF) and the MinimizeEnergy class
(see 'Minimize the energy' PDF).
"""

from __future__ import annotations
import numpy as np


# ============================================================
# Utility functions
# ============================================================
def _compute_kinetic_energy(velocities, masses):
    """
    Compute total kinetic energy of the system.
    
    The kinetic energy is evaluated as
    
        `KE = 1/2 \sum_{i=1}^N m_i |v_i|^2`

    where m_i is the mass of atom i and v_i is its velocity vector.
    Inputs may be nondimensional or SI; the returned energy is in the
    same unit system as the provided masses and velocities.
    
    Parameters
    ----------
    velocities : ndarray of shape (N, 3)
        Cartesian velocity components of all atoms.
    masses : ndarray of shape (N,)
        Mass of each atom.

    Returns
    -------
    float
        Total kinetic energy of the system.
    """
    return 0.5 * np.sum(masses[:, None] * velocities**2)


def _maxwell_boltzmann_random(temperature, masses, rng):
    """
    Draw atom velocities from the Maxwell–Boltzmann distribution.
    
    For each atom i, the velocity components are sampled independently from
    a normal (Gaussian) distribution:
    
        `v_i ~ Normal(0, sqrt(kT/m_i))`
    
    where T is the target temperature and m_i is the mass of atom i.
    In nondimensionalized units, k = 1. In SI units, the user should ensure
    that temperature and masses are consistent with the nondimensionalization
    used for the simulation.
    
    Parameters
    ----------
    temperature : float
        Target temperature. Interpreted either as a nondimensional temperature
        (if using reduced units) or as SI temperature depending on the units
        used in the simulation.
    masses : ndarray of shape (N,)
        Mass of each atom. Must match the unit convention of `temperature`.
    rng : np.random.Generator
        NumPy random number generator used to draw Gaussian samples.
    
    Returns
    -------
    velocities : ndarray of shape (N, 3)
        Random initial velocities for all atoms, where each component is drawn
        from a Maxwell–Boltzmann distribution consistent with the specified
        temperature.
    
    Notes
    -----
    - The generated velocities have zero mean but will *not* have exactly zero
      total momentum; many MD simulations subsequently subtract the center-of-mass
      velocity to enforce zero net momentum.
    - This routine is commonly used before equilibration or thermostatting.
    """
    sigma = np.sqrt(temperature / masses)[:, None]  # Standard deviation of velocities in the Maxwell–Boltzmann distribution, shape (N,1)
    return rng.normal(loc=0.0, scale=sigma, size=(len(masses), 3))


# ============================================================
# Main thermostat class
# ============================================================
class Thermostat:
    """
    Thermostat controller for MD simulation.
    
    The thermostat is not a standalone system, but an attached control module
    that operates directly on the main simulation through `self.sim`. Later,
    `Thermostat` can access all system variables through `self.sim.atoms_velocity`,
    `self.sim.atom_mass`, `self.sim.box_boundaries`, etc.
    
    Assumes these attributes exist on the simulation object:
    
    - atoms_velocity : (N, 3) array
        Per-atom Cartesian velocities.
    - atoms_mass : (N,) array
        Mass of each atom (consistent with simulation units).
    - ureg : pint.UnitRegistry, optional
        Unit registry used for physical-unit support.
    - compute_temperature() : callable
        Method returning the instantaneous kinetic temperature.
    - _compute_kinetic_energy() : callable
        Method computing system kinetic energy.
    """
    
    def __init__(self, sim, target_temperature):
        """
        Parameters
        ----------
        sim : object
            Simulation object (`InitializeSimulation`, `MinimizeEnergy`, etc.).
        target_temperature : float
            Target temperature. Interpreted in nondimensional units unless
            otherwise specified by the simulation setup.
        
        Raises
        ------
        AttributeError
            If the provided `sim` object does not contain an `atoms_velocity`
            attribute. Velocities must be initialized before using the thermostat.
        """
        self.sim = sim
        self.target_temperature = float(target_temperature)
        self.rng = np.random.default_rng()  # NumPy random number generator used for stochastic thermostatting (e.g., Maxwell–Boltzmann re-sampling).
        
        if not hasattr(sim, "atoms_velocity"):
            raise AttributeError(
                "Simulation object must have 'atoms_velocity' attribute. "
                "Initialize them before using the thermostat."
            )
    
    
    # -------------------------------------------------------
    # Temperature measurement
    # -------------------------------------------------------
    def instantaneous_temperature(self):
        """
        Compute instantaneous system temperature from kinetic energy.
        
        The temperature is estimated from the equipartition theorem:

            `T = (2 * KE) / (3 N k_B)`

        where KE is the total kinetic energy and N is the number of atoms.
        In nondimensional simulation units, the Boltzmann constant is taken
        to be k_B = 1, giving

            `T = (2 * KE) / (3 N)`
        
        Parameters
        ----------
        None
            This method operates directly on the simulation object stored in
            `self.sim` and does not accept external arguments.
        
        Returns
        -------
        float
            Instantaneous temperature of the system in nondimensional units.
        
        Notes
        -----
        - This formula assumes three translational degrees of freedom per atom.
        - The temperature estimate fluctuates due to finite system size and
          is meaningful only when averaged over time or ensemble samples.
        """
        vel = self.sim.atoms_velocity
        masses = self.sim.atom_mass
        N = len(masses)
        
        KE = _compute_kinetic_energy(vel, masses)
        T = (2.0 * KE) / (3.0 * N)
        return float(T)
    
    
    # -------------------------------------------------------
    # Simple rescaling thermostat
    # -------------------------------------------------------
    def velocity_rescale(self):
        """
        Rescale all atomic velocities so that the system temperature matches
        the target temperature exactly.
        
        The rescaling factor is computed from the instantaneous temperature:

            `v_new ← v_old * sqrt(T_target / T_current)`

        This operation enforces the desired kinetic temperature in a single step
        and is commonly referred to as *velocity rescaling*. It does not conserve
        energy and should be used with care (typically during system initialization
        or simple thermostats).
        
        Parameters
        ----------
        None
            Operates directly on `self.sim.atoms_velocity` and uses the
            thermostat's stored `target_temperature`.

        Returns
        -------
        None
            Velocities in `self.sim.atoms_velocity` are modified in-place.
        
        Raises
        ------
        ValueError
            If the instantaneous temperature is zero or negative, in which case
            rescaling is undefined.
        """
        current_T = self.instantaneous_temperature()
        if current_T <= 0:
            raise ValueError("Current temperature is zero or negative.")
        
        scale = np.sqrt(self.target_temperature / current_T)  # Rescaling factor of velocity
        self.sim.atoms_velocity *= scale
    
    
    # -------------------------------------------------------
    # Maxwell–Boltzmann velocity randomization
    # -------------------------------------------------------
    def randomize_velocities(self):
        """
        Assign new atomic velocities drawn from the Maxwell–Boltzmann distribution.
        
        This method replaces all atomic velocities in the simulation object with
        random samples consistent with the target temperature of the thermostat.
        Each velocity component is drawn independently from a normal distribution
        
            `v_i ~ Normal(0, sqrt(kT / m_i))`,
        
        where m_i is the mass of atom i. In nondimensional units, the Boltzmann
        constant k_B = 1.
        
        This procedure is typically performed at the beginning of NVT/NPT
        simulations or whenever a full re-thermalization of the system is desired.

        Parameters
        ----------
        None
            Operates directly on `self.sim.atoms_velocity` and does not require
            external arguments.

        Returns
        -------
        None
            Velocities stored in `self.sim.atoms_velocity` are overwritten
            in-place with newly generated Maxwell–Boltzmann velocities.
        """
        masses = self.sim.atom_mass
        self.sim.atoms_velocity = _maxwell_boltzmann_random(
            self.target_temperature, masses, rng=self.rng
        )
    
    
    # -------------------------------------------------------
    # Andersen thermostat
    # -------------------------------------------------------
    def andersen(self, collision_frequency, dt):
        """
        Apply the Andersen thermostat to randomly reassign particle velocities.
        
        In the Andersen thermostat, each atom undergoes a stochastic “collision”
        with a fictitious heat bath. During each timestep, atom i has a probability

            `p = collision_frequency * dt`

        of having its velocity reassigned. When a collision occurs, the new velocity
        is drawn from the Maxwell–Boltzmann distribution corresponding to the
        target temperature.
        
        Parameters
        ----------
        collision_frequency : float
            Collision rate (ν) controlling how often atoms collide with the
            heat bath. Larger values correspond to stronger stochastic coupling.
        dt : float
            Integration timestep. The collision probability is p = ν dt.

        Returns
        -------
        None
            Velocities of selected atoms are modified in-place via
            `self.sim.atoms_velocity`.
        
        Notes
        -----
        - The Andersen thermostat generates canonical (NVT) ensemble statistics,
          but destroys momentum conservation due to stochastic velocity resets.
        - New velocities are sampled using the helper function
          `_maxwell_boltzmann_random`.
        """
        N = len(self.sim.atom_mass)
        p = collision_frequency * dt  # Atom collision probability
        
        mask = self.rng.random(N) < p
        if np.any(mask):
            masses = self.sim.atom_mass
            new_vels = _maxwell_boltzmann_random(
                self.target_temperature, masses, rng=self.rng
            )
            self.sim.atoms_velocity[mask] = new_vels[mask]  # Draw velocities from MB distribution for collided atoms
    
    
    # -------------------------------------------------------
    # Berendsen thermostat
    # -------------------------------------------------------
    def berendsen(self, dt, tau):
        """
        Berendsen weak-coupling thermostat.
        
        The Berendsen thermostat rescales velocities to relax the system
        temperature toward a target temperature `T_target` according to

            `dT/dt = (T_target - T) / tau`

        which yields the velocity scaling factor

            `λ = sqrt(1 + (dt / tau) * (T_target / T - 1))`

        where `T` is the instantaneous temperature. Velocities are updated as

            `v_i ← λ v_i`
        
        Parameters
        ----------
        dt : float
            Integration timestep of the MD integrator.
        tau : float
            Thermostat relaxation timescale. Smaller values enforce faster
            temperature coupling; larger values lead to weaker coupling.
            Must be positive.

        Returns
        -------
        None
            The velocities stored in `self.sim.atoms_velocity` are updated
            in-place.
        
        Raises
        ------
        ValueError
            If `tau` is nonpositive.
        
        Notes
        -----
        - The Berendsen thermostat does *not* generate the correct canonical
          (NVT) ensemble but is widely used for equilibration because it
          smoothly drives the system toward the target temperature.
        - When the instantaneous temperature `T` is extremely small or zero,
          velocity scaling is skipped to avoid numerical overflow.
        """
        if tau <= 0:
            raise ValueError("Relaxation time tau must be positive.")
        
        T = self.instantaneous_temperature()
        if T <= 0:
            return
        
        lambda_factor = np.sqrt(1.0 + (dt / tau) * (self.target_temperature / T - 1.0))  # Velocity scaling factor
        self.sim.atoms_velocity *= lambda_factor
    
    
    # -------------------------------------------------------
    # Nose–Hoover thermostat (single chain)
    # -------------------------------------------------------
    def nose_hoover(self, dt, xi, Q):
        """
        Nose–Hoover thermostat update.
        
        This method integrates the Nose–Hoover extended equations of motion
        for the thermostat friction coefficient `xi` and rescales particle
        velocities accordingly. The Nose–Hoover thermostat generates
        canonical (NVT) ensemble dynamics by coupling the physical system
        to a fictitious heat bath with mass `Q`:

            `dxi/dt = (T / T_target - 1) / Q`
            `dv/dt  = -xi * v`

        where `T` is the instantaneous kinetic temperature of the system.
        
        Parameters
        ----------
        dt : float
            Integration timestep.
        xi : float
            Current Nose–Hoover friction coefficient. It will be updated and
            returned by the method.
        Q : float
            Thermostat mass parameter. Controls the coupling strength between
            the thermostat and the physical system: large `Q` → weak coupling,
            small `Q` → strong coupling.
        
        Returns
        -------
        float
            Updated Nose–Hoover friction coefficient `xi`.
        
        Notes
        -----
        - The update uses a simple first-order (Euler) integration of the
          extended variable `xi`.
        - Velocities are scaled as `v ← v * exp(-xi_new * dt)`, which is
          the exact integration of the linear friction equation.
        - This method updates only thermostat-related quantities; integration
          of atomic positions must occur separately in the MD integrator.
        """
        T = self.instantaneous_temperature()
        
        # Update xi
        dxi_dt = (T / self.target_temperature - 1.0) / Q
        xi_new = xi + dt * dxi_dt
        
        # Apply friction to velocities, dv/dt = -xi v
        self.sim.atoms_velocity *= np.exp(-xi_new * dt)
        
        return xi_new
