"""
integrator.py
-------------
Velocity–Verlet integrator for Molecular Dynamics simulation.

This module provides:
- Velocity–Verlet integrator
- Position & velocity updates
- Support for thermostats (optional)
- Periodic boundary wrapping
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
from initialization import InitializeSimulation
from potentials import EnergyComputationsMixin
from temperature import Thermostat, _compute_kinetic_energy
from pressure import PressureCalculator


# ============================================================
# Helper: remove center-of-mass (COM) drift
# ============================================================
def _remove_drift(velocities, masses):
    """
    Remove center-of-mass velocity (optional, but often useful).
    
    This operation enforces zero total linear momentum by subtracting the
    center-of-mass velocity from every particle. Although optional in MD,
    removing momentum drift is helpful for long simulations because 
    numerical integration errors can slowly accumulate and produce an
    unphysical translation of the entire system.
    
    Parameters
    ----------
    velocities : np.ndarray
        Array of shape (N, 3) containing the velocity vectors of all N particles.
    masses : np.ndarray
        Array of shape (N,) containing the mass of each particle.
    
    Returns
    -------
    None
        The function modifies `velocities` in place.
    """
    total_mass = np.sum(masses)
    v_cm = np.sum(masses[:, None] * velocities, axis=0) / total_mass  # Velocity of COM, (total momentum / total mass)
    velocities -= v_cm


# ============================================================
# Main integrator
# ============================================================
@dataclass
class VelocityVerletIntegrator(EnergyComputationsMixin, InitializeSimulation):
    """
    Velocity–Verlet integrator for MD simulations.
    
    Inherits:
    - InitializeSimulation (system setup, neighbor lists, etc.)
    - EnergyComputationsMixin (LJ potential & forces)
    
    Parameters (core)
    -----------------
    dt : float
        Time step (nondimensional).
    n_steps : int
        Total number of time steps.
    ensemble : {"NVE", "NVT"}
        Target ensemble; "NVT" activates thermostat coupling.
    thermostat_type : {"berendsen", "andersen", "nose-hoover", None}
        Thermostat algorithm to use when ensemble == "NVT".
    target_temperature : float, optional
        Required when ensemble == "NVT" and no external Thermostat is passed.
    tau_t : float, optional
        Relaxation time for Berendsen thermostat.

    All remaining *args and **kwargs are passed to InitializeSimulation.
    """
    
    dt: float = 1e-3
    n_steps: int = 1000
    ensemble: Literal["NVE", "NVT"] = "NVE"
    thermostat_type: Optional[Literal["berendsen", "andersen", "nose-hoover"]] = None
    target_temperature: Optional[float] = None
    tau_t: float = 0.1  # default thermostat time constant (for Berendsen)
    andersen_collision_freq: float = 0.1  # default Andersen frequency
    nose_hoover_Q: float = 10.0  # default Nose–Hoover mass
    
    # Internal variables
    _thermostat: Optional[Thermostat] = field(init=False, default=None)
    _forces: Optional[np.ndarray] = field(init=False, default=None)
    _xi: float = field(init=False, default=0.0)  # Nose–Hoover friction coefficient
    
    # Sampling and storing history
    sample_interval: int = 10
    time_history: List[float] = field(init=False, default_factory=list)
    temperature_history: List[float] = field(init=False, default_factory=list)
    kinetic_history: List[float] = field(init=False, default_factory=list)
    potential_history: List[float] = field(init=False, default_factory=list)
    total_energy_history: List[float] = field(init=False, default_factory=list)
    
    # Constructor (manual because of multiple inheritance)
    def __init__(self, dt: float, n_steps: int, ensemble: str = "NVE",
                 thermostat_type: Optional[str] = None,
                 target_temperature: Optional[float] = None,
                 tau_t: float = 0.1,
                 andersen_collision_freq: float = 0.1,
                 nose_hoover_Q: float = 10.0,
                 sample_interval: int = 10,
                 *args, **kwargs):
        
        # Save integrator parameters
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.ensemble = ensemble.upper()
        self.thermostat_type = thermostat_type
        self.target_temperature = target_temperature
        self.tau_t = float(tau_t)
        self.andersen_collision_freq = float(andersen_collision_freq)
        self.nose_hoover_Q = float(nose_hoover_Q)
        self.sample_interval = int(sample_interval)
        
        # Initialize mixin & simulation (positions, neighbor lists, etc.)
        InitializeSimulation.__init__(self, *args, **kwargs)
        
        # Initialize EnergyComputationsMixin (no state, but for clarity)
        EnergyComputationsMixin.__init__(self)
        
        # Initialize pressure calculation
        self.pressure_calculator = PressureCalculator(self)
        
        # Initialize neighbor lists & LJ cross coefficients
        self._update_neighbor_lists(force_update=True)
        self._update_cross_coefficients(force_update=True)
        
        # Initial force evaluation
        self._forces = self.compute_force(return_vector=True)
        
        # Create thermostat if needed
        self._thermostat = None
        if self.ensemble == "NVT":
            if self.target_temperature is None:
                raise ValueError(
                    "target_temperature must be provided for NVT ensemble."
                )
            self._thermostat = Thermostat(self, self.target_temperature)
        
        # Nose–Hoover friction coefficient
        self._xi = 0.0
        
        # Histories
        self.time_history = []
        self.temperature_history = []
        self.kinetic_history = []
        self.potential_history = []
        self.total_energy_history = []
        self.pressure_history = []
    
    
    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------
    def run(self):
        """
        Run the Velocity–Verlet integration for `n_steps` steps.
        
        This method advances the system for `n_steps` time steps using the
        velocity–Verlet algorithm, optionally applying a thermostat when
        the ensemble is NVT. Energies, temperature, and time are sampled
        periodically according to `sample_interval`.
        
        The integration loop performs:
        1. (Optional) removal of center-of-mass drift before dynamics begin.
        2. Initial sampling at time t = 0.
        3. Repeated velocity–Verlet updates of positions and velocities.
        4. Application of a thermostat (if enabled).
        5. Periodic storage of observables.
        
        After completion, time-dependent quantities are stored in:
            time_history
            temperature_history
            kinetic_history
            potential_history
            total_energy_history
        """
        # Optional: remove initial center-of-mass drift
        _remove_drift(self.atoms_velocity, self.atom_mass)
        
        # Initial sampling
        self._sample(0.0)
        
        for step in range(1, self.n_steps + 1):
            self._single_step()
            
            t = step * self.dt
            if step % self.sample_interval == 0:
                self._sample(t)
    
    
    # -------------------------------------------------------
    # Core VV step
    # -------------------------------------------------------
    def _single_step(self):
        """
        Perform one full Velocity–Verlet time integration step.
        
        This method advances positions and velocities according to the
        standard Velocity–Verlet algorithm:
        
            v(t+dt/2) = v(t) + (dt/2) * a(t)
            r(t+dt)   = r(t) + dt * v(t+dt/2)
            (rebuild neighbor lists, recompute forces)
            v(t+dt)   = v(t+dt/2) + (dt/2) * a(t+dt)
        
        After updating the atom velocities and positions, the method:
        - wraps particles into the simulation box,
        - rebuilds neighbor lists when needed,
        - recomputes Lennard–Jones forces,
        - applies a thermostat if the ensemble is NVT.
        
        Parameters
        ----------
        None
            All required data (positions, velocities, masses, forces, neighbor
            lists, thermostat state) are taken from the parent class instance.
        
        Returns
        -------
        None
            Updates `atoms_positions`, `atoms_velocity`, and `_forces` in place.
        """
        dt = self.dt
        half_dt = 0.5 * dt
        
        masses = self.atom_mass[:, None]  # shape (N,1)
        forces = self._forces             # shape (N,3)
        
        # --- First half-step for velocities ---
        self.atoms_velocity += half_dt * forces / masses
        
        # --- Full step for positions ---
        self.atoms_positions += dt * self.atoms_velocity
        self.wrap_in_box()
        
        # Update neighbor lists / cross coefficients as needed
        self._update_neighbor_lists()
        self._update_cross_coefficients()
        
        # --- Compute new forces at updated positions ---
        self._forces = self.compute_force(return_vector=True)
        
        # --- Second half-step for velocities ---
        self.atoms_velocity += half_dt * self._forces / masses
        
        # --- Apply thermostat if NVT ---
        if self.ensemble == "NVT" and self._thermostat is not None:
            self._apply_thermostat()  # Control tenperature through atom velocity
    
    
    # -------------------------------------------------------
    # Thermostat coupling
    # -------------------------------------------------------
    def _apply_thermostat(self):
        """
        Apply the chosen thermostat after the velocity update.
        
        This method is called once per integration step *after* the velocity
        update in the Velocity–Verlet scheme. It is active only when
        `ensemble="NVT"` and a thermostat has been constructed. The specific
        algorithm applied depends on `thermostat_type`:

        - "berendsen": exponential relaxation toward the target temperature.
        - "andersen": stochastic collisions with a heat bath.
        - "nose-hoover": deterministic extended-system temperature control.
        
        Parameters
        ----------
        None
            All thermostat parameters are taken from the class attributes.
        
        Returns
        -------
        None
            Velocities (and, for Nose–Hoover, the friction variable `self._xi`)
            are updated in place.
        """
        if self.thermostat_type is None:
            return
        
        tt = self.thermostat_type.lower()  # Convert to lowercase for easier string matching
        
        if tt == "berendsen":
            self._thermostat.berendsen(self.dt, tau=self.tau_t)
        
        elif tt == "andersen":
            self._thermostat.andersen(
                collision_frequency=self.andersen_collision_freq,
                dt=self.dt,
            )
        
        elif tt == "nose-hoover":
            self._xi = self._thermostat.nose_hoover(
                dt=self.dt,
                xi=self._xi,
                Q=self.nose_hoover_Q,
            )
        
        else:
            raise ValueError(f"Unknown thermostat_type: {self.thermostat_type}")
    
    
    # -------------------------------------------------------
    # Sampling utilities
    # -------------------------------------------------------
    def _sample(self, time_value: float):
        """
        Record instantaneous thermodynamic and energetic properties.
        
        This method evaluates kinetic energy, potential energy, total energy,
        and instantaneous temperature at the given simulation time and stores
        them in the corresponding history lists. It is typically called at
        fixed intervals during integration to enable trajectory diagnostics,
        energy conservation checks, and post-processing.

        Parameters
        ----------
        time_value : float
            Current simulation time at which the sample is taken.

        Returns
        -------
        None
            The function modifies internal history lists in place:
            ``time_history``, ``kinetic_history``, ``potential_history``,
            ``total_energy_history``, and ``temperature_history``.
        """
        # Kinetic energy
        KE = _compute_kinetic_energy(self.atoms_velocity, self.atom_mass)
        # Potential energy
        PE = self.compute_potential()
        # Total energy
        E_tot = KE + PE
        
        # Temperature (in reduced units: k_B = 1)
        N = len(self.atom_mass)
        T_inst = (2.0 * KE) / (3.0 * N)
        
        # System pressure
        P = self.pressure_calculator.instantaneous_pressure()
        
        # Writing history of time-dependent variables
        self.time_history.append(time_value)
        self.kinetic_history.append(KE)
        self.potential_history.append(PE)
        self.total_energy_history.append(E_tot)
        self.temperature_history.append(T_inst)
        self.pressure_history.append(P)
    
    
    # -------------------------------------------------------
    # Convenience getters
    # -------------------------------------------------------
    def get_trajectories(self):
        """
        Return a snapshot of the current atomic positions and velocities.
        
        This method is a convenience wrapper for extracting the system's
        instantaneous configuration during integration. Copies of the
        position and velocity arrays are returned to ensure that external
        modifications do not affect the internal simulation state.
        
        Parameters
        ----------
        self : VelocityVerletIntegrator
            The integrator instance containing the current particle
            positions (`atoms_positions`) and velocities (`atoms_velocity`).
        
        Returns
        -------
        positions : ndarray of shape (N, 3)
            Cartesian coordinates of all N particles at the current timestep.
            A copy is returned to avoid modifying the internal state.
        velocities : ndarray of shape (N, 3)
            Current velocities of the particles. Also returned as a copy.
        
        Notes
        -----
        The returned arrays are snapshots and do **not** remain linked
        to the simulation; later updates to the system will not alter
        previously returned data.
        """
        return self.atoms_positions.copy(), self.atoms_velocity.copy()
    
    
    def get_energy_time_series(self):
        """
        Retrieve the recorded thermodynamic time series from the simulation.
        
        This method returns all sampled quantities accumulated during the
        integration process, including time, instantaneous temperature,
        kinetic energy, potential energy, and total energy. The values are
        returned as NumPy arrays for convenient analysis, plotting, or
        post-processing.
        
        Returns
        -------
        dict of str -> np.ndarray
            A dictionary containing the following fields:

            - "time" : ndarray of shape (M,)
                Simulation times at sampling points.
            - "T" : ndarray of shape (M,)
                Instantaneous temperatures.
            - "KE" : ndarray of shape (M,)
                Kinetic energy history.
            - "PE" : ndarray of shape (M,)
                Potential energy history.
            - "E" : ndarray of shape (M,)
                Total energy (kinetic + potential) history.
            
            Here, `M` is the number of recorded sampling steps, controlled by
            `sample_interval`.
        """
        return {
            "time": np.asarray(self.time_history),
            "T": np.asarray(self.temperature_history),
            "KE": np.asarray(self.kinetic_history),
            "PE": np.asarray(self.potential_history),
            "E": np.asarray(self.total_energy_history),
            "P": np.asarray(self.pressure_history),
        }
    