from dataclasses import dataclass
from md_nucleation.io.validation import parse_typed_position
from md_nucleation.core.interaction_params import build_lj_interaction_table


@dataclass
class Particle:
    atom_type: str
    mass: float
    position: list
    velocity: list


@dataclass
class CubicBox:
    length: float


@dataclass
class System:
    particles: list
    box: CubicBox
    T: float
    S: int
    interaction_table: dict = None


def build_system_from_input(params, mass_table=None):
    """
    Construct the MD System object from validated input parameters.

    Parameters
    ----------
    params : dict
        Output of validate_parsed_input() after successful validation.

    mass_table : dict, optional
        Map atom types to masses, e.g. {"A": 1.0, "B": 2.0}.
        If not provided, all masses default to 1.0.

    Returns
    -------
    System
        Fully assembled MD system: particles, box, and simulation settings.
    """

    if mass_table is None:
        mass_table = {}

    # --- Parse scalar inputs --- #
    N = int(params["n"])
    L = float(params["l"])
    S = int(params["s"])
    T = float(params["t"])

    # --- Create box --- #
    box = CubicBox(length=L)

    # --- Build particles --- #
    particles = []
    for line in params["positions"]:
        atom_type, x, y, z = parse_typed_position(line)

        mass = mass_table.get(atom_type, 1.0)  # default mass = 1.0
        velocity = [0.0, 0.0, 0.0]  # initialize velocities to zero

        particles.append(
            Particle(
                atom_type=atom_type,
                mass=mass,
                position=[x, y, z],
                velocity=velocity,
            )
        )

    if len(particles) != N:
        raise ValueError(
            f"System builder error: Expected {N} particles but built {len(particles)}."
        )

    interaction_table = build_lj_interaction_table()

    return System(
        particles=particles, box=box, T=T, S=S, interaction_table=interaction_table
    )
