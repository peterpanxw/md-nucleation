import pytest
from md_nucleation.core.system_builder import (
    build_system_from_input,
    Particle,
    CubicBox,
    System
)


# --- Test System Builder: Valid Construction --- #

def test_build_system_valid():
    params = {
        "n": "2",
        "l": "10.0",
        "s": "100",
        "t": "300",
        "p": "1.0",
        "positions": [
            "A 0.0 0.0 0.0",
            "B 1.0 1.0 1.0"
        ]
    }

    mass_table = {"A": 2.0, "B": 3.0}

    system = build_system_from_input(params, mass_table)

    assert isinstance(system, System)
    assert len(system.particles) == 2

    p1, p2 = system.particles

    assert p1.atom_type == "A"
    assert p1.mass == 2.0
    assert p1.position == [0.0, 0.0, 0.0]
    assert p1.velocity == [0.0, 0.0, 0.0]

    assert p2.atom_type == "B"
    assert p2.mass == 3.0
    assert p2.position == [1.0, 1.0, 1.0]


# --- Test System Builder: Default Mass --- #

def test_build_system_default_mass():
    params = {
        "n": "1",
        "l": "10.0",
        "s": "10",
        "t": "300",
        "p": "1.0",
        "positions": ["A 1.0 2.0 3.0"],
    }

    system = build_system_from_input(params)

    assert system.particles[0].mass == 1.0  # default


# --- Test System Builder: Particle Count Mismatch --- #

def test_build_system_count_mismatch():
    params = {
        "n": "2",  # says 2 atoms
        "l": "10.0",
        "s": "100",
        "t": "300",
        "p": "1.0",
        "positions": ["A 0.0 0.0 0.0"],  # but only 1 provided
    }

    with pytest.raises(ValueError) as excinfo:
        build_system_from_input(params)

    assert "Expected 2 particles" in str(excinfo.value)