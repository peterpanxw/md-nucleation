import pytest
from md_nucleation.core.system_builder import build_system_from_input, System
from md_nucleation.core.default_params import UFF_LJ_PARAMS

# --- Test System Builder: Valid Construction --- #


def test_build_system_valid():
    params = {
        "n": "2",
        "l": "10.0",
        "s": "100",
        "t": "300",
        "p": "1.0",
        "positions": ["A 0.0 0.0 0.0", "B 1.0 1.0 1.0"],
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


# --- Test System Builder: LJ Interaction Table --- #


def test_system_has_lj_table():
    params = {
        "n": "2",
        "l": "10.0",
        "s": "100",
        "t": "300",
        "p": "1.0",
        "positions": ["H 0.0 0.0 0.0", "O 1.0 1.0 1.0"],
    }

    system = build_system_from_input(params)

    # LJ table must exist
    assert system.interaction_table is not None, "LJ interaction table is missing."

    # Check that H–O exists
    assert ("H", "O") in system.interaction_table

    # Verify defaults match UFF data
    sigma_expected = 0.5 * (UFF_LJ_PARAMS["H"]["sigma"] + UFF_LJ_PARAMS["O"]["sigma"])
    epsilon_expected = (
        UFF_LJ_PARAMS["H"]["epsilon"] * UFF_LJ_PARAMS["O"]["epsilon"]
    ) ** 0.5

    assert system.interaction_table[("H", "O")]["sigma"] == pytest.approx(
        sigma_expected
    )
    assert system.interaction_table[("H", "O")]["epsilon"] == pytest.approx(
        epsilon_expected
    )
