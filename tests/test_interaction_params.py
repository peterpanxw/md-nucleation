import pytest
from md_nucleation.core.interaction_params import (
    validate_type_params,
    lorentz_berthelot_mixing,
    build_lj_interaction_table,
)


# --- Test Validate Type Parameters --- #


def test_validate_type_params_valid():
    type_params = {
        "A": {"sigma": 1.0, "epsilon": 1.0},
        "B": {"sigma": 0.9, "epsilon": 0.5},
    }
    assert validate_type_params(type_params) is True


def test_validate_type_params_missing_sigma():
    type_params = {"A": {"epsilon": 1.0}}
    with pytest.raises(ValueError) as excinfo:
        validate_type_params(type_params)
    assert "Missing sigma" in str(excinfo.value)


def test_validate_type_params_missing_epsilon():
    type_params = {"A": {"sigma": 1.0}}
    with pytest.raises(ValueError) as excinfo:
        validate_type_params(type_params)
    assert "Missing epsilon" in str(excinfo.value)


def test_validate_type_params_non_numeric():
    type_params = {"A": {"sigma": "x", "epsilon": 1.0}}
    with pytest.raises(ValueError):
        validate_type_params(type_params)


# --- Test Lorentz-Berthelot Mixing --- #


def test_mixing_rules():
    sigma_ij, epsilon_ij = lorentz_berthelot_mixing(1.0, 0.5, 1.0, 0.25)

    assert sigma_ij == 0.75
    assert pytest.approx(epsilon_ij) == (1.0 * 0.25) ** 0.5


# --- Test Building LJ Interaction Table --- #


def test_build_lj_interaction_table():
    type_params = {
        "A": {"sigma": 1.0, "epsilon": 1.0},
        "B": {"sigma": 0.9, "epsilon": 0.5},
    }

    table = build_lj_interaction_table(type_params)

    # Check symmetric pairs
    assert ("A", "B") in table
    assert ("B", "A") in table

    # Check self pairs
    assert table[("A", "A")]["sigma"] == 1.0
    assert table[("B", "B")]["epsilon"] == 0.5

    # Check mixed interaction
    mixed = table[("A", "B")]
    assert mixed["sigma"] == pytest.approx(0.95)  # (1.0 + 0.9)/2
    assert mixed["epsilon"] == pytest.approx((1.0 * 0.5) ** 0.5)
