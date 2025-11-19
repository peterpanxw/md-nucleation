import pytest
from md_nucleation.core.default_params import UFF_LJ_PARAMS
from md_nucleation.core.interaction_params import build_lj_interaction_table


# --- Test Default UFF Parameters Are Loaded --- #

def test_default_params_exist():
    # Check we have at least the 22 elements defined
    assert len(UFF_LJ_PARAMS) == 22

    # Check some known keys
    for element in ["H", "C", "O", "Si", "Ti"]:
        assert element in UFF_LJ_PARAMS

    # Check sigma/epsilon are numeric and positive
    for element, params in UFF_LJ_PARAMS.items():
        assert isinstance(params["sigma"], (int, float))
        assert isinstance(params["epsilon"], (int, float))
        assert params["sigma"] > 0
        assert params["epsilon"] > 0


# --- Test Default Interaction Table Builds Correctly --- #

def test_default_interaction_table_builds():
    table = build_lj_interaction_table()  # no user params → UFF loads automatically

    # There should be 22 types, so 22 × 22 = 484 pairs
    assert len(table) == 22 * 22

    # Basic correctness for diagonal terms
    assert table[("H", "H")]["sigma"] == pytest.approx(UFF_LJ_PARAMS["H"]["sigma"])
    assert table[("C", "C")]["epsilon"] == pytest.approx(UFF_LJ_PARAMS["C"]["epsilon"])


# --- Test Symmetry of Mixing Rules --- #

def test_default_interaction_symmetry():
    table = build_lj_interaction_table()

    assert table[("H", "O")] == table[("O", "H")]
    assert table[("C", "N")] == table[("N", "C")]
    assert table[("Ti", "H")] == table[("H", "Ti")]


# --- Test Mixed Interaction Values Are Reasonable --- #

def test_default_mixed_interaction_values():
    table = build_lj_interaction_table()

    # H–O mixing
    H = UFF_LJ_PARAMS["H"]
    O = UFF_LJ_PARAMS["O"]

    expected_sigma = 0.5 * (H["sigma"] + O["sigma"])
    expected_epsilon = (H["epsilon"] * O["epsilon"]) ** 0.5

    assert table[("H", "O")]["sigma"] == pytest.approx(expected_sigma)
    assert table[("H", "O")]["epsilon"] == pytest.approx(expected_epsilon)


# --- Test All Elements Have Self-Interactions --- #

def test_self_interactions_exist():
    table = build_lj_interaction_table()

    for element in UFF_LJ_PARAMS.keys():
        assert (element, element) in table
        assert "sigma" in table[(element, element)]
        assert "epsilon" in table[(element, element)]