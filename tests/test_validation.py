import pytest
from md_nucleation.io.validation import (
    validate_parsed_input,
    parse_typed_position,
)


# --- Test parse_typed_position valid case --- #

def test_parse_typed_position_valid():
    atom_type, x, y, z = parse_typed_position("A 1.0 2.0 3.0")

    assert atom_type == "A"
    assert x == 1.0
    assert y == 2.0
    assert z == 3.0


# --- Test parse_typed_position invalid atom type --- #

def test_parse_typed_position_invalid_type():
    with pytest.raises(ValueError) as excinfo:
        parse_typed_position("1 0.0 0.0 0.0")

    assert "Invalid atom type" in str(excinfo.value)


# --- Test parse_typed_position non-numeric coordinates --- #

def test_parse_typed_position_non_numeric():
    with pytest.raises(ValueError) as excinfo:
        parse_typed_position("A x 1.0 2.0")

    assert "Coordinates must be numeric" in str(excinfo.value)


# --- Test parse_typed_position wrong format --- #

def test_parse_typed_position_wrong_format():
    with pytest.raises(ValueError) as excinfo:
        parse_typed_position("A 0.0 1.0")

    assert "Expected format: TYPE x y z" in str(excinfo.value)


# --- Test validate_parsed_input: valid case --- #

def test_validate_parsed_input_valid():
    params = {
        "n": "2",
        "l": "10.0",
        "s": "100",
        "t": "300",
        "p": "1.0",
        "positions": [
            "A 0.0 0.0 0.0",
            "A 1.0 1.0 1.0",
        ],
    }

    assert validate_parsed_input(params) is True


# --- Test validate_parsed_input: missing required field --- #

def test_validate_parsed_input_missing_s():
    params = {
        "n": "2",
        "l": "10.0",
        # "s" missing
        "t": "300",
        "p": "1.0",
        "positions": [
            "A 0.0 0.0 0.0",
            "A 1.0 1.0 1.0",
        ],
    }

    with pytest.raises(ValueError) as excinfo:
        validate_parsed_input(params)

    assert "number of time steps (S)" in str(excinfo.value)


# --- Test validate_parsed_input: invalid N type --- #

def test_validate_parsed_input_invalid_n():
    params = {
        "n": "abc",  # invalid
        "l": "10.0",
        "s": "100",
        "t": "300",
        "p": "1.0",
        "positions": [
            "A 0.0 0.0 0.0",
            "A 1.0 1.0 1.0",
        ],
    }

    with pytest.raises(ValueError) as excinfo:
        validate_parsed_input(params)

    assert "number of atoms (N)" in str(excinfo.value)


# --- Test validate_parsed_input: position count mismatch --- #

def test_validate_parsed_input_position_count_mismatch():
    params = {
        "n": "2",
        "l": "10.0",
        "s": "100",
        "t": "300",
        "p": "1.0",
        "positions": [
            "A 0.0 0.0 0.0"
            # missing second line
        ],
    }

    with pytest.raises(ValueError) as excinfo:
        validate_parsed_input(params)

    assert "does not match the number of atoms" in str(excinfo.value)


# --- Test validate_parsed_input: malformed position line --- #

def test_validate_parsed_input_malformed_position():
    params = {
        "n": "2",
        "l": "10.0",
        "s": "100",
        "t": "300",
        "p": "1.0",
        "positions": [
            "A 0.0 1.0",  # malformed
            "A 1.0 1.0 1.0",
        ],
    }

    with pytest.raises(ValueError) as excinfo:
        validate_parsed_input(params)

    assert "Expected format: TYPE x y z" in str(excinfo.value)


# --- Test validate_parsed_input: non-numeric coordinates --- #

def test_validate_parsed_input_non_numeric_position():
    params = {
        "n": "2",
        "l": "10.0",
        "s": "100",
        "t": "300",
        "p": "1.0",
        "positions": [
            "A x 1.0 2.0",  # non-numeric
            "A 1.0 1.0 1.0",
        ],
    }

    with pytest.raises(ValueError) as excinfo:
        validate_parsed_input(params)

    assert "Coordinates must be numeric" in str(excinfo.value)