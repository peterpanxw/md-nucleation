import json
import pathlib
import tempfile
from md_nucleation.io.filereadwrite import (
    parse_input,
    read_input_file,
    write_output_file,
    append_log_file,
    save_checkpoint,
)

# --- Test Parsing Basic Key-Fields --- #


def test_parse_basic_fields():
    data = ["N 5", "L 20.0", "s 1000", "T 300", "P 1.0"]
    params = parse_input(data)

    assert params["n"] == "5"
    assert params["l"] == "20.0"
    assert params["s"] == "1000"
    assert params["t"] == "300"
    assert params["p"] == "1.0"


# --- Test Parsing Positions Block --- #


def test_parse_positions_block():
    data = ["positions", "A 0.0 0.0 0.0", "A 1.0 1.0 1.0", "###"]

    params = parse_input(data)

    assert "positions" in params
    assert len(params["positions"]) == 2
    assert params["positions"][0] == "A 0.0 0.0 0.0"
    assert params["positions"][1] == "A 1.0 1.0 1.0"


# --- Test Reading from Real Input File --- #


def test_read_input_file():
    test_file = pathlib.Path("tests/data/test_input.txt")
    params = read_input_file(test_file)

    assert params["n"] == "3"
    assert params["l"] == "20.0"
    assert len(params["positions"]) == 3


# --- Test Writing an Output File --- #


def test_write_output_file():
    with tempfile.NamedTemporaryFile("r+", delete=True) as tmp:
        write_output_file(tmp.name, ["hello\n", "world\n"])
        tmp.seek(0)
        content = tmp.read()

    assert "hello" in content
    assert "world" in content


# --- Test Appending to a Log File --- #


def test_append_log_file():
    with tempfile.NamedTemporaryFile("r+", delete=True) as tmp:
        append_log_file(tmp.name, ["line1\n"])
        append_log_file(tmp.name, ["line2\n"])
        tmp.seek(0)
        lines = tmp.readlines()

    assert lines == ["line1\n", "line2\n"]


# --- Test Saving a Checkpoint --- #


def test_save_checkpoint():
    with tempfile.NamedTemporaryFile("r+", delete=True) as tmp:
        ok = save_checkpoint(
            tmp.name, positions=[[0, 0, 0]], velocities=[[1, 1, 1]], step=42
        )

        assert ok is True

        tmp.seek(0)
        checkpoint = json.load(tmp)

    assert checkpoint["step"] == 42
    assert checkpoint["positions"][0] == [0, 0, 0]
    assert checkpoint["velocities"][0] == [1, 1, 1]
