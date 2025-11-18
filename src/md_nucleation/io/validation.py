def parse_typed_position(line):
    """
    Parse a single typed position line of the form:
    TYPE x y z

    Returns
    -------
    tuple
        (atom_type : str, x : float, y : float, z : float)

    Raises
    ------
    ValueError
        If the format is wrong or values are not numeric.
    """
    parts = line.split()
    if len(parts) != 4:
        raise ValueError(
            f"Invalid position line '{line}'. Expected format: TYPE x y z"
        )

    atom_type, x, y, z = parts

    # Atom type: alphabetic only
    if not atom_type.isalpha():
        raise ValueError(
            f"Invalid atom type '{atom_type}' in line '{line}'. "
            "Atom type must contain only letters (e.g., A, B, C)."
        )

    # Coordinates: must be numeric
    try:
        x = float(x)
        y = float(y)
        z = float(z)
    except:
        raise ValueError(f"Coordinates must be numeric in line '{line}'.")

    return atom_type, x, y, z


def validate_parsed_input(params):
    """
    Validate parsed MD input parameters.
    Ensures required fields are present and consistent.
    """

    # --- Required scalar keys --- #
    required_scalars = {
        "n": "number of atoms (N)",
        "l": "box length (L)",
        "s": "number of time steps (S)",
        "t": "temperature (T)",
        "p": "pressure (P)"
    }

    # Check missing keys
    for key, description in required_scalars.items():
        if key not in params:
            raise ValueError(f"Missing required field: {description}")

    # --- Validate N --- #
    try:
        n = int(params["n"])
    except:
        raise ValueError("The number of atoms (N) must be an integer.")

    # --- Validate L --- #
    try:
        float(params["l"])
    except:
        raise ValueError("The box length (L) must be a float.")

    # --- Validate S --- #
    try:
        int(params["s"])
    except:
        raise ValueError("The number of time steps (S) must be an integer.")

    # --- Validate T --- #
    try:
        float(params["t"])
    except:
        raise ValueError("The temperature (T) must be a float.")

    # --- Validate P --- #
    try:
        float(params["p"])
    except:
        raise ValueError("The pressure (P) must be a float.")

    # --- Validate positions block exists --- #
    if "positions" not in params:
        raise ValueError("Missing positions block ('positions').")

    # --- Validate number of positions matches N --- #
    if len(params["positions"]) != n:
        raise ValueError(
            f"The number of positions ({len(params['positions'])}) "
            f"does not match the number of atoms N = {n}."
        )

    # --- Validate each typed position line --- #
    for i, line in enumerate(params["positions"]):
        parse_typed_position(line)   # If invalid, raises useful errors

    return True  # All checks passed