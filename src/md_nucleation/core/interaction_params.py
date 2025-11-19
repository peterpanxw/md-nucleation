def validate_type_params(type_params):
    """
    Validate that each atom type has numeric sigma and epsilon.

    Parameters
    ----------
    type_params : dict
        Example:
        {
           "A": {"sigma": 1.0, "epsilon": 1.0},
           "B": {"sigma": 0.9, "epsilon": 0.5}
        }

    Raises
    ------
    ValueError
        If sigma or epsilon is missing or non-numeric.
    """
    for atom_type, params in type_params.items():
        if "sigma" not in params:
            raise ValueError(f"Missing sigma for atom type '{atom_type}'.")
        if "epsilon" not in params:
            raise ValueError(f"Missing epsilon for atom type '{atom_type}'.")

        try:
            float(params["sigma"])
        except:
            raise ValueError(f"sigma for atom type '{atom_type}' must be numeric.")

        try:
            float(params["epsilon"])
        except:
            raise ValueError(f"epsilon for atom type '{atom_type}' must be numeric.")

    return True


def lorentz_berthelot_mixing(sigma_i, sigma_j, epsilon_i, epsilon_j):
    """
    Apply Lorentz-Berthelot mixing rules for LJ interactions.
    """
    mixed_sigma = 0.5 * (sigma_i + sigma_j)
    mixed_epsilon = (epsilon_i * epsilon_j) ** 0.5
    return mixed_sigma, mixed_epsilon


def build_lj_interaction_table(type_params):
    """
    Build a complete Lennard-Jones parameter table using mixing rules.

    Parameters
    ----------
    type_params : dict
        Example input:
        {
           "A": {"sigma": 1.0, "epsilon": 1.0},
           "B": {"sigma": 0.9, "epsilon": 0.5}
        }

    Returns
    -------
    dict
        Pairwise LJ parameters:
        {
            ("A","A"): {"sigma": 1.0, "epsilon": 1.0},
            ("A","B"): {"sigma": 0.95, "epsilon": 0.7071},
            ("B","B"): {"sigma": 0.9,  "epsilon": 0.5}
        }
    """

    validate_type_params(type_params)

    table = {}
    types = list(type_params.keys())

    for i in range(len(types)):
        for j in range(i, len(types)):
            ti = types[i]
            tj = types[j]

            sigma_i = float(type_params[ti]["sigma"])
            epsilon_i = float(type_params[ti]["epsilon"])

            sigma_j = float(type_params[tj]["sigma"])
            epsilon_j = float(type_params[tj]["epsilon"])

            sigma_ij, epsilon_ij = lorentz_berthelot_mixing(
                sigma_i, sigma_j, epsilon_i, epsilon_j
            )

            # Store symmetric pair
            table[(ti, tj)] = {"sigma": sigma_ij, "epsilon": epsilon_ij}
            table[(tj, ti)] = {"sigma": sigma_ij, "epsilon": epsilon_ij}

    return table