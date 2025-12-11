import numpy as np
import random


def compute_potential_energy(system):
    """
    Computes the potential energy of the system using Lennard-Jones potential
    with interaction parameters from the system's interaction table.

    Parameters
    ----------
    system : System
        MD system containing particles with positions, types, and interaction parameters.

    Returns
    -------
    float
        Total potential energy of the system.
    """
    energy = 0.0
    num_particles = len(system.particles)

    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            # Get particle positions
            pos_i = np.array(system.particles[i].position)
            pos_j = np.array(system.particles[j].position)

            # Compute distance with periodic boundary conditions
            delta = pos_i - pos_j
            box_length = system.box.length

            # Apply minimum image convention
            delta = delta - box_length * np.round(delta / box_length)
            r = np.linalg.norm(delta)

            if r > 0:
                # Get LJ parameters for this pair
                type_i = system.particles[i].atom_type
                type_j = system.particles[j].atom_type
                lj_params = system.interaction_table.get((type_i, type_j))

                if lj_params is None:
                    raise ValueError(
                        f"No interaction parameters found for pair ({type_i}, {type_j})"
                    )

                sigma = lj_params["sigma"]
                epsilon = lj_params["epsilon"]

                # Lennard-Jones potential: 4*epsilon*[(sigma/r)^12 - (sigma/r)^6]
                sr = sigma / r
                sr6 = sr**6
                sr12 = sr6**2
                energy += 4 * epsilon * (sr12 - sr6)

    return energy


def monte_carlo_move(system, max_displacement=0.1):
    """
    Performs a Monte Carlo move by randomly displacing one particle.
    Returns a new system with updated particle position.

    Parameters
    ----------
    system : System
        Current MD system.
    max_displacement : float
        Maximum displacement for the Monte Carlo move.

    Returns
    -------
    System
        New system with updated particle position.
    """
    import copy

    # Create a deep copy to avoid modifying the original
    new_system = copy.deepcopy(system)

    num_particles = len(new_system.particles)
    particle_index = random.randint(0, num_particles - 1)

    # Random displacement centered around 0, range [-0.5,0.5] * max
    displacement = (np.random.rand(3) - 0.5) * max_displacement 

    # Update position
    new_pos = np.array(new_system.particles[particle_index].position) + displacement

    # Periodic boundary conditions
    box_length = new_system.box.length
    new_pos = new_pos % box_length

    new_system.particles[particle_index].position = new_pos.tolist()

    return new_system


def monte_carlo_simulation(system, num_iterations=1000, max_displacement=0.1):
    """
    Runs the Monte Carlo simulation using the Metropolis algorithm.

    Parameters
    ----------
    system : System
        Initial MD system containing particles, box, temperature, etc.
    num_iterations : int
        Number of Monte Carlo iterations to perform.
    max_displacement : float
        Maximum displacement for Monte Carlo moves.

    Returns
    -------
    dict
        Results dictionary containing:
        - 'final_system' : System - Final system state
        - 'final_energy' : float - Final potential energy
        - 'accepted_moves' : int - Number of accepted moves
        - 'rejected_moves' : int - Number of rejected moves
        - 'acceptance_ratio' : float - Ratio of accepted to total moves
        - 'energy_history' : list - Energy at each iteration
    """
    import copy

    # Work with a copy to avoid modifying the input
    current_system = copy.deepcopy(system)
    temperature = system.T

    # Compute initial energy
    current_energy = compute_potential_energy(current_system)

    # Statistics
    accepted_moves = 0
    rejected_moves = 0
    energy_history = [current_energy]

    # Boltzmann constant (in appropriate units, here we use reduced units)
    kB = 1.0

    for iteration in range(num_iterations):
        # Propose a move
        new_system = monte_carlo_move(current_system, max_displacement)
        new_energy = compute_potential_energy(new_system)

        # Metropolis criterion
        delta_energy = new_energy - current_energy

        if delta_energy < 0:
            # Always accept moves that lower energy
            current_system = new_system
            current_energy = new_energy
            accepted_moves += 1
        else:
            # Accept with probability exp(-ΔE / kT)
            acceptance_probability = np.exp(-delta_energy / (kB * temperature))
            if random.random() < acceptance_probability:
                current_system = new_system
                current_energy = new_energy
                accepted_moves += 1
            else:
                rejected_moves += 1

        energy_history.append(current_energy)

    # Calculate acceptance ratio
    total_moves = accepted_moves + rejected_moves
    acceptance_ratio = accepted_moves / total_moves if total_moves > 0 else 0.0

    return {
        "final_system": current_system,
        "final_energy": current_energy,
        "accepted_moves": accepted_moves,
        "rejected_moves": rejected_moves,
        "acceptance_ratio": acceptance_ratio,
        "energy_history": energy_history,
    }


def histogram_energy(energies, num_bins):
    """
    Creates a histogram of the energy distribution.

    Parameters
    ----------
    energies : list[float]
        List of energy values.
    num_bins : int
        Number of bins for the histogram.

    Returns
    -------
    tuple : hist, bin_edges

    hist : numpy.ndarray
        Histogram counts.
    bin_edges : numpy.ndarray
        Edges of the histogram bins.
    """
    hist, bin_edges = np.histogram(energies, bins=num_bins)
    return hist, bin_edges


def save_results(results, system, file_path):
    """
    Saves the simulation results to a file.

    Parameters
    ----------
    results : dict
        Results dictionary from monte_carlo_simulation.
    system : System
        Initial system (for reference).
    file_path : str
        Path to the output file.

    Returns
    -------
    None
    """
    final_system = results["final_system"]

    with open(file_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MONTE CARLO SIMULATION RESULTS\n")
        f.write("=" * 60 + "\n\n")

        # System parameters
        f.write("System Parameters:\n")
        f.write(f"  Number of particles: {len(system.particles)}\n")
        f.write(f"  Box length: {system.box.length}\n")
        f.write(f"  Temperature: {system.T} K\n")
        f.write(f"  Pressure: {system.P} bar\n")
        f.write(f"  Total iterations: {system.S}\n\n")

        # Energy statistics
        f.write("Energy Statistics:\n")
        f.write(f"  Final Energy: {results['final_energy']:.6f}\n")
        f.write(f"  Average Energy: {np.mean(results['energy_history']):.6f}\n")
        f.write(f"  Min Energy: {np.min(results['energy_history']):.6f}\n")
        f.write(f"  Max Energy: {np.max(results['energy_history']):.6f}\n\n")

        # Move statistics
        f.write("Move Statistics:\n")
        f.write(f"  Accepted moves: {results['accepted_moves']}\n")
        f.write(f"  Rejected moves: {results['rejected_moves']}\n")
        f.write(f"  Acceptance ratio: {results['acceptance_ratio']:.4f}\n\n")

        # Final positions
        f.write("=" * 60 + "\n")
        f.write("Final Particle Positions:\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Type':<6} {'X':>12} {'Y':>12} {'Z':>12}\n")
        f.write("-" * 60 + "\n")

        for particle in final_system.particles:
            f.write(
                f"{particle.atom_type:<6} "
                f"{particle.position[0]:>12.6f} "
                f"{particle.position[1]:>12.6f} "
                f"{particle.position[2]:>12.6f}\n"
            )

        f.write("=" * 60 + "\n")


def run_monte_carlo_from_input(
    input_file, output_file="mc_results.txt", max_displacement=0.1
):
    """
    Complete workflow: read input file, build system, run MC simulation, save results.

    Parameters
    ----------
    input_file : str
        Path to the input configuration file.
    output_file : str
        Path to the output results file.
    max_displacement : float
        Maximum displacement for Monte Carlo moves.

    Returns
    -------
    dict
        Results dictionary from the Monte Carlo simulation.
    """
    from md_nucleation.io.filereadwrite import read_input_file
    from md_nucleation.io.validation import validate_parsed_input
    from md_nucleation.core.system_builder import build_system_from_input

    # Read and validate input
    print(f"Reading input file: {input_file}")
    params = read_input_file(input_file)
    validate_parsed_input(params)

    # Build system
    print("Building system...")
    system = build_system_from_input(params)

    # Run Monte Carlo simulation
    num_iterations = system.S  # Use S from input as number of MC steps
    print(f"Running Monte Carlo simulation with {num_iterations} iterations...")
    print(f"  Temperature: {system.T} K")
    print(f"  Number of particles: {len(system.particles)}")
    print(f"  Box length: {system.box.length}")

    results = monte_carlo_simulation(system, num_iterations, max_displacement)

    # Print summary
    print("\nSimulation complete!")
    print(f"  Final energy: {results['final_energy']:.6f}")
    print(f"  Acceptance ratio: {results['acceptance_ratio']:.4f}")
    print(f"  Accepted moves: {results['accepted_moves']}")
    print(f"  Rejected moves: {results['rejected_moves']}")

    # Save results
    print(f"\nSaving results to: {output_file}")
    save_results(results, system, output_file)

    return results
