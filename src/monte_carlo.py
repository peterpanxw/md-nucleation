import numpy as np
import random

def read_configuration(file_path):
    """
    Reads the configuration file and returns initial positions and box size.

    Parameters
    ----------
    file_path : str 
        Path to the configuration file.

    Returns
    ----------
    tuple : positions, box_size

    positions : numpy.ndarray
        Initial positions of the particles.
    box_size : float
        Size of the simulation box.
    """
    # Placeholder for reading configuration from a file
    # In practice, this would parse a file and extract positions, box size, etc.
    num_particles = 100
    box_size = 10.0
    positions = np.random.rand(num_particles, 3) * box_size
    return positions, box_size

def initialize_system(num_particles):
    """
    Initializes the system with random positions and velocities.

    Parameters
    ----------
    num_particles : int
        Number of particles in the system.
        
    Returns
    ----------

    tuple : positions, velocities
    positions : numpy.ndarray
        Initial positions of the particles.
    velocities : numpy.ndarray
        Initial velocities of the particles.
    """

    positions, _ = read_configuration("config.txt")
    velocities = np.random.rand(num_particles, 3) * 0.1  # Small initial velocities
    return positions, velocities

def compute_potential_energy(positions):
    """
    Computes the potential energy of the system using a simple Lennard-Jones potential.

    Parameters
    ----------
    positions : numpy.ndarray
        Positions of the particles.

    Returns
    -------
    float
        Total potential energy of the system.
    """
    energy = 0.0
    num_particles = positions.shape[0]
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 0:
                energy += 4 * ((1 / r)**12 - (1 / r)**6)  # Lennard-Jones potential
    return energy

def monte_carlo_move(positions, box_size):
    """
    Performs a Monte Carlo move by randomly displacing one particle.

    Parameters
    ----------
    positions : numpy.ndarray
        Current positions of the particles.
    box_size : float
        Size of the simulation box.

    Returns
    -------
    new_positions: numpy.ndarray
        New positions of the particles after the move.
    """
    num_particles = positions.shape[0]
    particle_index = random.randint(0, num_particles - 1)
    displacement = (np.random.rand(3) - 0.5) * 0.1  # Small random displacement
    new_positions = positions.copy()
    new_positions[particle_index] += displacement
    new_positions[particle_index] = new_positions[particle_index] % box_size  # Periodic boundary conditions
    return new_positions

def monte_carlo_simulation(num_particles, box_size, num_iterations, temperature):
    """
    Runs the Monte Carlo simulation.

    Parameters
    ----------
    num_particles : int
        Number of particles in the system.
    box_size : float
        Size of the simulation box.
    num_iterations : int
        Number of Monte Carlo iterations to perform.
    temperature : float
        Temperature of the system.

    Returns
    -------
    tuple : final_positions, final_energy, accepted_moves, rejected_moves

    final_positions : numpy.ndarray
        Final positions of the particles after simulation.
    final_energy : float
        Final potential energy of the system.
    accepted_moves : int
        Number of accepted moves.
    rejected_moves : int
        Number of rejected moves.
    """
    
    positions, velocities = initialize_system(num_particles)
    current_energy = compute_potential_energy(positions)

    acc = 0
    rej = 0
    
    for iteration in range(num_iterations):
        new_positions = monte_carlo_move(positions, box_size)
        new_energy = compute_potential_energy(new_positions)
        
        delta_energy = new_energy - current_energy
        if delta_energy < 0 or random.random() < np.exp(-delta_energy / temperature):
            positions = new_positions
            current_energy = new_energy
            acc += 1
        else:
            rej += 1

    return positions, current_energy, acc, rej

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

def save_results(positions, energy, ratio, file_path):
    """
    Saves the simulation results to a file.

    Parameters
    ----------
    positions : numpy.ndarray
        Final positions of the particles.
    energy : float
        Final potential energy of the system.
    ratio : float
        Acceptance ratio of the Monte Carlo moves.
    file_path : str
        Path to the output file.
    
    Returns
    -------
    None
    """
    # Placeholder for saving results to a file
    with open(file_path, 'w') as f:
        f.write(f"Final Energy: {energy}\n")
        f.write(f"Average Energy: {np.mean(energy)}\n")
        f.write(f"Acceptance Ratio: {ratio}\n")
        f.write("Final Positions:\n")
        for pos in positions:
            f.write(f"{pos[0]} {pos[1]} {pos[2]}\n")

# Example usage
if __name__ == "__main__":
    num_particles = 100
    box_size = 10.0
    num_iterations = 500
    temperature = 1.0
    
    final_positions, final_energy, acc, rej = monte_carlo_simulation(num_particles, box_size, num_iterations, temperature)
    print("Final Energy:", final_energy)
    save_results(final_positions, final_energy, acc / rej, "results.txt")
