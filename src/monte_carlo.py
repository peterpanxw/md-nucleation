## 3. Algorithm Steps
# 1. Initialize positions and velocities
# 2. Compute potential energy
# 3. Randomly move one particle (Monte Carlo Move) and compute the new potential energy
# 4. Accept/reject based on Boltzmann probability
# 5. Repeat N iterations

## 4. Input/Output
# - **Input:**
#   - Configuration file (positions, number of particles, temperature, box size)
#   - Interaction parameters (e.g., $\varepsilon$, $\sigma$ for Lennard-Jones)
# - **Output:**
#   - Average potential energy
#   - Energy distribution histogram
#   - Accepted/rejected move ratios
#   - Final molecular configurations

import numpy as np
import random

def read_configuration(file_path):
    # Placeholder for reading configuration from a file
    # In practice, this would parse a file and extract positions, box size, etc.
    num_particles = 100
    box_size = 10.0
    positions = np.random.rand(num_particles, 3) * box_size
    return positions, box_size

def initialize_system(num_particles):
    positions, _ = read_configuration("config.txt")
    velocities = np.random.rand(num_particles, 3) * 0.1  # Small initial velocities
    return positions, velocities

def compute_potential_energy(positions):
    energy = 0.0
    num_particles = positions.shape[0]
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 0:
                energy += 4 * ((1 / r)**12 - (1 / r)**6)  # Lennard-Jones potential
    return energy

def monte_carlo_move(positions, box_size):
    num_particles = positions.shape[0]
    particle_index = random.randint(0, num_particles - 1)
    displacement = (np.random.rand(3) - 0.5) * 0.1  # Small random displacement
    new_positions = positions.copy()
    new_positions[particle_index] += displacement
    new_positions[particle_index] = new_positions[particle_index] % box_size  # Periodic boundary conditions
    return new_positions

def monte_carlo_simulation(num_particles, box_size, num_iterations, temperature):
    positions, velocities = initialize_system(num_particles, box_size)
    current_energy = compute_potential_energy(positions)
    
    for iteration in range(num_iterations):
        new_positions = monte_carlo_move(positions, box_size)
        new_energy = compute_potential_energy(new_positions)
        
        delta_energy = new_energy - current_energy
        if delta_energy < 0 or random.random() < np.exp(-delta_energy / temperature):
            positions = new_positions
            current_energy = new_energy
            
    return positions, current_energy

def histogram_energy(energies, num_bins):
    hist, bin_edges = np.histogram(energies, bins=num_bins)
    return hist, bin_edges

def save_results(positions, energy, file_path):
    # Placeholder for saving results to a file
    with open(file_path, 'w') as f:
        f.write(f"Final Energy: {energy}\n")
        f.write("Final Positions:\n")
        for pos in positions:
            f.write(f"{pos[0]} {pos[1]} {pos[2]}\n")

# Example usage
if __name__ == "__main__":
    num_particles = 100
    box_size = 10.0
    num_iterations = 500
    temperature = 1.0
    
    final_positions, final_energy = monte_carlo_simulation(num_particles, box_size, num_iterations, temperature)
    print("Final Energy:", final_energy)
