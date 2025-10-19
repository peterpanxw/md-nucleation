# Monte Carlo Algorithm for Molecular Dynamics

## 1. Objective
Implement a **Monte Carlo (MC)** simulation to estimate thermodynamic properties during a **nucleation process**.  
The goal is to use random sampling to explore molecular configurations and evaluate equilibrium properties such as energy distributions and phase formation probabilities.

## 2. Files & Modules Required
- `monte_carlo.py`: mc algorithm implementation

## 3. Algorithm Steps
1. Initialize positions and velocities
2. Compute potential energy
3. Randomly move one particle (Monte Carlo Move) and compute the new potential energy
4. Accept/reject based on Boltzmann probability
5. Repeat N iterations

## 4. Input/Output
- **Input:**
  - Configuration file (positions, number of particles, temperature, box size)
  - Interaction parameters (e.g., $\varepsilon$, $\sigma$ for Lennard-Jones)
- **Output:**
  - Average potential energy
  - Energy distribution histogram
  - Accepted/rejected move ratios
  - Final molecular configurations

## 5. Notes
- Periodic boundary conditions (PBC) should be applied.

## 5. Next Steps
Chinmay: implement algorithm (`monte_carlo.py`)
