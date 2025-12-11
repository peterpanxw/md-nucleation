#!/usr/bin/env python3
"""
Example script demonstrating Monte Carlo simulation usage.

This script shows how to:
1. Create a simple input configuration
2. Run a Monte Carlo simulation
3. Analyze the results
"""

from md_nucleation.io.filereadwrite import read_input_file
from md_nucleation.io.validation import validate_parsed_input
from md_nucleation.core.system_builder import build_system_from_input
from md_nucleation.core.monte_carlo import (
    monte_carlo_simulation,
    compute_potential_energy,
    save_results,
)
import numpy as np
import os


def create_simple_system():
    """Create a simple test system programmatically."""
    params = {
        "n": "5",
        "l": "15.0",
        "s": "500",
        "t": "300",
        "p": "1.0",
        "positions": [
            "Ar 2.0 2.0 2.0",
            "Ar 5.0 5.0 5.0",
            "Ar 8.0 8.0 8.0",
            "Ar 11.0 11.0 11.0",
            "Ar 13.0 13.0 13.0",
        ],
    }

    validate_parsed_input(params)
    system = build_system_from_input(params)
    return system


def main():
    print("=" * 60)
    print("Monte Carlo Simulation Example")
    print("=" * 60)
    print()
    script_dir = ""


    try:
    # Option 1: Load from file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(script_dir, "input_minimal.txt")
        params = read_input_file(input_path)
        print("Option 1: Loading from input file...")
        validate_parsed_input(params)
        system_from_file = build_system_from_input(params)
        print(f"  Loaded {len(system_from_file.particles)} particles from file")
        print(f"  Box length: {system_from_file.box.length}")
        print(f"  Temperature: {system_from_file.T} K")
        system = system_from_file
    except FileNotFoundError:
    # Option 2: Create programmatic system
        print("  Input file not found, using programmatic system...")
        system = create_simple_system()
        print(f"Created system with {len(system.particles)} particles")

    print()

    # Compute initial energy
    print("Computing initial system energy...")
    initial_energy = compute_potential_energy(system)
    print(f"  Initial energy: {initial_energy:.6f}")
    print()

    # Run Monte Carlo simulation
    print("Running Monte Carlo simulation...")
    num_iterations = 1000
    max_displacement = 0.1

    print(f"  Iterations: {num_iterations}")
    print(f"  Max displacement: {max_displacement}")
    print(f"  Temperature: {system.T} K")
    print()

    results = monte_carlo_simulation(
        system=system,
        num_iterations=num_iterations,
        max_displacement=max_displacement,
    )

    # Print results
    print("Simulation Complete!")
    print("-" * 60)
    print(f"Final energy:      {results['final_energy']:12.6f}")
    print(f"Initial energy:    {initial_energy:12.6f}")
    print(f"Energy change:     {(results['final_energy'] - initial_energy):12.6f}")
    print()
    print(f"Accepted moves:    {results['accepted_moves']:6d}")
    print(f"Rejected moves:    {results['rejected_moves']:6d}")
    print(f"Acceptance ratio:  {results['acceptance_ratio']:8.4f}")
    print()

    # Energy statistics
    energy_hist = results["energy_history"]
    print("Energy Statistics:")
    print(f"  Minimum:  {np.min(energy_hist):.6f}")
    print(f"  Maximum:  {np.max(energy_hist):.6f}")
    print(f"  Average:  {np.mean(energy_hist):.6f}")
    print(f"  Median:   {np.median(energy_hist):.6f}")
    print()

    # Save results
    output_file = os.path.join(script_dir, "example_mc_results.txt")
    print(f"Saving results to '{output_file}'...")
    save_results(results, system, output_file)
    print("  Results saved")
    print()

    print("=" * 60)
    print("Example complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
