import pytest
import numpy as np
from md_nucleation.core.monte_carlo import (
    compute_potential_energy,
    monte_carlo_move,
    monte_carlo_simulation,
)
from md_nucleation.core.system_builder import build_system_from_input


def test_compute_potential_energy():
    """Test that potential energy calculation works with the System object."""
    params = {
        "n": "2",
        "l": "10.0",
        "s": "100",
        "t": "300",
        "p": "1.0",
        "positions": ["A 0.0 0.0 0.0", "A 5.0 0.0 0.0"],
    }
    
    system = build_system_from_input(params)
    energy = compute_potential_energy(system)
    
    # Energy should be a float
    assert isinstance(energy, float)
    # For two particles at distance 5.0 with A-A interaction, energy should be finite
    assert np.isfinite(energy)


def test_monte_carlo_move():
    """Test that MC move creates a valid new system."""
    params = {
        "n": "3",
        "l": "10.0",
        "s": "100",
        "t": "300",
        "p": "1.0",
        "positions": [
            "A 1.0 1.0 1.0",
            "A 2.0 2.0 2.0",
            "A 3.0 3.0 3.0",
        ],
    }
    
    system = build_system_from_input(params)
    new_system = monte_carlo_move(system, max_displacement=0.1)
    
    # Check that we get a new system
    assert new_system is not system
    
    # Check that number of particles is unchanged
    assert len(new_system.particles) == len(system.particles)
    
    # Check that at least one position changed
    positions_changed = False
    for i in range(len(system.particles)):
        if system.particles[i].position != new_system.particles[i].position:
            positions_changed = True
            break
    
    assert positions_changed, "Monte Carlo move should change at least one position"
    
    # Check that all positions are within box bounds
    for particle in new_system.particles:
        for coord in particle.position:
            assert 0 <= coord < system.box.length


def test_monte_carlo_simulation():
    """Test full Monte Carlo simulation."""
    params = {
        "n": "2",
        "l": "10.0",
        "s": "50",  # Small number of steps for quick test
        "t": "300",
        "p": "1.0",
        "positions": ["A 1.0 1.0 1.0", "A 2.0 2.0 2.0"],
    }
    
    system = build_system_from_input(params)
    results = monte_carlo_simulation(system, num_iterations=50, max_displacement=0.1)
    
    # Check that results dictionary has all expected keys
    assert "final_system" in results
    assert "final_energy" in results
    assert "accepted_moves" in results
    assert "rejected_moves" in results
    assert "acceptance_ratio" in results
    assert "energy_history" in results
    
    # Check types
    assert isinstance(results["final_energy"], float)
    assert isinstance(results["accepted_moves"], int)
    assert isinstance(results["rejected_moves"], int)
    assert isinstance(results["acceptance_ratio"], float)
    assert isinstance(results["energy_history"], list)
    
    # Check that total moves equals iterations
    total_moves = results["accepted_moves"] + results["rejected_moves"]
    assert total_moves == 50
    
    # Check that acceptance ratio is between 0 and 1
    assert 0 <= results["acceptance_ratio"] <= 1
    
    # Check energy history length
    assert len(results["energy_history"]) == 51  # Initial + 50 iterations


def test_monte_carlo_energy_conservation():
    """Test that energy changes are consistent with Metropolis criterion."""
    params = {
        "n": "3",
        "l": "10.0",
        "s": "100",
        "t": "1000",  # High temperature for more acceptances
        "p": "1.0",
        "positions": [
            "A 2.0 2.0 2.0",
            "A 5.0 5.0 5.0",
            "A 8.0 8.0 8.0",
        ],
    }
    
    system = build_system_from_input(params)
    results = monte_carlo_simulation(system, num_iterations=100, max_displacement=0.1)
    
    # At high temperature, we should accept a reasonable fraction of moves
    assert results["acceptance_ratio"] > 0.1
    
    # Energy history should have valid values
    for energy in results["energy_history"]:
        assert np.isfinite(energy)


def test_monte_carlo_with_different_atom_types():
    """Test MC simulation with mixed atom types."""
    params = {
        "n": "3",
        "l": "15.0",
        "s": "50",
        "t": "300",
        "p": "1.0",
        "positions": [
            "H 1.0 1.0 1.0",
            "O 8.0 8.0 8.0",
            "H 14.0 14.0 14.0",
        ],
    }
    
    system = build_system_from_input(params)
    
    # Should not raise an error with mixed types
    results = monte_carlo_simulation(system, num_iterations=50, max_displacement=0.2)
    
    assert results["final_energy"] is not None
    assert isinstance(results["final_energy"], float)
    assert np.isfinite(results["final_energy"])
