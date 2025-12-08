# md-nucleation
Simulating Nucleation using Molecular Dynamics

## How to Use

```bash
# Basic usage with default parameters
md-nucleation run examples/input_minimal.txt

# Specify output file
md-nucleation run examples/input_minimal.txt --output my_results.txt

# Adjust maximum displacement for MC moves
md-nucleation run examples/input_minimal.txt --max-displacement 0.2
```

## Input File Format

Your input file should follow the standard format:

```
N 10                    # Number of particles
L 20.0                  # Box length
S 1000                  # Number of MC steps (iterations)
T 300                   # Temperature (K)
P 1.0                   # Pressure (bar)

positions
A 0.0 0.0 0.0          # Type X Y Z
A 1.0 1.0 1.0
A 2.0 2.0 2.0
...
###
```

## Output Format

The simulation produces a detailed output file with:
- System parameters
- Energy statistics (final, average, min, max)
- Move statistics (accepted, rejected, acceptance ratio)
- Final particle positions

Example output:
```
============================================================
MONTE CARLO SIMULATION RESULTS
============================================================

System Parameters:
  Number of particles: 10
  Box length: 20.0
  Temperature: 300.0 K
  Pressure: 1.0 bar
  Total iterations: 1000

Energy Statistics:
  Final Energy: -12.345678
  Average Energy: -11.234567
  Min Energy: -13.456789
  Max Energy: -10.123456

Move Statistics:
  Accepted moves: 456
  Rejected moves: 544
  Acceptance ratio: 0.4560

============================================================
Final Particle Positions:
============================================================
Type              X            Y            Z
------------------------------------------------------------
A            1.234567    2.345678    3.456789
...
```

## Advanced Usage

### Working with Different Atom Types

The system automatically handles mixed atom types using UFF parameters:

```
positions
H 0.0 0.0 0.0
O 5.0 0.0 0.0
H 10.0 0.0 0.0
###
```

Supported elements: H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti
