# Demon Algorithm for the 2D Ideal Gas

This repository contains a Python implementation of the **Demon Algorithm** applied to a two‑dimensional ideal gas. The demon algorithm is a Monte Carlo method that samples the canonical ensemble without explicitly computing Boltzmann factors. Instead, it introduces a fictitious “demon” that exchanges energy with the system, conserving total energy.

## Overview

The simulation models `N` non‑interacting point particles in a square box with periodic boundaries.  
The demon stores an energy `E_d ≥ 0`. The total energy `E_total = kinetic_energy + E_d` is kept constant.  

At equilibrium, the demon energy follows the exponential distribution `P(E_d) ∝ exp(-E_d / T)`, so the average demon energy directly gives the temperature:  
**`⟨E_d⟩ = T`** (with `k_B = 1`).

The code reproduces three key results of the ideal gas:
- Equipartition: `⟨K⟩/N = T` (in 2D).
- Maxwell–Boltzmann speed distribution.
- Uniform spatial distribution of particles.

## Code Structure

### Main Functions

| Function | Description |
|----------|-------------|
| `initialise_system(T_init)` | Places particles randomly in the box and assigns velocities from a Maxwell–Boltzmann distribution at `T_init`. |
| `apply_periodic(pos)` | Wraps particle positions back into the box `[0, L)`. |
| `demon_velocity_move(...)` | Attempts to change a particle’s velocity using the demon. Accepts if `ΔK ≤ E_d`. Updates `kinetic` and `E_d`. |
| `position_move(...)` | Performs a random displacement of a particle (always accepted, because potential energy is zero). |
| `run_simulation(Ed_init)` | Runs the full Monte Carlo simulation for a given initial demon energy. Returns temperature, average kinetic energy, final positions/velocities, and speed distribution. |
| `maxwell_boltzmann_2d(v, T)` | Theoretical speed distribution for a 2D ideal gas. |

### Simulation Parameters (tunable)

```python
N = 100          # number of particles
L = 20.0         # box size (area = L²)
mass = 1.0       # particle mass (k_B = 1)
delta_v_max = 1.0   # max velocity change per Monte Carlo step
delta_r_max = 1.0   # max displacement step
thermal_steps = 200000   # steps for equilibration
prod_steps = 500000      # steps for production
