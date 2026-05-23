
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N = 100          # number of particles
L = 20.0         # box side length
mass = 1.0       # particle mass (k_B = 1)
d = 2            # dimensions

# Monte Carlo step sizes
delta_v_max = 1.0   # max velocity change
delta_r_max = 1.0   # max position displacement

# Simulation lengths
thermal_steps = 200000
prod_steps = 500000
total_steps = thermal_steps + prod_steps

def initialise_system(T_init):
    
    # positions: random in [0, L)
    positions = np.random.rand(N, 2) * L
    # velocities: each component ~ N(0, sqrt(T/mass))
    std = np.sqrt(T_init / mass)
    velocities = np.random.normal(0, std, (N, 2))
    # compute total kinetic energy
    kinetic = 0.5 * mass * np.sum(velocities**2)
    return positions, velocities, kinetic

def apply_periodic(pos):
    """Apply periodic boundary conditions in place."""
    pos[:, 0] %= L
    pos[:, 1] %= L


def demon_velocity_move(velocities, kinetic, Ed, i, delta_v_max, mass):

    v_old = velocities[i].copy()
    # propose new velocity by adding random vector
    dv = np.random.uniform(-delta_v_max, delta_v_max, size=2)
    v_new = v_old + dv
    deltaK = 0.5 * mass * (np.sum(v_new**2) - np.sum(v_old**2))
    if deltaK <= Ed:
        # accept
        velocities[i] = v_new
        kinetic += deltaK
        Ed -= deltaK
        return velocities, kinetic, Ed, True
    else:
        return velocities, kinetic, Ed, False

def position_move(positions, i, delta_r_max, L):
    """Random displacement of a particle (always accepted)."""
    dr = np.random.uniform(-delta_r_max, delta_r_max, size=2)
    positions[i] += dr
    apply_periodic(positions)
    return positions


def run_simulation(Ed_init, T_init=2.0):

    positions, velocities, kinetic = initialise_system(T_init)
    Ed = Ed_init
    # keep track of demon energy and kinetic energy during production
    demon_energies = []
    kinetic_energies = []
    speeds = []  # for speed distribution (only at the end)

    for step in range(total_steps):
        # choose random particle for velocity move
        i = np.random.randint(N)
        velocities, kinetic, Ed, _ = demon_velocity_move(
            velocities, kinetic, Ed, i, delta_v_max, mass)
        # position move for a different random particle (optional)
        j = np.random.randint(N)
        positions = position_move(positions, j, delta_r_max, L)

        # record after thermalisation
        if step >= thermal_steps:
            demon_energies.append(Ed)
            kinetic_energies.append(kinetic)
            # record speeds occasionally to reduce memory
            if step % 50 == 0:
                sp = np.linalg.norm(velocities, axis=1)
                speeds.extend(sp)

    T_mean = np.mean(demon_energies)
    mean_ke = np.mean(kinetic_energies)
    mean_ke_per_particle = mean_ke / N
    return T_mean, mean_ke_per_particle, positions, velocities, np.array(speeds)

# ==============================
#   MAXWELL-BOLTZMANN DISTRIBUTION (2D)
# ==============================
def maxwell_boltzmann_2d(v, T, mass=1.0):
    """Normalised speed distribution for 2D ideal gas."""
    return (mass * v / T) * np.exp(-mass * v**2 / (2 * T))

# ==============================
#   MAIN SIMULATION LOOP
# ==============================
def main():
    # ---- 1) Generate initial and final position plots (single run) ----
    print("Running a single simulation for configuration snapshots...")
    Ed_demo = 2.0   # gives T ~ 2.0
    T_final, _, final_positions, _, speeds = run_simulation(Ed_demo)
    # initial positions are from random initialisation (we need to re-initialise)
    _, _, init_positions, _, _ = initialise_system(2.0)  # dummy initialisation
    # correct initialisation for same seed? We'll just generate a random initial config.
    np.random.seed(42)   # for reproducibility
    init_positions = np.random.rand(N, 2) * L

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].scatter(init_positions[:, 0], init_positions[:, 1], s=10, alpha=0.7)
    axes[0].set_xlim(0, L); axes[0].set_ylim(0, L)
    axes[0].set_title('Initial random positions')
    axes[0].set_aspect('equal')
    axes[1].scatter(final_positions[:, 0], final_positions[:, 1], s=10, alpha=0.7)
    axes[1].set_xlim(0, L); axes[1].set_ylim(0, L)
    axes[1].set_title(f'Final positions at T ≈ {T_final:.2f}')
    axes[1].set_aspect('equal')
    plt.tight_layout()
    plt.savefig('initial_final_positions.pdf', dpi=150)
    plt.show()

    # ---- 2) Energy vs Temperature over a range of demon energies ----
    print("Running multiple simulations for E vs T...")
    Ed_vals = np.arange(0.5, 10.5, 0.5)
    n_runs = len(Ed_vals)
    temps = np.zeros(n_runs)
    ke_per_particle = np.zeros(n_runs)
    errors = np.zeros(n_runs)

    n_repeats = 5
    for idx, Ed in enumerate(tqdm(Ed_vals, desc="Temperature scan")):
        e_vals = []
        t_vals = []
        for rep in range(n_repeats):
            T, ke, _, _, _ = run_simulation(Ed)
            t_vals.append(T)
            e_vals.append(ke)
        temps[idx] = np.mean(t_vals)
        ke_per_particle[idx] = np.mean(e_vals)
        errors[idx] = np.std(e_vals) / np.sqrt(n_repeats)

    # Theoretical line: <K>/N = T (since d=2, mass=1, k_B=1)
    T_theory = np.linspace(0, 10, 100)
    theory_line = T_theory

    plt.figure(figsize=(7, 5))
    plt.errorbar(temps, ke_per_particle, yerr=errors, fmt='o', capsize=3,
                 label='Demon simulation', color='blue', ecolor='gray')
    plt.plot(T_theory, theory_line, 'r-', label='Equipartition: $\\langle K \\rangle/N = T$')
    plt.xlabel('Temperature $T$ (from demon)', fontsize=12)
    plt.ylabel('Average kinetic energy per particle $\\langle K \\rangle/N$', fontsize=12)
    plt.title('Demon Algorithm: Energy vs Temperature for 2D Ideal Gas')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('energy_vs_temperature.pdf', dpi=150)
    plt.show()

    # ---- 3) Speed distribution at a representative temperature ----
    # Use the run with Ed_init = 2.0 (already performed)
    # We already have 'speeds' from that run; re-run with fixed seed for consistency
    print("Generating speed distribution...")
    np.random.seed(123)
    T_fixed, _, _, _, speeds = run_simulation(2.0)
    v_max = np.max(speeds)
    bins = np.linspace(0, v_max, 40)
    hist, bin_edges = np.histogram(speeds, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    v_theory = np.linspace(0, v_max, 200)
    f_theory = maxwell_boltzmann_2d(v_theory, T_fixed, mass)

    plt.figure(figsize=(7, 5))
    plt.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0],
            alpha=0.6, label='Simulation histogram')
    plt.plot(v_theory, f_theory, 'r-', linewidth=2,
             label=f'Maxwell-Boltzmann, T = {T_fixed:.2f}')
    plt.xlabel('Speed $v$', fontsize=12)
    plt.ylabel('Probability density $f(v)$', fontsize=12)
    plt.title('Speed Distribution from Demon Algorithm')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('speed_distribution.pdf', dpi=150)
    plt.show()

    # Save data to file
    with open('ideal_gas_data.txt', 'w') as f:
        f.write("# T (demon)   <K>/N   error\n")
        for t, ke, err in zip(temps, ke_per_particle, errors):
            f.write(f"{t:.5f}   {ke:.6f}   {err:.6f}\n")
    print("Simulation completed. Data saved to ideal_gas_data.txt")

if __name__ == "__main__":
    main()
