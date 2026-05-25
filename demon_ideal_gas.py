"""
Demon Algorithm for the 2D Ideal Gas
Simulates N non-interacting particles in a square box with periodic boundaries.
The demon exchanges kinetic energy with particles, providing temperature.
Produces plots:
- initial vs final particle positions
- average kinetic energy per particle vs temperature (equipartition)
- speed distribution compared to Maxwell-Boltzmann
- demon energy distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


#   SYSTEM PARAMETERS
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

#   INITIALISATION
def initialise_system(T_init):
    """
    Create random positions and velocities sampled from a normal distribution with std = sqrt(T/m) and mean 0
    """
    # positions: random in [0, L)
    #np.random.seed(42) , you can uncomment this line to get the same results every run. 
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

#   DEMON STEP
def demon_velocity_move(velocities, kinetic, Ed, i, delta_v_max, mass):
    """
    Attempt to change velocity of particle i using the demon.
    Returns updated velocities, kinetic, Ed, and acceptance flag.
    """
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
    """
    Run one Demon algorithm simulation for a given initial demon energy.
    Returns:
        T: temperature from average demon energy
        mean_ke_per_particle: average kinetic energy per particle
        initial_positions, final_positions, final_velocities
        list of speeds for histogram and demon energies
    """
    initial_positions, velocities, kinetic = initialise_system(T_init)
    positions = initial_positions.copy()
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
            # record speeds occasionally
            if step % 50 == 0:
                sp = np.linalg.norm(velocities, axis=1)
                speeds.extend(sp)

    T_mean = np.mean(demon_energies)
    mean_ke = np.mean(kinetic_energies)
    mean_ke_per_particle = mean_ke / N
    return T_mean, mean_ke_per_particle,initial_positions, positions, velocities, np.array(speeds), demon_energies


def maxwell_boltzmann_2d(v, T, mass=1.0):
    """Normalised speed distribution for 2D ideal gas."""
    return (mass * v / T) * np.exp(-mass * v**2 / (2 * T))

def analyse_demon_distribution(demon_energies, T_means):
    fig, ax = plt.subplots(figsize=(7, 5))
    # histogram (normalised to probability density)
    counts, bins, patches = ax.hist(demon_energies, bins=50, density=True,
                                    alpha=0.6, label='Simulation data')
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Fit exponential: P(E) = (1/T) * exp(-E/T)
    # We fit a curve of the form a * exp(-b * E) and check b ≈ 1/T
    def exp_func(x, a, b):
        return a * np.exp(-b * x)

    popt, _ = curve_fit(exp_func, bin_centers, counts, p0=[1.0, 1.0/T_means])
    a_fit, b_fit = popt
    x_fit = np.linspace(0, max(bin_centers), 200)
    y_fit = exp_func(x_fit, a_fit, b_fit)

    # Theoretical curve: (1/T) * exp(-E/T)
    y_theory = (1.0 / T_means) * np.exp(-x_fit / T_means)

    ax.plot(x_fit, y_fit, 'r-', label=f'Exponential fit: $b = {b_fit:.3f}$')
    ax.plot(x_fit, y_theory, 'k--', label=f'Theory: $1/T \\cdot e^{{-E/T}}$, $T={T_means:.2f}$')
    ax.set_xlabel('Demon energy $E_d$', fontsize=12)
    ax.set_ylabel('Probability density $P(E_d)$', fontsize=12)
    ax.set_title('Demon Energy Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('demon_energy_distribution.pdf', dpi=150)
    plt.show()

    print(f"\nDemon energy distribution analysis:")
    print(f"Measured temperature from <Ed> = {T_means:.4f}")
    print(f"Fitted exponential decay constant b = {b_fit:.4f}")
    print(f"Theoretical b (1/T) = {1.0/T_means:.4f}")
    print(f"Ratio b / (1/T) = {b_fit * T_means:.4f} (should be ≈ 1)")


def energy_vs_temperature():
    """
    Runs 20 simulations with values of Ed from 0.5 to 10.5 in a 0.5 interval. 
    Saves data to ideal_gas_data.txt.
    """
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
            T, ke, _, _, _, _,_ = run_simulation(Ed)
            t_vals.append(T)
            e_vals.append(ke)
        temps[idx] = np.mean(t_vals)
        ke_per_particle[idx] = np.mean(e_vals)
        errors[idx] = np.std(e_vals) / np.sqrt(n_repeats)

    # Theoretical line: <K>/N = T (since d=2, mass=1, k_B=1)
    T_theory = np.linspace(0, 10, 100)
    theory_line = T_theory

      # Save data to file
    with open('ideal_gas_data.txt', 'w') as f:
        f.write("# T (demon)   <K>/N   error\n")
        for t, ke, err in zip(temps, ke_per_particle, errors):
            f.write(f"{t:.5f}   {ke:.6f}   {err:.6f}\n")
    print("Simulation completed. Data saved to ideal_gas_data.txt")

    plt.figure(figsize=(7, 5))
    plt.errorbar(temps, ke_per_particle, yerr=errors, fmt='o', capsize=3,
                 label='Demon simulation', color='blue', ecolor='gray')
    plt.plot(T_theory, theory_line, 'r-', label='Equipartition: ( K )/N = T$')
    plt.xlabel('Temperature $T$ (from demon)', fontsize=12)
    plt.ylabel('Average kinetic energy per particle ( K )/N$', fontsize=12)
    plt.title('Demon Algorithm: Energy vs Temperature for 2D Ideal Gas')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('energy_vs_temperature.pdf', dpi=150)
    plt.show()
    
def plot_positions(init_positions, final_positions, T_final):
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

def generate_speed_distribution(T_final, speeds):
    print("Generating speed distribution")
    T_fixed = T_final
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
    
def main():
    
    Ed_demo = 2.0   # gives T ~ 2.0

    #Run the simulation once with Ed = 2.0
    T_final, mean_ke, init_positions, final_positions, velocities, speeds, demon_energies = run_simulation(Ed_demo)

    # ---- 1) Plot the initial and final positions ----
    plot_positions(init_positions, final_positions, T_final)

    # ---- 2) Energy vs Temperature over a range of demon energies (Run this only once) , then use the saved data on the ideal_gas_data.txt for later use ----
    energy_vs_temperature()

    # ---- 3) Speed distribution at a representative temperature ----
    # Use the run with Ed_init = 2.0 (already performed)
    # We already have 'speeds' and t_final from that run
    generate_speed_distribution(T_final, speeds)

    # ---- 4) Speed distribution at a representative temperature ----
    analyse_demon_distribution(demon_energies, T_final)

if __name__ == "__main__":
    main()
