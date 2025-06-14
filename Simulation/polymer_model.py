import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the model parameters (to be adjusted externally)
N_MAX = 200  # maximum chain length
k_d = 1e-6   # initiator decomposition rate [1/s]
k_i = 5e2    # initiation rate [L/mol/s]
k_p = 2e3    # propagation rate [L/mol/s]
k_t = 5e5    # termination rate [L/mol/s]
k_tr = 1e1   # chain transfer to monomer [L/mol/s]
f = 0.5      # initiator efficiency

# Initial concentrations [mol/L]
M0 = 5.0     # monomer
I0 = 0.01    # initiator

# Indexing convention:
# y[0] = [M], y[1] = [I], y[2] = [R*], y[3:3+N_MAX] = [P1*, ..., PN*]
# y[3+N_MAX:3+2*N_MAX] = [P1, ..., PN] (dead polymers)

def polymer_pbe_rhs(t, y):
    dydt = np.zeros_like(y)
    M, I, R = y[0], y[1], y[2]
    P_rad = y[3:3 + N_MAX]
    P_dead = y[3 + N_MAX:]

    # Initiator decomposition
    dI_dt = -k_d * I
    R_gen = 2 * f * k_d * I

    # Initiation (R + M -> P1*)
    init = k_i * R * M

    # Propagation (Pn* + M -> Pn+1*)
    prop = k_p * P_rad * M

    # Termination (Pn* + Pm* -> dead polymer)
    term_matrix = np.outer(P_rad, P_rad)
    np.fill_diagonal(term_matrix, 0.5 * np.diag(term_matrix))  # avoid double-counting
    term_rates = k_t * term_matrix
    total_termination = np.sum(term_rates)

    # Chain transfer (Pn* + M -> Pn + R*)
    trans = k_tr * P_rad * M

    # === Fill RHS ===
    dydt[0] = -init - np.sum(prop) - np.sum(trans)  # d[M]/dt
    dydt[1] = dI_dt                                 # d[I]/dt
    dydt[2] = R_gen - init + np.sum(trans)          # d[R*]/dt

    for n in range(N_MAX):
        # d[Pn*]/dt
        growth = prop[n - 1] if n > 0 else init
        loss = prop[n] + trans[n]
        term_loss = k_t * P_rad[n] * np.sum(P_rad)
        dydt[3 + n] = growth - loss - term_loss

        # d[Pn]/dt (dead polymers)
        comb_terms = term_rates[n, :] + term_rates[:, n]
        transfer = trans[n]
        dydt[3 + N_MAX + n] = np.sum(comb_terms) + transfer

    return dydt

def run_simulation(t_end=2e6, N_max=N_MAX):
    y0 = np.zeros(3 + 2 * N_max)
    y0[0] = M0
    y0[1] = I0
    y0[2] = 0.0  # initial radicals

    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 500)

    sol = solve_ivp(polymer_pbe_rhs, t_span, y0, t_eval=t_eval,
                    method='BDF', events=stop_at_conversion)
    return sol


def compute_polymer_stats(y):
    P_dead = y[3 + N_MAX:]
    indices = np.arange(1, N_MAX + 1)
    total_polymer = np.sum(P_dead)
    Mn = np.sum(indices * P_dead) / total_polymer if total_polymer > 0 else 0
    Mw = np.sum(indices**2 * P_dead) / np.sum(indices * P_dead) if total_polymer > 0 else 0
    PDI = Mw / Mn if Mn > 0 else 0
    conversion = 1 - y[0] / M0
    return Mn, Mw, PDI, conversion

def stop_at_conversion(t, y):
    conversion = 1 - y[0] / M0
    return conversion - 0.8  # Stop when conversion reaches 80%

stop_at_conversion.terminal = True
stop_at_conversion.direction = 1


def plot_chain_length_distribution(y):
    P_dead = y[3 + N_MAX:]
    P_rad = y[3:3 + N_MAX]
    indices = np.arange(1, N_MAX + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(indices, P_dead, label="Dead polymers", linewidth=2)
    plt.plot(indices, P_rad, label="Radical chains", linestyle='--')
    plt.xlabel("Chain length")
    plt.ylabel("Concentration [mol/L]")
    plt.title("Chain Length Distribution at Final Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_chain_length_evolution(result, times=[100, 500, 1000, 5000, 10000]):
    indices = np.arange(1, N_MAX + 1)
    time_indices = [np.abs(result.t - t).argmin() for t in times if t <= result.t[-1]]

    plt.figure(figsize=(10, 6))
    for idx in time_indices:
        y = result.y[:, idx]
        P_dead = y[3 + N_MAX:]
        plt.plot(indices, P_dead, label=f"t = {result.t[idx]:.0f} s")

    plt.xlabel("Chain Length")
    plt.ylabel("Concentration [mol/L]")
    plt.title("Evolution of Chain Length Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


def plot_conversion_vs_time(result):
    M_vals = result.y[0, :]
    conversion = 1 - M_vals / M0

    plt.figure(figsize=(8, 4))
    plt.plot(result.t, conversion, linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Conversion")
    plt.title("Monomer Conversion Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


if __name__ == "__main__":
    result = run_simulation()
    final_state = result.y[:, -1]
    Mn, Mw, PDI, conv = compute_polymer_stats(final_state)
    print(f"Final conversion: {conv:.3f}")
    print(f"Mn: {Mn:.1f}, Mw: {Mw:.1f}, PDI: {PDI:.2f}")

    plot_chain_length_distribution(final_state)
    plot_chain_length_evolution(result, times=[100, 1000, 5000, 10000])
    plot_conversion_vs_time(result)

plt.show()