import numpy as np
from scipy.optimize import fsolve
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
import time as t

# === Parameter aus Tabelle 2 (IRT & IRTO) ===
k_d = 1e-5         # Initiatordissoziation [1/s]
k_p = 1e3          # Propagation [L/mol/s]
k_td = 1e7         # Termination durch Disproportionierung [L/mol/s]
k_tc = 1e7         # Termination durch Kombination [L/mol/s]
k_ct = 1e7         # Cross-Termination [L/mol/s]
k_a  = 1e6         # Addition [L/mol/s]
k_f  = 1e4         # Fragmentierung [1/s]
f    = 0.5         # Initiatoreffizienz

# === Anfangskonzentrationen ===
I0  = 5e-3         # Initiator [mol/L]
M0  = 5.0          # Monomer [mol/L]
CTA0 = 1e-2        # RAFT-Agent (alle TPs) [mol/L]

# === Simulationseinstellungen ===
dt = 0.01           # Zeitschritt [s]
t_max = 10        # max. Zeit [s]
n_steps = int(t_max / dt)
r_max = 300        # maximale Kettenlänge

# === Zustandsvariablen ===
I = I0
M = M0
QT_0 = 0.0
Q_0 = 0.0
TP_r = np.zeros(r_max + 1)  # Dormante Ketten TPr
TP_r[0] = CTA0

Y0 = 0.0  # initialer Guess
YT_0 = 0.0

# Arrays zur Speicherung der Konzentrationen
P_r = np.zeros(r_max + 1)             # Lebende Ketten P*
d_r = np.zeros(r_max + 1)             # partielle Momente d(r)
P_T_rs = np.zeros((r_max + 1, r_max + 1))  # Adduktradikale PrT*Ps

# === Hilfsfunktionen ===
def solve_Y0(I, QT_0):
    def eq(Y):
        term1 = Y**3
        term2 = ((2 * k_a * k_ct * QT_0 + k_f * (k_td + k_tc)) / (k_ct * (k_td + k_tc))) * Y**2
        term3 = -(2 * f * k_d * I) / (k_td + k_tc) * Y
        term4 = -k_f * 2 * f * k_d * I / (k_ct * (k_td + k_tc))
        return term1 + term2 + term3 + term4

    Y0_guess = 1e-8
    Y0_solution = fsolve(eq, Y0_guess)[0]
    return Y0_solution

def solve_YT0(Y0, I):
    return (2 * f * k_d * I - (k_td + k_tc) * Y0**2) / (2 * k_ct * Y0)

def update_Pr(Y0, YT_0):
    global P_r
    P_r_old = P_r.copy()
    for r in range(r_max + 1):
        if r == 0:
            numerator = 2 * f * k_d * I + 0.5 * k_f * d_r[0]
            denom = k_p * M + k_a * QT_0 + (k_tc + k_td) * Y0 + k_ct * YT_0
            P_r[r] = numerator / denom
        else:
            numerator = k_p * M * P_r_old[r - 1] + 0.5 * k_f * d_r[r]
            denom = k_p * M + k_a * QT_0 + (k_tc + k_td) * Y0 + k_ct * YT_0
            P_r[r] = numerator / denom

def update_P_T_rs(Y0):
    global P_T_rs, d_r
    for r in range(r_max + 1):
        d_sum = 0.0
        for s in range(r_max + 1):
            try:
                numer = k_a * P_r[r] * TP_r[s] + k_a * P_r[s] * TP_r[r]
                denom = k_f + k_ct * Y0
                if not np.isfinite(numer) or not np.isfinite(denom):
                    print(f"[Overflow Warning] r={r}, s={s}, P_r[r]={P_r[r]:.2e}, TP_r[s]={TP_r[s]:.2e}, P_r[s]={P_r[s]:.2e}, TP_r[r]={TP_r[r]:.2e}, numer={numer:.2e}, denom={denom:.2e}")
                P_T_rs[r, s] = numer / denom
                d_sum += P_T_rs[r, s]
            except FloatingPointError as e:
                print(f"FloatingPointError at r={r}, s={s}: {e}")
        d_r[r] = d_sum

    if np.any(np.isnan(d_r)) or np.any(np.isinf(d_r)):
        print("[NaN in d_r] after update_P_T_rs")
        for r in range(r_max + 1):
            if np.isnan(d_r[r]) or np.isinf(d_r[r]):
                print(f"r={r}, d_r={d_r[r]:.2e}")

def update_TP_r():
    global TP_r
    dTP = k_f * d_r - k_a * Y0 * TP_r

    if np.any(np.isnan(dTP)) or np.any(np.isinf(dTP)):
        print("[NaN in dTP] Detected at TP_r update:")
        for r in range(r_max + 1):
            if np.isnan(dTP[r]) or np.isinf(dTP[r]):
                print(f"r={r}, d_r={d_r[r]:.2e}, TP_r={TP_r[r]:.2e}, Y0={Y0:.2e}, dTP={dTP[r]:.2e}")

    TP_r[:] += dt * dTP

def update_QT0():
    global QT_0
    dQT0 = k_f * YT_0 - k_a * Y0 * QT_0
    QT_0 += dt * dQT0

def update_Q0():
    global Q_0
    dQ0 = (k_tc + k_td) * Y0**2 + k_ct * Y0 * YT_0
    Q_0 += dt * dQ0

def update_I():
    global I
    I -= dt * k_d * I

def update_M():
    global M
    M -= dt * k_p * M * np.sum(P_r)

# === Hauptzeitschleife ===
time_vals = []
DPn_list = []
PD_list = []

t_start = t.time()

for step in range(n_steps):
    if step % 10 == 0:
        print(f"Step {step}/{n_steps}")

    Y0 = solve_Y0(I, QT_0)
    YT_0 = solve_YT0(Y0, I)

    update_P_T_rs(Y0)
    update_Pr(Y0, YT_0)
    update_TP_r()
    update_QT0()
    update_Q0()
    update_I()
    update_M()

    if np.any(np.isnan(P_r)) or np.any(np.isnan(TP_r)):
        print(f"[NaN DETECTED] at step {step}, breaking simulation.")
        break

    time_vals.append(step * dt)
    Y1 = np.sum([r * P_r[r] for r in range(r_max + 1)])
    Y2 = np.sum([r**2 * P_r[r] for r in range(r_max + 1)])
    DPn = Y1 / np.sum(P_r)
    DPw = Y2 / Y1 if Y1 > 0 else 0
    PD = DPw / DPn if DPn > 0 else 0
    DPn_list.append(DPn)
    PD_list.append(PD)

t_end = t.time()
print(f"Simulation completed in {t_end - t_start:.2f} seconds.")

# === Plotten ===
plt.plot(time_vals, DPn_list, label='DPn')
plt.plot(time_vals, PD_list, label='PD')
plt.xlabel('Time [s]')
plt.ylabel('Values')
plt.legend()
plt.title('Number Average Degree of Polymerization (DPn) and Polydispersity Index (PD)')
plt.grid()
plt.show()
