import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')
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
dt = 0.1          # Zeitschritt [s]
t_max = 5000        # max. Zeit [s]
n_steps = int(t_max / dt)
r_max = 30        # maximale Kettenlänge

# === Initialisierung ===
I = I0
M = M0
Y0 = 5e-8  # Anfangswert für Y0 nicht 0, um Division durch Null zu vermeiden
YT0 = 0
QT_0 = 0.0
Q_0 = 0.0
TP_r = np.zeros(r_max + 1)
TP_r[0] = CTA0

P_r = np.zeros(r_max + 1)
d_r = np.zeros(r_max + 1)

# === Hilfsfunktionen ===
def solve_Y0(I, QT_0):
    def eq(Y):
        return (Y**3
                + ((2 * k_a * k_ct * QT_0 + k_f * (k_td + k_tc)) / (k_ct * (k_td + k_tc))) * Y**2
                - (2 * f * k_d * I) / (k_td + k_tc) * Y
                - k_f * 2 * f * k_d * I / (k_ct * (k_td + k_tc)))
    return fsolve(eq, 1e-8)[0]

def solve_YT0(Y0, I):
    return (2 * f * k_d * I - (k_td + k_tc) * Y0**2) / (2 * k_ct * Y0) if Y0 != 0 else 0.0

def compute_P_T_rs(Y0, P_r, TP_r):
    P_T_rs = np.zeros((r_max + 1, r_max + 1))
    d_r = np.zeros(r_max + 1)
    for r in range(r_max + 1):
        d_sum = 0.0
        for s in range(r_max + 1):
            numer = k_a * P_r[r] * TP_r[s] + k_a * P_r[s] * TP_r[r]
            denom = k_f + k_ct * Y0
            P_T_rs[r, s] = numer / denom if denom != 0 else 0.0
            d_sum += P_T_rs[r, s]
        d_r[r] = d_sum
    return d_r

def rhs(t, y):
    # Entpacken des Zustandsvektors
    I = y[0]
    M = y[1]
    QT_0 = y[2]
    Q_0 = y[3]
    P_r = y[4:4 + r_max + 1]
    TP_r = y[4 + r_max + 1:4 + 2 * (r_max + 1)]

    # Algebraische Variablen bestimmen
    Y0 = solve_Y0(I, QT_0)
    YT0 = solve_YT0(Y0, I)
    d_r = compute_P_T_rs(Y0, P_r, TP_r)

    # Ableitungen berechnen
    dI = -k_d * I
    dM = -k_p * M * np.sum(P_r)
    dQT_0 = k_f * YT0 - k_a * Y0 * QT_0
    dQ_0 = (k_tc + k_td) * Y0**2 + k_ct * Y0 * YT0

    dP_r = np.zeros(r_max + 1)
    for r in range(r_max + 1):
        de  nom = k_p * M + k_a * QT_0 + (k_tc + k_td) * Y0 + k_ct * YT0
        if r == 0:
            numer = 2 * f * k_d * I + 0.5 * k_f * d_r[0]
        else:
            numer = k_p * M * P_r[r - 1] + 0.5 * k_f * d_r[r]
        dP_r[r] = numer / denom - P_r[r] if denom != 0 else -P_r[r]

    dTP_r = k_f * d_r - k_a * Y0 * TP_r

    # Rückgabe als flacher Vektor
    return np.concatenate(([dI, dM, dQT_0, dQ_0], dP_r, dTP_r))

# Anfangsbedingungen als Vektor
y0 = np.zeros(4 + 2 * (r_max + 1))
y0[0] = I0
y0[1] = M0
y0[2] = CTA0
y0[3] = 0.0
y0[4 + r_max + 1] = CTA0  # TP_r[0] = CTA0

t_span = (0, t_max)
t_eval = np.linspace(0, t_max, int(t_max / dt))

t_start = t.time()
sol = solve_ivp(rhs, t_span, y0, method='BDF', t_eval=t_eval, vectorized=False, rtol=1e-6, atol=1e-9)
t_end = t.time()
print(f"Simulation completed in {t_end - t_start:.2f} seconds.")

# Extrahiere Größen für Plot
M_list = sol.y[1, :]
P_r_mat = sol.y[4:4 + r_max + 1, :]
TP_r_mat = sol.y[4 + r_max + 1:4 + 2 * (r_max + 1), :]
P_r_sum_list = np.sum(P_r_mat, axis=0)
TP_r_sum_list = np.sum(TP_r_mat, axis=0)

# DPn und PD berechnen
DPn_list = []
PD_list = []
for i in range(P_r_mat.shape[1]):
    P_r = P_r_mat[:, i]
    Y1 = np.sum([r * P_r[r] for r in range(r_max + 1)])
    Y2 = np.sum([r**2 * P_r[r] for r in range(r_max + 1)])
    DPn = Y1 / np.sum(P_r) if np.sum(P_r) > 0 else 0
    DPw = Y2 / Y1 if Y1 > 0 else 0
    PD = DPw / DPn if DPn > 0 else 0
    DPn_list.append(DPn)
    PD_list.append(PD)

# === Plotten ===
plt.plot(sol.t, DPn_list, label='DPn')
plt.plot(sol.t, PD_list, label='PD')
plt.xlabel('Time [s]')
plt.ylabel('Values')
plt.legend()
plt.title('Number Average Degree of Polymerization (DPn) and Polydispersity Index (PD)')
plt.grid()
plt.show()

plt.figure()
plt.plot(sol.t, M_list, label='[M] (Monomer)')
plt.plot(sol.t, P_r_sum_list, label='Σ[P_r] (aktive Ketten)')
plt.plot(sol.t, TP_r_sum_list, label='Σ[TP_r] (RAFT-Agent)')
plt.xlabel('Time [s]')
plt.ylabel('Concentration [mol/L]')
plt.yscale('log')
plt.legend()
plt.title('Verlauf der Spezies über die Zeit')
plt.grid()
plt.show()
