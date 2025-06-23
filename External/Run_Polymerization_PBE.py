import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Importiere Parameter und Funktionen aus Run_Polymerization
from Run_Polymerization import (
    V0, M0, I0, Mjm, T, Rcte, fkp, fktc, t0, tf, Nt
)   #Import parameters and reaction radte functions

# === Binning-Setup ===
N_bins = 100
max_chain_length = 4000
bin_edges = np.linspace(1, max_chain_length + 1, N_bins + 1, dtype=int)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:] - 1)

InputVar = np.zeros(3 + N_bins, dtype=np.float64)
InputVar[0] = V0
InputVar[1] = M0
InputVar[2] = I0
InputVar[3:] = 1e-20

def get_bin_index(chain_length):
    idx = np.searchsorted(bin_edges, chain_length, side='right') - 1
    if idx < 0:
        idx = 0
    if idx >= N_bins:
        idx = N_bins - 1
    return idx

def reac(t, Y):
    global Vantes
    V = max(Y[0], 1e-12)    #volume
    M = max(Y[1], 0.0)  #monomer concentration
    I = max(Y[2], 0.0)  #initiator concentration
    P_dead = np.maximum(Y[3:], 0.0) #avoid negative concentrations
    # Monomer/Polymer properties
    rhom = (0.968 - 1.225e-3 * (T - 273.15)) * 1000
    M0_local = rhom / Mjm
    e = 0.183 + 9e-4 * (T - 273.15)
    rhop = rhom * (1 + e)
    mm = M * V * rhom
    mm0 = M0_local * V0 * rhom
    mp = mm0 - mm
    wm = mm / (mm + mp + 1e-20) #monomer weight fraction
    phip = (mp / rhop) / (mp / rhop + mm / rhom + 1e-20)    #polymer volume fraction
    # Kinetik
    f = 0.58
    kp = fkp(T, 1e-8, wm, phip)
    ktc = fktc(T, 1e-8, phip, 1000, wm, kp, M)
    kf = 4.66e9 * np.exp(-76290 / (8.314 * T)) * 60
    ktd = ktc * (3.956e-4 * np.exp(4090 / (Rcte * T)))
    kd = 6.32e16 * np.exp(-30660 / (Rcte * T))
    kt = ktd + ktc
    dI_dt = -kd * I
    R_gen = 2 * f * kd * I
    lambda0 = np.sqrt(max(R_gen * M, 0) / (kt + 1e-20))
    dV = Vantes - V
    dy = np.zeros_like(Y)
    dy[0] = -V0 * e * (R_gen + kp * M * lambda0) * (1 - ((M0_local * V0 - M * V) / (M0_local * V0 + 1e-20)))    # Volumenänderung
    dy[1] = -R_gen * M - kp * M * lambda0 - (M / (V + 1e-20)) * dV  # Monomerkonzentration
    dy[2] = dI_dt - (I / (V + 1e-20)) * dV  # Initiator-Konzentration
    L = int(np.round(kp * M / (kt * lambda0 + 1e-20)))  # Kettenlänge (Anzahl Monomereinheiten pro Kette)
    if L < 1 or not np.isfinite(L): L = 1
    if L > max_chain_length: L = max_chain_length
    term_rate = kt * lambda0 ** 2
    if not np.isfinite(term_rate) or term_rate < 0:
        term_rate = 0.0

    # Verteile die Terminationsrate auf 3 Bins um L (optional, verbessert die Numerik)
    spread = 1  # 1 = auf 3 Bins verteilen (L-1, L, L+1)
    for i in range(-spread, spread + 1):
        idx = get_bin_index(L + i)
        dy[3 + idx] += term_rate / (2 * spread + 1)

    for n in range(N_bins):
        if np.isfinite(P_dead[n]) and np.isfinite(V) and np.isfinite(dV):
            dy[3 + n] -= (P_dead[n] / (V + 1e-20)) * dV
        else:
            dy[3 + n] += 0.0  # keine Änderung bei NaN/Inf

    Vantes = V  # Update des vorherigen Volumens für die nächste Iteration
    # Falls NaN/Inf entstehen, alles auf 0 setzen (Debug)
    if not np.all(np.isfinite(dy)):
        print(f"NaN/Inf detected at t={t}, Y={Y}")
        dy[:] = 0.0
    return dy

Vantes = V0

# Nutze solve_ivp mit BDF (steif) und feiner Zeitauflösung
t_eval = np.linspace(t0, tf, Nt)
sol = solve_ivp(reac, (t0, tf), InputVar, method='BDF', t_eval=t_eval, atol=1e-12, rtol=1e-8)

Y = sol.y.T

P_dead_bins = Y[-1, 3:]
Mn = np.sum(bin_centers * P_dead_bins) / (np.sum(P_dead_bins) + 1e-20)
Mw = np.sum((bin_centers ** 2) * P_dead_bins) / (np.sum(bin_centers * P_dead_bins) + 1e-20)
PDI = Mw / (Mn + 1e-20)
conversion = (M0 * V0 - Y[-1, 1] * Y[-1, 0]) / (M0 * V0)

# Gesamtkonzentration an Polymerketten (mol/L)
total_polymer_chains = np.sum(P_dead_bins)
# Gesamte Monomereinheiten in Polymeren (mol/L)
total_monomer_units_in_polymer = np.sum(bin_centers * P_dead_bins)

print(f"Endkonversion: {conversion:.3f}")
print(f"Mn: {Mn:.1f}, Mw: {Mw:.1f}, PDI: {PDI:.2f}")
print(f"Gesamtkonzentration Polymerketten: {total_polymer_chains:.4e} mol/L")
print(f"Gesamte Monomereinheiten in Polymeren: {total_monomer_units_in_polymer:.4e} mol/L")
print(f"Start-Monomerkonzentration: {M0:.4e} mol/L")

plt.figure(figsize=(8, 4))
plt.bar(bin_centers, P_dead_bins, width=(bin_edges[1] - bin_edges[0]))
plt.xlabel("Kettenlänge (binned)")
plt.ylabel("Konzentration [mol/L]")
plt.title("Kettenlängenverteilung (tot, gebinnt) am Ende")
plt.tight_layout()
plt.show()