import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time  # <-- hinzugefügt

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

reac_call_count = 0
reac_total_time = 0.0

# Arrays zur Speicherung von kp, ktc und lambda0
kp_array = np.zeros(len(np.linspace(t0, tf, Nt)))
ktc_array = np.zeros(len(np.linspace(t0, tf, Nt)))
lambda0_array = np.zeros(len(np.linspace(t0, tf, Nt)))  # <-- hinzugefügt

def reac(t, Y):
    global Vantes, reac_call_count, reac_total_time
    t_start = time.perf_counter()
    V = max(Y[0], 1e-12)
    M = max(Y[1], 1e-20)
    I = max(Y[2], 1e-20)
    P_dead = np.maximum(Y[3:], 0.0)

    # Momente aus der Populationsbilanz berechnen
    mu0 = np.sum(P_dead)
    mu1 = np.sum(bin_centers * P_dead)
    mu2 = np.sum((bin_centers ** 2) * P_dead)
    mu3 = np.sum((bin_centers ** 3) * P_dead)
    mu4 = np.sum((bin_centers ** 4) * P_dead)

    # Polymerparameter wie in Run_Polymerization
    rhom = (0.968 - 1.225e-3 * (T - 273.15)) * 1000
    M0_local = rhom / Mjm
    e = 0.183 + 9e-4 * (T - 273.15)
    rhop = rhom * (1 + e)
    mm = M * V * rhom
    mm0 = M0_local * V0 * rhom
    mp = mm0 - mm
    wm = mm / (mm + mp + 1e-20)
    phip = (mp / rhop) / (mp / rhop + mm / rhom + 1e-20)

    # Mw wie in Run_Polymerization
    Mw = mu2 / (mu1 + 1e-20)

    f = 0.58

    # Fixpunktiteration für lambda0 und kt wie in Run_Polymerization
    kd = 6.32e16 * np.exp(-30660 / (Rcte * T))
    kf = 4.66e9 * np.exp(-76290 / (8.314 * T)) * 60
    ki_PR = 2 * f * kd * I / (M + 1e-20)
    lambda0_fp = 1.e-8
    tol = 1e-10
    max_iter = 20
    for _ in range(max_iter):
        kp_fp = fkp(T, max(lambda0_fp, 1e-20), wm, phip)
        Mw_fp = mu2 / (mu1 + 1e-20)
        ktc_fp = fktc(T, max(lambda0_fp, 1e-20), phip, Mw_fp, wm, kp_fp, M)
        ktd_fp = ktc_fp * (3.956e-4 * np.exp(4090 / (Rcte * T)))
        kt_fp = ktd_fp + ktc_fp
        if not np.isfinite(kt_fp) or kt_fp <= 0:
            kt_fp = 1e-12
        if ki_PR > 0:
            lambda0_new = np.sqrt(ki_PR * M / (kt_fp + 1e-20))
        else:
            lambda0_new = 1.e-20
        if not np.isfinite(lambda0_new) or lambda0_new <= 0:
            lambda0_new = 1.e-20
        if abs(lambda0_new - lambda0_fp) < tol:
            break
        lambda0_fp = lambda0_new
    lambda0 = max(lambda0_fp, 1e-20)
    kp = max(kp_fp, 1e-20)
    ktc = max(ktc_fp, 1e-20)
    ktd = max(ktd_fp, 0.0)
    kt = max(kt_fp, 1e-20)

    dI_dt = -kd * I
    R_gen = 2 * f * kd * I
    dV = Vantes - V
    dy = np.zeros_like(Y)
    dy[0] = -V0 * e * (R_gen + kp * M * lambda0) * (1 - ((M0_local * V0 - M * V) / (M0_local * V0 + 1e-20)))
    dy[1] = -R_gen * M - kp * M * lambda0 - (M / (V + 1e-20)) * dV
    dy[2] = dI_dt - (I / (V + 1e-20)) * dV
    denom = kt * lambda0 + 1e-20
    if not np.isfinite(denom) or denom <= 0:
        denom = 1e-20
    L_float = kp * M / denom
    if not np.isfinite(L_float) or L_float <= 0:
        L_float = 1
    L = int(np.round(L_float))
    if L < 1: L = 1
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
    t_end = time.perf_counter()
    if reac_call_count < 100:
        reac_total_time += (t_end - t_start)
    reac_call_count += 1

    # kp, ktc und lambda0 für Plot speichern (Index anhand von t_eval)
    if hasattr(reac, "step_idx"):
        idx = reac.step_idx
        if idx < len(kp_array):
            kp_array[idx] = kp
            ktc_array[idx] = ktc
            lambda0_array[idx] = lambda0
        reac.step_idx += 1
    else:
        reac.step_idx = 1
        kp_array[0] = kp
        ktc_array[0] = ktc
        lambda0_array[0] = lambda0

    return dy

Vantes = V0

# Zeitmessung für solve_ivp
t0_solve = time.perf_counter()
t_eval = np.linspace(t0, tf, Nt)
# Reset step_idx für reac
if hasattr(reac, "step_idx"):
    del reac.step_idx
sol = solve_ivp(
    reac, (t0, tf), InputVar,
    method='BDF', t_eval=t_eval,
    atol=1e-8, rtol=1e-6  # <- Toleranzen gelockert
)
t1_solve = time.perf_counter()

print(f"solve_ivp Laufzeit: {t1_solve - t0_solve:.2f} Sekunden")
if reac_call_count > 0:
    print(f"Mittlere reac()-Laufzeit (erste 100 Aufrufe): {reac_total_time / min(reac_call_count,100):.6f} Sekunden")

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

# Nach der Integration: Plots nur mit den Arrays aus der ODE-Funktion
# (kp_array, ktc_array, lambda0_array werden bereits in reac() gefüllt und sind identisch zur ODE-Berechnung)

# Plot für kp über die Zeit
plt.figure()
plt.plot(t_eval, kp_array, label="kp")
plt.xlabel("Zeit [min]")
plt.ylabel("Wachstumsrate $k_p$ [L/(mol·min)]")
plt.title("Verlauf von $k_p$ über die Zeit")
plt.legend()
plt.tight_layout()
plt.yscale("log")
plt.show()

# Plot für ktc über die Zeit
plt.figure()
plt.plot(t_eval, ktc_array, label="ktc")
plt.xlabel("Zeit [min]")
plt.ylabel("Terminierungsrate $k_{tc}$ [L/(mol·min)]")
plt.title("Verlauf von $k_{tc}$ über die Zeit")
plt.legend()
plt.tight_layout()
plt.yscale("log")
plt.show()

# Plot für lambda0 über die Zeit
plt.figure()
plt.plot(t_eval, lambda0_array, label="lambda0")
plt.xlabel("Zeit [min]")
plt.ylabel(r"$\lambda_0$ [mol/L]")
plt.title(r"Verlauf von $\lambda_0$ über die Zeit")
plt.legend()
plt.tight_layout()
plt.show()
