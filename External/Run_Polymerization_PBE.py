import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===== Parameter wie in Run_Polymerization =====
Rcte = 1.982
T = 70 + 273.15
V0 = 1
Mjm = 100.13
rhom = (0.968 - 1.225e-3 * (T - 273.15)) * 1000
Vm_esp = 0.822
Mjp = 150
Vp_esp = 0.77
Mji = 68
Vi_esp = 0.913
jc0 = 0.874
delta = 6.9e-8
Na = 6.032e23

I0 = 0.01548
M0 = rhom / Mjm

tf = 100  # [min]
Nt = 8 * tf
t_eval = np.linspace(0, tf, Nt)

# ===== Kettenlängen-Binning =====
Nmax = 10000
n_bins = 200
bin_edges = np.linspace(1, Nmax+1, n_bins+1, dtype=int)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:] - 1)

# ===== Kinetik wie in Run_Polymerization =====
def fkp(T, lambda0, wm, phip):
    kp0 = 4.92e5 * np.exp(-4353 / (Rcte * T)) * 60
    invjc = 1 / jc0 + 2 * phip / 100
    jc = 1 / invjc
    tau = np.sqrt(3 / (2 * jc * (delta ** 2)))
    rt = np.sqrt(np.log(1000 * (tau ** 3) / (Na * lambda0 * (3.1415 ** 1.5)))) / tau
    rm = rt
    wp = 1 - wm
    gamma = 0.763
    Vfm = 0.149 + 2.9e-4 * (T - 273.15)
    Vfp = 0.0194 + 1.3e-4 * (T - 273.15 - 105)
    Vf = wm * Vm_esp * Vfm + wp * Vp_esp * Vfp
    factor = -gamma * Vm_esp * Mjm * (wm / Mjm + wp / Mjp) / Vf
    Dm0 = 0.827e-10
    Dm = Dm0 * np.exp(factor) * 60
    tauDp = (rm ** 2) / (3 * Dm)
    tauRp = 1 / (kp0 * lambda0)
    kp = kp0 / (1 + tauDp / tauRp)
    return kp

def fktc(T, lambda0, phip, Mw, wm, kp, M):
    ktc0 = 9.80e7 * np.exp(-701.0 / (Rcte * T)) * 60
    Mwlin = 4293.76
    kB = 1.3806e-23
    RH = 1.3e-9 * (Mwlin ** 0.574)
    etas = np.exp(-0.099 + (496 / T) - 1.5939 * np.log(T))
    Dplin = kB * T / (6 * 3.1415 * etas * RH) * 1e6 * 60
    wp = 1 - wm
    gamma = 0.763
    Vfm = 0.149 + 2.9e-4 * (T - 273.15)
    Vfp = 0.0194 + 1.3e-4 * (T - 273.15 - 105)
    Vf = wm * Vm_esp * Vfm + wp * Vp_esp * Vfp
    epsillon13 = Vi_esp * Mji / (Vp_esp * Mjp)
    factor = -(gamma / epsillon13) * (wm * Vm_esp + wp * Vp_esp * epsillon13) / Vf - (1 / Vfm)
    Dp = Dplin * ((Mwlin ** 2) / (Mw ** 2 + 1e-20)) * np.exp(factor)
    Fseg = 0.12872374
    Dpe = Fseg * Dp
    invjc = 1 / jc0 + 2 * phip / 100
    jc = 1 / invjc
    tau = np.sqrt(3 / (2 * jc * (delta ** 2)))
    rt = np.sqrt(np.log(1000 * (tau ** 3) / (Na * lambda0 * (3.1415 ** 1.5)))) / tau
    rm = rt
    tauRt = 1 / (ktc0 * lambda0)
    tauDt = (rm ** 2) / (3 * Dpe)
    ktc = ktc0 / (1 + tauDt / tauRt)
    A = 8 * 3.1415 * (delta ** 3) * np.sqrt(jc) * Na / (3000)
    ktres = A * kp * M
    kte = ktc + ktres
    return kte

# ===== PBE-Rechte Seite =====
def pbe_rhs(t, y):
    # y = [M, I, P_rad[0:n_bins], P_dead[0:n_bins]]
    M, I = y[0], y[1]
    P_rad = y[2:2+n_bins]
    P_dead = y[2+n_bins:]

    # Momente für Ratenberechnung
    mu0 = np.sum(P_dead)
    mu1 = np.sum(bin_centers * P_dead)
    mu2 = np.sum((bin_centers**2) * P_dead)
    mu3 = np.sum((bin_centers**3) * P_dead)
    mu4 = np.sum((bin_centers**4) * P_dead)
    Mw = mu2 / (mu1 + 1e-20)

    mm = M * V0 * rhom
    mm0 = M0 * V0 * rhom
    mp = mm0 - mm
    wm = mm / (mm + mp)
    rhop = rhom * (1 + 0.183 + 9e-4 * (T - 273.15))
    phip = (mp / rhop) / (mp / rhop + mm / rhom)

    # Kinetik
    f = 0.58
    kd = 6.32e16 * np.exp(-30660 / (Rcte * T))
    kf = 4.66e9 * np.exp(-76290 / (8.314 * T)) * 60
    ki_PR = 2 * f * kd * I / (M + 1e-20)
    # Fixpunktiteration für lambda0, kp, kt
    lambda0_fp = 1e-8
    for _ in range(10):
        kp_fp = fkp(T, lambda0_fp, wm, phip)
        ktc_fp = fktc(T, lambda0_fp, phip, Mw, wm, kp_fp, M)
        ktd_fp = ktc_fp * (3.956e-4 * np.exp(4090 / (Rcte * T)))
        kt_fp = ktd_fp + ktc_fp
        if ki_PR > 0:
            lambda0_new = np.sqrt(ki_PR * M / (kt_fp + 1e-20))
        else:
            lambda0_new = 1e-20
        if abs(lambda0_new - lambda0_fp) < 1e-10:
            break
        lambda0_fp = lambda0_new
    lambda0 = lambda0_fp
    kp = kp_fp
    ktc = ktc_fp
    ktd = ktd_fp
    kt = kt_fp

    # Kettentransfer (nur zu Monomer)
    # Termination: Kombination + Disproportionierung
    # Initiatorzerfall
    dI_dt = -kd * I
    R_gen = 2 * f * kd * I

    # Population balances
    dydt = np.zeros_like(y)
    # Monomerbilanz
    dydt[0] = -kp * M * np.sum(P_rad) - kf * M * np.sum(P_rad)
    # Initiatorbilanz
    dydt[1] = dI_dt

    # Radikalische Ketten
    for i in range(n_bins):
        n = bin_centers[i]
        # Wachstum
        growth = kp * M * P_rad[i-1] if i > 0 else R_gen
        loss = kp * M * P_rad[i]
        # Kettentransfer
        transfer = kf * M * P_rad[i]
        # Termination (Summe über alle Radikale)
        term_comb = ktc * P_rad[i] * np.sum(P_rad)
        term_disp = ktd * P_rad[i] * np.sum(P_rad)
        dydt[2+i] = growth - loss - transfer - term_comb - term_disp

    # Tote Ketten
    for i in range(n_bins):
        # Termination durch Kombination (Summe aller Paare, die in diesen Bin fallen)
        # Näherung: alle Kombinationen, die in diesen Bin fallen, werden gleichverteilt
        # (Genauer: Bin-zu-Bin-Matrix, hier: einfache Approximation)
        comb_sum = 0
        for j in range(n_bins):
            k = i - j
            if 0 <= k < n_bins:
                comb_sum += 0.5 * ktc * P_rad[j] * P_rad[k]
        # Disproportionierung: alle Radikale in diesen Bin
        disp_sum = ktd * P_rad[i] * np.sum(P_rad)
        # Kettentransfer: alle Radikale in diesen Bin
        transfer_sum = kf * M * P_rad[i]
        dydt[2+n_bins+i] = comb_sum + disp_sum + transfer_sum

    return dydt

# ===== Anfangswerte =====
y0 = np.zeros(2 + 2 * n_bins)
y0[0] = M0
y0[1] = I0
# Start: keine Radikale, keine toten Ketten

# ===== Integration und Speicherung von kp, kt, Conversion =====
def pbe_rhs_with_rates(t, y):
    # ...wie pbe_rhs, aber ohne Speicherung von kp, kt, Conversion...
    M, I = y[0], y[1]
    P_rad = y[2:2+n_bins]
    P_dead = y[2+n_bins:]

    mu0 = np.sum(P_dead)
    mu1 = np.sum(bin_centers * P_dead)
    mu2 = np.sum((bin_centers**2) * P_dead)
    mu3 = np.sum((bin_centers**3) * P_dead)
    mu4 = np.sum((bin_centers**4) * P_dead)
    Mw = mu2 / (mu1 + 1e-20)

    mm = M * V0 * rhom
    mm0 = M0 * V0 * rhom
    mp = mm0 - mm
    wm = mm / (mm + mp)
    rhop = rhom * (1 + 0.183 + 9e-4 * (T - 273.15))
    phip = (mp / rhop) / (mp / rhop + mm / rhom)

    f = 0.58
    kd = 6.32e16 * np.exp(-30660 / (Rcte * T))
    kf = 4.66e9 * np.exp(-76290 / (8.314 * T)) * 60
    ki_PR = 2 * f * kd * I / (M + 1e-20)
    lambda0_fp = 1e-8
    for _ in range(10):
        kp_fp = fkp(T, lambda0_fp, wm, phip)
        ktc_fp = fktc(T, lambda0_fp, phip, Mw, wm, kp_fp, M)
        ktd_fp = ktc_fp * (3.956e-4 * np.exp(4090 / (Rcte * T)))
        kt_fp = ktd_fp + ktc_fp
        if ki_PR > 0:
            lambda0_new = np.sqrt(ki_PR * M / (kt_fp + 1e-20))
        else:
            lambda0_new = 1e-20
        if abs(lambda0_new - lambda0_fp) < 1e-10:
            break
        lambda0_fp = lambda0_new
    # Fix: Werte zuweisen, damit sie unten verwendet werden können
    kp = kp_fp
    ktc = ktc_fp
    ktd = ktd_fp

    dI_dt = -kd * I
    R_gen = 2 * f * kd * I

    dydt = np.zeros_like(y)
    dydt[0] = -kp * M * np.sum(P_rad) - kf * M * np.sum(P_rad)
    dydt[1] = dI_dt

    for i in range(n_bins):
        n = bin_centers[i]
        growth = kp * M * P_rad[i-1] if i > 0 else R_gen
        loss = kp * M * P_rad[i]
        transfer = kf * M * P_rad[i]
        term_comb = ktc * P_rad[i] * np.sum(P_rad)
        term_disp = ktd * P_rad[i] * np.sum(P_rad)
        dydt[2+i] = growth - loss - transfer - term_comb - term_disp

    for i in range(n_bins):
        comb_sum = 0
        for j in range(n_bins):
            k = i - j
            if 0 <= k < n_bins:
                comb_sum += 0.5 * ktc * P_rad[j] * P_rad[k]
        disp_sum = ktd * P_rad[i] * np.sum(P_rad)
        transfer_sum = kf * M * P_rad[i]
        dydt[2+n_bins+i] = comb_sum + disp_sum + transfer_sum

    return dydt

# ===== Anfangswerte =====
y0 = np.zeros(2 + 2 * n_bins)
y0[0] = M0
y0[1] = I0
# Start: keine Radikale, keine toten Ketten

# ===== Integration =====
sol = solve_ivp(pbe_rhs_with_rates, (0, tf), y0, t_eval=t_eval, method='BDF')

# ===== Nach der Integration: Raten und Conversion berechnen =====
kp_array = np.zeros_like(t_eval)
kt_array = np.zeros_like(t_eval)
conversion_array = np.zeros_like(t_eval)

for idx in range(len(t_eval)):
    y = sol.y[:, idx]
    M = y[0]
    I = y[1]
    P_rad = y[2:2+n_bins]
    P_dead = y[2+n_bins:]

    mu0 = np.sum(P_dead)
    mu1 = np.sum(bin_centers * P_dead)
    mu2 = np.sum((bin_centers**2) * P_dead)
    Mw = mu2 / (mu1 + 1e-20)

    mm = M * V0 * rhom
    mm0 = M0 * V0 * rhom
    mp = mm0 - mm
    wm = mm / (mm + mp)
    rhop = rhom * (1 + 0.183 + 9e-4 * (T - 273.15))
    phip = (mp / rhop) / (mp / rhop + mm / rhom)

    f = 0.58
    kd = 6.32e16 * np.exp(-30660 / (Rcte * T))
    kf = 4.66e9 * np.exp(-76290 / (8.314 * T)) * 60
    ki_PR = 2 * f * kd * I / (M + 1e-20)
    lambda0_fp = 1e-8
    for _ in range(10):
        kp_fp = fkp(T, lambda0_fp, wm, phip)
        ktc_fp = fktc(T, lambda0_fp, phip, Mw, wm, kp_fp, M)
        ktd_fp = ktc_fp * (3.956e-4 * np.exp(4090 / (Rcte * T)))
        kt_fp = ktd_fp + ktc_fp
        if ki_PR > 0:
            lambda0_new = np.sqrt(ki_PR * M / (kt_fp + 1e-20))
        else:
            lambda0_new = 1e-20
        if abs(lambda0_new - lambda0_fp) < 1e-10:
            break
        lambda0_fp = lambda0_new
    kp_array[idx] = kp_fp
    kt_array[idx] = kt_fp
    conversion_array[idx] = (M0 - M) / (M0 + 1e-20)

# ===== Ergebnis: Kettenlängenverteilung am Ende =====
final_P_dead = sol.y[2+n_bins:, -1]
plt.figure(figsize=(10, 5))
plt.bar(bin_centers, final_P_dead, width=bin_centers[1]-bin_centers[0])
plt.xlabel("Kettenlänge (binned)")
plt.ylabel("Anzahl Ketten (mol/L)")
plt.title("Kettenlängenverteilung (PBE, t = %.1f min)" % t_eval[-1])
plt.yscale("log")
plt.tight_layout()
plt.show()

# ===== Monomerumsatz (Conversion) =====
plt.figure()
plt.plot(t_eval, conversion_array, label="Monomerumsatz (Conversion)")
plt.xlabel("Zeit [min]")
plt.ylabel("Conversion")
plt.title("Monomerumsatz über die Zeit")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
print(f"Endgültiger Monomerumsatz (Conversion): {conversion_array[-1]:.4f}")

# ===== Verlauf von kp und kt =====
plt.figure()
plt.plot(t_eval, kp_array, label="kp")
plt.plot(t_eval, kt_array, label="kt")
plt.xlabel("Zeit [min]")
plt.ylabel("Reaktionsrate [L/(mol·min)]")
plt.title("Verlauf von $k_p$ und $k_t$ über die Zeit")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()

def pbe_rhs_qssa(t, y):
    # y = [M, I, P_dead[0:n_bins]]
    M, I = y[0], y[1]
    P_dead = y[2:]

    # Momente für Ratenberechnung
    mu0 = np.sum(P_dead)
    mu1 = np.sum(bin_centers * P_dead)
    mu2 = np.sum((bin_centers**2) * P_dead)
    Mw = mu2 / (mu1 + 1e-20)

    mm = M * V0 * rhom
    mm0 = M0 * V0 * rhom
    mp = mm0 - mm
    wm = mm / (mm + mp)
    rhop = rhom * (1 + 0.183 + 9e-4 * (T - 273.15))
    phip = (mp / rhop) / (mp / rhop + mm / rhom)

    # Kinetik
    f = 0.58
    kd = 6.32e16 * np.exp(-30660 / (Rcte * T))
    kf = 4.66e9 * np.exp(-76290 / (8.314 * T)) * 60
    ki_PR = 2 * f * kd * I / (M + 1e-20)

    # Fixpunktiteration für lambda0, kp, kt
    lambda0_fp = 1e-8
    for _ in range(10):
        kp_fp = fkp(T, lambda0_fp, wm, phip)
        ktc_fp = fktc(T, lambda0_fp, phip, Mw, wm, kp_fp, M)
        ktd_fp = ktc_fp * (3.956e-4 * np.exp(4090 / (Rcte * T)))
        kt_fp = ktd_fp + ktc_fp
        if ki_PR > 0:
            lambda0_new = np.sqrt(ki_PR * M / (kt_fp + 1e-20))
        else:
            lambda0_new = 1e-20
        if abs(lambda0_new - lambda0_fp) < 1e-10:
            break
        lambda0_fp = lambda0_new
    lambda0 = lambda0_fp
    kp = kp_fp
    ktc = ktc_fp
    ktd = ktd_fp

    # QSSA für Radikalketten: P_rad[i] = lambda0 * Verteilungsfunktion
    # Annahme: exponentielle Verteilung (wie in klassischen Systemen)
    # Die Summe aller P_rad[i] ergibt lambda0
    # Verteilungsfunktion: p_n = (1 - p) * p**(n-1), mit Mittelwert <n> = 1/(1-p)
    # Hier: Approximativ gleichverteilt auf alle Bins (vereinfachte Annahme)
    P_rad = np.full(n_bins, lambda0 / n_bins)

    dydt = np.zeros_like(y)
    # Monomerbilanz
    dydt[0] = -kp * M * np.sum(P_rad) - kf * M * np.sum(P_rad)
    # Initiatorbilanz
    dydt[1] = -kd * I

    # Tote Ketten
    for i in range(n_bins):
        # Termination durch Kombination (Summe aller Paare, die in diesen Bin fallen)
        comb_sum = 0
        for j in range(n_bins):
            k = i - j
            if 0 <= k < n_bins:
                comb_sum += 0.5 * ktc * P_rad[j] * P_rad[k]
        # Disproportionierung: alle Radikale in diesen Bin
        disp_sum = ktd * P_rad[i] * np.sum(P_rad)
        # Kettentransfer: alle Radikale in diesen Bin
        transfer_sum = kf * M * P_rad[i]
        dydt[2+i] = comb_sum + disp_sum + transfer_sum

    return dydt

# ===== Anfangswerte für QSSA-Variante =====
y0_qssa = np.zeros(2 + n_bins)
y0_qssa[0] = M0
y0_qssa[1] = I0
# P_dead startet mit 0

# ===== Integration (QSSA) =====
sol = solve_ivp(pbe_rhs_qssa, (0, tf), y0_qssa, t_eval=t_eval, method='BDF')

# ===== Nach der Integration: Raten und Conversion berechnen (QSSA) =====
kp_array = np.zeros_like(t_eval)
kt_array = np.zeros_like(t_eval)
conversion_array = np.zeros_like(t_eval)

for idx in range(len(t_eval)):
    y = sol.y[:, idx]
    M = y[0]
    I = y[1]
    P_dead = y[2:]

    mu0 = np.sum(P_dead)
    mu1 = np.sum(bin_centers * P_dead)
    mu2 = np.sum((bin_centers**2) * P_dead)
    Mw = mu2 / (mu1 + 1e-20)

    mm = M * V0 * rhom
    mm0 = M0 * V0 * rhom
    mp = mm0 - mm
    wm = mm / (mm + mp)
    rhop = rhom * (1 + 0.183 + 9e-4 * (T - 273.15))
    phip = (mp / rhop) / (mp / rhop + mm / rhom)

    f = 0.58
    kd = 6.32e16 * np.exp(-30660 / (Rcte * T))
    kf = 4.66e9 * np.exp(-76290 / (8.314 * T)) * 60
    ki_PR = 2 * f * kd * I / (M + 1e-20)
    lambda0_fp = 1e-8
    for _ in range(10):
        kp_fp = fkp(T, lambda0_fp, wm, phip)
        ktc_fp = fktc(T, lambda0_fp, phip, Mw, wm, kp_fp, M)
        ktd_fp = ktc_fp * (3.956e-4 * np.exp(4090 / (Rcte * T)))
        kt_fp = ktd_fp + ktc_fp
        if ki_PR > 0:
            lambda0_new = np.sqrt(ki_PR * M / (kt_fp + 1e-20))
        else:
            lambda0_new = 1e-20
        if abs(lambda0_new - lambda0_fp) < 1e-10:
            break
        lambda0_fp = lambda0_new
    kp_array[idx] = kp_fp
    kt_array[idx] = kt_fp
    conversion_array[idx] = (M0 - M) / (M0 + 1e-20)

# ===== Ergebnis: Kettenlängenverteilung am Ende (QSSA) =====
final_P_dead = sol.y[2:, -1]
plt.figure(figsize=(10, 5))
plt.bar(bin_centers, final_P_dead, width=bin_centers[1]-bin_centers[0])
plt.xlabel("Kettenlänge (binned)")
plt.ylabel("Anzahl Ketten (mol/L)")
plt.title("Kettenlängenverteilung (PBE QSSA, t = %.1f min)" % t_eval[-1])
plt.yscale("log")
plt.tight_layout()
plt.show()

# ===== Monomerumsatz (Conversion) =====
plt.figure()
plt.plot(t_eval, conversion_array, label="Monomerumsatz (Conversion, QSSA)")
plt.xlabel("Zeit [min]")
plt.ylabel("Conversion")
plt.title("Monomerumsatz über die Zeit (QSSA)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
print(f"Endgültiger Monomerumsatz (Conversion, QSSA): {conversion_array[-1]:.4f}")

# ===== Verlauf von kp und kt =====
plt.figure()
plt.plot(t_eval, kp_array, label="kp (QSSA)")
plt.plot(t_eval, kt_array, label="kt (QSSA)")
plt.xlabel("Zeit [min]")
plt.ylabel("Reaktionsrate [L/(mol·min)]")
plt.title("Verlauf von $k_p$ und $k_t$ über die Zeit (QSSA)")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()
