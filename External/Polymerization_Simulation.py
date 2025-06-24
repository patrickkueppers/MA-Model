# Brazilian approach to polymerization simulation with population balance

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt

# === Systemparameter wie bei dir ===
T = 70 + 273.15  # K
Mjm = 100.13     # g/mol
Mji = 68         # g/mol
Rcte = 1.982     # cal/(mol.K)
Na = 6.032e23    # wie Run_Polymerization
delta = 6.9e-8   # cm
jc0 = 0.874
Xc0 = 100
Vm_esp = 0.822  # [cm3/g]
Mjp = 150  # [g/mol]
Vp_esp = 0.77  # [cm3/g]
Vi_esp = 0.913  # [cm3/g]

# === Diskretisierung der Kettenlängenverteilung ===
N_bins = 200
bin_width = 40
L = N_bins

# === Anfangskonzentrationen ===
rhom = 0.968 - 1.225e-3 * (T - 273.15)
rhom = rhom * 1000  # [g/L]
M0 = rhom / Mjm  # [mol/L] wie Run_Polymerization
I0 = 0.01548  # mol/L
R0 = 0.0
P0 = np.zeros(L)
Rn = np.zeros(L)


def fkp(T, lambda0, wm, phip):  # T in [K]  #refer to equations in table 3 of the paper
    global Rcte, Mjm, Mjp, Xc0, jc0, delta, Na, Vm_esp, Vp_esp

    kp0 = 4.92 * (1.e5) * np.exp(-4353 / (Rcte * T)) * 60  # [L/(mol.min)] --> Arrhenius equation

    invjc = 1 / jc0 + 2 * phip / Xc0  # [ADM]
    jc = 1 / invjc  # [ADM]
    tau = np.sqrt(3 / (2 * jc * (delta ** 2)))  # [1/cm]
    rt = np.sqrt(np.log(1000 * (tau ** 3) / (Na * lambda0 * (3.1415 ** (1.5))))) / tau  # [cm] (1000 converts L -> cm3)

    rm = rt

    wp = 1 - wm  # [ADM]
    gamma = 0.763  # [ADM]
    Vfm = 0.149 + 2.9 * (1.e-4) * (T - 273.15)
    Vfp = 0.0194 + 1.3 * (1.e-4) * (T - 273.15 - 105)
    Vf = wm * Vm_esp * Vfm + wp * Vp_esp * Vfp
    factor = -gamma * Vm_esp * Mjm * (wm / Mjm + wp / Mjp) / Vf  # [ADM]
    Dm0 = 0.827 * (1.e-10)  # [cm2/s]
    Dm = Dm0 * np.exp(factor)  # [cm2/s]
    Dm = Dm * 60  # [cm2/min]

    tauDp = (rm ** 2) / (3 * Dm)  # [min]
    tauRp = 1 / (kp0 * lambda0)  # [min]

    kp = kp0 / (1 + tauDp / tauRp)

    return kp  # [L/(mol.min)]


def fktc(T, lambda0, phip, Mw, wm, kp, M):  # T in [K]
    global Rcte, Mji, Mjp, Xc0, jc0, delta, Na, Vm_esp, Vp_esp, Vi_esp

    ktc0 = 9.80 * (1.e7) * np.exp(-701.0 / (Rcte * T)) * 60  # [L/(mol.min)] # Arrhenius equation

    Mwlin = 4293.76  # [g/mol]
    kB = 1.3806 * (1.e-23)  # [J/K]
    RH = 1.3 * (1.e-9) * (Mwlin ** 0.574)  # [cm]
    etas = np.exp(-00.099 + (496 / T) - 1.5939 * np.log(T))  # [Pa.s]
    Dplin = kB * T / (6 * 3.1415 * etas * RH)  # [m3/(s.cm)] = [cm2/s * 1.e-6]
    Dplin = Dplin * (1.e6) * 60  # [cm2/min]

    wp = 1 - wm  # [ADM]
    gamma = 0.763  # [ADM]
    Vfm = 0.149 + 2.9 * (1.e-4) * (T - 273.15)
    Vfp = 0.0194 + 1.3 * (1.e-4) * (T - 273.15 - 105)
    Vf = wm * Vm_esp * Vfm + wp * Vp_esp * Vfp
    epsillon13 = Vi_esp * Mji / (Vp_esp * Mjp)  # [ADM]
    factor = -(gamma / epsillon13) * (wm * Vm_esp + wp * Vp_esp * epsillon13) / Vf - (1 / Vfm)  # [ADM]

    Dp = Dplin * ((Mwlin ** 2) / (Mw ** 2 + 1.e-20)) * np.exp(factor)  # [cm2/min]

    Fseg = 0.12872374  # [ADM]
    Dpe = Fseg * Dp  # [cm2/min]

    invjc = 1 / jc0 + 2 * phip / Xc0  # [ADM]
    jc = 1 / invjc  # [ADM]
    tau = np.sqrt(3 / (2 * jc * (delta ** 2)))  # [1/cm]
    rt = np.sqrt(np.log(1000 * (tau ** 3) / (Na * lambda0 * (3.1415 ** (1.5))))) / tau  # [cm] (1000 converts L -> cm3)
    rm = rt  # [cm]

    tauRt = 1 / (ktc0 * lambda0)  # [min]
    tauDt = (rm ** 2) / (3 * Dpe)  # [min]

    ktc = ktc0 / (1 + tauDt / tauRt)  # [L/(mol.min)]

    A = 8 * 3.1415 * (delta ** 3) * np.sqrt(jc) * Na / (3000)

    ktres = A * kp * M

    kte = ktc + ktres

    return kte  # [L/(mol.min)]
# === Ableitungssystem ===
def dYdt(t, y):
    M, I = y[0], y[1]
    Rn = y[2:2 + L]
    Pn = y[2 + L:]

    # === Dynamische Ratenberechnung wie Run_Polymerization ===
    mm = M * rhom  # [g/L]
    mm0 = M0 * rhom  # [g/L]
    mp = mm0 - mm
    wm = mm / (mm + mp + 1e-20)
    e = 0.183 + 9e-4 * (T - 273.15)
    rhop = rhom * (1 + e)
    phip = (mp / rhop) / (mp / rhop + mm / rhom + 1e-20)
    f = 0.58

    # Fixpunktiteration für lambda0 und Raten
    kd = 6.32e16 * np.exp(-30660 / (Rcte * T))  # [1/min]
    kf = 4.66e9 * np.exp(-76290 / (8.314 * T)) * 60  # [L/(mol.min)]
    ki_PR = 2 * f * kd * I / (M + 1e-20)
    lambda0_fp = 1e-8
    tol = 1e-10
    max_iter = 20
    for _ in range(max_iter):
        kp_fp = fkp(T, lambda0_fp, wm, phip)
        # Momente aus Pn berechnen
        mu0 = np.sum(Pn)
        mu1 = np.sum((np.arange(L) + 1) * Pn)
        mu2 = np.sum((np.arange(L) + 1)**2 * Pn)
        Mw_fp = mu2 / (mu1 + 1e-20)
        ktc_fp = fktc(T, lambda0_fp, phip, Mw_fp, wm, kp_fp, M)
        ktd_fp = ktc_fp * (3.956e-4 * np.exp(4090 / (Rcte * T)))
        kt_fp = ktd_fp + ktc_fp
        if ki_PR > 0:
            lambda0_new = np.sqrt(ki_PR * M / (kt_fp + 1e-20))
        else:
            lambda0_new = 1e-20
        if abs(lambda0_new - lambda0_fp) < tol:
            break
        lambda0_fp = lambda0_new
    lambda0 = lambda0_fp
    kp = kp_fp
    ktc = ktc_fp
    ktd = ktd_fp
    kt = kt_fp

    dM = 0
    dI = -kd * I
    dRn = np.zeros(L)
    dPn = np.zeros(L)

    # Initiator-Zerfall
    dRn[0] += 2 * f * kd * I

    # Propagation und Kettenübertragung
    for i in range(L):
        # Propagation
        if i < L - 1:
            dRn[i] -= kp * Rn[i] * M
            dRn[i+1] += kp * Rn[i] * M
            dM -= kp * Rn[i] * M
        # Chain transfer
        dRn[i] -= kf * Rn[i] * M
        dRn[0] += kf * Rn[i] * M
        dM -= kf * Rn[i] * M
        dPn[i] += kf * Rn[i] * M

    # Termination (Kombination und Disproportionierung)
    for i in range(L):
        for j in range(L):
            rate = Rn[i] * Rn[j]
            if i + j < L:
                dPn[i+j] += 0.5 * ktc * rate  # Kombination
            dPn[i] += 0.5 * ktd * rate  # Disproportionierung
            dPn[j] += 0.5 * ktd * rate
            dRn[i] -= (ktc + ktd) * Rn[i] * Rn[j]

    return np.concatenate([[dM, dI], dRn, dPn])

# === Startwerte-Vektor ===
y0 = np.concatenate([[M0, I0], Rn, P0])

# === Integration ===
t_span = (0, 100)
t_eval = np.linspace(*t_span, 500)
sol = solve_ivp(dYdt, t_span, y0, t_eval=t_eval, method="BDF")

# === Ergebnisse plotten ===
Pn_end = sol.y[2+L:, -1]
chain_lengths = (np.arange(L) + 1)  # Kettenlänge = Index+1
mu0 = np.sum(Pn_end)
mu1 = np.sum(chain_lengths * Pn_end)
mu2 = np.sum((chain_lengths**2) * Pn_end)
Mn = Mjm * mu1 / (mu0 + 1e-20)
Mw = Mjm * mu2 / (mu1 + 1e-20)
PDI = Mw / (Mn + 1e-20)

plt.bar(chain_lengths, Pn_end, width=1.0)
plt.xlabel("Kettenlänge n")
plt.ylabel("Konzentration [mol/L]")
plt.title(f"Mn={Mn:.1f}, Mw={Mw:.1f}, PDI={PDI:.2f}")
plt.yscale("log")
plt.tight_layout()
plt.show()
