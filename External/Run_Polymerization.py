import numpy as np
import os
import sys
from scipy.integrate import ode
from graphs_plot import plotgraphs
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# ================================================================================== #
# ================================ INITIAL PARAMETERS ============================== #
# ================================================================================== #

global Rcte, V0, Vantes, Mji, Mjm, Mjp, Xc0, jc0, delta, Na, Vm_esp, Vp_esp, Vi_esp, T

# Integration interval definition
t0 = 0.0  # [min] reaction initial time
tf = 100  # [min] reaction end time, default value is 100min
Nt = 8 * tf  # [min] quantity of integration interval
tArray = np.linspace(t0, tf, Nt)  # Time vector

# Operational conditions
T = 70  # [oC] Temperature
T = T + 273.15  # [K] Temperature
V0 = 1  # [L] Solution volume

# Monomer properties
Mjm = 100.13  # [g/mol] monomer molecular weight
rhom = 0.968 - 1.225 * (1.e-3) * (T - 273.15)  # [g/cm3] monomer density (T in oC)
rhom = rhom * 1000  # [g/L] monomer density
Vm_esp = 0.822  # [cm3/g]   # monomer specific volume

# Polymer properties
Mjp = 150  # [g/mol]    # polymer molecular weight
Vp_esp = 0.77  # [cm3/g]    # polymer specific volume

# Initiator properties (AIBN)
Mji = 68  # [g/mol] initiator molecular weight
Vi_esp = 0.913  # [cm3/g]   # initiator specific volume

# Other parameters
Rcte = 1.982  # Gas constant [cal/(mol.K)]
Xc0 = 100  # [ADM]  # initial conversion
jc0 = 0.874  # [ADM]    # initial number density of radicals
delta = 6.9 * (1.e-8)  # [cm]   # distance between radicals (assumed spherical volume)
Na = 6.032 * (1.e23)  # [1/mol] # Avogadro's number

# OED initial values
I0 = 0.01548  # [mol/L] initial initiator concentration
Vm0 = V0  # [L] initial monomer volume
M0 = rhom / Mjm  # [mol/L] initial monomer concentration

NInputVar = 8  # jetzt 8 Variablen (mu3, mu4 hinzugefügt)
InputVar = np.zeros(NInputVar)  # input variables vector for integration

InputVar[0] = V0  # [L] V
InputVar[1] = M0  # [mol/L] M
InputVar[2] = I0  # [mol/L] I
InputVar[3] = 1.e-20  # [mol/L] mu0
InputVar[4] = 1.e-20  # [mol/L] mu1
InputVar[5] = 1.e-20  # [mol/L] mu2
InputVar[6] = 1.e-20  # [mol/L] mu3
InputVar[7] = 1.e-20  # [mol/L] mu4


# ================================================================================== #
# ============================= KINETIC CONST FUNCTIONS ============================ #
# ================================================================================== #

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


# ================================================================================== #
# ================================ REACTION FUNCTION =============================== #
# ================================================================================== #

Vantes = V0


def reac(t, Y):
    global Vantes, lambda0, Rcte, V0, Mjm, T

    # Monomer properties
    rhom = 0.968 - 1.225 * (1.e-3) * (T - 273.15)  # [g/cm3] monomer density (T in oC)
    rhom = rhom * 1000  # [g/L] monomer density
    M0 = rhom / Mjm  # [mol/L] initial monomer concentration

    # Polymer properties
    e = 0.183 + 9 * (1.e-4) * (T - 273.15)  # [ADM] volume contraction factor
    rhop = rhom * (1 + e)  # [g/L] monomer density

    # Current variables value
    V = Y[0]    # [L] solution volume
    M = Y[1]    # [mol/L] monomer concentration
    I = Y[2]    # [mol/L] initiator concentration
    mu0 = Y[3]  # [mol/L] dead polymer chain
    mu1 = Y[4]  # [mol/L] polymer chain of length 1
    mu2 = Y[5]      # [mol/L] polymer chain of length 2
    mu3 = Y[6]      # [mol/L] polymer chain of length 3
    mu4 = Y[7]      # [mol/L] polymer chain of length 4
    X = (M0 * V0 - M * V) / (M0 * V0)   # [ADM] conversion
    dV = Vantes - V # [L] change in solution volume

    if (V < 0):
        V = 0
    if (M < 0):
        M = 0
    if (I < 0):
        I = 0
    if (mu0 < 0):
        mu0 = 0
    if (mu1 < 0):
        mu1 = 0
    if (mu2 < 0):
        mu2 = 0
    if (mu3 < 0):
        mu3 = 0
    if (mu4 < 0):
        mu4 = 0
    if (mu0 == 1.e-20):
        lambda0 = 1.e-20

    Mw = mu2 / (mu1 + 1.e-20)

    mm = M * V * rhom  # [g] monomer mass
    mm0 = M0 * V0 * rhom  # [g] monomer initial mass
    mp = mm0 - mm  # [g] polymer mass
    wm = mm / (mm + mp)  # [ADM]
    phip = (mp / rhop) / (mp / rhop + mm / rhom)  # [ADM]

    # kinetic paremeters
    f = 0.58  # [ADM]

    # Fixpunktiteration für lambda0 und kt
    kd = 6.32 * (1.e16 * np.exp(-30660 / (Rcte * T)))  # [1/min] initiator decomposition rate constant
    kf = 4.66 * (1.e9) * np.exp(-76290 / (8.314 * T)) * 60  # [L/(mol.min)] # chain transfer rate constant
    ki_PR = 2 * f * kd * I / (M + 1.e-20)   # [1/min] radical generation rate constant

    # Initialwerte für die Iteration
    lambda0_fp = 1.e-8
    tol = 1e-10
    max_iter = 20
    for _ in range(max_iter):
        kp_fp = fkp(T, lambda0_fp, wm, phip)
        Mw_fp = mu2 / (mu1 + 1.e-20)
        ktc_fp = fktc(T, lambda0_fp, phip, Mw_fp, wm, kp_fp, M)
        ktd_fp = ktc_fp * (3.956 * (1.e-4) * np.exp(4090 / (Rcte * T)))
        kt_fp = ktd_fp + ktc_fp
        if ki_PR > 0:
            lambda0_new = np.sqrt(ki_PR * M / (kt_fp + 1.e-20))
        else:
            lambda0_new = 1.e-20
        if abs(lambda0_new - lambda0_fp) < tol:
            break
        lambda0_fp = lambda0_new
    # Nach der Iteration: konsistente Werte verwenden
    lambda0 = lambda0_fp
    kp = kp_fp
    ktc = ktc_fp
    ktd = ktd_fp
    kt = kt_fp

    # QSSA applied on R radical and lambda 0, 1 and 2
    ki_PR = 2 * f * kd * I / (M + 1.e-20)   # [1/min] radical generation rate constant
    if (ki_PR < 0):
        ki_PR = 0
#lambda: Moments of the distribution of active chains
    lambda0 = np.sqrt(ki_PR * M / kt)  # [mol/L] dlambda0

    lambda1 = (ki_PR * M + kp * M * lambda0 + kf * M * lambda0) / (kf * M + kt * lambda0)  # [mol/L] dlambda1

    lambda2 = (ki_PR * M + kp * M * (2 * lambda1 + lambda0) + kf * M * lambda0) / (kt * lambda0 + kf * M)
    lambda3 = (ki_PR * M + kp * M * (3 * lambda2 + 3 * lambda1 + lambda0) + kf * M * lambda2) / (kt * lambda0 + kf * M)
    lambda4 = (ki_PR * M + kp * M * (4 * lambda3 + 6 * lambda2 + 4 * lambda1 + lambda0) + kf * M * lambda3) / (kt * lambda0 + kf * M)

    # other constants
    eps = 1.e-20    # [L] small value to avoid division by zero

    # Balance equations
    dy = np.zeros(8)

    dy[0] = -V0 * e * (ki_PR + (kp + kf) * lambda0) * (1 - X)  # [L] dV  # [L] change in solution volume
    dy[1] = -ki_PR * M - (kp + kf) * M * lambda0 - (M / (V + eps)) * dV  # [mol/L] dM   # [mol/L] change in monomer concentration
    dy[2] = -kd * I - (I / (V + eps)) * dV  # [mol/L] dI    # [mol/L] change in initiator concentration
    dy[3] = (ktd + 0.5 * ktc) * (lambda0 ** 2) + kf * M * lambda0 - (mu0 / (V + eps)) * dV  # [mol/L] dmu0  # [mol/L] change in dead polymer chains concentration
    dy[4] = kt * lambda0 * lambda1 + kf * M * lambda1 - (mu1 / (V + eps)) * dV  # [mol/L] dmu1  # [mol/L] change in mu1 concentration
    dy[5] = kt * lambda0 * lambda2 + ktc * (lambda1 ** 2) + kf * M * lambda2 - (mu2 / (V + eps)) * dV  # [mol/L] dmu2
    dy[6] = kt * lambda0 * lambda3 + 3 * ktc * lambda1 * lambda2 + kf * M * lambda3 - (mu3 / (V + eps)) * dV  # dmu3
    dy[7] = kt * lambda0 * lambda4 + 4 * ktc * lambda1 * lambda3 + 3 * ktc * (lambda2 ** 2) + kf * M * lambda4 - (mu4 / (V + eps)) * dV  # dmu4

    Vantes = V
    return dy


# ================================================================================== #
# =================================== INTEGRATION  ================================= #
# ================================================================================== #

if __name__ == "__main__":
    YY = np.zeros(NInputVar, dtype=float)   # Initializing the array for the integration results
    YY = ode(reac).set_integrator('dopri5') # Using the dopri5 method for integration
    YY.set_initial_value(InputVar, t0)  # Setting the initial values for the integration
    Y = np.zeros((int(Nt), len(YY.y)), dtype=float) # Array to store the results of the integration
    dt = (tf - t0) / (Nt)  # [min] integration interval
    j = 0

    # Arrays zur Speicherung der Reaktionsraten
    kp_array = np.zeros(int(Nt))
    kt_array = np.zeros(int(Nt))
    lambda0_array = np.zeros(int(Nt))  # <-- Array für lambda0

    sys.stdout.write('\r' + '00%')  # Initial output for progress tracking
    sys.stdout.flush()

    while YY.successful() and YY.t < tf and j < Nt:

        Y[j, :] = YY.y[:]

        # Reaktionsraten für aktuellen Schritt berechnen und speichern
        # Werte aus Y[j, :] extrahieren
        V = Y[j, 0]
        M = Y[j, 1]
        I = Y[j, 2]
        mu0 = Y[j, 3]
        mu1 = Y[j, 4]
        mu2 = Y[j, 5]
        mu3 = Y[j, 6]
        mu4 = Y[j, 7]

        # Für kp und kt benötigen wir lambda0, Mw, wm, phip
        rhom = 0.968 - 1.225 * (1.e-3) * (T - 273.15)   # [g/cm3] monomer density (T in oC)
        rhom = rhom * 1000  # [g/L] monomer density
        M0 = rhom / Mjm # [mol/L] initial monomer concentration
        e = 0.183 + 9 * (1.e-4) * (T - 273.15)  # [ADM] volume contraction factor
        rhop = rhom * (1 + e)   # [g/L]
        Mw = mu2 / (mu1 + 1.e-20)   # [g/mol] average molecular weight of polymer chains
        mm = M * V * rhom   # [g] monomer mass
        mm0 = M0 * V0 * rhom    # [g] monomer initial mass
        mp = mm0 - mm   # [g] polymer mass
        wm = mm / (mm + mp) # [ADM] monomer weight fraction
        phip = (mp / rhop) / (mp / rhop + mm / rhom)    # [ADM] polymer volume fraction
        # lambda0 wie im reac()
        f = 0.58    # [ADM] propagation factorv --> initiator efficiency
        kd = 6.32 * (1.e16 * np.exp(-30660 / (Rcte * T)))   # [1/min] initiator decomposition rate constant
        ki_PR = 2 * f * kd * I / (M + 1.e-20)   # [1/min] radical generation rate constant
        ktc = 0  # Platzhalter, wird in fktc berechnet
        kf = 4.66 * (1.e9) * np.exp(-76290 / (8.314 * T)) * 60  # [L/(mol.min)] chain transfer rate constant
        lambda0 = np.sqrt(ki_PR * M / (1.0)) if ki_PR > 0 else 1.e-20  # Platzhalter für kt
        kp = fkp(T, lambda0, wm, phip)
        # Für kt brauchen wir Mw, kp, M
        kt = fktc(T, lambda0, phip, Mw, wm, kp, M)
        kp_array[j] = kp
        kt_array[j] = kt

        # Berechnung von lambda0 für Plot
        ki_PR = 2 * f * kd * I / (M + 1.e-20)
        lambda0_val = np.sqrt(ki_PR * M / (kt + 1.e-20)) if ki_PR > 0 else 1.e-20
        lambda0_array[j] = lambda0_val

        j = j + 1

        if (YY.t > 10000):
            os.system("PAUSE")

        ctrl = int(YY.t / tf * 100)
        sys.stdout.write('\r' + str(ctrl) + '%')
        sys.stdout.flush()

        YY.integrate(YY.t + dt)

    sys.stdout.write('\r' + '100%')
    print('\tFim integracao, t = ' + str(YY.t))

    # ================================================================================== #
    # ==================================== RESULTS ===================================== #
    # ================================================================================== #

    Vt = Y[:, 0]
    Mt = Y[:, 1]
    I = Y[:, 2]
    Mu0 = Y[:, 3]
    Mu1 = Y[:, 4]
    Mu2 = Y[:, 5]
    Mu3 = Y[:, 6]
    Mu4 = Y[:, 7]

    # PDI/X
    X = np.zeros(Nt)
    PDI = np.zeros(Nt)
    Mn = np.zeros(Nt)
    Mw = np.zeros(Nt)
    for i in range(0, Nt):
        X[i] = (M0 * V0 - Mt[i] * Vt[i]) / (M0 * V0)
        Mn[i] = Mjm * Mu1[i] / (Mu0[i] + 1.e-20)
        Mw[i] = Mjm * Mu2[i] / (Mu1[i] + 1.e-20)
        PDI[i] = Mw[i] / (Mn[i] + 1.e-20)

    # ================================================================================== #
    # ==================================== GRAPHS ====================================== #
    # ================================================================================== #
    if 1==2:
        plotgraphs(tArray, X, PDI, Mn, Mw)
    print("Mn:", Mn[-1], "Mw:", Mw[-1], "PDI:", PDI[-1])
    # ================================================================================== #
    # =========== KETTENLÄNGENVERTEILUNG via MAXIMUM ENTROPY mit n Momenten ============ #
    # ================================================================================== #
    def maxent_chain_length_dist(mus, Nmax=None):
        """
        Approximiert die Kettenlängenverteilung p_n (n=1...Nmax) aus den Momenten mus[0], mus[1], ..., mus[k]
        mittels Maximum Entropy Ansatz.
        mus: Liste der Momente [mu0, mu1, mu2, ...]
        """
        order = len(mus) - 1
        mean_n = mus[1] / (mus[0] + 1e-20)
        if Nmax is None:
            Nmax = int(5 * mean_n)
        n = np.arange(1, Nmax + 1)

        def get_pn(lambdas):
            expo = lambdas[0]
            for i in range(1, order + 1):
                expo = expo + lambdas[i] * n**i
            expo -= np.max(expo)
            pn = np.exp(expo)
            pn /= pn.sum()
            # Numerische Untergrenze setzen
            pn = np.clip(pn, 1e-30, None)
            pn /= pn.sum()
            return pn

        def objective(lambdas):
            pn = get_pn(lambdas)
            return np.sum(pn * np.log(pn + 1e-20))

        # Nebenbedingungen: Momente müssen passen
        def make_constraint(i):
            def constraint(lambdas):
                pn = get_pn(lambdas)
                return (pn * n**i).sum() - mus[i] / (mus[0] + 1e-20)
            return constraint

        cons = [{'type': 'eq', 'fun': make_constraint(i)} for i in range(order + 1)]

        # Startwerte: zufällig im Bereich [-1, 0]
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-1, 0, order + 1)
        if order >= 1:
            x0[1] = -1.0 / (mean_n + 1e-20)

        res = minimize(objective, x0, constraints=cons, method='SLSQP', options={'ftol':1e-10, 'maxiter':5000})

        if not res.success:
            print("Maximum Entropy Optimierung nicht konvergiert:", res.message)
            # Fallback: Exponentialverteilung mit Mittelwert
            l1 = -1.0 / (mean_n + 1e-20)
            pn = np.exp(l1 * n)
            pn /= pn.sum()
            pn = np.clip(pn, 1e-30, None)
            pn /= pn.sum()
            pn_abs = pn * mus[0]
            return n, pn_abs

        pn = get_pn(res.x)
        pn_abs = pn * mus[0]
        return n, pn_abs

    # Beispiel: Kettenlängenverteilung am Ende der Reaktion plotten (bis zu 5 Momente)
    mus = [Mu0[-1], Mu1[-1], Mu2[-1], Mu3[-1], Mu4[-1]]
    n, pn = maxent_chain_length_dist(mus, Nmax=3*int(Mu1[-1]/(Mu0[-1]+1e-20)))

    plt.figure()
    plt.bar(n, pn, width=1.0)
    plt.xlabel("Kettenlänge n")
    plt.ylabel("Anzahl Ketten (MaxEnt, 5 Momente)")
    plt.title("Kettenlängenverteilung (Maximum Entropy, t = %.1f min)" % tArray[-1])
    if np.any(pn > 0):
        plt.yscale("log")
    else:
        print("Warnung: Keine positiven Werte für log-Plot.")
    plt.tight_layout()
    plt.show()

    # Plot der Reaktionsraten
    plt.figure()
    plt.plot(tArray, kp_array, label="Wachstumsrate $k_p$")
    plt.xlabel("Zeit [min]")
    plt.ylabel("Reaktionsrate [L/(mol��min)]")
    plt.title("Veränderliche Reaktionsraten über die Zeit")
    plt.legend()
    plt.tight_layout()
    plt.yscale("log")
    plt.show()

    plt.figure()
    plt.plot(tArray, kt_array, label="Terminierungsrate $k_t$")
    plt.xlabel("Zeit [min]")
    plt.ylabel("Reaktionsrate [L/(mol·min)]")
    plt.title("Veränderliche Reaktionsraten über die Zeit")
    plt.legend()
    plt.tight_layout()
    plt.yscale("log")
    plt.show()

    # Plot für lambda0 über die Zeit
    plt.figure()
    plt.plot(tArray, lambda0_array, label="lambda0")
    plt.xlabel("Zeit [min]")
    plt.ylabel(r"$\lambda_0$ [mol/L]")
    plt.title(r"Verlauf von $\lambda_0$ über die Zeit")
    plt.legend()
    plt.tight_layout()
    plt.show()
