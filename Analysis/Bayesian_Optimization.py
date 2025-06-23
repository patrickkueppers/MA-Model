# Copy of Jupyter-Notebook
#%%
# %% [Zelle 1] Imports und Pfade setzen
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from Simulation import Polymer_Model_new

from skopt import gp_minimize
from skopt.space import Real



#%%

# %% [Zelle 2] Parameterraum definieren
space = [
    Real(1e-4, 1e-1, name='kd'),       # Initiator Dissoziation
    Real(500, 5000, name='kip'),       # Initiierung + Propagation (gleicher Wert)
    Real(0.0, 1.0, name='ktr'),        # Kettenübertragung
    Real(1e4, 1e6, name='kt'),         # Termination
]



#%%
# %% [Zelle 3] Zielkriterium definieren – ohne use_named_args
def objective(params):
    kd, kip, ktr, kt = params

    print(f"\nRunning simulation with kd={kd:.2e}, kip={kip:.2e}, ktr={ktr:.2e}, kt={kt:.2e}")
    start = time.time()

    try:
        result = Polymer_Model_new.run_simulation_with_params(
            kd=kd, ki=kip, kp=kip, ktr=ktr, kt=kt, t_end=1000
        )
    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1e6  # Bestrafe fehlgeschlagene Simulationen

    final_state = result.y[:, -1]
    P_dead = final_state[3 + Polymer_Model_new.N_MAX:]

    indices = np.arange(1, Polymer_Model_new.N_MAX + 1)
    total = np.sum(P_dead)
    if total == 0:
        return 1e6  # Ungültige Lösung

    normalized = P_dead / total
    mean = np.sum(indices * normalized)
    std = np.sqrt(np.sum(((indices - mean)**2) * normalized))

    end = time.time()
    print(f"→ Simulation took {end - start:.2f} seconds. mean={mean:.1f}, std={std:.2f}")

    loss = (mean - 100)**2 + std**2
    return loss  # Minimierung der Verlustfunktion



#%%
# %% [Zelle 4] Optimierung starten
res = gp_minimize(
    func=objective,
    dimensions=space,
    acq_func="EI",       # Alternativen: "LCB", "PI"
    n_calls=20,
    n_initial_points=5,
    random_state=42,
    verbose=True
)

#%%
# %% [Zelle 5] Beste Parameter testen und visualisieren
best_params = res.x
kd_best, kip_best, ktr_best, kt_best = best_params
print(f"Beste Parameter: kd={kd_best:.2e}, kip={kip_best:.2e}, ktr={ktr_best:.2e}, kt={kt_best:.2e}")

result = Polymer_Model_new.run_simulation_with_params(
    kd=kd_best, ki=kip_best, kp=kip_best, ktr=ktr_best, kt=kt_best, t_end=1000
)
final_state = result.y[:, -1]
Polymer_Model_new.plot_chain_length_distribution(final_state)
