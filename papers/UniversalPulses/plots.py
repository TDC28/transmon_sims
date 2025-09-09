import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from objects.quantum_systems import Transmon

plt.style.use("science")

# Fig 2a
n = 40
ej = 34.08  # GHz
ec = 0.096  # GHz

transmon = Transmon(ej=ej, ec=ec, n=n)
energies, eigenstates = transmon.get_eigen()

# Plot of detuning
detuning_abs = []
for i in range(8):
    detuning_abs.append(np.abs(energies[i] - 2 * energies[i+1] + energies[i+2]) * 1000)


plt.figure(figsize=(9, 5))
plt.scatter(np.arange(8), detuning_abs)
plt.suptitle("Detuning between the target subspace and the leakage levels in the rotating frame")
plt.title(f"$\\omega_q = 5 \\text{{ GHz}}$, $\\frac{{E_J}}{{E_C}} = {int(ej / ec)}$")
plt.xlabel("k")
plt.ylabel("$\\left| \\Delta_k \\right| / 2 \\pi$ (MHz)")
plt.savefig("figs/papers/UniversalPulses/fig2a.png", dpi=1200)
plt.show()


# Fig 2b
n = 40
wq = 5  # GHz

ks = range(1, 6)
ratios = {k: np.linspace(80 * (k + 2) / 3, 2e3, 500) for k in ks}
deltas = {k: [] for k in ks}

for k in ks:
    for ejec_ratio in ratios[k]:
        ec = wq / (np.sqrt(8 * ejec_ratio) - 1)
        ej = ejec_ratio * ec
    
        transmon = Transmon(ej=ej, ec=ec, n=n)
        energies, _= transmon.get_eigen()

        delta = -energies[k-1] + 3 * energies[k] - 3 * energies[k+1] + energies[k+2]
        deltas[k].append(delta * -1e3)


plt.figure(figsize=(9, 5))

for k in ks:
    plt.plot(ratios[k], deltas[k], label=f"({k}, {k+1})")

plt.xscale("log")
plt.xlabel("$E_J/E_C$")
plt.ylabel("$\\delta_{{k-1, k+2}} / (2 \\pi)$ (MHz)")
plt.suptitle("Energy gap between leakage levels $|k-1 \\rangle$ and $|k+2 \\rangle$")

def r_to_alpha_mag_mhz(r):
    return wq * 1000 / (np.sqrt(8 * r) - 1)

def alpha_mag_mhz_to_r(a_mag_mhz):
    ec = a_mag_mhz / 1000
    return ((1 + wq / ec) ** 2) / 8

alpha_ticks = [200, 150, 100, 50]
secax = plt.gca().secondary_xaxis("top", functions=(r_to_alpha_mag_mhz, alpha_mag_mhz_to_r))
secax.set_xlabel(r"Anharmonicity $\alpha/(2\pi)=\Delta_1$ (MHz)")
secax.xaxis.set_major_locator(FixedLocator(alpha_ticks))
secax.xaxis.set_major_formatter(FixedFormatter([f"−{t}" for t in alpha_ticks]))
secax.xaxis.set_minor_locator(NullLocator())
secax.set_xticklabels([f"−{t}" for t in alpha_ticks])
secax.set_xlim(200, 50)

plt.legend()
plt.savefig("figs/papers/UniversalPulses/fig2b.png", dpi=1200)
plt.show()
