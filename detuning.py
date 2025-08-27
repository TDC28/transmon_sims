"""Plot of the detuning for a transmon qudit (https://arxiv.org/abs/2005.13165)."""
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from objects.quantum_systems import Transmon

plt.style.use("science")

if __name__ == "__main__":
    n = 40
    ej = 34.08 # GHz
    ec = 0.096 # GHz

    transmon = Transmon(ej=ej, ec=ec, n=n)
    energies, eigenstates = transmon.get_eigen()

    # Plot of detuning
    detuning_abs = []
    for i in range(8):
        detuning_abs.append(np.abs(energies[i] - 2 * energies[i+1] + energies[i+2]) * 1000)

    plt.figure(figsize=(6, 4))
    plt.scatter(np.arange(8), detuning_abs)
    plt.suptitle("Detuning between the target subspace and the leakage levels in the rotating frame")
    plt.title(f"$\\omega_q = 5 \\text{{ GHz}}$, $\\frac{{E_J}}{{E_C}} = {int(ej / ec)}$")
    plt.xlabel("k")
    plt.ylabel("$\\left| \\Delta_k \\right| / 2 \\pi$ (MHz)")
    plt.savefig("figs/detuning/detuning.png", dpi=1200)
    plt.show()
