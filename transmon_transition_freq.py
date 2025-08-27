"""Plots of the transition frequency of a transmon with various Ej/Ec ratios (https://arxiv.org/abs/2005.12667)."""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from objects.quantum_systems import CPB

plt.style.use("science")


if __name__ == "__main__":
    ngs = np.linspace(-1, 1, 200)
    ratios = [2, 10, 50]

    for ratio in ratios:
        e00 = np.zeros(200, dtype=np.float64)
        e01 = np.zeros(200, dtype=np.float64)
        e02 = np.zeros(200, dtype=np.float64)

        for i, ng in enumerate(ngs):
            # Define Ej and Ec so the qubit plasma frequency is 5 GHz
            ec = 5 / np.sqrt(8 * ratio)
            ej = ec * ratio

            th = CPB(10, ej, ec, ng)
            evals, _ = th.get_eigen()

            e00[i] = evals[0] - evals[0]
            e01[i] = evals[1] - evals[0]
            e02[i] = evals[2] - evals[0]

        plt.figure(figsize=(6, 6))
        plt.suptitle("Frequency difference for $E_0$, $E_1$, and $E_2$")
        plt.title(f"$\\frac{{E_J}}{{E_C}} = {ratio}$, $\\omega_p = 5$ GHz")
        plt.plot(ngs, e00, label="$E_{00}$")
        plt.plot(ngs, e01, label="$E_{01}$")
        plt.plot(ngs, e02, label="$E_{02}$")
        plt.xlabel("$n_g$")
        plt.ylabel("Energy (GHz)")
        plt.legend()
        plt.savefig(f"figs/transmon_transition_freq/ratio_{ratio}.png", dpi=1200)
        plt.show()

