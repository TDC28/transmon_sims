"""Plots of the control of a capacitively driven transmon (https://arxiv.org/abs/2005.13165)."""
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from objects.hamiltonians import Transmon

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
    plt.savefig("figs/square_pulse/detuning.png", dpi=1200)
    plt.show()

    n_op = qt.Qobj(np.diag(np.arange(-n, n+1)))
    n_op_eigenbasis = np.zeros((4, 4), dtype=np.complex128)

    for i in range(4):
        for j in range(4):
            n_op_eigenbasis[i, j] = eigenstates[i].dag() * n_op * eigenstates[j]

    # Weird corrections
    # n_op_eigenbasis[0, 3] *= -1
    # n_op_eigenbasis[3, 0] *= -1
    # n_op_eigenbasis[2, 3] *= -1
    # n_op_eigenbasis[3, 2] *= -1

    n_op_eigenbasis = qt.Qobj(n_op_eigenbasis)

    # Driven transmon hamiltonian in RWA form
    wk = energies[:4]
    wd = (wk[1] - wk[0])

    # Square pulse
    rabi_freq = 0.017
    detuning_p = wk - np.arange(4) * wd

    print("|0> -> |1> transition time: t =", np.pi / (rabi_freq * np.abs(n_op_eigenbasis[0, 1])), "ns")

    tlist = np.linspace(0, 200, 4001)
    transmon_rwa = qt.Qobj(np.diag(wk - np.arange(4) * wd)) + rabi_freq / 2 * n_op_eigenbasis

    plt.figure(figsize=(6, 6))

    for i in range(4):
        result = qt.mesolve(transmon_rwa, qt.basis(4, 0), tlist, e_ops=qt.basis(4, i) @ qt.basis(4, i).dag())

        plt.plot(result.times, result.expect[0], label=f"$P_{i}$")

    plt.suptitle("Control of qubit (square pulse)")
    plt.xlabel("Pulse duration (ns)")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig("figs/square_pulse/0to1_transition.png", dpi=1200)
    plt.show()

