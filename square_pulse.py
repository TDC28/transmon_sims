"""Plot of the control of a capacitively driven transmon with square pulses."""
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

    wk = energies[:4]
    wd = wk[1] - wk[0]
    detuning_p = wk - np.arange(4) * wd

    # Square pulse
    tg = 100  # ns
    rabi_freq = np.pi / (tg * np.abs(n_op_eigenbasis[0, 1]))

    tlist = np.linspace(0, tg, 100 * tg)
    transmon_rwa = qt.Qobj(np.diag(wk - np.arange(4) * wd)) + rabi_freq / 2 * n_op_eigenbasis

    plt.figure(figsize=(6, 6))

    for i in range(4):
        result = qt.mesolve(transmon_rwa, qt.basis(4, 0), tlist, e_ops=qt.basis(4, i) @ qt.basis(4, i).dag())

        plt.plot(result.times, result.expect[0], label=f"$P_{i}$")

    plt.suptitle("Transmon driven at $\\omega_{01}$ by a square pulse")
    plt.title(f"$t_g = {tg}$ ns, $\\Omega (t) = {rabi_freq:.4f}$")
    plt.xlabel("Pulse duration (ns)")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig("figs/square_pulse/0to1_transition.png", dpi=1200)
    plt.show()

