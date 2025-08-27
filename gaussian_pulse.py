"""Plots of the control of a capacitively driven transmon with gaussian pulses."""
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from objects.quantum_systems import Transmon
from scipy.special import erf

plt.style.use("science")


def gaussian_pulse(t, args):
    return args["amp"] * np.exp(-0.5 * ((t - args["tg"] / 2) / args["sigma"]) ** 2)


def gaussian_pulse_area(sigma, tg):
    return  np.sqrt(2 * np.pi) * sigma * erf(tg / (2 * np.sqrt(2) * sigma))

if __name__ == "__main__":
    n = 40
    ej = 34.08 # GHz
    ec = 0.96 # GHz

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

    tg = 50  # ns
    sigma = tg / 8
    amp = np.pi / (gaussian_pulse_area(sigma, tg) * np.abs(n_op_eigenbasis[0, 1]))

    transmon_rwa_hamiltonian = qt.Qobj(np.diag(wk - np.arange(4) * wd))
    drive_hamiltonian = [n_op_eigenbasis / 2, gaussian_pulse]
    full_hamiltonian = [transmon_rwa_hamiltonian, drive_hamiltonian]

    t = np.linspace(0, tg, 100 * tg)
    drive_args = {"tg": tg, "amp": amp, "sigma": sigma}

    plt.figure(figsize=(6, 6))
    plt.plot(t, gaussian_pulse(t, drive_args))
    plt.suptitle("Gaussian drive envelope")
    plt.xlabel("$t$ [ns]")
    plt.ylabel("$\\Omega(t)$")
    plt.savefig("figs/gaussian_pulse/drive_envelope.png", dpi=1200)
    plt.show()

    plt.figure(figsize=(6, 6))

    for i in range(4):
        result = qt.mesolve(full_hamiltonian, qt.basis(4, 0), t, e_ops=qt.basis(4, i) @ qt.basis(4, i).dag(), args=drive_args)

        plt.plot(result.times, result.expect[0], label=f"$P_{i}$")

    plt.suptitle("Transmon driven at $\\omega_{01}$ by a gaussian pulse")
    plt.title(f"$t_g = {tg}$ ns")
    plt.xlabel("Pulse duration (ns)")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig("figs/gaussian_pulse/0to1_transition.png", dpi=1200)
    plt.show()

