import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")

if __name__ == "__main__":
    b = qt.destroy(2)
    b_dag = qt.create(2)
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    # Ej / Ec = 50
    ej = 12.5
    ec = 0.25
    omega_q = np.sqrt(8 * ej * ec) - ec

    phi_d = np.pi / 2
    rabi_freq = 1

    plt.figure(figsize=(6, 6))
    for detuning in [0, 1, 3]:
        h_q = detuning / 2 * sz + rabi_freq / 2 * (np.cos(phi_d) * sx + np.sin(phi_d) * sy)

        evolution = qt.mesolve(h_q, qt.basis(2, 0), np.linspace(0, 10, 100), e_ops=[b_dag @ b])

        plt.plot(evolution.times, evolution.expect[0], label=f"$\\Delta = {detuning}$ GHz")

    plt.title(f"$\\phi_d = \\frac{{\\pi}}{{{np.round(np.pi / phi_d).astype(int)}}}, \\Omega_R(t) = {{{rabi_freq}}}$")
    plt.xlabel("$t$")
    plt.ylabel("$\\langle N \\rangle$")
    plt.legend()
    plt.show()

