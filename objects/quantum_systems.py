import qutip as qt
import numpy as np


class QuantumSystem:
    hamiltonian: qt.Qobj

    def get_eigen(self):
        return self.hamiltonian.eigenstates()


class CPB(QuantumSystem):
    """Cooper-pair box qubit.

    Parameters
    ----------
    n : int
        Sets number of charge states, from -n to n+1.
    ej : flaot
        Josephson energy of the CPB qubit.
    ec : float
        Charging energy of the CPB qubit.
    ng : float, optional
        Charge of the transmon qubit. Default is 0.

    Methods
    -------
    build_hamiltonian
    """
    def __init__(self, n, ej, ec, ng=0):
        self.n_dim = 2 * n + 1
        self.ej = ej
        self.ec = ec
        self.ng = ng

        self.ncp_op = qt.charge(n)
        self.cos_phase_op = 1/2 * qt.Qobj(np.diag(np.ones(2 * n), 1) + np.diag(np.ones(2 * n), -1))

        self.hamiltonian = self.build_hamiltonian()

    def build_hamiltonian(self):
        return 4 * self.ec * (self.ncp_op - self.ng) ** 2 - self.ej * self.cos_phase_op


class Transmon(QuantumSystem):
    """Hamiltonian for a transmon qubit. ng is removed as it is assumed Ej >> Ec.

    Parameters
    ----------
    n : int
        Sets dimension of the Hilbert space to 2n+1.
    ej : flaot
        Josephson energy of the transmon qubit.
    ec : float
        Charging energy of the transmon qubit.

    Methods
    -------
    build_hamiltonian
    """
    def __init__(self, n, ej, ec):
        self.n = n
        self.ej = ej
        self.ec = ec

        self.build_hamiltonian()

    def build_hamiltonian(self):
        charge = qt.Qobj(np.diag(np.arange(-self.n, self.n + 1)))
        cos_phase = qt.Qobj(np.diag(np.ones(2 * self.n), 1) + np.diag(np.ones(2 * self.n), -1)) / 2

        self.hamiltonian = 4 * self.ec * charge ** 2 - self.ej * cos_phase


class ResonatorHamiltonian(QuantumSystem):
    """Hamiltonian for a resonator.

    Parameters
    ----------
    n : int
        Sets dimension of the Hilbert space.
    omega_r : float
        Frequency of the resonator.

    Methods
    -------
    build_hamiltonian
    """
    def __init__(self, n_dim, omega_r):
        self.n_dim = n_dim
        self.omega_r = omega_r

        self.create = qt.create(n_dim)
        self.destroy = qt.destroy(n_dim)

        self.build_hamiltonian()
