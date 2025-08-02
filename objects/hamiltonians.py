import qutip as qt
import numpy as np

class Hamiltonian:
    n_dim: int | list[int]
    hamiltonian: qt.Qobj
    create: qt.Qobj
    destroy: qt.Qobj

    def __add__(self, h2):
        pass

    def get_eigen(self):
        return self.hamiltonian.eigenenergies(), self.hamiltonian.eigenstates()


class CPBHamiltonian(Hamiltonian):
    """Hamiltonian for a Cooper-pair box qubit.

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


class TransmonHamiltonian(Hamiltonian):
    """Hamiltonian for a transmon qubit. ng is removed as it is assumed Ej >> Ec.

    Parameters
    ----------
    n_dim : int
        Sets dimension of the Hilbert space.
    ej : flaot
        Josephson energy of the transmon qubit.
    ec : float
        Charging energy of the transmon qubit.

    Methods
    -------
    build_hamiltonian
    """
    def __init__(self, n_dim, ej, ec):
        self.n_dim = n_dim
        self.ej = ej
        self.ec = ec

        self.create = qt.create(n_dim)
        self.destroy = qt.destroy(n_dim)

        # self.phase_op = (2 * ec / ej) ** (1/4) * (self.create + self.destroy)
        # self.ncp_op = 1j / 2 * (ej / (2 * ec)) ** (1/4) * (self.create - self.destroy)

        self.build_hamiltonian()

    def build_hamiltonian(self):
        self.hamiltonian = (np.sqrt(8 * self.ec * self.ej) - self.ec) * self.create * self.destroy - self.ec / 2 * self.create * self.create * self.destroy * self.destroy


def ResonatorHamiltonian(Hamiltonian):
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

    def build_hamiltonian(self):
        self.hamiltonian = self.omega_r/2 * self.create * self.destroy
