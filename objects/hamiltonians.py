import qutip as qt
import numpy as np

class Hamiltonian:
    def __init__(self, h) -> None:
        self.hamiltonian = h
    def __add__(self, h2):
        print("You are adding 2 hamiltonians")

        return h2

    def get_eigen(self):
        return self.hamiltonian.eigenenergies()


class CPBHamiltonian(Hamiltonian):
    """Hamiltonian for a Cooper-pair box qubit.

    Parameters
    ----------
    n : int
        Sets number of charge states, from -n to n+1.
    ej : flaot
        Josephson energy of the transmon qubit.
    ec : float
        Charging energy of the transmon qubit.
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

        self.n_op = qt.charge(n)
        self.phase_op = 1/2 * qt.Qobj(np.diag(np.ones(2 * n), 1) + np.diag(np.ones(2 * n), -1))

        self.hamiltonian = self.build_hamiltonian()

    def build_hamiltonian(self):
        return 4 * self.ec * (self.n_op - self.ng) ** 2 - self.ej * self.phase_op


class TransmonHamiltonian(Hamiltonian):
    """Hamiltonian for a transmon qubit.

    Parameters
    ----------
    n : int
        Sets number of charge states, from -n to n+1.
    ej : flaot
        Josephson energy of the transmon qubit.
    ec : float
        Charging energy of the transmon qubit.

    Methods
    -------
    build_hamiltonian
    """
    def __init__(self, n, ej, ec):
        self.n_dim = 2 * n + 1
        self.ej = ej
        self.ec = ec

        self.n_op = qt.charge(n)
        self.phase_op = 1/2 * qt.Qobj(np.diag(np.ones(2 * n), 1) + np.diag(np.ones(2 * n), -1))

        self.hamiltonian = self.build_hamiltonian()

    def build_hamiltonian(self):
        return 4 * self.ec * self.n_op ** 2 - self.ej * self.phase_op
