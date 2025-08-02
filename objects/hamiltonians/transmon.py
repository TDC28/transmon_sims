import qutip as qt

from .hamiltonian import Hamiltonian


class Transmon(Hamiltonian):
    def __init__(self, n_dim, ej, ec, ng=0):
        self.n_dim = n_dim
        self.ej = ej
        self.ec = ec
        self.ng = ng

    def build_hamiltonian(self):
        pass
