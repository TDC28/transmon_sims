import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from objects.quantum_systems import Transmon
from qutip.control import pulseoptim


plt.style.use("science")

def dirac_picture_hamiltonian(t, args, H):
    u = qt.Qobj(-1j * args["wd"] * t * np.diag(np.arange(4))).expm()
    u_dot = qt.Qobj(-1j * args["wd"] * np.diag(np.arange(4))) @ u

    return u.dag() @ H @ u - 1j * u.dag() @ u_dot

ej = 11.67 * 2 * np.pi  # rad GHz
ec = 0.1988 * 2 * np.pi # rad GHz
n = 40

transmon = Transmon(ej=ej, ec=ec, n=n)
evals, evecs = transmon.get_eigen()

evals4 = evals[:4] - evals[0]
evecs4 = []

for i in range(4):
    evecs4.append(evecs[i])

# Transition frequencies
for i in range(3):
    print(f"|{i}> -> |{i+1}> :", (evals[i+1] - evals[i]) / (2 * np.pi), "GHz")

# Raising and lowering operators in Transmon eigenbasis
n_op = qt.charge(Nmax=n)
raise_op_np = np.zeros((4, 4), dtype=np.complex128)

for i in range(3):
    raise_op_np[i + 1, i] = evecs[i + 1].dag() @ n_op @ evecs[i]

# Weird corrections
raise_op_np[1, 0] *= -1
raise_op_np[3, 0] *= -1

raise_op = qt.Qobj(raise_op_np / raise_op_np[1, 0])
lower_op = raise_op.dag()

# Moving to interaction picture
h_0_rot = qt.Qobj(np.diag(evals4 - evals4[1] * np.arange(4)))

# u = lambda t: qt.Qobj(-1j * evals4[1] * t * np.diag(np.arange(4))).expm()

h_1 = lower_op + raise_op
h_2 = -1j * (lower_op - raise_op)

# Optimizing drive
tg = 150  # ns
resolution = 1 / 2 # ns
n_bins = int(tg / resolution)
a_max = 2 * np.pi * 6e-3  # Max drive amplitude of 6 MHz rotating frame
u0 = qt.qeye(4)
swap02 = qt.Qobj([[0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1]])

result = pulseoptim.optimize_pulse(drift=h_0_rot,
                                   ctrls=[h_1, h_2],
                                   initial=u0,
                                   target=swap02,
                                   num_tslots=n_bins,
                                   evo_time=tg,
                                   amp_lbound=-a_max,
                                   amp_ubound=a_max,
                                   )

print(result.termination_reason)
u1, u2 = result.final_amps.T

plt.figure(figsize=(6, 6))
plt.plot(u1, label="u1")
plt.plot(u2, label="u2")
plt.legend()
plt.show()

# ---------- 1) Piecewise-constant I/Q envelopes ----------
dt = tg / n_bins
def bin_index(t):
    # clamp so t=tg maps to last bin
    i = int(np.floor(t / dt))
    return min(max(i, 0), n_bins - 1)

def u1_of_t(t, args=None):
    return float(u1[bin_index(t)])

def u2_of_t(t, args=None):
    return float(u2[bin_index(t)])

# ---------- 2) Rotating-frame Hamiltonian ----------
# H(t) = H0_rot + u1(t) H1 + u2(t) H2
H_t = [h_0_rot, [h_1, u1_of_t], [h_2, u2_of_t]]

# time grid for solver (>= ~10 steps per bin is safe)
steps_per_bin = 10
tlist = np.linspace(0.0, tg, n_bins * steps_per_bin + 1)

# ---------- 3) Evolve populations from |0> (sanity plot) ----------
psi0 = qt.basis(4, 0)
res0 = qt.mesolve(H_t, psi0, tlist,
                  e_ops=[qt.basis(4,k) * qt.basis(4,k).dag() for k in range(4)])

import matplotlib.pyplot as plt
plt.figure(figsize=(6,5))
for k in range(4):
    plt.plot(tlist, res0.expect[k], label=fr"$P_{k}$")
plt.xlabel("time [ns]"); plt.ylabel("Population"); plt.title("Driven transmon (rotating frame)")
plt.legend(); plt.tight_layout(); plt.show()

# ---------- 4) Get full propagator U(T) in the rotating frame ----------
# Build columns by evolving basis kets
cols = []
for k in range(4):
    out = qt.sesolve(H_t, qt.basis(4, k), tlist)
    cols.append(out.states[-1].full())
U_T = qt.Qobj(np.hstack(cols))  # 4x4 unitary in the ω01 rotating frame

# ---------- 5) Gate fidelity on the {0,1,2} subspace ----------
P = qt.Qobj(np.diag([1,1,1,0]))
U_sub = (P * U_T * P).extract_states([0,1,2])  # 3x3 block
U_targ_3 = qt.Qobj([[0,0,1],[0,1,0],[1,0,0]])

d = 3
Fg = abs((U_targ_3.dag() * U_sub).tr()) / d   # phase-insensitive PSU if you used that in GRAPE
print(f"Subspace gate fidelity Fg: {Fg:.6f}")

# ---------- 6) (Optional) Leakage metric during the gate ----------
Proj3 = qt.basis(4,3) * qt.basis(4,3).dag()
def avg_leak_from(ket):
    out = qt.mesolve(H_t, ket, tlist, e_ops=[Proj3])
    # time-averaged P3
    return np.trapz(out.expect[0], tlist) / (tlist[-1] - tlist[0])

L = np.mean([avg_leak_from(qt.basis(4,k)) for k in (0,1,2)])
print(f"Avg leakage ⟨P3⟩ during gate: {L:.6e}")

