import numpy as np

def total_energy_drift(y_history, masses, N, eps=1e4):
   

    G = 6.67430e-11
    energies = []

    for state in y_history:

        reshaped_state = state.reshape((N, 6))

        positions = reshaped_state[:, 0:3]
        velocities = reshaped_state[:, 3:6]

        # ---------- Kinetic Energy ----------
        v_sq = np.sum(velocities**2, axis=1)
        K = 0.5 * np.sum(masses * v_sq)

        # ---------- Potential Energy ----------
        U = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                r_vec = positions[j] - positions[i]
                dist = np.sqrt(np.dot(r_vec, r_vec) + eps**2)
                U -= G * masses[i] * masses[j] / dist

        energies.append(K + U)

    E = np.array(energies)

    # ---------- Energy Drift ----------
    E0 = E[0]
    dE = E - E0
    dE_rel = dE / abs(E0)
    

    return np.array(dE_rel)




def Angular_momentum(y,masses,N):
    states = y.reshape((-1, N, 6))
    L_vec = []

    for state in states:
        r = state[:, :3]        # (N,3)
        v = state[:, 3:]        # (N,3)

        # The total angular momentum 
        L = np.sum(np.cross(r, masses[:, None] * v), axis=0)
        L_vec.append(L)

    L_vec = np.array(L_vec)               # (T,3)
    L_mag = np.linalg.norm(L_vec, axis=1)

    L0 = L_mag[0]
    delta_L_over_L = np.abs(L_mag - L0) / np.abs(L0)

    return L_mag,delta_L_over_L
