import numpy as np

def positify(psi: np.ndarray):
    if psi.ndim == 1:
        psimax = np.amax(abs(psi))
        m = np.where(abs(psi) > psimax * 0.9)[0][0]
        eta = np.conj(psi[m]) / abs(psi[m])

        if np.abs(psimax) < 0.001:
            print("Warning: small phase factor")
        return psi * eta
    else:
        nalpha = psi.shape[1]
        psi1 = psi.copy()
        for alpha in range(nalpha):
            psi1[:, alpha] = positify(psi[:, alpha])
        return psi1