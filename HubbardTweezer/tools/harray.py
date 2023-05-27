import numpy as np


class harray(np.ndarray):

    @property
    def H(self):
        return self.conj().T