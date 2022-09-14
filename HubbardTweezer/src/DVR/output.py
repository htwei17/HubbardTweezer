import h5py
import numpy as np


class DVRdynaOutput:

    def __init__(self,
                 t=None,
                 gs=None,
                 wavefunc=False,
                 trap=(None, None)) -> None:
        self.t = t
        self.rho_gs = gs
        self.wavefunc = wavefunc
        if wavefunc:
            self.rho_trap = trap[0]
            self.psi = trap[1]

    def write_file(self, fn: str):
        with h5py.File(fn, "a") as f:
            append_to_table(f, 't', self.t)
            # if __debug__:
            #     print('t OK')
            append_to_table(f, 'rho_gs', self.rho_gs)
            # if __debug__:
            #     print('gs OK')
            if self.wavefunc:
                append_to_table(f, 'rho_trap', self.rho_trap)
                # if __debug__:
                #     print('trap OK')
                append_to_table(f,
                                'psi',
                                self.psi.astype(np.complex),
                                dtype=np.complex)
                # if __debug__:
                #     print('wavefunc OK')

    def read_file(self, fn: str, path: str = '../output/'):
        with h5py.File(path + fn, 'r') as f:
            self.t = np.array(f['t'])
            self.rho_gs = np.array(f['rho_gs'])
            if self.wavefunc:
                self.rho_trap = np.array(f['rho_trap'])
                self.psi = np.array(f['psi'])
            else:
                self.rho_trap = None
                self.psi = None


def append_to_table(f: h5py.File, dset: str, t, dtype=np.float):
    t = np.array([t]).reshape(1, -1)
    if dset in f.keys():
        t_table = f[dset]
        t_table.resize(t_table.shape[0] + 1, axis=0)
        t_table[-1, :] = t
    else:
        f.create_dataset(dset,
                         data=t,
                         dtype=dtype,
                         chunks=True,
                         maxshape=(None, t.shape[1]))