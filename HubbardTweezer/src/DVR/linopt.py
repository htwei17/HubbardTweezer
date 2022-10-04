import torch
import numpy as np
from xitorch import LinearOperator

torch.set_default_dtype(torch.float64)


class LinearOperator(LinearOperator):

    def __init__(self, T: list[torch.Tensor], V: torch.Tensor, no: np.ndarray[int]):
        self.no = no
        N = np.prod(self.no)
        super().__init__(shape=(N, N), is_hermitian=True)
        self.T = T
        self.V = V

    def _mv(self, x: torch.Tensor):
        x = x.reshape(*self.no).double()
        # delta_xx' delta_yy' delta_zz' V(x,y,z)
        psi: torch.Tensor = self.V * x
        # T_xx' delta_yy' delta_zz'
        psi += torch.einsum('ij,jkl->ikl', self.T[0], x)
        # delta_xx' T_yy' delta_zz'
        psi += torch.einsum('jl,ilk->ijk', self.T[1], x)
        # delta_xx' delta_yy' T_zz'
        psi += torch.einsum('ij,klj->kli', self.T[2], x)
        return psi.reshape(-1)

    def _mm(self, m: torch.Tensor):
        return torch.stack([self._mv(x) for x in m.T], dim=1)

    def _getparamnames(self, prefix=""):
        return [prefix+"T", prefix+"V", prefix+"no"]
