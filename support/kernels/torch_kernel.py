import logging
import torch

from ...core import GpuMode, default
from ...support.kernels.abstract_kernel import AbstractKernel

logger = logging.getLogger(__name__)


def gaussian(r2, s):
    return torch.exp(-r2 / (s * s))


def binet(prs):
    return prs * prs


class TorchKernel(AbstractKernel):
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, gpu_mode=default.gpu_mode, kernel_width=None, **kwargs):
        super().__init__('torch', gpu_mode, kernel_width)

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def convolve(self, x, y, p, mode='gaussian'):
        res = None

        if mode in ['gaussian', 'pointcloud']:
            # move tensors with respect to gpu_mode
            x, y, p = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, y, p])
            assert x.device == y.device == p.device, 'x, y and p must be on the same device'

            sq = self._squared_distances(x, y)
            res = torch.mm(torch.exp(-sq / (self.kernel_width ** 2)), p)
            # res = torch.mm(1.0 / (1 + sq / self.kernel_width ** 2), p)

        elif mode == 'varifold':
            assert isinstance(x, tuple), 'x must be a tuple'
            assert len(x) == 2, 'tuple length must be 2'
            assert isinstance(y, tuple), 'y must be a tuple'
            assert len(y) == 2, 'tuple length must be 2'

            # tuples are immutable, mutability is needed to mode to device
            x = list(x)
            y = list(y)

            # move tensors with respect to gpu_mode
            x[0], x[1], y[0], y[1], p = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x[0], x[1], y[0], y[1], p])
            assert x[0].device == y[0].device == p.device, 'x, y and p must be on the same device'
            assert x[1].device == y[1].device == p.device, 'x, y and p must be on the same device'

            sq = self._squared_distances(x[0], y[0])
            res = torch.mm(gaussian(sq, self.kernel_width) * binet(torch.mm(x[1], torch.t(y[1]))), p)
        else:
            raise RuntimeError('Unknown kernel mode.')

        return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

    def convolve_gradient(self, px, x, y=None, py=None):
        if y is None:
            y = x
        if py is None:
            py = px

        # move tensors with respect to gpu_mode
        x, px, y, py = (self._move_to_device(t, gpu_mode=self.gpu_mode) for t in [x, px, y, py])
        assert px.device == x.device == y.device == py.device, 'tensors must be on the same device'

        # A=exp(-(x_i - y_j)^2/(ker^2)).
        sq = self._squared_distances(x, y)
        A = torch.exp(-sq / (self.kernel_width ** 2))
        # A = 1.0 / (1 + sq / self.kernel_width ** 2) ** 2

        # B=(x_i - y_j)*exp(-(x_i - y_j)^2/(ker^2))/(ker^2).
        B = self._differences(x, y) * A

        res = (- 2 * torch.sum(px * (torch.matmul(B, py)), 2) / (self.kernel_width ** 2)).t()
        return res.cpu() if self.gpu_mode is GpuMode.KERNEL else res

    ####################################################################################################################
    ### Auxiliary methods:
    ####################################################################################################################

    @staticmethod
    def _differences(x, y):
        """
        Returns the matrix of $(x_i - y_j)$.
        Output is of size (D, M, N).
        """
        x_col = x.t().unsqueeze(2)  # (M,D) -> (D,M,1)
        y_lin = y.t().unsqueeze(1)  # (N,D) -> (D,1,N)
        return x_col - y_lin
