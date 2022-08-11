from abc import ABC, abstractmethod
import torch

from ...core import default

import logging
logger = logging.getLogger(__name__)


class AbstractKernel(ABC):
    def __init__(self, kernel_type='undefined', gpu_mode=default.gpu_mode, kernel_width=None):
        self.kernel_type = kernel_type
        self.kernel_width = kernel_width
        self.gpu_mode = gpu_mode
        logger.debug('instantiating kernel %s with kernel_width %s and gpu_mode %s. addr: %s',
                     self.kernel_type, self.kernel_width, self.gpu_mode, hex(id(self)))

    def __eq__(self, other):
        return self.kernel_type == other.kernel_type \
               and self.gpu_mode == other.gpu_mode \
               and self.kernel_width == other.kernel_width

    @staticmethod
    def hash(kernel_type, cuda_type, gpu_mode, *args, **kwargs):
        return hash((kernel_type, cuda_type, gpu_mode, frozenset(args), frozenset(kwargs.items())))

    def __hash__(self, **kwargs):
        return AbstractKernel.hash(self.kernel_type, None, self.gpu_mode, **kwargs)

    @abstractmethod
    def convolve(self, x, y, p, mode=None):
        raise NotImplementedError

    @abstractmethod
    def convolve_gradient(self, px, x, y=None, py=None):
        raise NotImplementedError

    def get_kernel_matrix(self, x, y=None):
        """
        returns the kernel matrix, A_{ij} = exp(-|x_i-x_j|^2/sigma^2)
        """
        if y is None:
            y = x
        assert (x.size(0) == y.size(0))
        sq = self._squared_distances(x, y)
        return torch.exp(-sq / (self.kernel_width ** 2))

    @staticmethod
    def _squared_distances(x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        return dist

    @staticmethod
    def _move_to_device(t, gpu_mode):
        """
        AUTO, FULL, NONE, KERNEL

        gpu_mode:
        - auto: use get_best_gpu_mode(model)
        - full: tensors should already be on gpu, if not, move them and do not return to cpu
        - none: tensors should be on cpu, if not, move them
        - kernel: tensors should be on cpu. Move them to cpu and return to cpu
        """

        from ...core import GpuMode
        from ...support import utilities

        if gpu_mode in [GpuMode.FULL, GpuMode.KERNEL]:
            # tensors should already be on the target device, if not, they will be moved
            dev, _ = utilities.get_best_device(gpu_mode=gpu_mode)
            t = utilities.move_data(t, device=dev)

        elif gpu_mode is GpuMode.NONE:
            # tensors should be on cpu. If not they should be moved to cpu.
            t = utilities.move_data(t, device='cpu')

        return t
