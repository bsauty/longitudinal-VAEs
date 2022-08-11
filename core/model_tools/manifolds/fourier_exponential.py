import numpy as np
import torch
from torch.autograd import Variable

from ....core.model_tools.manifolds.exponential_interface import ExponentialInterface
from ....support.utilities.general_settings import Settings

import logging
logger = logging.getLogger(__name__)

"""
Class with a parametric inverse metric in Fourier form, with not so natural condition to ensure positivity...
"""

class FourierExponential(ExponentialInterface):

    def __init__(self):
        ExponentialInterface.__init__(self)
        self.dimension = Settings().dimension
        self.number_of_frequencies = 5
        self.coefficients = Variable(torch.from_numpy(np.random.uniform(0, 1, self.number_of_frequencies)).type(Settings().tensor_scalar_type))
        self.frequencies = Variable(torch.from_numpy(np.arange(1, self.number_of_frequencies + 1))).type(Settings().tensor_scalar_type)
        self.sigma = 5.

        self.has_closed_form = False
        self.has_closed_form_dp = False
        self.has_closed_form_parallel_transport = False

    # def set_fourier_coefficients(self, fourier_coefficients):
    #     """
    #     Torch tensors
    #     """
    #     self.fourier_coefficients = fourier_coefficients
    #     self.is_modified = True

    def inverse_metric(self, q):
        differences = q.view(self.dimension, -1).expand(self.dimension, self.dimension) - q
        coeffs = self.coefficients.view(-1, 1, 1).expand(self.number_of_frequencies, self.dimension, self.dimension)
        cosinuses = torch.cos(self.frequencies.view(-1, 1, 1).expand(self.number_of_frequencies, self.dimension, self.dimension) *
                            differences.view(self.dimension, self.dimension, 1).expand(self.dimension, self.dimension, self.number_of_frequencies)
                              .contiguous().view(self.number_of_frequencies, self.dimension, self.dimension))
        out = torch.sum(coeffs * cosinuses, 0) * torch.exp(-0.5 * differences**2 * self.sigma**2)

        return out

    # def dp(self, q, p):
    #     squared_distances = (self.interpolation_points_torch - q)**2.
    #     A = torch.exp(-1.*squared_distances/self.width**2.)
    #     differences = self.interpolation_points_torch - q
    #     return 1./self.width**2. * torch.sum(self.interpolation_values_torch*differences*A) * p**2

    # def set_parameters(self, extra_parameters):
    #     """
    #     In this case, the parameters are the fourier coefficients
    #     """
    #     logger.info("Is this implementation right ? In Fourier exponential ?")
    #     assert extra_parameters.size() == self.fourier_coefficients.size(),\
    #         "Wrong format of parameters"
    #     self.fourier_coefficients = extra_parameters
    #     self.is_modified = True


# def f(x, coefs):
#    ...:     kx = np.array([k * x for k in range(len(coefs))])
#    ...:     return np.sum(coefs * np.sin(kx*np.pi))

# def generate_random_coefs(num=10):
#    ...:     coefs = [1.]
#    ...:     for i in range(num-1):
#    ...:         r = np.random.uniform(0., coefs[-1]/2)
#    ...:         coefs.append(r)
#    ...:     return coefs

