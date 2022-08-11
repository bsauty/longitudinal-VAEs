from math import sqrt

import numpy as np
import torch
from scipy.stats import truncnorm

from ...support import utilities


class MultiScalarTruncatedNormalDistribution:

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, variance=None, std=None):
        self.mean = None

        if variance is not None:
            self.set_variance(variance)
        elif std is not None:
            self.set_variance_sqrt(std)

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def get_mean(self):
        return self.mean

    def set_mean(self, m):
        self.mean = m

    def get_variance_sqrt(self):
        return self.variance_sqrt

    def set_variance_sqrt(self, std):
        self.variance_sqrt = std
        self.variance_inverse = 1.0 / std ** 2

    def set_variance(self, var):
        self.variance_sqrt = sqrt(var)
        self.variance_inverse = 1.0 / var

    def get_expected_mean(self):
        assert len(self.mean) == 1  # Only coded case for now.
        mean = self.mean[0]
        return float(truncnorm.stats(- mean / self.variance_sqrt, 100.0 * self.variance_sqrt,
                                     loc=mean, scale=self.variance_sqrt, moments='m'))

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self):
        out = np.zeros(self.mean.shape)
        for index, mean in np.ndenumerate(self.mean):
            out[index] = truncnorm.rvs(- mean / self.variance_sqrt, float('inf'), loc=mean, scale=self.variance_sqrt)
        return out

    def compute_log_likelihood(self, observation):
        """
        Fully numpy method.
        Returns only the part that includes the observation argument.
        """
        assert self.mean.size == 1 or self.mean.shape == observation.shape
        if np.min(observation) < 0.0:
            return - float('inf')
        else:
            delta = observation.ravel() - self.mean.ravel()
            return - 0.5 * self.variance_inverse * np.sum(delta ** 2)

    def compute_log_likelihood_torch(self, observation, tensor_scalar_type, device='cpu'):
        """
        Fully torch method.
        Returns only the part that includes the observation argument.
        """
        mean = utilities.move_data(self.mean, dtype=tensor_scalar_type, requires_grad=False, device=device)
        observation = utilities.move_data(observation, dtype=tensor_scalar_type, device=device)
        assert mean.detach().cpu().numpy().size == observation.detach().cpu().numpy().size, \
            'mean.detach().cpu().numpy().size = %d, \t observation.detach().cpu().numpy().size = %d' \
            % (mean.detach().cpu().numpy().size, observation.detach().cpu().numpy().size)

        if np.min(observation.detach().cpu().numpy()) < 0.0:
            return torch.sum(tensor_scalar_type([- float('inf')]))
        else:
            delta = observation.contiguous().view(-1, 1) - mean.contiguous().view(-1, 1)
            return -0.5 * torch.sum(delta ** 2) * self.variance_inverse
