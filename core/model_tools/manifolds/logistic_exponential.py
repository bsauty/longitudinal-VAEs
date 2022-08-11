import torch
import numpy as np

from ....core.model_tools.manifolds.exponential_interface import ExponentialInterface

import logging
logger = logging.getLogger(__name__)


"""
Exponential on \R for 1/(q**2(1-q)) metric i.e. logistic curves.
"""

class LogisticExponential(ExponentialInterface):

    def __init__(self):
        # Mother class constructor
        ExponentialInterface.__init__(self)
        self.has_closed_form = True
        self.has_closed_form_parallel_transport = True

    def inverse_metric(self, q):
        return torch.diag((q*(1-q))**2)

    def closed_form(self, q, v, t):
        return 1./(1 + (1/q - 1) * torch.exp(-1.*v/(q * (1-q)) * t))

    def closed_form_velocity(self, q, v, t):
        aux = torch.exp(-1. * v * t / (q * (1 - q)))
        return v/q**2 * aux/(1 + (1/q - 1) * aux)**2
        
    def parallel_transport_closed_form(self, vector_to_transport, t, with_tangential_components=True):
        """
        This is dirty but only serves as a proof of concept for the starmen dataset
        Since we chose v0 to be orthogonal to the modulation matrix directions transport is only identity
        """
        if with_tangential_components:
            return vector_to_transport
        else:
            sp = torch.dot(self.initial_velocity, vector_to_transport)
            vector_to_transport_orthogonal = vector_to_transport - sp * self.initial_velocity / torch.dot(self.initial_velocity, self.initial_velocity)
            return vector_to_transport_orthogonal