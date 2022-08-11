import torch
from torch.autograd import Variable

from ....core.model_tools.manifolds.exponential_interface import ExponentialInterface
from ....support.utilities.general_settings import Settings

import logging
logger = logging.getLogger(__name__)

"""
Straight lines.
"""


# This implementation is a bit dirty ... the parallel transport method works in Z space
# only the closed_form method does the conversion to the Y space.
class EuclideanExponential(ExponentialInterface):

    def __init__(self, dimension=2):
        # Mother class constructor
        ExponentialInterface.__init__(self)
        self.has_closed_form = True
        self.has_closed_form_parallel_transport = True
        self.dimension = dimension
        logger.info(f"Setting the Euclidean exponential dimension to {dimension} from the settings")

    def inverse_metric(self, q):
        return Variable(torch.eye(self.dimension).type(Settings().tensor_scalar_type))

    def closed_form(self, q, v, t):
        return q+v*t

    def parallel_transport_closed_form(self, vector_to_transport, t, with_tangential_components=True):
        if with_tangential_components:
            return vector_to_transport
        else:
            sp = torch.dot(self.initial_velocity, vector_to_transport)
            vector_to_transport_orthogonal = vector_to_transport - sp * self.initial_velocity / torch.dot(self.initial_velocity, self.initial_velocity)
            return vector_to_transport_orthogonal

