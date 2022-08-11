from ....core.model_tools.manifolds.euclidean_exponential import EuclideanExponential
from ....core.model_tools.manifolds.fourier_exponential import FourierExponential
from ....core.model_tools.manifolds.logistic_exponential import LogisticExponential
from ....core.model_tools.manifolds.parametric_exponential import ParametricExponential
from ....support.utilities.general_settings import Settings

import logging
logger = logging.getLogger(__name__)

"""
Reads a dictionary of parameters, and returns the corresponding exponential object.
"""

class ExponentialFactory:
    def __init__(self):
        self.manifold_type = None
        self.manifold_parameters = None

    def set_manifold_type(self, manifold_type):
        self.manifold_type = manifold_type

    def set_parameters(self, manifold_parameters):
        self.manifold_parameters = manifold_parameters

    def create(self):
        """
        Returns an exponential for a manifold of a given type, using the parameters
        """
        if self.manifold_type == 'parametric':
            out = ParametricExponential()
            out.width = self.manifold_parameters['width']
            out.number_of_interpolation_points = self.manifold_parameters['interpolation_points_torch'].size()[0]
            out.interpolation_points_torch = self.manifold_parameters['interpolation_points_torch']
            out.interpolation_values_torch = self.manifold_parameters['interpolation_values_torch']
            out.dimension = out.interpolation_points_torch.size()[1]
            return out

        if self.manifold_type == 'fourier':
            return FourierExponential()

        if self.manifold_type == 'logistic':
            out = LogisticExponential()
            return out

        if self.manifold_type == 'deep':
            out = EuclideanExponential(dimension=self.manifold_parameters['latent_space_dimension'])
            return out

        if self.manifold_type == 'euclidean':
            out = EuclideanExponential(dimension=Settings().dimension)
            return out

        raise ValueError("Unrecognized manifold type in exponential factory")


if __name__ == '__main__':
    import torch
    import numpy as np
    from torch.autograd import Variable
    from deformetrica.core.model_tools.manifolds.generic_geodesic import GenericGeodesic
    from deformetrica.support.utilities.general_settings import Settings
    import matplotlib.pyplot as plt

    Settings().dimension = 2
    factory = ExponentialFactory()
    factory.set_manifold_type("fourier")
    geodesic = GenericGeodesic(factory)
    geodesic.set_t0(1.)
    geodesic.set_tmin(1.)
    geodesic.set_tmax(2.)
    geodesic.concentration_of_time_points = 20

    p0 = np.zeros(2)
    p0[0] = 0.2

    p0 = Variable(torch.from_numpy(p0)).type(Settings().tensor_scalar_type)
    v0 = Variable(torch.from_numpy(np.ones(2))).type(Settings().tensor_scalar_type)
    geodesic.set_position_t0(p0)
    geodesic.set_velocity_t0(v0)

    for i in range(10):
        geodesic.forward_exponential.coefficients = Variable(torch.from_numpy(np.random.uniform(0, 1, geodesic.forward_exponential.number_of_frequencies)).type(Settings().tensor_scalar_type))
        geodesic.is_modified = True
        geodesic.update()
        traj = np.array([elt.data.numpy() for elt in geodesic.get_geodesic_trajectory()])
        times = geodesic.get_times()
        for d in range(len(traj[0])):
            plt.plot(times, traj[:,d])
        plt.show()
