import numpy as np
import torch
from torch.autograd import Variable

from ....core.model_tools.manifolds.exponential_interface import ExponentialInterface
from ....support.utilities.general_settings import Settings

import logging
logger = logging.getLogger(__name__)


"""
Class with a parametric inverse metric: $$g_{\theta}(q) = \sum_{i=1}^n \alpha_i \exp{-\frac {\|x-q\|^2} {2 \sigma^2}$$ 
The metric_parameters for this class is the set of symmetric positive definite matrices from which we interpolate. 
It is a (nb_points, dimension*(dimension+1)/2) tensor, enumerated by lines.
"""

class ParametricExponential(ExponentialInterface):

    def __init__(self):
        ExponentialInterface.__init__(self)
        self.dimension = None
        self.number_of_interpolation_points = None
        self.width = None
        # List of points in the space, from which the metric is interpolated
        self.interpolation_points_torch = None
        # Tensor, shape (number of points, dimension, dimension)
        self.interpolation_values_torch = None

        self.diagonal_indices = None

        self.has_closed_form = False
        self.has_closed_form_dp = True
        self.has_closed_form_parallel_transport = False

    def inverse_metric(self, q):
        squared_distances = torch.sum(((self.interpolation_points_torch - q)**2.).view(self.number_of_interpolation_points, self.dimension), 1)
        self.last_inverse_metric = torch.sum(self.interpolation_values_torch
                         * torch.exp(-1.*squared_distances/self.width**2)
                         .view(self.number_of_interpolation_points, 1, 1)
                         .expand(-1, -1, self.dimension), 0)
        #val = np.sqrt(.2 / self.number_of_interpolation_points)
        #self.last_inverse_metric += torch.tensor(np.array([[val,0],[0,val]]))
        return self.last_inverse_metric

    def dp(self, q, p):
        differences = (q - self.interpolation_points_torch).view(self.number_of_interpolation_points, self.dimension)
        psp = torch.bmm(p.view(1, 1, self.dimension).expand(self.number_of_interpolation_points, 1, self.dimension),
                         torch.bmm(self.interpolation_values_torch,
                    p.view(1, self.dimension, 1).expand(self.number_of_interpolation_points, self.dimension, 1)))\
            .view(self.number_of_interpolation_points, 1)\
            .expand(self.number_of_interpolation_points, self.dimension)

        norm_gradient = differences
        weight = torch.exp(- torch.sum(differences**2, 1)/self.width**2).view(self.number_of_interpolation_points, -1).expand(self.number_of_interpolation_points, self.dimension)
        return -1/self.width**2 * torch.sum(psp * norm_gradient * weight, 0)

    def set_parameters(self, extra_parameters):
        """
        In this case, the parameters are the interpolation values
        """

        dim = Settings().dimension
        size = extra_parameters.size()
        assert size[0] == self.interpolation_points_torch.size()[0]
        assert size[1] == dim * (dim + 1) / 2
        symmetric_matrices = ParametricExponential.uncholeskify(extra_parameters, dim)
        if self.dimension is None:
            self.dimension = dim

        self.interpolation_values_torch = symmetric_matrices
        self.number_of_interpolation_points = extra_parameters.size()[0]
        self.is_modified = True

    @staticmethod
    def uncholeskify(l, dim):
        """
        Takes a tensor l of shape (n_cp, dimension*(dimension+1)/2)
        and returns out a tensor of shape (n_cp, dimension, dimension)
        such that out[i] = Upper(l[i]).transpose() * Upper(l[i])
        """
        out = Variable(torch.from_numpy(np.zeros((l.size()[0], dim, dim))).type(Settings().tensor_scalar_type))
        ones = torch.ones(dim, dim).type(Settings().tensor_integer_type)
        for i in range(l.size()[0]):
            aux = Variable(torch.from_numpy(np.zeros((dim, dim))).type(Settings().tensor_scalar_type))
            aux[torch.triu(ones) == 1] = l[i]
            out[i] = aux
        return torch.bmm(torch.transpose(out, 1, 2), out)

    # @staticmethod
    # def choleskify(l):
    #     logger.info("Choleskify needs to be checked !")
    #     return l[:, torch.triu(torch.ones(3, 3)) == 1]

    def _get_diagonal_indices(self):
        if self.diagonal_indices is None:
            diagonal_indices = []
            spacing = 0
            pos_in_line = 0
            dim = Settings().dimension
            for j in range(int(dim*(dim+1)/2) - 1, -1, -1):
                if pos_in_line == spacing:
                    spacing += 1
                    pos_in_line = 0
                    diagonal_indices.append(j)
                else:
                    pos_in_line += 1
            self.diagonal_indices = np.array(diagonal_indices)

        return self.diagonal_indices

    def project_metric_parameters(self, metric_parameters):
        """
        :param metric_parameters: numpy array of shape (number of points, dimension*(dimension+1)/2)
        we must check positivity of the diagonal coefficients of the lower matrices.
        :return:
        """
        diagonal_indices = self._get_diagonal_indices()

        # Positivity for each diagonal coefficient.
        for i in range(len(metric_parameters)):
            for j in diagonal_indices:
                if metric_parameters[i][j] < 0:
                    metric_parameters[i][j] = 0

        # Sum to one for each diagonal coefficient.
        for j in diagonal_indices:
            metric_parameters[:, j] /= np.sqrt(np.sum(metric_parameters[:, j]**2))

        return metric_parameters

    def project_metric_parameters_gradient(self, metric_parameters, metric_parameters_gradient):
        """
        Projection to ensure identifiability of the geodesic parametrizations.
        """
        diagonal_indices = self._get_diagonal_indices()

        out = metric_parameters_gradient

        for j in diagonal_indices:
            orthogonal_gradient = metric_parameters[:, j]
            orthogonal_gradient /= np.linalg.norm(orthogonal_gradient)
            out[:, j] -= np.dot(out[:, j], orthogonal_gradient)

        return out