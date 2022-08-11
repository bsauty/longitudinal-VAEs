import numpy as np


class InverseWishartDistribution:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        self.degrees_of_freedom = 1
        self.scale_matrix = np.ones((1, 1))
        # self.scale_matrix_inverse = np.ones((1, 1))

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_scale_matrix(self, scale_matrix):
        self.scale_matrix = scale_matrix
        # self.scale_matrix_inverse = np.linalg.inv(scale_matrix)

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self):
        raise RuntimeError('The "sample" method is not implemented yet for the inverse Wishart distribution.')
        pass

    def compute_log_likelihood(self, observation_inverse):
        """
        Careful: the input is the inverse matrix already, and is assumed symmetric.
        See "A Bayesian Framework for Joint Morphometry of Surface and Curve meshes in Multi-Object Complexes",
        Gori et al. 2016 (equations 8 and 9).
        """
        return - 0.5 * self.degrees_of_freedom * (
            np.trace(np.dot(observation_inverse, self.scale_matrix)) - np.linalg.slogdet(observation_inverse)[1])
