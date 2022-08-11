import torch

from .....core import default
from .....core.observations.deformable_objects.landmarks.landmark import Landmark
from .....support import utilities

import logging
logger = logging.getLogger(__name__)


class PointCloud(Landmark):
    """
    Points in 2D or 3D space, seen as measures
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, points, weights=None):
        Landmark.__init__(self, points)
        self.type = 'PointCloud'

        self.connectivity = weights

        # All of these are torch tensor attributes.
        self.centers, self.normals = PointCloud._get_centers_and_normals(torch.from_numpy(points))

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    @staticmethod
    def _get_centers_and_normals(points,
                                 tensor_scalar_type=default.tensor_scalar_type,
                                 tensor_integer_type=default.tensor_integer_type,
                                 device='cpu'):

        centers = utilities.move_data(points, dtype=tensor_scalar_type, device=device)
        normals = utilities.move_data(torch.ones(points.size(0), 1) / points.size(0),
                                      device=device, dtype=tensor_scalar_type)

        assert torch.device(device) == centers.device == normals.device
        return centers, normals

    def get_centers_and_normals(self, points=None,
                                tensor_scalar_type=default.tensor_scalar_type,
                                tensor_integer_type=default.tensor_integer_type,
                                device='cpu'):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals (which are tangents in this case) and centers
        """
        if points is None:
            if not self.is_modified:
                return (utilities.move_data(self.centers, dtype=tensor_scalar_type, device=device),
                        utilities.move_data(self.normals, dtype=tensor_scalar_type, device=device))

            else:
                logger.debug('Call of PointCloud.get_centers_and_normals with is_modified=True flag.')
                points = torch.from_numpy(self.points)

        return PointCloud._get_centers_and_normals(
            points,  tensor_scalar_type=tensor_scalar_type, tensor_integer_type=tensor_integer_type, device=device)
