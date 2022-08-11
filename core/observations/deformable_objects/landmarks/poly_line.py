import torch

from .....core import default
from .....core.observations.deformable_objects.landmarks.landmark import Landmark
from .....support import utilities

import logging
logger = logging.getLogger(__name__)


class PolyLine(Landmark):
    """
    Lines in 2D or 3D space
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, points, segments):
        Landmark.__init__(self, points)
        self.type = 'PolyLine'

        self.connectivity = segments

        # All these attributes are torch tensors.
        self.centers, self.normals = PolyLine._get_centers_and_normals(
            torch.from_numpy(points), torch.from_numpy(segments))

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    @staticmethod
    def _get_centers_and_normals(points, segments,
                                 tensor_scalar_type=default.tensor_scalar_type,
                                 tensor_integer_type=default.tensor_integer_type,
                                 device='cpu'):

        points = utilities.move_data(points, dtype=tensor_scalar_type, device=device)
        segments = utilities.move_data(segments, dtype=tensor_integer_type, device=device)

        a = points[segments[:, 0]]
        b = points[segments[:, 1]]
        centers = (a + b) / 2.
        normals = b - a

        assert torch.device(device) == centers.device == normals.device
        return centers, normals

    def get_centers_and_normals(self, points=None,
                                tensor_scalar_type=default.tensor_scalar_type,
                                tensor_integer_type=default.tensor_integer_type,
                                device='cpu'):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals (which are tangents in this case) and centers
        It's also a lazy initialization of those attributes !
        """
        if points is None:
            if not self.is_modified:
                return (utilities.move_data(self.centers, dtype=tensor_scalar_type, device=device),
                        utilities.move_data(self.normals, dtype=tensor_scalar_type, device=device))

            else:
                logger.debug('Call of PolyLine.get_centers_and_normals with is_modified=True flag.')
                points = torch.from_numpy(self.points)

        return PolyLine._get_centers_and_normals(
            points, torch.from_numpy(self.connectivity),
            tensor_scalar_type=tensor_scalar_type, tensor_integer_type=tensor_integer_type, device=device)

