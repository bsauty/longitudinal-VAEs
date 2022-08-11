import numpy as np
import torch

from .....core import default
from .....core.observations.deformable_objects.landmarks.landmark import Landmark
from .....support import utilities

import logging
logger = logging.getLogger(__name__)


class SurfaceMesh(Landmark):
    """
    3D Triangular mesh.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, points, triangles):
        Landmark.__init__(self, points)
        self.type = 'SurfaceMesh'

        self.connectivity = triangles

        # All of these are torch tensor attributes.
        self.centers, self.normals = SurfaceMesh._get_centers_and_normals(
            torch.from_numpy(points), torch.from_numpy(triangles))

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def remove_null_normals(self):
        _, normals = self.get_centers_and_normals()
        triangles_to_keep = torch.nonzero(torch.norm(normals, 2, 1) != 0)

        if len(triangles_to_keep) < len(normals):
            logger.info(
                'I detected {} null area triangles, I am removing them'.format(len(normals) - len(triangles_to_keep)))
            new_connectivity = self.connectivity[triangles_to_keep.view(-1)]
            new_connectivity = np.copy(new_connectivity)
            self.connectivity = new_connectivity

            # Updating the centers and normals consequently.
            self.centers, self.normals = SurfaceMesh._get_centers_and_normals(
                torch.from_numpy(self.points), torch.from_numpy(self.connectivity))

    @staticmethod
    def _get_centers_and_normals(points, triangles,
                                 tensor_scalar_type=default.tensor_scalar_type,
                                 tensor_integer_type=default.tensor_integer_type,
                                 device='cpu'):

        points = utilities.move_data(points, dtype=tensor_scalar_type, device=device)
        triangles = utilities.move_data(triangles, dtype=tensor_integer_type, device=device)

        a = points[triangles[:, 0]]
        b = points[triangles[:, 1]]
        c = points[triangles[:, 2]]
        centers = (a + b + c) / 3.
        normals = torch.cross(b - a, c - a) / 2

        assert torch.device(device) == centers.device == normals.device
        return centers, normals

    def get_centers_and_normals(self, points=None,
                                tensor_scalar_type=default.tensor_scalar_type,
                                tensor_integer_type=default.tensor_integer_type,
                                device='cpu'):
        """
        Given a new set of points, use the corresponding connectivity available in the polydata
        to compute the new normals, all in torch
        """
        if points is None:
            if not self.is_modified:
                return (utilities.move_data(self.centers, dtype=tensor_scalar_type, device=device),
                        utilities.move_data(self.normals, dtype=tensor_scalar_type, device=device))

            else:
                logger.debug('Call of SurfaceMesh.get_centers_and_normals with is_modified=True flag.')
                points = torch.from_numpy(self.points)

        return SurfaceMesh._get_centers_and_normals(
            points, torch.from_numpy(self.connectivity),
            tensor_scalar_type=tensor_scalar_type, tensor_integer_type=tensor_integer_type, device=device)
