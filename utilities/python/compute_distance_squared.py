import os
import sys

import torch
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

from in_out.deformable_object_reader import DeformableObjectReader
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
import support.kernels as kernel_factory


def compute_distance_squared(path_to_mesh_1, path_to_mesh_2, deformable_object_type, attachment_type,
                             kernel_width=None):
    reader = DeformableObjectReader()
    object_1 = reader.create_object(path_to_mesh_1, deformable_object_type.lower())
    object_2 = reader.create_object(path_to_mesh_2, deformable_object_type.lower())

    multi_object_1 = DeformableMultiObject([object_1])
    multi_object_2 = DeformableMultiObject([object_2])
    multi_object_attachment = MultiObjectAttachment([attachment_type], [kernel_factory.factory('torch', kernel_width)])

    return multi_object_attachment.compute_distances(
        {key: torch.from_numpy(value) for key, value in multi_object_1.get_points().items()},
        multi_object_1, multi_object_2).data.cpu().numpy()


if __name__ == '__main__':

    """
    Basic info printing.
    """

    logger.info('')
    logger.info('##############################')
    logger.info('##### PyDeformetrica 1.0 #####')
    logger.info('##############################')
    logger.info('')

    """
    Read command line.
    """

    assert len(sys.argv) in [5, 6], \
        'Usage: ' + sys.argv[0] \
        + " <path_to_mesh_1.vtk> <path_to_mesh_2.vtk> <deformable_object_type> <attachment_type> " \
          "[optional kernel_width]"

    path_to_mesh_1 = sys.argv[1]
    path_to_mesh_2 = sys.argv[2]
    deformable_object_type = sys.argv[3]
    attachment_type = sys.argv[4]

    kernel_width = None
    if len(sys.argv) == 6:
        kernel_width = float(sys.argv[5])

    if not os.path.isfile(path_to_mesh_1):
        raise RuntimeError('The specified source file ' + path_to_mesh_1 + ' does not exist.')
    if not os.path.isfile(path_to_mesh_2):
        raise RuntimeError('The specified source file ' + path_to_mesh_2 + ' does not exist.')

    """
    Core part.
    """

    logger.info(compute_distance_squared(
        path_to_mesh_1, path_to_mesh_2, deformable_object_type, attachment_type, kernel_width))
