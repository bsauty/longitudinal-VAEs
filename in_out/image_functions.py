import os.path
import sys

from ..core import default

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import numpy as np


def points_to_voxels_transform(points, affine):
    """
    Only useful for image + mesh cases. Not implemented yet.
    """
    return points


def metric_to_image_radial_length(length, affine):
    """
    Only useful for image + mesh cases. Not implemented yet.
    """
    return length


def normalize_image_intensities(intensities):

    dtype = str(intensities.dtype)
    if dtype == 'uint8':
        return np.array(intensities / 255.0, dtype=default.dtype), dtype
    else:
        return np.array(intensities, dtype=default.dtype), dtype


def rescale_image_intensities(intensities, dtype):
    tol = 1e-10
    if dtype == 'uint8':
        return (np.clip(intensities, tol, 1 - tol) * 255).astype('uint8')
    else:
        return intensities.astype('float32')
