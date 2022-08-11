import logging
import warnings
import os

# Image readers
import PIL.Image as pimg
import nibabel as nib
import numpy as np

# Mesh readers
from vtk import vtkPolyDataReader, vtkSTLReader
from vtk.util import numpy_support as nps

from ..core.observations.deformable_objects.image import Image
from ..core.observations.deformable_objects.landmarks.landmark import Landmark
from ..core.observations.deformable_objects.landmarks.point_cloud import PointCloud
from ..core.observations.deformable_objects.landmarks.poly_line import PolyLine
from ..core.observations.deformable_objects.landmarks.surface_mesh import SurfaceMesh
from ..in_out.image_functions import normalize_image_intensities

logger = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING)


class DeformableObjectReader:
    """
    Creates PyDeformetrica objects from specified filename and object type.

    """

    connectivity_degrees = {'LINES': 2, 'VERTICES': 2, 'POLYGONS': 3}

    # Create a PyDeformetrica object from specified filename and object type.
    @staticmethod
    def create_object(object_filename, object_type, dimension=None):

        if object_type.lower() in ['SurfaceMesh'.lower(), 'PolyLine'.lower(), 'PointCloud'.lower(), 'Landmark'.lower()]:

            if object_type.lower() == 'SurfaceMesh'.lower():
                points, dimension, connectivity = DeformableObjectReader.read_file(
                    object_filename, dimension, extract_connectivity=True)
                out_object = SurfaceMesh(points, connectivity)
                out_object.remove_null_normals()

            elif object_type.lower() == 'PolyLine'.lower():
                points, dimension, connectivity = DeformableObjectReader.read_file(
                    object_filename, dimension, extract_connectivity=True)
                out_object = PolyLine(points, connectivity)

            elif object_type.lower() == 'PointCloud'.lower():
                try:
                    points, dimension, connectivity = DeformableObjectReader.read_file(
                        object_filename, dimension, extract_connectivity=True)
                    out_object = PointCloud(points, connectivity)

                except KeyError:
                    points, dimension = DeformableObjectReader.read_file(
                        object_filename, dimension, extract_connectivity=False)
                    out_object = PointCloud(points)

            elif object_type.lower() == 'Landmark'.lower():
                try:
                    points, dimension, connectivity = DeformableObjectReader.read_file(
                        object_filename, dimension, extract_connectivity=True)
                    out_object = Landmark(points)
                    out_object.set_connectivity(connectivity)

                except KeyError:
                    points, dimension = DeformableObjectReader.read_file(
                        object_filename, dimension, extract_connectivity=False)
                    out_object = Landmark(points)
            else:
                raise TypeError('Object type ' + object_type + ' was not recognized.')

        elif object_type.lower() == 'Image'.lower():
            if object_filename.find(".png") > 0:
                img_data = np.array(pimg.open(object_filename))
                dimension = len(img_data.shape)
                img_affine = np.eye(dimension + 1)
                if len(img_data.shape) > 2:
                    warnings.warn('Multi-channel images are not managed (yet). Defaulting to the first channel.')
                    dimension = 2
                    img_data = img_data[:, :, 0]

            elif object_filename.find(".npy") > 0:
                img_data = np.load(object_filename)
                dimension = len(img_data.shape)
                img_affine = np.eye(dimension + 1)

            elif object_filename.find(".nii") > 0 or object_filename.find(".nii.gz") > 0:
                img = nib.load(object_filename)
                img_data = img.get_data()
                dimension = len(img_data.shape)
                img_affine = img.affine
                assert len(img_data.shape) == 3, "Multi-channel images not available (yet!)."

            else:
                raise TypeError('Unknown image extension for file: %s' % object_filename)

            # Rescaling between 0. and 1.
            img_data, img_data_dtype = normalize_image_intensities(img_data)
            out_object = Image(img_data, img_data_dtype, img_affine)

            dimension_image = len(img_data.shape)
            if dimension_image != dimension:
                logger.warning(
                    'I am reading a {}d image but the dimension is set to {}'.format(dimension_image, dimension))

            # out_object.update()

        else:
            raise RuntimeError('Unknown object type: ' + object_type)

        return out_object

    @staticmethod
    def read_file(filename, dimension=None, extract_connectivity=False):
        """
        Routine to read VTK files based on the VTK library (available from conda).
        """
        assert os.path.isfile(filename), 'File does not exist: %s' % filename

        # choose vtk reader depending on file extension
        if filename.find(".vtk") > 0:
            poly_data_reader = vtkPolyDataReader()
        elif filename.find(".stl") > 0:
            poly_data_reader = vtkSTLReader()
        else:
            raise RuntimeError("Unrecognized file extension: " + filename)

        poly_data_reader.SetFileName(filename)
        poly_data_reader.Update()
        poly_data = poly_data_reader.GetOutput()

        points = nps.vtk_to_numpy(poly_data.GetPoints().GetData()).astype('float64')

        if dimension is None:
            dimension = 3
            if np.std(points[:, 2]) < 1e-10:
                dimension = 2
        assert dimension in [2, 3], 'The ambient-space dimension should be either 2 ou 3.'

        points = points[:, :dimension]

        if not extract_connectivity:
            return points, dimension

        else:
            lines = nps.vtk_to_numpy(poly_data.GetLines().GetData()).reshape((-1, 3))[:, 1:]
            polygons = nps.vtk_to_numpy(poly_data.GetPolys().GetData()).reshape((-1, 4))[:, 1:]

            if len(lines) == 0 and len(polygons) == 0:
                connectivity = None
            elif len(lines) > 0 and len(polygons) == 0:
                connectivity = lines
            elif len(lines) == 0 and len(polygons) > 0:
                connectivity = polygons
            else:
                if dimension == 2:
                    connectivity = lines
                else:
                    connectivity = polygons

            return points, dimension, connectivity
