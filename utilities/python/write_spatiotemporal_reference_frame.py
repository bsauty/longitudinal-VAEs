import os
import sys

# logger.info(sys.path)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../../../')

import torch

from api.deformetrica import Deformetrica
from deformetrica import get_model_options

from in_out.xml_parameters import XmlParameters
from core.model_tools.deformations.spatiotemporal_reference_frame import SpatiotemporalReferenceFrame
from in_out.dataset_functions import create_template_metadata
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
import support.kernels as kernel_factory
from in_out.array_readers_and_writers import *

if __name__ == '__main__':

    logger.info('')
    logger.info('##############################')
    logger.info('##### PyDeformetrica 1.0 #####')
    logger.info('##############################')
    logger.info('')

    """
    Read command line, prepare output folder, read model xml file.
    """

    assert len(sys.argv) >= 2, 'Usage: ' + sys.argv[0] + " <model.xml> <optional --output-dir=path_to_output>"

    model_xml_path = sys.argv[1]

    if len(sys.argv) > 2:
        output_dir = sys.argv[2][len("--output-dir="):]
        logger.info(">> Setting output directory to:", output_dir)
    else:
        output_dir = 'output'

    if not os.path.exists(output_dir):
        logger.info('>> Creating the output directory: "' + output_dir + '"')
        os.makedirs(output_dir)

    xml_parameters = XmlParameters()
    xml_parameters._read_model_xml(model_xml_path)

    deformetrica = Deformetrica(output_dir=output_dir)
    template_specifications, model_options, _ = deformetrica.further_initialization(
        'Shooting', xml_parameters.template_specifications, get_model_options(xml_parameters))

    """
    Load the template, control points, momenta, modulation matrix.
    """

    # Template.
    t_list, objects_name, objects_name_extension, _, _ = create_template_metadata(template_specifications)

    template = DeformableMultiObject(t_list)
    template_data = {key: torch.from_numpy(value).type(model_options['tensor_scalar_type'])
                     for key, value in template.get_data().items()}
    template_points = {key: torch.from_numpy(value).type(model_options['tensor_scalar_type'])
                       for key, value in template.get_points().items()}

    # Control points.
    control_points = read_2D_array(model_options['initial_control_points'])
    logger.info('>> Reading ' + str(len(control_points)) + ' initial control points from file: '
          + model_options['initial_control_points'])
    control_points = torch.from_numpy(control_points).type(model_options['tensor_scalar_type'])

    # Momenta.
    momenta = read_3D_array(model_options['initial_momenta'])
    logger.info('>> Reading initial momenta from file: ' + model_options['initial_momenta'])
    momenta = torch.from_numpy(momenta).type(model_options['tensor_scalar_type'])

    # Modulation matrix.
    modulation_matrix = read_2D_array(model_options['initial_modulation_matrix'])
    if len(modulation_matrix.shape) == 1:
        modulation_matrix = modulation_matrix.reshape(-1, 1)
    logger.info('>> Reading ' + str(modulation_matrix.shape[1]) + '-source initial modulation matrix from file: '
          + model_options['initial_modulation_matrix'])
    modulation_matrix = torch.from_numpy(modulation_matrix).type(model_options['tensor_scalar_type'])

    """
    Instantiate the spatiotemporal reference frame, update and write.
    """

    spatiotemporal_reference_frame = SpatiotemporalReferenceFrame(
        kernel=kernel_factory.factory(kernel_type=model_options['deformation_kernel_type'],
                                      kernel_width=model_options['deformation_kernel_width']),
        concentration_of_time_points=model_options['concentration_of_time_points'],
        number_of_time_points=model_options['number_of_time_points'],
        use_rk2_for_shoot=model_options['use_rk2_for_shoot'],
        use_rk2_for_flow=model_options['use_rk2_for_flow'],
    )

    spatiotemporal_reference_frame.set_template_points_t0(template_points)
    spatiotemporal_reference_frame.set_control_points_t0(control_points)
    spatiotemporal_reference_frame.set_momenta_t0(momenta)
    spatiotemporal_reference_frame.set_modulation_matrix_t0(modulation_matrix)
    spatiotemporal_reference_frame.set_t0(model_options['t0'])
    spatiotemporal_reference_frame.set_tmin(model_options['tmin'])
    spatiotemporal_reference_frame.set_tmax(model_options['tmax'])
    spatiotemporal_reference_frame.update()

    spatiotemporal_reference_frame.write('SpatioTemporalReferenceFrame',
                                         objects_name, objects_name_extension, template, template_data, output_dir,
                                         write_adjoint_parameters=True, write_exponential_flow=False)
