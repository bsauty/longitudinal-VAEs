import torch
from torch.autograd import Variable
import time
from ....core.model_tools.manifolds.generic_geodesic import GenericGeodesic
from ....support.utilities.general_settings import Settings

import logging
logger = logging.getLogger(__name__)


class GenericSpatiotemporalReferenceFrame:
    """
    Spatiotemporal reference frame based on exp-parallelization.
    It uses a geodesic and an exponential + the parallel transport. It needs an exponential factory to construct itself.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, exponential_factory):
        self.geodesic = GenericGeodesic(exponential_factory)
        self.exponential = exponential_factory.create()
        self.factory = exponential_factory

        self.modulation_matrix_t0 = None
        self.projected_modulation_matrix_t0 = None
        self.number_of_sources = None
        self.transport_is_modified = True
        self.no_parallel_transport = None

        self.times = None
        self.position_t = None
        self.projected_modulation_matrix_t = None

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_concentration_of_time_points(self, ctp):
        self.geodesic.concentration_of_time_points = ctp

    def set_number_of_time_points(self, ntp):
        self.exponential.number_of_time_points = ntp

    def set_position_t0(self, td):
        self.geodesic.set_position_t0(td)
        self.transport_is_modified = True

    def set_velocity_t0(self, v):
        self.geodesic.set_velocity_t0(v)
        self.transport_is_modified = True

    def set_modulation_matrix_t0(self, mm):
        self.modulation_matrix_t0 = mm
        self.number_of_sources = mm.size()[1]
        if self.number_of_sources > 0:
            self.transport_is_modified = True

    def set_t0(self, t0):
        self.geodesic.set_t0(t0)
        self.transport_is_modified = True

    def set_tmin(self, tmin):
        self.geodesic.set_tmin(tmin)
        self.transport_is_modified = True

    def set_tmax(self, tmax):
        self.geodesic.set_tmax(tmax)
        self.transport_is_modified = True

    def set_metric_parameters(self, metric_parameters):
        self.geodesic.set_parameters(metric_parameters)
        self.exponential.set_parameters(metric_parameters)
        self.transport_is_modified = True

    def get_times(self):
        return self.times

    def get_position_exponential(self, t, sources=None):

        # Assert for coherent length of attribute lists.
        assert len(self.position_t) == len(self.times)
        if not self.no_parallel_transport:
            assert len(self.times) == len(self.projected_modulation_matrix_t)

        # Initialize the returned exponential.
        exponential = self.factory.create()

        #Case without sources:
        if sources is None:
            raise RuntimeError("I cannot multithread when no parallel transport is required.")

        # Deal with the special case of a geodesic reduced to a single point.
        if len(self.times) == 1:
            logger.info('>> The spatiotemporal reference frame geodesic seems to be reduced to a single point.')
            exponential.set_initial_position(self.position_t[0])
            if self.exponential_has_closed_form:
                exponential.set_initial_velocity(torch.mm(self.projected_modulation_matrix_t[0],
                                                     sources.unsqueeze(1)).view(self.geodesic.momenta_t0.size()))
            else:
                exponential.set_initial_momenta(torch.mm(self.projected_modulation_matrix_t[0],
                                                     sources.unsqueeze(1)).view(self.geodesic.momenta_t0.size()))
            return exponential

        # Standard case.
        index, weight_left, weight_right = self.geodesic.get_interpolation_index_and_weights(t)
        position = weight_left * self.position_t[index - 1] + weight_right * self.position_t[index]
        modulation_matrix = weight_left * self.projected_modulation_matrix_t[index - 1] \
                            + weight_right * self.projected_modulation_matrix_t[index]
        space_shift = torch.mm(modulation_matrix, sources.unsqueeze(1)).view(self.geodesic.velocity_t0.size())

        exponential.set_initial_position(position)
        if exponential.has_closed_form:
            exponential.set_initial_velocity(space_shift)
        else:
            exponential.set_initial_momenta(space_shift)

        return exponential

    def get_position(self, t, sources=None):

        # Case of no transport (e.g. dimension = 1)
        if sources is None:
            # assert self.no_parallel_transport, "Should not happen. (Or could it :o ?)"
            return self.geodesic.get_geodesic_point(t)

        # General case
        else:
            # Assert for coherent length of attribute lists.
            assert len(self.position_t) == len(self.projected_modulation_matrix_t) == len(self.times)

            # Deal with the special case of a geodesic reduced to a single point.
            if len(self.times) == 1:
                logger.info('>> The spatiotemporal reference frame geodesic seems to be reduced to a single point.')
                self.exponential.set_initial_position(self.position_t[0])

                # Little subtlety here (not so clean btw): closed_form exponential returns transported velocities
                # Non closed form exponential returns transported momenta.
                if self.exponential.has_closed_form:
                    self.exponential.set_initial_velocity(torch.mm(self.projected_modulation_matrix_t[0],
                                                              sources.unsqueeze(1)).view(self.geodesic.velocity_t0.size()))
                else:
                    self.exponential.set_initial_momenta(torch.mm(self.projected_modulation_matrix_t[0],
                                                                   sources.unsqueeze(1)))
                self.exponential.update()
                return self.exponential.get_final_position()

            # Standard case.
            index, weight_left, weight_right = self.geodesic.get_interpolation_index_and_weights(t)
            position = weight_left * self.position_t[index - 1] + weight_right * self.position_t[index]
            modulation_matrix = weight_left * self.projected_modulation_matrix_t[index - 1] \
                                + weight_right * self.projected_modulation_matrix_t[index]
            space_shift = torch.mm(modulation_matrix, sources.unsqueeze(1)).view(self.geodesic.velocity_t0.size())

            self.exponential.set_initial_position(position)
            if self.exponential.has_closed_form:
                self.exponential.set_initial_velocity(space_shift)
            else:
                self.exponential.set_initial_momenta(space_shift)
            self.exponential.update()

            return self.exponential.get_final_position()

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Update the geodesic, and compute the parallel transport of each column of the modulation matrix along
        this geodesic, ignoring the tangential components.
        """
        # Update the geodesic.
        self.geodesic.update()
        
        # Convenient attributes for later use.
        self.times = self.geodesic.get_times()
        self.position_t = self.geodesic.get_geodesic_trajectory()

        if not self.no_parallel_transport and self.transport_is_modified:
            # Initializes the projected_modulation_matrix_t attribute size.
            self.projected_modulation_matrix_t = \
                [Variable(torch.zeros(self.modulation_matrix_t0.size()).type(Settings().tensor_scalar_type),
                          requires_grad=False) for _ in range(len(self.position_t))]

            # Transport each column, ignoring the tangential components.
            for s in range(self.number_of_sources):
                space_shift_t0 = self.modulation_matrix_t0[:, s].contiguous().view(self.geodesic.velocity_t0.size())
                space_shift_t = self.geodesic.parallel_transport(space_shift_t0, with_tangential_component=False)

                #assert len(space_shift_t) == len(self.projected_modulation_matrix_t)

                # Set the result correctly in the projected_modulation_matrix_t attribute.
                # TODO : here we delete the project modulation matrix part to test a constant modulation matrix
                for t, space_shift in enumerate(space_shift_t):
                    if torch.sum(space_shift).isnan().any():
                        print('OK the NAN comes from here')
                    self.projected_modulation_matrix_t[t][:, s] = space_shift.view(-1)

                #for t in range(len(self.projected_modulation_matrix_t)):
                #    self.projected_modulation_matrix_t[t][:, s] = space_shift_t0.view(-1)

            self.transport_is_modified = False

        # Because in multi-threading we instantiate exponentials.
        if self.factory.manifold_type == 'parametric':
            self.factory.manifold_parameters['interpolation_points_torch'] = self.exponential.interpolation_points_torch
            self.factory.manifold_parameters['interpolation_values_torch'] = self.exponential.interpolation_values_torch

    def project_metric_parameters(self, metric_parameters):
        return self.exponential.project_metric_parameters(metric_parameters)

    def project_metric_parameters_gradient(self, metric_parameters, metric_parameters_gradient):
        return self.exponential.project_metric_parameters_gradient(metric_parameters, metric_parameters_gradient)

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, root_name, objects_name, objects_extension, template,
              write_adjoint_parameters=False, write_exponential_flow=False):
        pass
        # # Write the geodesic -------------------------------------------------------------------------------------------
        # self.geodesic.write(root_name, objects_name, objects_extension, template, write_adjoint_parameters)
        #
        # # Write the exp-parallel curves --------------------------------------------------------------------------------
        # # Initialization.
        # template_data_memory = template.get_points()
        #
        # # Core loop.
        # times = self.geodesic.get_times()
        # for t, (time, modulation_matrix) in enumerate(zip(times, self.projected_modulation_matrix_t)):
        #     for s in range(self.number_of_sources):
        #         space_shift = modulation_matrix[:, s].contiguous().view(self.geodesic.momenta_t0.size())
        #         self.exponential.set_initial_template_data(self.template_data_t[t])
        #         self.exponential.set_initial_control_points(self.control_points_t[t])
        #

        #         if self.exponential.has_closed_form:
        #             self.exponential.set_initial_velocity(space_shift)
        #         else:
        #             self.exponential.set_initial_momenta(space_shift)
        #         self.exponential.update()
        #         deformed_points = self.exponential.get_template_data()
        #
        #         names = []
        #         for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
        #             name = root_name + '__IndependentComponent_' + str(s) + '__' + object_name + '__tp_' + str(t) \
        #                    + ('__age_%.2f' % time) + object_extension
        #             names.append(name)
        #         template.set_data(deformed_points.data.numpy())
        #         template.write(names)
        #
        #         # Massive writing.
        #         if write_exponential_flow:
        #             names = []
        #             for k, (object_name, object_extension) in enumerate(zip(objects_name, objects_extension)):
        #                 name = root_name + '__IndependentComponent_' + str(s) + '__' + object_name + '__tp_' + str(t) \
        #                        + ('__age_%.2f' % time) + '__ExponentialFlow'
        #                 names.append(name)
        #             self.exponential.write_flow(names, objects_extension, template, write_adjoint_parameters)
        #
        # # Finalization.
        # template.set_data(template_data_memory)



