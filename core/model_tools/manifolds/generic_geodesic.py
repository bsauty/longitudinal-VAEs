import os.path
import warnings

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from ....in_out.array_readers_and_writers import *

import logging
logger = logging.getLogger(__name__)


"""
Generic geodesic. It wraps a manifold (e.g. OneDimensionManifold) and uses 
its exponential attributes to make manipulations more convenient (e.g. backward and forward) 
The handling is radically different between exponential with closed form and the ones without.

The velocity is the initial criterion here. The translation to momenta is done, if needed, in the exponential objects.
"""

class GenericGeodesic:
    def __init__(self, exponential_factory):
        self.t0 = None
        self.tmax = None
        self.tmin = None
        self.concentration_of_time_points = 10

        self.position_t0 = None
        self.velocity_t0 = None

        self.forward_exponential = exponential_factory.create()
        self.backward_exponential = exponential_factory.create()

        self.manifold_type = exponential_factory.manifold_type

        self.is_modified = True
        self._times = None
        self._geodesic_trajectory = None

    def set_t0(self, t0):
        self.t0 = t0
        self.is_modified = True

    def set_tmin(self, tmin):
        self.tmin = tmin
        self.is_modified = True

    def set_tmax(self, tmax):
        self.tmax = tmax
        self.is_modified = True

    def set_position_t0(self, position_t0):
        self.position_t0 = position_t0
        self.is_modified = True

    def set_velocity_t0(self, velocity_t0):
        self.velocity_t0 = velocity_t0
        self.is_modified = True

    def set_concentration_of_time_points(self, ctp):
        self.concentration_of_time_points = ctp
        self.is_modified = True

    def get_geodesic_point(self, time):
        if self.forward_exponential.has_closed_form:
            return self.forward_exponential.closed_form(self.position_t0, self.velocity_t0, time - self.t0)
        else:
            j, weight_left, weight_right = self.get_interpolation_index_and_weights(time)
            geodesic_t = self.get_geodesic_trajectory()
            geodesic_point = weight_left * geodesic_t[j - 1] + weight_right * geodesic_t[j]
            return geodesic_point

    def get_interpolation_index_and_weights(self, t):
        time_np = t.cpu().data.numpy()
        times = self.get_times()
        if time_np <= self.t0:
            if self.backward_exponential.number_of_time_points <= 2:
                j = 1
            else:
                dt = (self.t0 - self.tmin) / (self.backward_exponential.number_of_time_points - 1)
                j = int((time_np - self.tmin) / dt) + 1
                # These tests are no longer relevant with the current if/else structure -> TODO : fix this
                #assert times[j - 1] <= time_np + 1e-3, "{} {} {}".format(j, time_np, times[j-1])
                #assert times[j] + 1e-3 >= time_np, "{} {} {}".format(j, time_np, times[j])
        else:
            if self.forward_exponential.number_of_time_points <= 2:
                j = len(times) - 1
            else:
                dt = (self.tmax - self.t0) / (self.forward_exponential.number_of_time_points - 1)
                j = min(len(times) - 1,
                        int((time_np - self.t0) / dt) + self.backward_exponential.number_of_time_points)

                assert times[j - 1] <= time_np
        #we have to add this test
        if time_np <= self.tmin:
            #logger.info('Want to estimate timepoints below tmin')
            j = 1
            weight_left = 1
            weight_right = 0
        elif self.tmax <= time_np:
            #logger.info('Want to estimate timepoints above tmax')
            j = len(times) - 1
            weight_left = 0
            weight_right = 1
        else:
            weight_left = (times[j] - t) / (times[j] - times[j - 1])
            weight_right = (t - times[j - 1]) / (times[j] - times[j - 1])

        if weight_left <= 0:
            weight_left = 0
        if weight_right <= 0:
            weight_right = 0

        return j, weight_left, weight_right

    def update(self):
        assert self.t0 >= self.tmin, "tmin should be smaller than t0"
        assert self.t0 <= self.tmax, "tmax should be larger than t0"
        # if not self.forward_exponential.has_closed_form:
            # Backward exponential -----------------------------------------------------------------------------------------
        delta_t = self.t0 - self.tmin
        self.backward_exponential.number_of_time_points = max(1, int(delta_t * self.concentration_of_time_points + 1.5))
        if self.is_modified:
            self.backward_exponential.set_initial_position(self.position_t0)
            self.backward_exponential.set_initial_velocity(- self.velocity_t0 * delta_t)
        if self.backward_exponential.number_of_time_points > 1:
            self.backward_exponential.update()
        else:
            self.backward_exponential._update_norm_squared()

        # Forward exponential ------------------------------------------------------------------------------------------
        # assert not self.forward_exponential.has_closed_form
        delta_t = self.tmax - self.t0
        self.forward_exponential.number_of_time_points = max(1, int(delta_t * self.concentration_of_time_points + 1.5))
        if self.is_modified:
            self.forward_exponential.set_initial_position(self.position_t0)
            self.forward_exponential.set_initial_velocity(self.velocity_t0 * delta_t)
        if self.forward_exponential.number_of_time_points > 1:
            self.forward_exponential.update()
        else:
            self.forward_exponential._update_norm_squared()
        
        #else:
         #   dt = 1/self.concentration_of_time_points
         #   self._times = [self.tmin + i * dt for i in range(int((self.tmax - self.tmin)*self.concentration_of_time_points))]
         #   self._geodesic_trajectory = self.get_geodesic_trajectory()

        self._update_geodesic_trajectory()
        self._update_times()
        self.is_modified = False

    def _update_times(self):
        # if not self.forward_exponential.has_closed_form:
        times_backward = [self.t0]
        if self.backward_exponential.number_of_time_points > 1:
            times_backward = np.linspace(
                self.t0, self.tmin, num=self.backward_exponential.number_of_time_points).tolist()

        times_forward = [self.t0]
        if self.forward_exponential.number_of_time_points > 1:
            times_forward = np.linspace(
                self.t0, self.tmax, num=self.forward_exponential.number_of_time_points).tolist()

        self._times = times_backward[::-1] + times_forward[1:]

        # else:
        #     delta_t = self.tmax - self.tmin
        #     self._times = np.linspace(self.tmin, self.tmax, delta_t * self.concentration_of_time_points+2)

    def get_times(self):
        if self.is_modified:
            msg = "Asking for geodesic times but the geodesic was modified and not updated"
            warnings.warn(msg)

        return self._times

    def _update_geodesic_trajectory(self):
        if not self.forward_exponential.has_closed_form:
            backward_geodesic_t = [self.backward_exponential.get_initial_position()]
            if self.backward_exponential.number_of_time_points > 1:
                backward_geodesic_t = self.backward_exponential.position_t

            forward_geodesic_t = [self.forward_exponential.get_initial_position()]
            if self.forward_exponential.number_of_time_points > 1:
                forward_geodesic_t = self.forward_exponential.position_t

            self._geodesic_trajectory = backward_geodesic_t[::-1] + forward_geodesic_t[1:]

    def get_geodesic_trajectory(self):
        if self.is_modified and not self.forward_exponential.has_closed_form:
            msg = "Trying to get geodesic trajectory in non updated geodesic."
            warnings.warn(msg)

        if self.forward_exponential.has_closed_form:
            trajectory = [self.get_geodesic_point(time)
                          for time in self.get_times()]
            return trajectory

        return self._geodesic_trajectory

    def set_parameters(self, extra_parameters):
        """
        Setting extra parameters for the exponentials
        e.g. parameters for the metric
        """
        self.forward_exponential.set_parameters(extra_parameters)
        self.backward_exponential.set_parameters(extra_parameters)
        self.is_modified = True

    def save_metric_plot(self):
        """
        Plot the metric (if it's 1D)
        """
        if Settings().dimension == 1:
            times = np.linspace(-0.4, 1.2, 300)
            times_torch = Variable(torch.from_numpy(times)).type(torch.DoubleTensor)
            metric_values = [self.forward_exponential.inverse_metric(t).data.numpy() for t in times_torch]
            # square_root_metric_values = [np.sqrt(elt) for elt in metric_values]
            plt.plot(times, metric_values)
            plt.ylim(0., 1.)
            plt.savefig(os.path.join(Settings().output_dir, "inverse_metric_profile.pdf"))
            plt.clf()

    def parallel_transport(self, vector_to_transport_t0, with_tangential_component=True):
        """
        :param vector_to_transport_t0: the vector to parallel transport, given at t0 and carried at position_t0
        :returns: the full trajectory of the parallel transport, from tmin to tmax
        ACHTUNG the returned trajectory is of velocities of closed form is available, and of momenta otherwise.
        """

        if self.is_modified:
            msg = "Trying to parallel transport but the geodesic object was modified, please update before."
            warnings.warn(msg)

        if self.backward_exponential.number_of_time_points > 1:
            backward_transport = self.backward_exponential.parallel_transport(vector_to_transport_t0,
                                                                              with_tangential_component)
        else:
            backward_transport = [vector_to_transport_t0]

        if self.forward_exponential.number_of_time_points > 1:
            forward_transport = self.forward_exponential.parallel_transport(vector_to_transport_t0,
                                                                            with_tangential_component)
        else:
            forward_transport = []

        return backward_transport[::-1] + forward_transport[1:]

    def _write(self):
        logger.info("Write method not implemented for the generic geodesic !")