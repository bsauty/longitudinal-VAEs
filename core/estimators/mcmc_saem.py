import torch
import logging
import os.path
import _pickle as pickle
from tqdm import tqdm
import time

from ...core import default
from ...core.estimator_tools.samplers.srw_mhwg_sampler import SrwMhwgSampler
from ...core.estimators.abstract_estimator import AbstractEstimator
from ...core.estimators.gradient_ascent import GradientAscent
from ...in_out.dataset_functions import create_scalar_dataset
from ...in_out.array_readers_and_writers import *
from ...support.utilities.general_settings import Settings

logger = logging.getLogger(__name__)

class McmcSaem(AbstractEstimator):
    """
    GradientAscent object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, statistical_model, dataset, optimization_method_type='undefined', individual_RER={},
                 max_iterations=default.max_iterations,
                 print_every_n_iters=default.print_every_n_iters, save_every_n_iters=default.save_every_n_iters,
                 sampler=default.sampler,
                 individual_proposal_distributions=default.individual_proposal_distributions,
                 sample_every_n_mcmc_iters=default.sample_every_n_mcmc_iters,
                 convergence_tolerance=default.convergence_tolerance,
                 callback=None, output_dir=default.output_dir,
                 scale_initial_step_size=default.scale_initial_step_size, initial_step_size=default.initial_step_size,
                 max_line_search_iterations=default.max_line_search_iterations,
                 line_search_shrink=default.line_search_shrink, line_search_expand=default.line_search_expand,
                 load_state_file=default.load_state_file, state_file=default.state_file,
                 **kwargs):

        super().__init__(statistical_model=statistical_model, dataset=dataset, name='McmcSaem',
                         # optimized_log_likelihood=optimized_log_likelihood,
                         max_iterations=max_iterations,
                         convergence_tolerance=convergence_tolerance,
                         print_every_n_iters=print_every_n_iters, save_every_n_iters=save_every_n_iters,
                         individual_RER=individual_RER,
                         callback=callback, state_file=state_file, output_dir=output_dir)

        assert optimization_method_type.lower() == self.name.lower()

        assert sampler.lower() == 'SrwMhwg'.lower(), \
            "The only available sampler for now is the Symmetric-Random-Walk Metropolis-Hasting-within-Gibbs " \
            "(SrwMhhwg) sampler."
        self.sampler = SrwMhwgSampler(individual_proposal_distributions=individual_proposal_distributions)

        self.current_mcmc_iteration = 0
        self.sample_every_n_mcmc_iters = sample_every_n_mcmc_iters
        self.save_every_n_iters = save_every_n_iters
        self.number_of_burn_in_iterations = None
        self._initialize_number_of_burn_in_iterations()  # Number of iterations without memory.
        self.memory_window_size = 1  # Size of the averaging window for the acceptance rates.

        self.number_of_trajectory_points = min(self.max_iterations, 500)
        self.save_model_parameters_every_n_iters = max(1, int(self.max_iterations / float(self.number_of_trajectory_points)))

        # Initialization of the gradient-based optimizer.
        # TODO let the possibility to choose all options (e.g. max_iterations, or ScipyLBFGS optimizer).
        self.gradient_based_estimator = GradientAscent(
            statistical_model, dataset,
            optimized_log_likelihood='class2',
            max_iterations=99, convergence_tolerance=convergence_tolerance,
            print_every_n_iters=1, save_every_n_iters=100000,
            scale_initial_step_size=scale_initial_step_size, initial_step_size=initial_step_size,
            max_line_search_iterations=max_line_search_iterations,
            line_search_shrink=line_search_shrink,
            line_search_expand=line_search_expand,
            output_dir=output_dir, individual_RER=individual_RER,
            optimization_method_type='GradientAscent',
            callback=callback
        )

        # If the load_state_file flag is active, restore context.
        if load_state_file:
            (self.current_iteration, parameters, self.sufficient_statistics, proposal_stds,
             self.current_acceptance_rates, self.average_acceptance_rates,
             self.current_acceptance_rates_in_window, self.average_acceptance_rates_in_window,
             self.model_parameters_trajectory, self.individual_random_effects_samples_stack) = self._load_state_file()
            self._set_parameters(parameters)
            self.sampler.set_proposal_standard_deviations(proposal_stds)
            logger.info("State file loaded, it was at iteration %d." % self.current_iteration)

        else:
            self.current_iteration = 0
            self.sufficient_statistics = None  # Dictionary of numpy arrays.
            self.current_acceptance_rates = {}  # Acceptance rates of the current iteration.
            self.average_acceptance_rates = {}  # Mean acceptance rates, computed over all past iterations.
            self.current_acceptance_rates_in_window = None  # Memory of the last memory_window_size acceptance rates.
            self.average_acceptance_rates_in_window = None  # Moving average of current_acceptance_rates_in_window.
            self.model_parameters_trajectory = None  # Memory of the model parameters along the estimation.
            self.individual_random_effects_samples_stack = None  # Stack of the last individual random effect samples.

            self._initialize_acceptance_rate_information()
            sufficient_statistics = self._initialize_sufficient_statistics()
            self._initialize_model_parameters_trajectory()
            self._initialize_individual_random_effects_samples_stack()

            # Ensures that all the model fixed effects are initialized.
            self.statistical_model.update_fixed_effects(self.dataset, sufficient_statistics)


    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Runs the MCMC-SAEM algorithm and updates the statistical model.
        """

        # Print initial console information.
        logger.info(f"---------------------------------- MCMC Iteration: {self.current_iteration} -------------------------------------")
        logger.info(f">> MCMC-SAEM algorithm launched for {self.max_iterations} iterations ({self.number_of_burn_in_iterations}  iterations of burn-in).")
        self.statistical_model.print(self.individual_RER)

        # Initialization of the average random effects realizations.
        averaged_population_RER = {key: np.zeros(value.shape) for key, value in self.population_RER.items()}
        averaged_individual_RER = {key: np.zeros(value.shape) for key, value in self.individual_RER.items()}

        # Main loop ----------------------------------------------------------------------------------------------------
        while self.callback_ret and self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            step = self._compute_step_size()
            # Simulation.
            current_model_terms = None
            
            # After a few MCMC steps with training of the network, only keep the maximize procedure
            if self.current_iteration == 100:
                self.statistical_model.has_maximization_procedure = False
            if self.current_iteration > 20:
                self.sample_every_n_mcmc_iters = 3
            for n in range(self.sample_every_n_mcmc_iters):
                self.current_mcmc_iteration += 1

                # Single iteration of the MCMC
                self.current_acceptance_rates, current_model_terms = self.sampler.sample(
                    self.statistical_model, self.dataset, self.population_RER, self.individual_RER,
                    current_model_terms)

                # Adapt proposal variances.
                self._update_acceptance_rate_information()
                if not (self.current_mcmc_iteration % self.memory_window_size):
                    self.average_acceptance_rates_in_window = {
                        key: np.mean(self.current_acceptance_rates_in_window[key])
                        for key in self.sampler.individual_proposal_distributions.keys()}
                    self.sampler.adapt_proposal_distributions(
                        self.average_acceptance_rates_in_window,
                        self.current_mcmc_iteration,
                        not self.current_iteration % self.print_every_n_iters and n == self.sample_every_n_mcmc_iters - 1)

            # Maximization for the class 1 fixed effects.
            sufficient_statistics = self.statistical_model.compute_sufficient_statistics(
                self.dataset, self.population_RER, self.individual_RER, model_terms=current_model_terms)
            self.sufficient_statistics = {key: value + step * (sufficient_statistics[key] - value) for key, value in
                                          self.sufficient_statistics.items()}
            self.statistical_model.update_fixed_effects(self.dataset, self.sufficient_statistics)

            #if (self.current_iteration < 50) :
             #   self.gradient_based_estimator.max_iterations = 2
              #  self.gradient_based_estimator.max_line_search_iterations = 4
            #else:
              #  self.gradient_based_estimator.max_iterations = 0
            #if (self.gradient_based_estimator.step is not None) and (self.gradient_based_estimator.step['v0'] < 1e-10):
               # self.gradient_based_estimator.max_iterations = 1

            if self.gradient_based_estimator.max_iterations :
                # Maximization for the class 2 fixed effects.
                fixed_effects_before_maximization = self.statistical_model.get_fixed_effects()
                self._maximize_over_fixed_effects()
                fixed_effects_after_maximization = self.statistical_model.get_fixed_effects()
                fixed_effects = {key: value + step * (fixed_effects_after_maximization[key] - value) for key, value in
                                 fixed_effects_before_maximization.items()}
                self.statistical_model.set_fixed_effects(fixed_effects)

            # Try to not normalize the parameters to not mess with the gradient descent
            self._normalize_individual_parameters(center_sources=True)

            # Averages the random effect realizations in the concentration phase.
            if step < 1.0:
                coefficient_1 = float(self.current_iteration + 1 - self.number_of_burn_in_iterations)
                coefficient_2 = (coefficient_1 - 1.0) / coefficient_1
                averaged_population_RER = {key: value * coefficient_2 + self.population_RER[key] / coefficient_1 for
                                           key, value in averaged_population_RER.items()}
                averaged_individual_RER = {key: value * coefficient_2 + self.individual_RER[key] / coefficient_1 for
                                           key, value in averaged_individual_RER.items()}
                self._update_individual_random_effects_samples_stack()

            else:
                averaged_individual_RER = self.individual_RER
                averaged_population_RER = self.population_RER

            # Saving, printing, writing.
            if not (self.current_iteration % self.save_model_parameters_every_n_iters):
                self._update_model_parameters_trajectory()
            if not (self.current_iteration % self.print_every_n_iters):
                self.print()
            if not (self.current_iteration % self.save_every_n_iters):
                self.write()

        # Finalization -------------------------------------------------------------------------------------------------
        self.population_RER = averaged_population_RER
        self.individual_RER = averaged_individual_RER
        #self._normalize_individual_parameters()
        self._update_model_parameters_trajectory()
        self.print()
        self.write()

    def print(self):
        """
        Prints information.
        """
        # Iteration number.
        logger.info('')
        logger.info(f"---------------------------------- MCMC Iteration: {self.current_iteration} -------------------------------------")

        # Averaged acceptance rates over all the past iterations.
        logger.info('>> Average acceptance rates (all past iterations):')
        for random_effect_name, average_acceptance_rate in self.average_acceptance_rates.items():
            logger.info(f"       {average_acceptance_rate}   {random_effect_name}")

        # Let the model under optimization print information about itself.
        self.statistical_model.print(self.individual_RER)

    def save_model_xml(self, path):
        """"""""""""""""""""""""""""""""
        """Creating a xml file"""
        """"""""""""""""""""""""""""""""

        model_xml = et.Element('data-set')
        model_xml.set('deformetrica-min-version', "3.0.0")

        model_type = et.SubElement(model_xml, 'model-type')
        model_type.text = "LongitudinalMetricLearning"

        dimension = et.SubElement(model_xml, 'dimension')
        dimension.text = str(Settings().dimension)

        estimated_alphas = np.loadtxt(os.path.join(path, 'LongitudinalMetricModel_alphas.txt'))
        estimated_onset_ages = np.loadtxt(
            os.path.join(path, 'LongitudinalMetricModel_onset_ages.txt'))

        initial_time_shift_std = et.SubElement(model_xml, 'initial-time-shift-std')
        initial_time_shift_std.text = str(np.std(estimated_onset_ages))

        initial_log_acceleration_std = et.SubElement(model_xml, 'initial-log-acceleration-std')
        initial_log_acceleration_std.text = str(np.std(np.log(estimated_alphas)))

        deformation_parameters = et.SubElement(model_xml, 'deformation-parameters')

        exponential_type = et.SubElement(deformation_parameters, 'exponential-type')
        exponential_type.text = xml_parameters.exponential_type

        interpolation_points = et.SubElement(deformation_parameters, 'interpolation-points-file')
        interpolation_points.text = os.path.join(path,
                                                 'LongitudinalMetricModel_interpolation_points.txt')
        kernel_width = et.SubElement(deformation_parameters, 'kernel-width')
        kernel_width.text = str(xml_parameters.deformation_kernel_width)

        concentration_of_timepoints = et.SubElement(deformation_parameters,
                                                    'concentration-of-timepoints')
        concentration_of_timepoints.text = str(xml_parameters.concentration_of_time_points)

        number_of_timepoints = et.SubElement(deformation_parameters,
                                             'number-of-timepoints')
        number_of_timepoints.text = str(xml_parameters.number_of_time_points)

        estimated_fixed_effects = np.load(os.path.join(path,
                                                       'LongitudinalMetricModel_all_fixed_effects.npy'),
                                          allow_pickle=True)[()]

        metric_parameters_file = et.SubElement(deformation_parameters,
                                               'metric-parameters-file')
        metric_parameters_file.text = os.path.join(path,
                                                   'LongitudinalMetricModel_metric_parameters.txt')

        if xml_parameters.number_of_sources is not None and xml_parameters.number_of_sources > 0:
            initial_sources_file = et.SubElement(model_xml, 'initial-sources')
            initial_sources_file.text = os.path.join(path, 'LongitudinalMetricModel_sources.txt')
            number_of_sources = et.SubElement(deformation_parameters, 'number-of-sources')
            number_of_sources.text = str(xml_parameters.number_of_sources)
            initial_modulation_matrix_file = et.SubElement(model_xml, 'initial-modulation-matrix')
            initial_modulation_matrix_file.text = os.path.join(path,
                                                               'LongitudinalMetricModel_modulation_matrix.txt')

        t0 = et.SubElement(deformation_parameters, 't0')
        t0.text = str(estimated_fixed_effects['reference_time'])

        v0 = et.SubElement(deformation_parameters, 'v0')
        v0.text = os.path.join(path, 'LongitudinalMetricModel_v0.txt')

        p0 = et.SubElement(deformation_parameters, 'p0')
        p0.text = os.path.join(path, 'LongitudinalMetricModel_p0.txt')

        initial_onset_ages = et.SubElement(modexl_xml, 'initial-onset-ages')
        initial_onset_ages.text = os.path.join(path,
                                               "LongitudinalMetricModel_onset_ages.txt")

        initial_log_accelerations = et.SubElement(model_xml, 'initial-log-accelerations')
        initial_log_accelerations.text = os.path.join(path, "LongitudinalMetricModel_log_accelerations.txt")

        model_xml_path = 'model_fitted.xml'
        doc = parseString((et.tostring(model_xml).decode('utf-8').replace('\n', '').replace('\t', ''))).toprettyxml()
        np.savetxt(model_xml_path, [doc], fmt='%s')

    def write(self, population_RER=None, individual_RER=None):
        """
        Save the current results.
        """
        # Call the write method of the statistical model.
        if population_RER is None:
            population_RER = self.population_RER
        if individual_RER is None:
            individual_RER = self.individual_RER
        self.statistical_model.write(self.dataset, population_RER, individual_RER, Settings().output_dir,
                                      update_fixed_effects=False, iteration=self.current_iteration)

        # Save the recorded model parameters trajectory.
        # self.model_parameters_trajectory is a list of dictionaries
        np.save(os.path.join(Settings().output_dir, self.statistical_model.name + '__EstimatedParameters__Trajectory.npy'),
                np.array(
                    {key: value[:(1 + int(self.current_iteration / float(self.save_model_parameters_every_n_iters)))]
                     for key, value in self.model_parameters_trajectory.items()}), allow_pickle=True)

        # Save the memorized individual random effects samples.
        if self.current_iteration > self.number_of_burn_in_iterations:
            np.save(os.path.join(Settings().output_dir,
                                 self.statistical_model.name + '__EstimatedParameters__IndividualRandomEffectsSamples.npy'),
                    {key: value[:(self.current_iteration - self.number_of_burn_in_iterations)] for key, value in
                     self.individual_random_effects_samples_stack.items()}, allow_pickle=True)

        # Dump state file.
        self._dump_state_file()

    ####################################################################################################################
    ### Private_maximize_over_remaining_fixed_effects() method and associated utilities:
    ####################################################################################################################

    def _maximize_over_fixed_effects(self):
        """
        Update the model fixed effects for which no closed-form update is available (i.e. based on sufficient
        statistics).
        """

        # Default optimizer, if not initialized in the launcher.
        # Should better be done in a dedicated initializing method. TODO.
        if self.statistical_model.has_maximization_procedure is not None \
                and self.statistical_model.has_maximization_procedure:
            self.statistical_model.maximize(individual_RER=self.individual_RER)
            # Update the longitudinal dataset 
            new_dataset = create_scalar_dataset(self.statistical_model.full_labels, self.statistical_model.full_encoded.numpy(), \
                                                self.statistical_model.full_timepoints)
            self.dataset = new_dataset

        self.gradient_based_estimator.initialize()

        if self.gradient_based_estimator.verbose > 0:
            logger.info('')
            logger.info('[ maximizing over the fixed effects with the %s optimizer ]'
                        % self.gradient_based_estimator.name)

        success = False
        while not success:
            try:
                self.gradient_based_estimator.update()
                success = True
            except RuntimeError as error:
                logger.info('>> ' + str(error.args[0]) + ' [ in mcmc_saem ]')
                self.statistical_model.adapt_to_error(error.args[1])

        if self.gradient_based_estimator.verbose > 0:
            logger.info('')
            logger.info('[ end of the gradient-based maximization ]')

        #if self.current_iteration < self.number_of_burn_in_iterations:
        #    self.statistical_model.preoptimize(self.dataset, self.individual_RER)

    ####################################################################################################################
    ### Other private methods:
    ####################################################################################################################

    def _compute_step_size(self):
        aux = self.current_iteration - self.number_of_burn_in_iterations + 1
        if aux <= 0:
            return 1.0
        else:
            return 1.0 / aux

    def _initialize_number_of_burn_in_iterations(self):
        if self.number_of_burn_in_iterations is None:
            # Because some models will set it manually (e.g. deep Riemannian models)
            if self.max_iterations > 400:
                self.number_of_burn_in_iterations = self.max_iterations - 100
            else:
                self.number_of_burn_in_iterations = int(self.max_iterations * 0.5)

    def _initialize_acceptance_rate_information(self):
        # Initialize average_acceptance_rates.
        self.average_acceptance_rates = {key: 0.0 for key in self.sampler.individual_proposal_distributions.keys()}
        # Initialize current_acceptance_rates_in_window.
        self.current_acceptance_rates_in_window = {key: np.zeros((self.memory_window_size,)) for key in self.sampler.individual_proposal_distributions.keys()}
        self.average_acceptance_rates_in_window = {key: 0.0 for key in self.sampler.individual_proposal_distributions.keys()}

    def _update_acceptance_rate_information(self):
        # Update average_acceptance_rates.
        coefficient_1 = float(self.current_mcmc_iteration)
        coefficient_2 = (coefficient_1 - 1.0) / coefficient_1
        self.average_acceptance_rates = {key: value * coefficient_2 + self.current_acceptance_rates[key] / coefficient_1
                                         for key, value in self.average_acceptance_rates.items()}

        # Update current_acceptance_rates_in_window.
        for key in self.current_acceptance_rates_in_window.keys():
            self.current_acceptance_rates_in_window[key][(self.current_mcmc_iteration - 1) % self.memory_window_size] = \
                self.current_acceptance_rates[key]

    def _initialize_sufficient_statistics(self):
        sufficient_statistics = self.statistical_model.compute_sufficient_statistics(self.dataset, self.population_RER,
                                                                                     self.individual_RER)
        self.sufficient_statistics = {key: np.zeros(value.shape) for key, value in sufficient_statistics.items()}
        return sufficient_statistics

    # Normalize the ip -------------------------------------------------------------------------------------------------
    def _normalize_individual_parameters(self, center_sources=False):
        """"We want the acceleration factor and sources to be centered and the sources to have variance 1
        For the sources there is a subtelty and to improve convergence we add the 0.3 factor to smooth a little
        the change in p0 and avoid hard oscillations, especially in the beggining"""
        if  center_sources:
            # Center the sources by taking the exponential of the average spaceshift
            sources_mean = 0.6 * self.individual_RER['sources'].mean(axis=0)
            self.individual_RER['sources'] -= sources_mean

            """
            reference_frame = self.statistical_model.spatiotemporal_reference_frame
            # Right now this only works for scalar data, for higher dimensions, sources_mean is a vector
            spaceshift = torch.mm(reference_frame.modulation_matrix_t0, torch.tensor([sources_mean]).float().unsqueeze(1)).view(reference_frame.geodesic.velocity_t0.size())
            position = torch.tensor(self.statistical_model.fixed_effects['p0']).float()
            reference_frame.exponential.set_initial_position(position)
            if reference_frame.exponential.has_closed_form:
                reference_frame.exponential.set_initial_velocity(spaceshift)
            else:
                reference_frame.exponential.set_initial_momenta(spaceshift)
            reference_frame.exponential.update()
            self.statistical_model.fixed_effects['p0'] = np.array(reference_frame.exponential.get_final_position())
            """
        # Normalize the sources
        assert self.statistical_model.individual_random_effects['sources'].variance_sqrt == 1;
        sources_std = self.individual_RER['sources'].std(axis=0)
        self.statistical_model.individual_random_effects['sources'].set_variance(1.0)
        self.individual_RER['sources'] /= sources_std
        # This is a semi-dirty trick to avoid the modulation matrix to grow arbitrarily large 
        #self.statistical_model.fixed_effects['modulation_matrix'] -= self.statistical_model.fixed_effects['modulation_matrix'].mean(axis=1).reshape((4,1))

        #self.statistical_model.fixed_effects['modulation_matrix'] *= sources_std
        #if not self.statistical_model.is_frozen['modulation_matrix']:
            #self.gradient_based_estimator.step['modulation_matrix'] /= sources_std

        # Center the log_acceleration
        log_acceleration_mean = self.individual_RER['log_acceleration'].mean()
        self.individual_RER['log_acceleration'] -= log_acceleration_mean
        self.statistical_model.fixed_effects['v0'] *= np.exp(log_acceleration_mean)
        #self.gradient_based_estimator.step['v0'] /= np.exp(log_acceleration_mean)

    ####################################################################################################################
    ### Model parameters trajectory saving methods:
    ####################################################################################################################

    def _initialize_model_parameters_trajectory(self):
        self.model_parameters_trajectory = {}
        for (key, value) in self.statistical_model.get_fixed_effects().items():
            self.model_parameters_trajectory[key] = np.zeros((self.number_of_trajectory_points + 1, value.size))
            self.model_parameters_trajectory[key][0, :] = value.flatten()

    def _update_model_parameters_trajectory(self):
        for (key, value) in self.statistical_model.get_fixed_effects().items():
            self.model_parameters_trajectory[key][
            int(self.current_iteration / float(self.save_model_parameters_every_n_iters)), :] = value.flatten()

    def _get_vectorized_individual_RER(self):
        return np.concatenate([value.flatten() for value in self.individual_RER.values()])

    def _initialize_individual_random_effects_samples_stack(self):
        number_of_concentration_iterations = self.max_iterations - self.number_of_burn_in_iterations
        self.individual_random_effects_samples_stack = {}
        for (key, value) in self.individual_RER.items():
            if number_of_concentration_iterations > 0:
                self.individual_random_effects_samples_stack[key] = np.zeros(
                    (number_of_concentration_iterations, value.size))
                self.individual_random_effects_samples_stack[key][0, :] = value.flatten()

    def _update_individual_random_effects_samples_stack(self):
        for (key, value) in self.individual_RER.items():
            self.individual_random_effects_samples_stack[key][
            self.current_iteration - self.number_of_burn_in_iterations - 1, :] = value.flatten()

    ####################################################################################################################
    ### Pickle dump methods.
    ####################################################################################################################

    def _get_parameters(self):
        out = self.statistical_model.get_fixed_effects()
        out.update(self.population_RER)
        out.update(self.individual_RER)
        assert len(out) == len(self.statistical_model.get_fixed_effects()) \
               + len(self.population_RER) + len(self.individual_RER)
        return out

    def _set_parameters(self, parameters):
        fixed_effects = {key: parameters[key] for key in self.statistical_model.get_fixed_effects().keys()}
        self.statistical_model.set_fixed_effects(fixed_effects)
        self.population_RER = {key: parameters[key] for key in self.population_RER.keys()}
        self.individual_RER = {key: parameters[key] for key in self.individual_RER.keys()}

    def _load_state_file(self):
        with open(self.state_file, 'rb') as f:
            d = pickle.load(f)
            return (d['current_iteration'],
                    d['current_parameters'],
                    d['current_sufficient_statistics'],
                    d['current_proposal_stds'],
                    d['current_acceptance_rates'],
                    d['average_acceptance_rates'],
                    d['current_acceptance_rates_in_window'],
                    d['average_acceptance_rates_in_window'],
                    d['trajectory'],
                    d['samples'])

    def _dump_state_file(self):
        if self.state_file is None:
            self.state_file = Settings().output_dir + '/state_file.txt'
        d = {
            'current_iteration': self.current_iteration,
            'current_parameters': self._get_parameters(),
            'current_sufficient_statistics': self.sufficient_statistics,
            'current_proposal_stds': self.sampler.get_proposal_standard_deviations(),
            'current_acceptance_rates': self.current_acceptance_rates,
            'average_acceptance_rates': self.average_acceptance_rates,
            'current_acceptance_rates_in_window': self.current_acceptance_rates_in_window,
            'average_acceptance_rates_in_window': self.average_acceptance_rates_in_window,
            'trajectory': self.model_parameters_trajectory,
            'samples': self.individual_random_effects_samples_stack
        }
        with open(self.state_file, 'wb') as f:
            pickle.dump(d, f)
