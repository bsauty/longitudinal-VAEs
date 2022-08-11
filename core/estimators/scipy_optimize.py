import _pickle as pickle
import logging
from decimal import Decimal

import numpy as np
from scipy.optimize import minimize, brute, basinhopping

from ...core import default
from ...core.estimators.abstract_estimator import AbstractEstimator
from ...support.utilities.general_settings import Settings

logger = logging.getLogger(__name__)


class ScipyOptimize(AbstractEstimator):
    """
    ScipyOptimize object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, statistical_model, dataset, optimization_method_type='undefined', individual_RER={},
                 optimized_log_likelihood=default.optimized_log_likelihood,
                 max_iterations=default.max_iterations, convergence_tolerance=default.convergence_tolerance,
                 print_every_n_iters=default.print_every_n_iters, save_every_n_iters=default.save_every_n_iters,
                 memory_length=default.memory_length,
                 # parameters_shape, parameters_order, gradient_memory,
                 max_line_search_iterations=default.max_line_search_iterations,
                 output_dir=default.output_dir, verbose=default.verbose,
                 callback=None,
                 load_state_file=default.load_state_file, state_file=default.state_file,
                 **kwargs):

        super().__init__(statistical_model=statistical_model, dataset=dataset, name='ScipyOptimize', verbose=verbose,
                         optimized_log_likelihood=optimized_log_likelihood,
                         max_iterations=max_iterations, convergence_tolerance=convergence_tolerance,
                         print_every_n_iters=print_every_n_iters, save_every_n_iters=save_every_n_iters,
                         individual_RER=individual_RER,
                         callback=callback, state_file=state_file, output_dir=output_dir)

        assert optimization_method_type.lower() in ['ScipyLBFGS'.lower(), 'ScipyPowell'.lower(),
                                                    'GridSearch'.lower(), 'BasinHopping'.lower()]

        # If the load_state_file flag is active, restore context.
        if load_state_file:
            self.x0, self.current_iteration, self.parameters_shape, self.parameters_order = self._load_state_file()
            self._set_parameters(self._unvectorize_parameters(self.x0))  # Propagate the parameter values.
            logger.info("State file loaded, it was at iteration", self.current_iteration)

        else:
            parameters = self._get_parameters()
            self.current_iteration = 1
            self.parameters_shape = {key: value.shape for key, value in parameters.items()}
            self.parameters_order = [key for key in parameters.keys()]
            self.x0 = self._vectorize_parameters(parameters)
            self._gradient_memory = None

        if optimization_method_type.lower() == 'ScipyLBFGS'.lower():
            self.method = 'L-BFGS-B'
        elif optimization_method_type.lower() == 'ScipyPowell'.lower():
            self.method = 'Powell'
        elif optimization_method_type.lower() == 'GridSearch'.lower():
            self.method = 'GridSearch'
        elif optimization_method_type.lower() == 'BasinHopping'.lower():
            self.method = 'BasinHopping'
        else:
            raise RuntimeError('Unexpected error.')

        self.memory_length = memory_length
        self.max_line_search_iterations = max_line_search_iterations


    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def initialize(self):
        parameters = self._get_parameters()
        self.current_iteration = 1
        self.x0 = self._vectorize_parameters(parameters)
        self._gradient_memory = None

    def update(self):
        """
        Runs the scipy optimize routine and updates the statistical model.
        """
        super().update()

        # Main loop ----------------------------------------------------------------------------------------------------
        # self.current_iteration = 1
        if self.verbose > 0:
            logger.info('')
            logger.info('>> Scipy optimization method: ' + self.method)
            self.print()

        try:
            if self.method == 'L-BFGS-B':
                result = minimize(self._cost_and_derivative, self.x0.astype('float64'),
                                  method='L-BFGS-B', jac=True, callback=self._callback,
                                  options={
                                      'maxiter': self.max_iterations + 10,
                                      'maxls': self.max_line_search_iterations,
                                      'ftol': self.convergence_tolerance,
                                      # Number of previous gradients used to approximate the Hessian.
                                      'maxcor': self.memory_length,
                                      'disp': False
                                  })
                msg = result.message
                if msg == 'ABNORMAL_TERMINATION_IN_LNSRCH':
                    logger.info('>> Number of line search loops exceeded. Stopping.')

                else:
                    logger.info('>> ' + msg)

            elif self.method == 'Powell':
                result = minimize(self._cost, self.x0.astype('float64'),
                                  method='Powell', tol=self.convergence_tolerance, callback=self._callback,
                                  options={
                                      'maxiter': self.max_iterations + 10,
                                      'maxfev': 10e4,
                                      'disp': True
                                  })

            elif self.method == 'BasinHopping':
                raise RuntimeError('The BasinHopping algorithm is not available yet.')
                # result = basinhopping(self._cost_and_derivative, self.x0, niter=25, disp=True,
                #                       minimizer_kwargs={
                #                           'method': 'L-BFGS-B',
                #                           'jac': True,
                #                           'bounds': self._get_bounds(),
                #                           'tol': self.convergence_tolerance,
                #                           'options': {'maxiter': self.max_iterations},
                #                           'args': (True,)
                #                       })
                # self._set_parameters(self._unvectorize_parameters(result.x))

            elif self.method == 'GridSearch':
                raise RuntimeError('The GridSearch algorithm is not available yet.')
                # x = brute(self._cost, self._get_parameters_range(self.x0), Ns=3, disp=True)
                # self._set_parameters(self._unvectorize_parameters(x))

            else:
                raise RuntimeError('Unknown optimization method.')

        # Finalization -------------------------------------------------------------------------------------------------
        except StopIteration:
            logger.info('>> STOP: TOTAL NO. of ITERATIONS EXCEEDS LIMIT')
        finally:
            self.statistical_model.cleanup()

    def print(self):
        """
        Print information.
        """
        logger.info('')
        logger.info('------------------------------------- Iteration: '
              + str(self.current_iteration) + ' -------------------------------------')

        if self.method == 'Powell':
            try:
                attachment, regularity = self.statistical_model.compute_log_likelihood(
                    self.dataset, self.population_RER, self.individual_RER, with_grad=False)
                logger.info('>> Log-likelihood = %.3E \t [ attachment = %.3E ; regularity = %.3E ]' %
                      (Decimal(str(attachment + regularity)),
                       Decimal(str(attachment)),
                       Decimal(str(regularity))))
            except ValueError as error:
                logger.info('>> ' + str(error) + ' [ in scipy_optimize ]')
                self.statistical_model.clear_memory()

    def write(self):
        """
        Save the results.
        """
        self.statistical_model.write(self.dataset, self.population_RER, self.individual_RER, self.output_dir)
        self._dump_state_file(self._vectorize_parameters(self._get_parameters()))


    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _cost(self, x):
        # Propagates the parameter value to all necessary attributes.
        self._set_parameters(self._unvectorize_parameters(x))

        # Call the model method.
        try:
            attachment, regularity = self.statistical_model.compute_log_likelihood(
                self.dataset, self.population_RER, self.individual_RER,
                mode=self.optimized_log_likelihood, with_grad=False)

        except ValueError as error:
            logger.info('>> ' + str(error) + ' [ in scipy_optimize ]')
            self.statistical_model.clear_memory()
            return np.float64(float('inf'))

        # Prepare the outputs: notably linearize and concatenates the gradient.
        cost = - attachment - regularity

        # Return.
        return cost.astype('float64')

    def _cost_and_derivative(self, x):
        # Propagates the parameter value to all necessary attributes.
        self._set_parameters(self._unvectorize_parameters(x))

        # Call the model method.
        try:
            attachment, regularity, gradient = self.statistical_model.compute_log_likelihood(
                self.dataset, self.population_RER, self.individual_RER,
                mode=self.optimized_log_likelihood, with_grad=True)

        except ValueError as error:
            logger.info('>> ' + str(error))
            self.statistical_model.clear_memory()
            if self._gradient_memory is None:
                raise RuntimeError('Failure of the scipy_optimize L-BFGS-B algorithm: '
                                   'the initial gradient of the model log-likelihood fails to be computed.')
            else:
                return np.float64(float('inf')), self._gradient_memory

        # Print.
        if self.verbose > 0 and not self.current_iteration % self.print_every_n_iters:
            logger.info('>> Log-likelihood = %.3E \t [ attachment = %.3E ; regularity = %.3E ]' %
                  (Decimal(str(attachment + regularity)),
                   Decimal(str(attachment)),
                   Decimal(str(regularity))))

        # Call user callback function
        if self.callback is not None:
            self._call_user_callback(float(attachment + regularity), float(attachment), float(regularity), gradient)

        # Prepare the outputs: notably linearize and concatenates the gradient.
        cost = - attachment - regularity
        gradient = - np.concatenate([gradient[key].flatten() for key in self.parameters_order])

        # Memory for exception handling.
        self._gradient_memory = gradient.astype('float64')

        # Return.
        return cost.astype('float64'), gradient.astype('float64')

    def _callback(self, x):
        # Propagate the parameters to all necessary attributes.
        self._set_parameters(self._unvectorize_parameters(x))

        # Print and save.
        self.current_iteration += 1
        if not self.current_iteration % self.save_every_n_iters:
            self.write()
        if not self.current_iteration % self.save_every_n_iters:
            self._dump_state_file(x)

        if not self.callback_ret or self.current_iteration == self.max_iterations + 1:
            raise StopIteration
        else:
            if self.verbose > 0 and not self.current_iteration % self.print_every_n_iters:
                self.print()

    def _get_parameters(self):
        """
        Return a dictionary of numpy arrays.
        """
        out = self.statistical_model.get_fixed_effects()
        if self.optimized_log_likelihood == 'complete':
            out.update(self.population_RER)
            out.update(self.individual_RER)
        return out

    def _get_bounds(self):
        """
        If one of the optimized parameters is called "acceleration", it should respect a zero lower bound.
        """
        parameters = self._get_parameters()
        bounds = []
        for key in self.parameters_order:
            for _ in parameters[key].flatten():
                if key == 'acceleration':
                    bounds.append((0.0, None))
                else:
                    bounds.append((None, None))
        return bounds

    def _vectorize_parameters(self, parameters):
        """
        Returns a 1D numpy array from a dictionary of numpy arrays.
        """
        return np.concatenate([parameters[key].flatten() for key in self.parameters_order])

    def _unvectorize_parameters(self, x):
        """
        Recover the structure of the parameters
        """
        parameters = {}
        cursor = 0
        for key in self.parameters_order:
            shape = self.parameters_shape[key]
            length = np.prod(shape)
            parameters[key] = x[cursor:cursor + length].reshape(shape)
            cursor += length
        return parameters

    def _set_parameters(self, parameters):
        """
        Updates the model and the random effect realization attributes.
        """
        fixed_effects = {key: parameters[key] for key in self.statistical_model.get_fixed_effects().keys()}
        self.statistical_model.set_fixed_effects(fixed_effects)
        if self.optimized_log_likelihood == 'complete':
            self.population_RER = {key: parameters[key] for key in self.population_RER.keys()}
            self.individual_RER = {key: parameters[key] for key in self.individual_RER.keys()}

    ####################################################################################################################
    ### Pickle dump and load methods:
    ####################################################################################################################

    def _load_state_file(self):
        """
        loads Settings().state_file and returns what's necessary to restart the scipy optimization.
        """
        with open(self.state_file, 'rb') as f:
            d = pickle.load(f)
            return d['parameters'], d['current_iteration'], d['parameters_shape'], d['parameters_order']

    def _dump_state_file(self, parameters):
        """
        Dumps the state file with the new value of $x_0$ as argument.
        """
        if self.state_file is None:
            self.state_file = Settings().output_dir + '/state_file.txt'
        d = {'parameters': parameters, 'current_iteration': self.current_iteration,
             'parameters_shape': self.parameters_shape, 'parameters_order': self.parameters_order}
        with open(self.state_file, 'wb') as f:
            pickle.dump(d, f)
