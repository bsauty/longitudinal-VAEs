import logging
import os
from abc import ABC, abstractmethod

from ...core import default

logger = logging.getLogger(__name__)


class AbstractEstimator(ABC):

    """
    AbstractEstimator object class.
    An estimator is an algorithm which updates the fixed effects of a statistical model.

    """

    ################################################################################
    ### Constructor:
    ################################################################################

    def __init__(self, statistical_model=None, dataset=None, name='undefined', verbose=default.verbose,
                 optimized_log_likelihood=default.optimized_log_likelihood,
                 max_iterations=default.max_iterations, convergence_tolerance=default.convergence_tolerance,
                 print_every_n_iters=default.print_every_n_iters, save_every_n_iters=default.save_every_n_iters,
                 population_RER={}, individual_RER={},
                 callback=None, state_file=None, output_dir=default.output_dir):

        self.statistical_model = statistical_model
        self.dataset = dataset
        self.name = name
        self.verbose = verbose  # If 0, don't print nothing.
        self.optimized_log_likelihood = optimized_log_likelihood
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.print_every_n_iters = print_every_n_iters
        self.save_every_n_iters = save_every_n_iters

        # RER = random effects realization.
        self.population_RER = population_RER
        self.individual_RER = individual_RER

        self.callback = callback
        self.callback_ret = True
        self.output_dir = output_dir
        self.state_file = state_file

    @abstractmethod
    def update(self):
        if self.statistical_model is None:
            raise RuntimeError('statistical_model has not been set')

    @abstractmethod
    def write(self):
        pass

    def _call_user_callback(self, current_log_likelihood, current_attachment, current_regularity, gradient):
        if self.callback is not None:
            try:
                self.callback_ret = self.callback(self.__format_callback_data(current_log_likelihood, current_attachment, current_regularity, gradient))
            except Exception as e:
                logger.error(e)
        else:
            logger.warning('Trying to call user callback that has not been specified')

    def __format_callback_data(self, current_log_likelihood, current_attachment, current_regularity, gradient):
        return {
            'current_iteration': self.current_iteration,
            'current_log_likelihood': current_log_likelihood,
            'current_attachment': current_attachment,
            'current_regularity': current_regularity,
            'gradient': gradient
        }
