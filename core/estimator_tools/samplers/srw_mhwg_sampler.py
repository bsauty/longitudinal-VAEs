import logging
logger = logging.getLogger(__name__)

import math

import numpy as np

from ....core import default


class SrwMhwgSampler:
    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self,
                 individual_proposal_distributions=default.individual_proposal_distributions,
                 acceptance_rates_target=33.0):

        # Dictionary of probability distributions.
        self.population_proposal_distributions = {}
        self.individual_proposal_distributions = individual_proposal_distributions

        self.acceptance_rates_target = acceptance_rates_target  # Percentage.

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def sample(self, statistical_model, dataset, population_RER, individual_RER, current_model_terms=None):

        # Initialization -----------------------------------------------------------------------------------------------

        # Initialization of the memory of the current model terms.
        # The contribution of each subject is stored independently.
        if current_model_terms is None:
            current_model_terms = self._compute_model_log_likelihood(statistical_model, dataset,
                                                                     population_RER, individual_RER)

        # Acceptance rate metrics initialization.
        acceptance_rates = {key: 0.0 for key in self.individual_proposal_distributions.keys()}

        # Main loop ----------------------------------------------------------------------------------------------------
        for random_effect_name, proposal_RED in self.individual_proposal_distributions.items():
            # RED: random effect distribution.
            model_RED = statistical_model.individual_random_effects[random_effect_name]

            # Initialize subject lists.
            current_regularity_terms = []
            candidate_regularity_terms = []
            current_RER = []
            candidate_RER = []

            # Shape parameters of the current random effect realization.
            shape_parameters = individual_RER[random_effect_name][0].shape

            for i in range(dataset.number_of_subjects):
                # Evaluate the current part.
                current_regularity_terms.append(model_RED.compute_log_likelihood(individual_RER[random_effect_name][i]))
                current_RER.append(individual_RER[random_effect_name][i].flatten())

                # Draw the candidate.
                proposal_RED.mean = current_RER[i]
                sample = proposal_RED.sample()
                # Dirty but saves time by avoiding stupid sampling
                if random_effect_name == 'log_acceleration' and sample > 2:
                    candidate_RER.append(np.array([2]))
                else:
                    candidate_RER.append(sample)

                # Evaluate the candidate part.
                individual_RER[random_effect_name][i] = candidate_RER[i].reshape(shape_parameters)
                candidate_regularity_terms.append(model_RED.compute_log_likelihood(candidate_RER[i]))

            # Evaluate the candidate terms for all subjects at once, since all contributions are independent.
            candidate_model_terms = self._compute_model_log_likelihood(
                statistical_model, dataset, population_RER, individual_RER, modified_individual_RER=random_effect_name)

            for i in range(dataset.number_of_subjects):

                # if i == np.random.randint(0, dataset.number_of_subjects) and random_effect_name == 'log_acceleration':
                #     logger.info("Sampling summary", random_effect_name,
                #           "from", candidate_RER[i], "to", current_RER[i],
                #           "attachments:", candidate_model_terms[i], current_model_terms[i],
                #           "regularities:", candidate_regularity_terms[i], current_regularity_terms[i])

                # Acceptance rate.
                tau = candidate_model_terms[i] + candidate_regularity_terms[i] \
                      - current_model_terms[i] - current_regularity_terms[i]

                # TODO : assess the use of different 'model terms' for tau
                #tau = candidate_model_terms[i] - current_model_terms[i]

                # Reject.
                if math.log(np.random.uniform()) > tau or math.isnan(tau):
                    individual_RER[random_effect_name][i] = current_RER[i].reshape(shape_parameters)

                # Accept.
                else:
                    current_model_terms[i] = candidate_model_terms[i]
                    current_regularity_terms[i] = candidate_regularity_terms[i]
                    acceptance_rates[random_effect_name] += 1.0

            # Acceptance rate final scaling for the considered random effect.
            acceptance_rates[random_effect_name] *= 100.0 / float(dataset.number_of_subjects)

        return acceptance_rates, current_model_terms

    ####################################################################################################################
    ### Auxiliary methods:
    ####################################################################################################################

    def _compute_model_log_likelihood(self, statistical_model, dataset, population_RER, individual_RER,
                                      modified_individual_RER='all'):
        try:
            return statistical_model.compute_log_likelihood(
                dataset, population_RER, individual_RER, mode='model', modified_individual_RER=modified_individual_RER)

        except ValueError as error:
            logger.info('>> ' + str(error) + ' \t[ in srw_mhwg_sampler ]')
            statistical_model.clear_memory()
            return np.zeros((dataset.number_of_subjects,)) - float('inf')

    def adapt_proposal_distributions(self, current_acceptance_rates_in_window, iteration_number, verbose):
        goal = self.acceptance_rates_target
        msg = '>> Proposal std re-evaluated from:\n'

        for random_effect_name, proposal_distribution in self.individual_proposal_distributions.items():
            ar = current_acceptance_rates_in_window[random_effect_name]
            std = proposal_distribution.get_variance_sqrt()
            msg += '\t\t %.3f ' % std
            if ar > self.acceptance_rates_target:
                std *= 1 + (ar - goal) / ((100 - goal) * math.sqrt(iteration_number + 1))
            else:
                std *= 1 - (goal - ar) / (goal * math.sqrt(iteration_number + 1))

            msg += '\tto\t%.3f \t[ %s ]\n' % (std, random_effect_name)
            proposal_distribution.set_variance_sqrt(std)

        if verbose > 0: logger.info(msg[:-1])

    ####################################################################################################################
    ### Pickle dump methods.
    ####################################################################################################################

    def get_proposal_standard_deviations(self):
        out = {}
        for random_effect_name, proposal_distribution in self.individual_proposal_distributions.items():
            out[random_effect_name] = proposal_distribution.get_variance_sqrt()
        return out

    def set_proposal_standard_deviations(self, stds):
        for random_effect_name, std in stds.items():
            self.individual_proposal_distributions[random_effect_name].set_variance_sqrt(std)