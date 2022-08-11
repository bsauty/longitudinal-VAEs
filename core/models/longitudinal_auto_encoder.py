import math
import os.path
import shutil
import warnings
from copy import deepcopy
import time
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from random import randint
from sklearn.decomposition import FastICA
from numpy.linalg import lstsq

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from ...core.models.abstract_statistical_model import AbstractStatisticalModel
from ...in_out.array_readers_and_writers import *
from ...support.probability_distributions.multi_scalar_inverse_wishart_distribution import \
    MultiScalarInverseWishartDistribution
from ...support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution
from ...support.utilities.general_settings import Settings
from ..model_tools.neural_networks.networks_LVAE import Dataset

import logging

logger = logging.getLogger(__name__)
writer = SummaryWriter()

class LongitudinalAutoEncoder(AbstractStatisticalModel):
    """
    Longitudinal auto-encoders.
    """
    

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self):
        AbstractStatisticalModel.__init__(self)

        # Instantiate the images and first encoded representation (splitting test/train for cross validation)
        self.number_of_subjects = None
        self.number_of_objects = None
        self.train_labels = None             # id of patients
        self.test_labels = None
        self.full_labels = None
        self.train_images = None            # images
        self.test_images = None
        self.full_images = None
        self.train_timepoints = None        # timepoints
        self.test_timepoints = None
        self.full_timepoints = None
        self.train_encoded = None            # latent representations
        self.test_encoded = None
        self.full_encoded = None            

        # Instantiate the auto-encoders
        self.CAE = None
        self.LAE = None

        # Whether there is a parallel transport to compute (not in 1D for instance.)
        self.no_parallel_transport = False
        self.number_of_sources = 0
        self.spatiotemporal_reference_frame = None
        self.latent_space_dimension = None
        self.has_maximization_procedure = True
        self.observation_type = None  # image of scalar, to compute the distances efficiently in here.

        # Template object, used to have information about the deformables and to write.
        self.template = None

        # Dictionary of numpy arrays.
        self.fixed_effects['p0'] = None
        self.fixed_effects['reference_time'] = None
        self.fixed_effects['v0'] = None
        self.fixed_effects['onset_age_variance'] = None
        self.fixed_effects['log_acceleration_variance'] = None
        self.fixed_effects['noise_variance'] = None
        self.fixed_effects['modulation_matrix'] = None

        # Dictionary of prior distributions
        self.priors['onset_age_variance'] = MultiScalarInverseWishartDistribution()
        self.priors['log_acceleration_variance'] = MultiScalarInverseWishartDistribution()
        self.priors['noise_variance'] = MultiScalarInverseWishartDistribution()
        self.priors['modulation_matrix'] = MultiScalarNormalDistribution()

        # Dictionary of probability distributions.
        self.individual_random_effects['sources'] = MultiScalarNormalDistribution()
        self.individual_random_effects['onset_age'] = MultiScalarNormalDistribution()
        self.individual_random_effects['log_acceleration'] = MultiScalarNormalDistribution()

        # Dictionary of booleans
        self.is_frozen = {}
        self.is_frozen['v0'] = False
        self.is_frozen['p0'] = False
        self.is_frozen['reference_time'] = False
        self.is_frozen['onset_age_variance'] = False
        self.is_frozen['log_acceleration_variance'] = False
        self.is_frozen['noise_variance'] = False
        self.is_frozen['modulation_matrix'] = False

    ####################################################################################################################
    ### Encapsulation methods:
    ####################################################################################################################

    def set_v0(self, v0):
        self.fixed_effects['v0'] = np.array([v0]).flatten()

    def get_v0(self):
        return self.fixed_effects['v0']

    def get_p0(self):
        return self.fixed_effects['p0']

    def set_p0(self, p0):
        self.fixed_effects['p0'] = np.array([p0]).flatten()

    # Reference time ---------------------------------------------------------------------------------------------------
    def get_reference_time(self):
        return self.fixed_effects['reference_time']

    def set_reference_time(self, rt):
        self.fixed_effects['reference_time'] = np.float64(rt)
        self.individual_random_effects['onset_age'].mean = np.array([rt])

    def get_modulation_matrix(self):
        return self.fixed_effects['modulation_matrix']

    def set_modulation_matrix(self, mm):
        self.fixed_effects['modulation_matrix'] = mm

    # Log-acceleration variance ----------------------------------------------------------------------------------------
    def get_log_acceleration_variance(self):
        return self.fixed_effects['log_acceleration_variance']

    def set_log_acceleration_variance(self, lav):
        assert lav is not None
        self.fixed_effects['log_acceleration_variance'] = np.float64(lav)
        self.individual_random_effects['log_acceleration'].set_variance(lav)

    # Time-shift variance ----------------------------------------------------------------------------------------------
    def get_onset_age_variance(self):
        return self.fixed_effects['onset_age_variance']

    def set_onset_age_variance(self, tsv):
        assert tsv is not None
        self.fixed_effects['onset_age_variance'] = np.float64(tsv)
        self.individual_random_effects['onset_age'].set_variance(tsv)

    # Noise variance ---------------------------------------------------------------------------------------------------
    def get_noise_variance(self):
        return self.fixed_effects['noise_variance']

    def set_noise_variance(self, nv):
        self.fixed_effects['noise_variance'] = np.float64(nv)

    # Full fixed effects -----------------------------------------------------------------------------------------------
    def get_fixed_effects(self):
        out = {}
        if not self.is_frozen['p0']:
            out['p0'] = np.array([self.fixed_effects['p0']])
        if not self.is_frozen['v0']:
            out['v0'] = np.array([self.fixed_effects['v0']])
        if not self.is_frozen['modulation_matrix'] and not self.no_parallel_transport:
            out['modulation_matrix'] = self.fixed_effects['modulation_matrix']
        return deepcopy(out)

    def set_fixed_effects(self, fixed_effects):
        if not self.is_frozen['p0']:
            self.set_p0(fixed_effects['p0'])
        if not self.is_frozen['v0']:
            self.set_v0(fixed_effects['v0'])
        if not self.is_frozen['modulation_matrix'] and not self.no_parallel_transport:
            self.set_modulation_matrix(fixed_effects['modulation_matrix'])

    ####################################################################################################################
    ### Public methods:
    ####################################################################################################################

    def update(self):
        """
        Initializations of prior parameters
        """
        self.initialize_noise_variables()
        self.initialize_onset_age_variables()
        self.initialize_log_acceleration_variables()

        if self.number_of_sources < 1:
            self.no_parallel_transport = True

        if not self.no_parallel_transport:
            self.initialize_source_variables()

        if self.no_parallel_transport:
            self.spatiotemporal_reference_frame.no_transport_needed = True
        else:
            self.spatiotemporal_reference_frame.no_transport_needed = False

        # Uncomment to see the frozen parameters
        for (key, val) in self.is_frozen.items():
            logger.info(f"{key, val}")

    def longitudinal_loss(self, data, encoded, reconstructed, individual_RER):
        """
        This is where all the longitudinal stuff happens : we both penalize bad reconstruction and
        bad alignment on the prescribed geodesics.

        Need to put the id and timepoints in one single key of the dictionnary to access them simultaneously
        """
        # Compute the error of alignment
        alignment_loss = 0
        ids, timepoints = data[1], data[2]
    
        onset_ages, log_accelerations, sources = individual_RER['onset_age'],\
                                                 individual_RER['log_acceleration'], \
                                                 individual_RER['sources']

        # TODO : tensorize this to do only one operation
        for i in range(len(timepoints)):
            idx, t = ids[i], timepoints[i]
            onset_age, log_acceleration = onset_ages[idx], log_accelerations[idx]
            absolute_time = np.exp(log_acceleration) * (t - onset_age) + self.fixed_effects['reference_time']
            expected_position = self.spatiotemporal_reference_frame.get_position(absolute_time, sources=torch.Tensor(sources[idx]).to(Settings().device))
            alignment_loss += torch.sum((expected_position - encoded[i])**2)

        # Compute the error of reconstruction
        return alignment_loss

    # Compute the functional. Numpy input/outputs.
    def compute_log_likelihood(self, dataset, population_RER, individual_RER,
                               mode='complete', with_grad=False, modified_individual_RER='all'):
        """
        Compute the log-likelihood of the dataset, given parameters fixed_effects and random effects realizations
        population_RER and indRER.

        :param dataset: LongitudinalDataset instance
        :param fixed_effects: Dictionary of fixed effects.
        :param population_RER: Dictionary of population random effects realizations.
        :param individual_RER: Dictionary of individual random effects realizations.
        :param mode: Indicates which log_likelihood should be computed, between 'complete', 'model', and 'class2'.
        :param with_grad: Flag that indicates wether the gradient should be returned as well.
        :return:
        """

        v0, p0, modulation_matrix = self._fixed_effects_to_torch_tensors(with_grad)
        onset_ages, log_accelerations, sources = self._individual_RER_to_torch_tensors(individual_RER, with_grad)

        residuals = self._compute_residuals(dataset, v0, p0, modulation_matrix,
                                            log_accelerations, onset_ages, sources, with_grad=with_grad)

        if mode == 'complete':
            sufficient_statistics = self.compute_sufficient_statistics(dataset, population_RER, individual_RER,
                                                                       residuals)
            self.update_fixed_effects(dataset, sufficient_statistics)

        attachments = self._compute_individual_attachments(residuals)
        attachment = torch.sum(attachments)
        # if with_grad:
        #    attachment.backward(retain_graph=False)
        regularity = self._compute_random_effects_regularity(log_accelerations, onset_ages, sources)

        if mode == 'complete':
            class1_reg = self._compute_class1_priors_regularity()
            class2_reg = self._compute_class2_priors_regularity(modulation_matrix)
            regularity += class1_reg
            regularity += class2_reg
        if mode == 'class2':
            class2_reg = self._compute_class2_priors_regularity(modulation_matrix)
            regularity += class2_reg

        if with_grad:
            total = attachment + regularity
            total.backward(retain_graph=False)

            # Gradients of the effects with no closed form update.
            gradient = {}
            if not self.is_frozen['v0']:
                gradient['v0'] = v0.grad.data.cpu().numpy()
            if not self.is_frozen['p0']:
                gradient['p0'] = p0.grad.data.cpu().numpy()
            if not self.is_frozen['modulation_matrix'] and not self.no_parallel_transport:
                gradient['modulation_matrix'] = modulation_matrix.grad.data.cpu().numpy()

            if mode == 'complete':
                gradient['onset_age'] = onset_ages.grad.data.cpu().numpy()
                gradient['log_acceleration'] = log_accelerations.grad.data.cpu().numpy()
                if not self.no_parallel_transport:
                    gradient['sources'] = sources.grad.data.cpu().numpy()

            if mode in ['complete', 'class2']:
                return attachment.data.cpu().numpy(), regularity.data.cpu().numpy(), gradient
            elif mode == 'model':
                return attachments.data.cpu().numpy(), gradient

        else:
            if mode in ['complete', 'class2']:
                return attachment.data.cpu().numpy(), regularity.data.cpu().numpy()
            elif mode == 'model':
                return attachments.data.cpu().numpy()

    def maximize(self, individual_RER):
        """
        This is where we update Longitudinal CAE between each MCMC iteration.
        After training, we update the longitudinal dataset of encoded images
        """
        
        logger.info(f"Into the maximize procedure of {self.name}")
        if self.CAE.epoch == 0:
            self.CAE.lr = 5e-5
        elif not(self.CAE.epoch % 10):
            self.CAE.lr *= .97
           
        self.CAE.gamma *= 1.01
        logger.info(f"Multiplied the gamma factor by 1.02 to reach {self.CAE.gamma}")
        
        optimizer_fn = optim.Adam
        train_data = Dataset(self.train_images, self.train_labels, self.train_timepoints)
        test_data = Dataset(self.test_images, self.test_labels, self.test_timepoints)
        trainloader = DataLoader(train_data, batch_size=12, shuffle=False)

        if 'GAN' in self.CAE.name:
            vae_optimizer = optimizer_fn(self.CAE.VAE.parameters(), lr=self.CAE.lr)
            d_optimizer = optimizer_fn(self.CAE.discriminator.parameters(), lr=self.CAE.lr)
            self.CAE.train_(data_loader=trainloader, test=test_data, vae_optimizer=vae_optimizer, d_optimizer=d_optimizer,\
                        num_epochs=2, longitudinal=self.longitudinal_loss, individual_RER=individual_RER, writer=writer)
        else:
            optimizer = optimizer_fn(self.CAE.parameters(), lr=self.CAE.lr)
            self.CAE.train_(data_loader=trainloader, test=test_data, optimizer=optimizer,\
                num_epochs=1, longitudinal=self.longitudinal_loss, individual_RER=individual_RER, writer=writer)

        if not(self.CAE.epoch % 10):
            torch.save(self.CAE.state_dict(), 'CVAE_longitudinal')
            logger.info(f"Saving the model at CVAE_longitudinal")


        # Then update the latent representation
        _, self.train_encoded = self.CAE.evaluate(self.train_images)
        _, self.test_encoded = self.CAE.evaluate(self.test_images)
        self.full_encoded = torch.cat((self.train_encoded, self.test_encoded))

        # We compute a set of directions in the latent space that are orthogonal to v0
        encoded_images = torch.zeros(2*Settings().number_of_sources+1, 7, Settings().dimension)
        encoded_gradient = torch.zeros(Settings().number_of_sources+1, Settings().dimension)
        
        """v0, _, modulation_matrix = self._fixed_effects_to_torch_tensors(False)
        v0 = v0.cpu().unsqueeze(0)
        proj_v0 = torch.mm(v0.T,v0) / torch.mm(v0,v0.T)                                     # projector on vect(v0)
        proj_v0_ortho = torch.eye(proj_v0.shape[0]) - proj_v0                               # projector on vect(v0)_orthogonal
        """
        
        source = torch.zeros(Settings().number_of_sources)  
        #times = [-4, -3, -2, 0, 2, 3, 4]
        t0 = self.get_reference_time()
        times = [t0+i*8 for i in range(-3,4)]
        #times = [65, 70, 75, 80, 85, 90, 95]
        for i in range(len(times)):
            t = times[i]
            encoded_images[0][i] = self.spatiotemporal_reference_frame.get_position(torch.FloatTensor([t]).to(Settings().device),\
                                                                                        sources=torch.FloatTensor(source).to(Settings().device))
        encoded_gradient[0] = self.spatiotemporal_reference_frame.get_position(torch.FloatTensor([t0+2.5]).to(Settings().device),\
                                                                                        sources=torch.FloatTensor(source).to(Settings().device))

        """
        # Then the difference sources that are projected versions of the latent space orthogonal basis vectors on the orgthogonal of v0
        smallest_norm = 1e5
        smallest_norm_idx = -1
        projected_spaceshifts = []
        for i in range(Settings().dimension):
            spaceshift = torch.zeros(Settings().dimension)
            spaceshift[i] = 1
            proj_spaceshift = torch.mm(proj_v0_ortho, spaceshift.unsqueeze(1))
            proj_spaceshift_norm = torch.norm(proj_spaceshift, p=1)
            if proj_spaceshift_norm < smallest_norm:
                smallest_norm_idx = i
                smallest_norm = proj_spaceshift_norm
            projected_spaceshifts.append(proj_spaceshift)

        assert smallest_norm_idx >= 0
        assert smallest_norm_idx >= 0
        del(projected_spaceshifts[smallest_norm_idx])"""

        for i in range(Settings().number_of_sources):
            source = torch.zeros(Settings().number_of_sources)  
            for j in range(2):
                source[i] =  2 * (j - 1/2)
                for idx in range(len(times)):
                    encoded_images[2*i+j+1][idx] = self.spatiotemporal_reference_frame.get_position(torch.FloatTensor([times[idx]]).to(Settings().device),\
                                                                                    sources=torch.FloatTensor(source).to(Settings().device))
            encoded_gradient[i+1] = self.spatiotemporal_reference_frame.get_position(torch.FloatTensor([t0]).to(Settings().device),\
                                                                                    sources=torch.FloatTensor(source/2).to(Settings().device))
        
        if 'GAN' in self.CAE.name:
            self.CAE.VAE.plot_images_longitudinal(encoded_images, writer=writer)
            self.CAE.VAE.plot_images_gradient(encoded_gradient, writer=writer)
        else:
            if not(self.CAE.epoch%6):
                self.CAE.plot_images_longitudinal(encoded_images, writer=writer)
                self.CAE.plot_images_gradient(encoded_gradient, writer=writer)

    def _fixed_effects_to_torch_tensors(self, with_grad):
        v0_torch = Variable(torch.from_numpy(self.fixed_effects['v0']),
                            requires_grad=((not self.is_frozen['v0']) and with_grad)) \
            .type(Settings().tensor_scalar_type)

        p0_torch = Variable(torch.from_numpy(self.fixed_effects['p0']),
                            requires_grad=((not self.is_frozen['p0']) and with_grad)) \
            .type(Settings().tensor_scalar_type)

        modulation_matrix = Variable(torch.from_numpy(self.fixed_effects['modulation_matrix']),
                                     requires_grad=((not self.is_frozen['modulation_matrix']) and with_grad))\
            .type(Settings().tensor_scalar_type)


        if not self.no_parallel_transport:
            modulation_matrix = Variable(torch.from_numpy(self.get_modulation_matrix()),
                                         requires_grad=((not self.is_frozen['modulation_matrix']) and with_grad)) \
                .type(Settings().tensor_scalar_type)

        if with_grad:
            if not (self.is_frozen['v0']):
                v0_torch.retain_grad()
            if not (self.is_frozen['p0']):
                p0_torch.retain_grad()
            if not (self.is_frozen['modulation_matrix']):
                modulation_matrix.retain_grad()

        return v0_torch, p0_torch, modulation_matrix

    def _individual_RER_to_torch_tensors(self, individual_RER, with_grad):
        onset_ages = Variable(torch.from_numpy(individual_RER['onset_age']).type(Settings().tensor_scalar_type),
                              requires_grad=with_grad)
        log_accelerations = Variable(torch.from_numpy(individual_RER['log_acceleration'])
                                     .type(Settings().tensor_scalar_type),
                                     requires_grad=with_grad)

        if not self.no_parallel_transport:
            sources = Variable(torch.from_numpy(individual_RER['sources']).type(Settings().tensor_scalar_type),
                               requires_grad=with_grad)

            return onset_ages, log_accelerations, sources

        return onset_ages, log_accelerations, None

    def _compute_residuals(self, dataset, v0, p0, modulation_matrix,
                           log_accelerations, onset_ages, sources, with_grad=True):

        targets = dataset.deformable_objects  # A list of list
        absolute_times = self._compute_absolute_times(dataset.times, log_accelerations, onset_ages)

        self._update_spatiotemporal_reference_frame(absolute_times, p0, v0, modulation_matrix)
        number_of_subjects = dataset.number_of_subjects

        residuals = []
        for i in range(number_of_subjects):
            targets_torch = targets[i]
            predicted_latent_values_i = torch.zeros_like(targets_torch)

            for j, t in enumerate(absolute_times[i]):
                if sources is not None:
                    predicted_latent_values_i[j] = self.spatiotemporal_reference_frame.get_position(t,
                                                                                             sources=sources[i])
                else:
                    predicted_latent_values_i[j] = self.spatiotemporal_reference_frame.get_position(t)

            residuals_i = torch.sum((targets_torch-predicted_latent_values_i)**2)
            residuals.append(residuals_i)

        return residuals

    def _compute_individual_attachments(self, residuals):

        number_of_subjects = len(residuals)
        attachments = Variable(torch.zeros((number_of_subjects,)).type(Settings().tensor_scalar_type),
                               requires_grad=False)
        noise_variance_torch = Variable(torch.from_numpy(np.array([self.fixed_effects['noise_variance']]))
                                        .type(Settings().tensor_scalar_type),
                                        requires_grad=False)

        for i in range(number_of_subjects):
            attachments[i] = - 0.5 * torch.sum(residuals[i])
        print(f"Average residuals : {torch.sum(attachments) / number_of_subjects}")
        return attachments / noise_variance_torch

    def _compute_absolute_times(self, times, log_accelerations, onset_ages):
        """
        Fully torch.
        """
        reference_time = self.get_reference_time()
        accelerations = torch.exp(log_accelerations)

        upper_threshold = 500.
        lower_threshold = 1e-5
        #logger.info(
        #    f"Acceleration factors max/min: {np.max(accelerations.data.numpy()), np.argmax(accelerations.data.numpy()), np.min(accelerations.data.numpy()), np.argmin(accelerations.data.numpy())}")
        if np.max(accelerations.cpu().data.numpy()) > upper_threshold or np.min(
                accelerations.cpu().data.numpy()) < lower_threshold:
            raise ValueError('Absurd numerical value for the acceleration factor. Exception raised.')

        absolute_times = []
        for i in range(len(times)):
            absolute_times_i = (Variable(torch.from_numpy(times[i]).type(Settings().tensor_scalar_type)) - onset_ages[
                i]) * accelerations[i] + reference_time
            absolute_times.append(absolute_times_i)
        return absolute_times

    def _compute_absolute_time(self, time, acceleration, onset_age, reference_time):
        return (Variable(torch.from_numpy(np.array([time])).type(
            Settings().tensor_scalar_type)) - onset_age) * acceleration + reference_time

    ####################################################################################################################
    ### Private methods:
    ####################################################################################################################

    def _compute_random_effects_regularity(self, log_accelerations, onset_ages, sources):
        """
        Fully torch.
        """
        number_of_subjects = onset_ages.shape[0]
        regularity = 0.0

        # Onset age random effect.
        for i in range(number_of_subjects):
            regularity += self.individual_random_effects['onset_age'].compute_log_likelihood_torch(onset_ages[i],
                                                                                                   Settings().tensor_scalar_type)

        # Log-acceleration random effect.
        for i in range(number_of_subjects):
            regularity += \
                self.individual_random_effects['log_acceleration'].compute_log_likelihood_torch(log_accelerations[i],
                                                                                                Settings().tensor_scalar_type)
        # Sources random effect
        if sources is not None:
            for i in range(number_of_subjects):
                regularity += self.individual_random_effects['sources'].compute_log_likelihood_torch(sources[i],
                                                                                                     Settings().tensor_scalar_type)

        return regularity

    def compute_sufficient_statistics(self, dataset, population_RER, individual_RER, residuals=None, model_terms=None):
        sufficient_statistics = {}

        if residuals is None:
            v0, p0, modulation_matrix = self._fixed_effects_to_torch_tensors(False)

            onset_ages, log_accelerations, sources = self._individual_RER_to_torch_tensors(individual_RER, False)

            residuals = self._compute_residuals(dataset, v0, p0, modulation_matrix,
                                                log_accelerations, onset_ages, sources, with_grad=False)

        if not self.is_frozen['noise_variance']:
            sufficient_statistics['S1'] = 0.
            for i in range(len(residuals)):
                sufficient_statistics['S1'] += torch.sum(residuals[i]).cpu().data.numpy()

        if not self.is_frozen['log_acceleration_variance']:
            log_accelerations = individual_RER['log_acceleration']
            sufficient_statistics['S2'] = np.sum(log_accelerations ** 2)

        if not self.is_frozen['reference_time'] or not self.is_frozen['onset_age_variance']:
            onset_ages = individual_RER['onset_age']
            sufficient_statistics['S3'] = np.sum(onset_ages)

        if not self.is_frozen['onset_age_variance']:
            onset_ages = individual_RER['onset_age']
            ref_time = sufficient_statistics['S3'] / dataset.number_of_subjects
            sufficient_statistics['S4'] = np.sum((onset_ages - ref_time) ** 2)

        onset_ages, log_accelerations, sources = self._individual_RER_to_torch_tensors(individual_RER, False)
        
        if len(log_accelerations) > 1:
            if (log_accelerations.device != 'cpu'):
                sufficient_statistics['S5'] = np.exp(np.mean(log_accelerations.data.cpu().numpy()))
            else:
                sufficient_statistics['S5'] = np.exp(np.mean(log_accelerations.data.numpy()))
        return sufficient_statistics

    def update_fixed_effects(self, dataset, sufficient_statistics):
        """
        Updates the fixed effects based on the sufficient statistics, maximizing the likelihood.
        """
        number_of_subjects = dataset.number_of_subjects
        total_number_of_observations = dataset.total_number_of_observations

        # Updating the noise variance
        if not self.is_frozen['noise_variance']:
            prior_scale = self.priors['noise_variance'].scale_scalars[0]
            prior_dof = self.priors['noise_variance'].degrees_of_freedom[0]
            if self.observation_type == 'scalar':
                noise_dimension = Settings().dimension
            else:
                if Settings().dimension == 2:
                    noise_dimension = 64 * 64
                else:
                    noise_dimension = 64 * 64 * 64
            #  TODO : evaluate the effect of this different estimation of noise_variance
            # noise_variance = (sufficient_statistics['S1'] + prior_dof * prior_scale) \
            #                            / (noise_dimension * total_number_of_observations + prior_dof) # Dimension of objects is 1

            noise_variance = sufficient_statistics['S1'] / (noise_dimension * total_number_of_observations)
            self.set_noise_variance(noise_variance)

        # Updating the log acceleration variance
        if not self.is_frozen['log_acceleration_variance']:
            prior_scale = self.priors['log_acceleration_variance'].scale_scalars[0]
            prior_dof = self.priors['log_acceleration_variance'].degrees_of_freedom[0]
            log_acceleration_variance = (sufficient_statistics["S2"] + prior_dof * prior_scale) \
                                        / (number_of_subjects + prior_dof)
            self.set_log_acceleration_variance(log_acceleration_variance)

        # logger.info(f"log acceleration variance : {log_acceleration_variance}")
        # logger.info("Un-regularized log acceleration variance : ", sufficient_statistics['S2']/number_of_subjects)

        # Updating the reference time
        if not self.is_frozen['reference_time']:
            reftime = sufficient_statistics['S3'] / number_of_subjects
            self.set_reference_time(reftime)

        # Updating the onset ages variance
        if not self.is_frozen['onset_age_variance']:
            reftime = self.get_reference_time()
            onset_age_prior_scale = self.priors['onset_age_variance'].scale_scalars[0]
            onset_age_prior_dof = self.priors['onset_age_variance'].degrees_of_freedom[0]
            onset_age_variance = (sufficient_statistics['S4'] + onset_age_prior_dof * onset_age_prior_scale) \
                                 / (number_of_subjects + onset_age_prior_scale)
            self.set_onset_age_variance(onset_age_variance)
        if not self.is_frozen['v0']:
            if 'S5' in sufficient_statistics.keys():
                v0 = self.get_v0()
                self.set_v0(v0 * sufficient_statistics['S5'])

    def _compute_class1_priors_regularity(self):
        """
        Fully torch.
        Prior terms of the class 1 fixed effects, i.e. those for which we know a close-form update. No derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # Time-shift variance prior
        if not self.is_frozen['onset_age_variance']:
            regularity += \
                self.priors['onset_age_variance'].compute_log_likelihood(self.fixed_effects['onset_age_variance'])

        # Log-acceleration variance prior
        if not self.is_frozen['log_acceleration_variance']:
            regularity += self.priors['log_acceleration_variance'].compute_log_likelihood(
                self.fixed_effects['log_acceleration_variance'])

        # Noise variance prior
        if not self.is_frozen['noise_variance']:
            print('Noise variance prior regularity',
                  self.priors['noise_variance'].compute_log_likelihood(self.fixed_effects['noise_variance']))
            regularity += self.priors['noise_variance'].compute_log_likelihood(self.fixed_effects['noise_variance'])

        return regularity

    def _compute_class2_priors_regularity(self, modulation_matrix):
        """
        Fully torch.
        Prior terms of the class 2 fixed effects, i.e. those for which we do not know a close-form update. Derivative
        wrt those fixed effects will therefore be necessary.
        """
        regularity = 0.0

        # Prior on modulation_matrix fixed effects (if not frozen).
        if not self.is_frozen['modulation_matrix'] and not self.no_parallel_transport:
            assert not self.no_parallel_transport, "Should not happen"
            assert modulation_matrix is not None, "Should not happen"
            regularity += self.priors['modulation_matrix'].compute_log_likelihood_torch(modulation_matrix,
                                                                                        Settings().tensor_scalar_type)

        return regularity

    def _update_spatiotemporal_reference_frame(self, absolute_times, p0, v0,
                                               modulation_matrix):

        t0 = self.get_reference_time()

        self.spatiotemporal_reference_frame.set_t0(t0)
        #tmin = max(35, min([subject_times[0].cpu().data.numpy() for subject_times in absolute_times] + [t0]))
        tmin = min([subject_times[0].cpu().data.numpy() for subject_times in absolute_times] + [t0])
        #tmax = min(120, max([subject_times[-1].cpu().data.numpy() for subject_times in absolute_times] + [t0]))
        tmax = max([subject_times[-1].cpu().data.numpy() for subject_times in absolute_times] + [t0])
        self.spatiotemporal_reference_frame.set_tmin(tmin)
        self.spatiotemporal_reference_frame.set_tmax(tmax)
        self.spatiotemporal_reference_frame.set_position_t0(p0)
        self.spatiotemporal_reference_frame.set_velocity_t0(v0)

        if modulation_matrix is not None:
            self.spatiotemporal_reference_frame.set_modulation_matrix_t0(modulation_matrix)

        t_begin = time.time()
        self.spatiotemporal_reference_frame.update()
        t_end = time.time()
        logger.info(f"Tmin {tmin} Tmax {tmax} Update of the spatiotemporalframe: {round((t_end - t_begin) * 1000)} ms")

    def _get_lsd_observations(self, individual_RER, dataset):
        """
        Returns the latent positions of the observations, all concatenated in a one-dimensional array
        """

        alphas = np.exp(individual_RER['log_acceleration'])
        onset_ages = individual_RER['onset_age']
        sources = individual_RER['sources']
        sources = sources.reshape(len(sources), self.latent_space_dimension - 1)
        lsd_observations = []
        v0 = self.get_v0()
        p0 = self.get_p0()
        reference_time = self.get_reference_time()
        modulation_matrix = self.get_modulation_matrix()

        for i, elt in enumerate(dataset.times):
            for j, t in enumerate(elt):
                abs_time = alphas[i] * (t - onset_ages[i]) + reference_time
                lsd_obs = (abs_time - reference_time) * v0 + p0 + np.matmul(modulation_matrix, sources[i])
                lsd_observations.append(lsd_obs)

        return np.array(lsd_observations)

    def initialize_noise_variables(self):
        initial_noise_variance = self.get_noise_variance()
        if np.isnan(initial_noise_variance):
            initial_noise_variance = 1
        assert initial_noise_variance > 0
        if len(self.priors['noise_variance'].scale_scalars) == 0:
            self.priors['noise_variance'].scale_scalars.append(0.01 * initial_noise_variance)

    def initialize_onset_age_variables(self):
        # Check that the onset age random variable mean has been set.
        if self.individual_random_effects['onset_age'].mean is None:
            raise RuntimeError('The set_reference_time method of a LongitudinalAtlas model should be called before '
                               'the update one.')
        # Check that the onset age random variable variance has been set.
        if self.individual_random_effects['onset_age'].variance_sqrt is None:
            raise RuntimeError('The set_time_shift_variance method of a LongitudinalAtlas model should be called '
                               'before the update one.')

        if not self.is_frozen['onset_age_variance']:
            # Set the time_shift_variance prior scale to the initial time_shift_variance fixed effect.
            self.priors['onset_age_variance'].scale_scalars.append(self.get_onset_age_variance())
            logger.info(
                f">> The time shift variance prior degrees of freedom parameter is set to {self.number_of_subjects}")
            self.priors['onset_age_variance'].degrees_of_freedom.append(self.number_of_subjects)

    def initialize_log_acceleration_variables(self):
        # Set the log_acceleration random variable mean.
        self.individual_random_effects['log_acceleration'].mean = np.zeros((1,))
        # Set the log_acceleration_variance fixed effect.
        if self.get_log_acceleration_variance() is None:
            logger.info(">> The initial log-acceleration std fixed effect is ARBITRARILY set to 0.5")
            log_acceleration_std = 0.5
            self.set_log_acceleration_variance(log_acceleration_std ** 2)

        if not self.is_frozen["log_acceleration_variance"]:
            # Set the log_acceleration_variance prior scale to the initial log_acceleration_variance fixed effect.
            self.priors['log_acceleration_variance'].scale_scalars.append(self.get_log_acceleration_variance())
            # Arbitrarily set the log_acceleration_variance prior dof to 1.
            logger.info(
                f">> The log-acceleration variance prior degrees of freedom parameter is set to the number of subjects: {self.number_of_subjects}")
            self.priors['log_acceleration_variance'].degrees_of_freedom.append(self.number_of_subjects)

    def initialize_source_variables(self):
        # Set the sources random effect mean.
        if self.number_of_sources is None:
            raise RuntimeError('The number of sources must be set before calling the update method '
                               'of the LongitudinalAtlas class.')
        if self.no_parallel_transport:
            del self.individual_random_effects['sources']

        else:
            self.individual_random_effects['sources'].mean = np.zeros((self.number_of_sources,))
            self.individual_random_effects['sources'].set_variance(1.0)

    def initialize_modulation_matrix_variables(self):
        # Is the modulation matrix needed ?
        if self.no_parallel_transport:
            del self.fixed_effects['modulation_matrix']
            self.is_frozen['modulation_matrix'] = True
            return

        else:
            if self.fixed_effects['modulation_matrix'] is None:
                assert self.number_of_sources > 0, "Something went wrong."
                # We initialize it to number_of_sources components of an orthonormal basis to v0.

                self.fixed_effects['modulation_matrix'] = np.zeros((len(self.get_p0()), self.number_of_sources))

            else:
                assert self.number_of_sources == self.get_modulation_matrix().shape[
                    1], "The number of sources should be set somewhere"

        if not self.is_frozen['modulation_matrix']:
            # Set the modulation_matrix prior mean as the initial modulation_matrix.
            self.priors['modulation_matrix'].mean = self.get_modulation_matrix()
            # Set the modulation_matrix prior standard deviation to the deformation kernel width.
            self.priors['modulation_matrix'].set_variance_sqrt(1.)

    ####################################################################################################################
    ### Writing methods:
    ####################################################################################################################

    def write(self, dataset, population_RER, individual_RER, sample=False, update_fixed_effects=False, iteration=''):
        #self._write_model_predictions(dataset, individual_RER, sample=sample)
        self._write_model_parameters()
        self._write_individual_RER(dataset, individual_RER)
        self._write_individual_RER(dataset, individual_RER)
        #self._write_geodesic_and_parallel_trajectories()

    def _write_model_parameters(self):
        if not self.no_parallel_transport:
            write_2D_array(self.get_modulation_matrix(), Settings().output_dir, self.name + "_modulation_matrix.txt")

        write_2D_array(self.get_v0(), Settings().output_dir, self.name + '_v0.txt')
        write_2D_array(self.get_p0(), Settings().output_dir, self.name + '_p0.txt')

        np.save(os.path.join(Settings().output_dir, self.name + "_all_fixed_effects.npy"), self.fixed_effects)

    def _write_individual_RER(self, dataset, individual_RER):
        onset_ages = individual_RER['onset_age']
        write_2D_array(onset_ages, Settings().output_dir, self.name + "_onset_ages.txt")
        write_2D_array(np.exp(individual_RER['log_acceleration']), Settings().output_dir, self.name + "_alphas.txt")
        write_2D_array(individual_RER['log_acceleration'], Settings().output_dir, self.name + "_log_accelerations.txt")
        write_2D_array(np.array(dataset.subject_ids), Settings().output_dir, self.name + "_subject_ids_unique.txt",
                       fmt='%s')
        if not self.no_parallel_transport:
            write_2D_array(individual_RER['sources'], Settings().output_dir, self.name + "_sources.txt")

    def _write_geodesic_and_parallel_trajectories(self):
        """
        Plot the trajectory corresponding to each different source.
        """

        # Getting the geodesic trajectory
        times_geodesic = np.array(self.spatiotemporal_reference_frame.geodesic.get_times())
        geodesic_values = self.spatiotemporal_reference_frame.geodesic.get_geodesic_trajectory()
        labels = Settings().labels

        geodesic_values = np.array([elt.cpu().data.numpy() for elt in geodesic_values])

        # Saving a plot of this trajectory
        if self.observation_type == 'scalar':
            self._plot_scalar_trajectory(times_geodesic, geodesic_values, names=labels)
            self._save_and_clean_plot("reference_geodesic.pdf")

        else:
            # Emptying the geodesic_trajectory directory
            self._clean_and_create_directory(os.path.join(Settings().output_dir, "geodesic_trajectory"))
            self._write_image_trajectory(range(len(times_geodesic)), geodesic_values,
                                         os.path.join(Settings().output_dir, "geodesic_trajectory"), "geodesic")

        if self.observation_type == 'image':
            times_parallel_curves = np.linspace(np.min(times_geodesic) + 1e-5, np.max(times_geodesic) - 1e-5, 20)

        else:
            times_parallel_curves = np.linspace(np.min(times_geodesic) + 1e-5, np.max(times_geodesic) - 1e-5, 200)

        # Saving a txt file with the trajectory.
        write_2D_array(times_geodesic, Settings().output_dir, self.name + "_reference_geodesic_trajectory_times.txt")
        if self.observation_type == 'scalar':
            write_2D_array(geodesic_values, Settings().output_dir,
                           self.name + "_reference_geodesic_trajectory_values.txt")

        # If there are parallel curves, we save them too
        if not self.no_parallel_transport and self.number_of_sources > 0:

            times_torch = Variable(torch.from_numpy(times_parallel_curves).type(Settings().tensor_scalar_type))

            self._plot_scalar_trajectory(times_geodesic, geodesic_values, names=labels)

            for i in range(self.number_of_sources):
                for s in [0.2, 0.6, 0.9]:
                    source_pos = np.zeros(self.number_of_sources)
                    source_pos[i] = s
                    source_neg = np.zeros(self.number_of_sources)
                    source_neg[i] = -s
                    source_pos_torch = Variable(torch.from_numpy(source_pos).type(Settings().tensor_scalar_type))
                    source_neg_torch = Variable(torch.from_numpy(source_neg).type(Settings().tensor_scalar_type))

                    trajectory_pos = [self.spatiotemporal_reference_frame.get_position(t, sources=source_pos_torch) for
                                      t in
                                      times_torch]
                    trajectory_neg = [self.spatiotemporal_reference_frame.get_position(t, sources=source_neg_torch) for
                                      t in
                                      times_torch]

                    trajectory_pos = np.array([elt.cpu().data.numpy() for elt in trajectory_pos])
                    trajectory_neg = np.array([elt.cpu().data.numpy() for elt in trajectory_neg])

                    write_2D_array(trajectory_pos, Settings().output_dir,
                                   self.name + "_source_" + str(i) + "_pos_" + str(s) + ".txt")
                    write_2D_array(trajectory_neg, Settings().output_dir,
                                   self.name + "_source_" + str(i) + "_neg_" + str(s) + ".txt")
                    write_2D_array(times_parallel_curves, Settings().output_dir,
                                   self.name + "_times_parallel_curves.txt")
                    self._plot_scalar_trajectory(times_parallel_curves, trajectory_pos,
                                                 linestyles=['dashed'] * len(trajectory_pos[0]),
                                                 linewidth=0.8 * (1 - s), names=labels)
                    self._plot_scalar_trajectory(times_parallel_curves, trajectory_neg,
                                                 linestyles=['dotted'] * len(trajectory_neg[0]),
                                                 linewidth=0.8 * (1 - s), names=labels)

                self._save_and_clean_plot("plot_source_" + str(i) + '.pdf')

    def _write_model_predictions(self, dataset, individual_RER, sample=False):
        """
        Compute the model predictions
        if sample is On, it will compute predictions, noise them and save them
        else it will save the predictions.
        """

        v0, p0, modulation_matrix = self._fixed_effects_to_torch_tensors(False)
        onset_ages, log_accelerations, sources = self._individual_RER_to_torch_tensors(individual_RER, False)
        t0 = self.get_reference_time()

        absolute_times = self._compute_absolute_times(dataset.times, log_accelerations, onset_ages)

        absolute_times_to_write = []
        for elt in absolute_times:
            for e in elt.cpu().data.numpy():
                absolute_times_to_write.append(e)

        np.savetxt(os.path.join(Settings().output_dir, self.name + "_absolute_times.txt"),
                   np.array(absolute_times_to_write))

        accelerations = torch.exp(log_accelerations)

        self._update_spatiotemporal_reference_frame(absolute_times, p0, v0, modulation_matrix)

        linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

        pos = 0
        nb_plot_to_make = 10000
        if sample:
            nb_plot_to_make = float("inf")
        subjects_per_plot = 3
        predictions = []
        subject_ids = []
        times = []

        targets = dataset.deformable_objects

        number_of_subjects = dataset.number_of_subjects
        residuals = []

        already_plotted_patients = 0

        for i in range(number_of_subjects):
            predictions_i = []
            for j, t in enumerate(absolute_times[i]):
                if sources is not None:
                    prediction = self.spatiotemporal_reference_frame.get_position(t, sources=sources[i])
                else:
                    prediction = self.spatiotemporal_reference_frame.get_position(t)
                predictions_i.append(prediction.cpu().data.numpy())
                predictions.append(prediction.cpu().data.numpy())
                subject_ids.append(dataset.subject_ids[i])
                times.append(dataset.times[i][j])

            targets_i = targets[i].cpu().data.numpy()

            residuals.append(np.mean(np.linalg.norm(predictions_i - targets_i, axis=0, ord=1)))

            if already_plotted_patients >= 10:
                continue
            # Now plotting the real data.
            if nb_plot_to_make > 0:
                # We also make a plot of the trajectory and save it.
                times_subject = dataset.times[i]  # np.linspace(dataset.times[i][0], dataset.times[i][-1], 10)
                absolute_times_subject = [self._compute_absolute_time(t, accelerations[i], onset_ages[i], t0) for t in
                                          times_subject]

                if sources is None:
                    trajectory = [self.spatiotemporal_reference_frame.get_position(t) for t in absolute_times_subject]
                else:
                    trajectory = [self.spatiotemporal_reference_frame.get_position(t, sources=sources[i]) for t in
                                  absolute_times_subject]

                trajectory = np.array([elt.cpu().data.numpy() for elt in trajectory])

                targets_i = targets_i.reshape(len(targets_i), Settings().dimension)
                self._plot_scalar_trajectory(times_subject, trajectory,
                                             names=['subject ' + str(int(float(dataset.subject_ids[i])))],
                                             linestyles=linestyles)
                self._plot_scalar_trajectory(times_subject, targets_i, linewidth=.05)
                self._plot_scalar_trajectory([t for t in dataset.times[i]], targets_i, linestyles=linestyles,
                                             linewidth=0.2)
                pos += 1
                if pos >= subjects_per_plot or i == number_of_subjects - 1:
                    self._save_and_clean_plot("plot_subject_" + str(i - pos + 1) + '_to_' + str(i) + '.pdf')
                    pos = 0
                    nb_plot_to_make -= 1
                    already_plotted_patients += 1

        # Saving the predictions, un-noised
        if self.observation_type == 'scalar':
            write_2D_array(np.array(predictions), Settings().output_dir, self.name + "_reconstructed_values.txt")

        write_2D_array(np.array(subject_ids), Settings().output_dir, self.name + "_subject_ids.txt", fmt='%s')
        write_2D_array(np.array(times), Settings().output_dir, self.name + "_times.txt")

        write_2D_array(np.array(residuals), Settings().output_dir, self.name + '_residuals.txt')

    def print(self, individual_RER):
        logger.info('>> Model parameters:')

        # Noise variance.
        msg = '\t\t noise_std         ='
        noise_variance = self.get_noise_variance()
        msg += '\t%.4f\t ; ' % (math.sqrt(noise_variance))
        logger.info(f"{msg[:-4]}")

        # Empirical distributions of the individual parameters.
        logger.info('\t\t onset_ages        =\t%.3f\t[ mean ]\t+/-\t%.4f\t[std]' %
                    (np.mean(individual_RER['onset_age']), np.std(individual_RER['onset_age'])))
        logger.info('\t\t log_accelerations =\t%.4f\t[ mean ]\t+/-\t%.4f\t[std]' %
                    (np.mean(individual_RER['log_acceleration']), np.std(individual_RER['log_acceleration'])))
        logger.info('\t\t sources           =\t%.4f\t[ mean ]\t+/-\t%.4f\t[std]' %
                    (np.mean(individual_RER['sources']), np.std(individual_RER['sources'])))

    def _write_image_trajectory(self, times, images, folder, name):
        # We use the template object to write the images.
        if Settings().dimension == 2:
            for j, t in enumerate(times):
                self.template.set_intensities(images[j])
                # self.template.update()
                self.template.write(os.path.join(folder, self.name + "_" + name + "_t__" + str(t) + ".npy"))
        elif Settings().dimension == 3:
            for j, t in enumerate(times):
                self.template.set_intensities(images[j])
                # self.template.update()
                self.template.write(os.path.join(folder, self.name + "_" + name + "_t__" + str(t) + ".nii"))
        else:
            raise RuntimeError("Not a proper dimension for an image.")

    def _plot_scalar_trajectory(self, times, trajectory, names=['memory', 'language', 'praxis', 'concentration'],
                                linestyles=None, linewidth=1.):
        # names = ['MDS','SBR'] # for ppmi
        colors = [('#%06X' % randint(0, 0xFFFFFF)) for i in range(10)]
        for d in range(len(trajectory[0])):
            if d >= len(names):
                if linestyles is not None:
                    plt.plot(times, trajectory[:, d], c=colors[d], linestyle=linestyles[d], linewidth=linewidth)
                else:
                    plt.plot(times, trajectory[:, d], c=colors[d], linestyle='solid', linewidth=linewidth)
            else:
                if linestyles is not None:
                    plt.plot(times, trajectory[:, d], label=names[d], c=colors[d], linestyle=linestyles[d],
                             linewidth=linewidth)
                else:
                    plt.plot(times, trajectory[:, d], label=names[d], c=colors[d], linestyle='solid',
                             linewidth=linewidth)

    def _save_and_clean_plot(self, name):
        plt.legend()
        plt.savefig(os.path.join(Settings().output_dir, name))
        plt.clf()

    def _clean_and_create_directory(self, path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)

    def _write_lsd_coordinates(self, individual_RER, dataset):
        """
        Saves the position in the latent space
        """
        lsd_observations = self._get_lsd_observations(individual_RER, dataset)
        write_2D_array(lsd_observations, self.name + '_latent_space_positions.txt')
        plt.scatter(lsd_observations[:, 0], lsd_observations[:, 1])
        # plt.savefig(os.path.join(Settings().output_dir, self.name+'_all_latent_space_positions.pdf'))
        plt.clf()


