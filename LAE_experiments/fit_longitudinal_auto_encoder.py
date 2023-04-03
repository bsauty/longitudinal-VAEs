#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import logging
import os.path
import time
from numpy import number

import torch
import sys

from copy import deepcopy
import torch.nn as nn
import torch.optim as optim

sys.path.append('/home/benoit.sautydechalon/deformetrica')
import deformetrica as dfca

from deformetrica.core.estimator_tools.samplers.srw_mhwg_sampler import SrwMhwgSampler
from deformetrica.core.estimators.gradient_ascent import GradientAscent
from deformetrica.core.estimators.mcmc_saem import McmcSaem
# Estimators
from deformetrica.core.estimators.scipy_optimize import ScipyOptimize
from deformetrica.core.model_tools.manifolds.exponential_factory import ExponentialFactory
from deformetrica.core.model_tools.manifolds.generic_spatiotemporal_reference_frame import GenericSpatiotemporalReferenceFrame
from deformetrica.core.model_tools.neural_networks.networks_LVAE import Dataset, CVAE_2D, CVAE_3D, VAE_GAN
from deformetrica.in_out.array_readers_and_writers import *
from deformetrica.core import default
from deformetrica.in_out.dataset_functions import create_scalar_dataset
from deformetrica.support.probability_distributions.multi_scalar_normal_distribution import MultiScalarNormalDistribution
from deformetrica.support.utilities.general_settings import Settings
from deformetrica.core.models import LongitudinalAutoEncoder

from deformetrica.support.utilities.general_settings import Settings

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Settings().device = device

def initialize_spatiotemporal_reference_frame(model, logger, observation_type='image'):
    """
    Initialize everything which is relative to the geodesic its parameters.
    """
    exponential_factory = ExponentialFactory()
    exponential_factory.set_manifold_type('euclidean')
    logger.info('Initialized the Euclidean metric for latent space')

    model.spatiotemporal_reference_frame = GenericSpatiotemporalReferenceFrame(exponential_factory)
    model.spatiotemporal_reference_frame.set_concentration_of_time_points(default.concentration_of_time_points)
    model.spatiotemporal_reference_frame.set_number_of_time_points(default.number_of_time_points)
    model.no_parallel_transport = False
    model.spatiotemporal_reference_frame.no_parallel_transport = False

def initialize_CAE(logger, model, path_CAE=None):

    if (path_CAE is not None) and (os.path.isfile(path_CAE)):
        checkpoint =  torch.load(path_CAE, map_location='cpu')
        if '2D' in path_CAE:
            autoencoder = CVAE_2D()
        elif 'GAN' in path_CAE:
            autoencoder = VAE_GAN()
        else:
            autoencoder = CVAE_3D()
        autoencoder.beta = 1
        autoencoder.load_state_dict(checkpoint)
        logger.info(f">> Loaded CAE network from {path_CAE}")

    else:
        logger.info(">> Training the CAE network")
        epochs = 200
        batch_size = 10
        lr = 1e-5

        if '2D' in path_CAE:
            autoencoder = CVAE_2D()
        elif 'GAN' in path_CAE:
            autoencoder = VAE_GAN()
        else:
            autoencoder = CVAE_3D()
                
        #checkpoint =  torch.load('VAE_GAN', map_location='cpu')
        #autoencoder.load_state_dict(checkpoint)
            
        logger.info(f"Learning rate is {lr}")
        logger.info(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)} parameters")
        #logger.info(f"Including {sum(p.numel() for p in autoencoder.VAE.parameters() if p.requires_grad)} for the VAE and {sum(p.numel() for p in autoencoder.discriminator.parameters() if p.requires_grad)} for the discriminator.")
        
        # Load data
        train_loader = torch.utils.data.DataLoader(model.train_images, batch_size=batch_size,
                                                   shuffle=True, drop_last=True)
        autoencoder.beta = 1
        if 'GAN' in path_CAE:
            vae_optimizer = optim.Adam(autoencoder.VAE.parameters(), lr=lr)               # optimizer for the vae generator
            d_optimizer = optim.Adam(autoencoder.discriminator.parameters(), lr=4*lr)     # optimizer for the discriminator
            autoencoder.train_(train_loader, test=model.test_images, vae_optimizer=vae_optimizer, d_optimizer=d_optimizer, num_epochs=epochs)
        else:
            optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
            autoencoder.train_(train_loader, test=model.test_images, optimizer=optimizer, num_epochs=epochs)
        
        logger.info(f"Saving the model at {path_CAE}")
        torch.save(autoencoder.state_dict(), path_CAE)
        logger.info(f"Freezing the convolutionnal layers to train only the linear layers for longitudinal analysis")
    
    #for layer in autoencoder.named_children():
    #    if 'conv' in layer[0] or 'bn' in layer[0]:
    #        for param in layer[1].parameters():
    #            param.requires_grad = False
    
    logger.info(f"Model has a total of {sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)} parameters")
    return autoencoder

def instantiate_longitudinal_auto_encoder_model(logger, path_data, path_CAE=None,
                                                dataset=None, xml_parameters=None, number_of_subjects=None):

    model = LongitudinalAutoEncoder()

    # Load the train/test data
    torch_data = torch.load(path_data, map_location='cpu')
    torch_data = Dataset(torch_data['data'].unsqueeze(1).float(), torch_data['labels'], torch_data['timepoints'])
    train, test = torch_data[:len(torch_data) - 100], torch_data[len(torch_data) - 100:]
    train, test = Dataset(train[0], train[1], train[2]), Dataset(test[0], test[1], test[2])

    logger.info(f"Loaded {len(train.data)} train images and {len(test.data)} test images")
    model.train_images, model.test_images, model.full_images = train.data, test.data, torch_data.data
    model.train_labels, model.test_labels, model.full_labels = train.labels, test.labels, torch_data.labels
    model.train_timepoints, model.test_timepoints, model.full_timepoints = train.timepoints, test.timepoints, torch_data.timepoints

    # Initialize the CAE
    model.CAE = initialize_CAE(logger, model, path_CAE=path_CAE)

    # Then initialize the first latent representation
    _, model.train_encoded = model.CAE.evaluate(model.train_images)
    _, model.test_encoded = model.CAE.evaluate(model.test_images)
    model.full_encoded = torch.cat((model.train_encoded, model.test_encoded))

    model.observation_type = 'scalar'
    group, timepoints = model.full_labels, model.full_timepoints
    dataset = create_scalar_dataset(group, model.full_encoded.numpy(), timepoints)

    # Then initialize the longitudinal model for the latent space
    if dataset is not None:
        template = dataset.deformable_objects[0][0] 
        model.template = deepcopy(template)

    # Initialize the fixed effects, either from files or to arbitrary values
    if xml_parameters is not None:
        # Reference time
        model.set_reference_time(xml_parameters.t0)
        model.is_frozen['reference_time'] = xml_parameters.freeze_reference_time
        # Initial velocity
        initial_velocity_file = xml_parameters.v0
        model.set_v0(read_2D_array(initial_velocity_file))
        model.is_frozen['v0'] = xml_parameters.freeze_v0
        # Initial position
        initial_position_file = xml_parameters.p0
        model.set_p0(read_2D_array(initial_position_file))
        model.is_frozen['p0'] = xml_parameters.freeze_p0
        # Time shift variance
        model.set_onset_age_variance(xml_parameters.initial_time_shift_variance)
        model.is_frozen['onset_age_variance'] = xml_parameters.freeze_onset_age_variance
        # Log acceleration variance
        model.set_log_acceleration_variance(xml_parameters.initial_acceleration_variance)
        model.is_frozen["log_acceleration_variance"] = xml_parameters.freeze_acceleration_variance
        # Noise variance
        model.set_noise_variance(xml_parameters.initial_noise_variance)
        # Modulation matrix
        modulation_matrix = read_2D_array(xml_parameters.initial_modulation_matrix)
        if len(modulation_matrix.shape) == 1:
            modulation_matrix = modulation_matrix.reshape(Settings().dimension, 1)
        logger.info(f">> Reading {str(modulation_matrix.shape[1]) }-source initial modulation matrix from file: {xml_parameters.initial_modulation_matrix}")
        assert xml_parameters.number_of_sources == modulation_matrix.shape[1], "Please set correctly the number of sources"
        model.set_modulation_matrix(modulation_matrix)
        model.number_of_sources = modulation_matrix.shape[1]
        model.is_frozen['modulation_matrix'] = xml_parameters.freeze_modulation_matrix

    else:
        # All these initial paramaters are pretty arbitrary TODO: find a better way to initialize
        model.set_reference_time(75)
        #model.set_reference_time(0)
        v0 = np.zeros(Settings().dimension)
        v0[0] = 1/30
        #v0[0] = 1/5
        #v0 = np.ones(Settings().dimension)/6
        model.set_v0(v0)
        model.set_p0(np.zeros(Settings().dimension))
        model.set_onset_age_variance(7)
        model.set_log_acceleration_variance(0.3)
        model.number_of_sources = Settings().number_of_sources
        modulation_matrix = np.zeros((Settings().dimension, model.number_of_sources))
        for i in range(model.number_of_sources):
            modulation_matrix[i+1,i] = .4
        #modulation_matrix = np.random.normal(0,.1,(Settings().dimension, model.number_of_sources))
        model.set_modulation_matrix(modulation_matrix)
        model.initialize_modulation_matrix_variables()
        model.is_frozen['p0'] = True
        model.is_frozen['v0'] = True
        model.is_frozen['modulation_matrix'] = True

    # Initializations of the individual random effects
    assert not (dataset is None and number_of_subjects is None), "Provide at least one info"

    if dataset is not None:
        number_of_subjects = dataset.number_of_subjects

    # Initialization the individual parameters
    if xml_parameters is not None:
        logger.info(f"Setting initial onset ages from {xml_parameters.initial_onset_ages} file")
        onset_ages = read_2D_array(xml_parameters.initial_onset_ages).reshape((len(dataset.times),))
        logger.info(f"Setting initial log accelerations from { xml_parameters.initial_accelerations} file")
        log_accelerations = read_2D_array(xml_parameters.initial_accelerations).reshape((len(dataset.times),))

    else:
        logger.info("Initializing all the onset_ages to the reference time.")
        onset_ages = np.zeros((number_of_subjects,))
        onset_ages += model.get_reference_time()
        logger.info("Initializing all log-accelerations to zero.")
        log_accelerations = np.zeros((number_of_subjects,))

    individual_RER = {}
    individual_RER['onset_age'] = onset_ages
    individual_RER['log_acceleration'] = log_accelerations

    # Initialization of the spatiotemporal reference frame.
    initialize_spatiotemporal_reference_frame(model, logger, observation_type=model.observation_type)

    # Sources initialization
    if xml_parameters is not None:
        logger.info(f"Setting initial sources from {xml_parameters.initial_sources} file")
        individual_RER['sources'] = read_2D_array(xml_parameters.initial_sources).reshape(len(dataset.times), model.number_of_sources)

    elif model.number_of_sources > 0:
        # Actually here we initialize the sources to almost zero values to avoid renormalization issues (div 0)
        logger.info("Initializing all sources to zero")
        individual_RER['sources'] = np.random.normal(0,0.1,(number_of_subjects, model.number_of_sources))
    model.initialize_source_variables()

    if dataset is not None:
        total_number_of_observations = dataset.total_number_of_observations
        model.number_of_subjects = dataset.number_of_subjects

        if model.get_noise_variance() is None:

            v0, p0, modulation_matrix = model._fixed_effects_to_torch_tensors(False)
            onset_ages, log_accelerations, sources = model._individual_RER_to_torch_tensors(individual_RER, False)
                        
            residuals = model._compute_residuals(dataset, v0, p0, modulation_matrix,
                                log_accelerations, onset_ages, sources)

            total_residual = 0.
            for i in range(len(residuals)):
                total_residual += torch.sum(residuals[i]).cpu().data.numpy()

            dof = total_number_of_observations
            nv = total_residual / dof
            model.set_noise_variance(nv)
            logger.info(f">> Initial noise variance set to {nv} based on the initial mean residual value.")

        if not model.is_frozen['noise_variance']:
            dof = total_number_of_observations
            model.priors['noise_variance'].degrees_of_freedom.append(dof)

    else:
        if model.get_noise_variance() is None:
            logger.info("I can't initialize the initial noise variance: no dataset and no initialization given.")

    # Only MCMC for the mixed effect and no network training
    #model.has_maximization_procedure = False
    model.update()

    return model, dataset, individual_RER

def estimate_longitudinal_auto_encoder_model(logger, path_data, path_CAE, xml_parameters=None, number_of_subjects=None):
    logger.info('')
    logger.info('[ estimate_longitudinal_auto_encoder_model function ]')

    if number_of_subjects is None:
        torch_data = torch.load(path_data, map_location='cpu')
        image_data = Dataset(torch_data['data'].unsqueeze(1).float(), torch_data['labels'], torch_data['timepoints'])
        number_of_subjects = len(np.unique(image_data.labels))

    model, dataset, individual_RER = instantiate_longitudinal_auto_encoder_model(logger, path_data, path_CAE=path_CAE,
                                                                        number_of_subjects=number_of_subjects, xml_parameters=xml_parameters)

    sampler = SrwMhwgSampler()
    estimator = McmcSaem(model, dataset, 'McmcSaem', individual_RER, max_iterations=Settings().max_iterations,
             print_every_n_iters=1, save_every_n_iters=10)
    estimator.sampler = sampler

    # Onset age proposal distribution.
    onset_age_proposal_distribution = MultiScalarNormalDistribution()
    onset_age_proposal_distribution.set_variance_sqrt(10)
    sampler.individual_proposal_distributions['onset_age'] = onset_age_proposal_distribution

    # Log-acceleration proposal distribution.
    log_acceleration_proposal_distribution = MultiScalarNormalDistribution()
    log_acceleration_proposal_distribution.set_variance_sqrt(.7)
    sampler.individual_proposal_distributions['log_acceleration'] = log_acceleration_proposal_distribution

    # Sources proposal distribution

    if model.number_of_sources > 0:
        sources_proposal_distribution = MultiScalarNormalDistribution()
        # Here we impose the sources variance to be 1
        sources_proposal_distribution.set_variance_sqrt(1)
        sampler.individual_proposal_distributions['sources'] = sources_proposal_distribution

    estimator.sample_every_n_mcmc_iters = 3
    estimator._initialize_acceptance_rate_information()

    # Gradient-based estimator.
    estimator.gradient_based_estimator = GradientAscent(model, dataset, 'GradientAscent', individual_RER)
    estimator.gradient_based_estimator.statistical_model = model
    estimator.gradient_based_estimator.dataset = dataset
    estimator.gradient_based_estimator.optimized_log_likelihood = 'class2'
    estimator.gradient_based_estimator.max_iterations = 1
    estimator.gradient_based_estimator.max_line_search_iterations = 1
    estimator.gradient_based_estimator.convergence_tolerance = 1e-4
    estimator.gradient_based_estimator.initial_step_size = 1e-1
    estimator.gradient_based_estimator.print_every_n_iters = 5
    estimator.gradient_based_estimator.save_every_n_iters = 100000
    estimator.gradient_based_estimator.scale_initial_step_size = True

    estimator.print_every_n_iters = 1
    estimator.save_every_n_iters = 1

    estimator.dataset = dataset
    estimator.statistical_model = model

    # Initial random effects realizations
    estimator.individual_RER = individual_RER

    if not os.path.exists(Settings().output_dir): os.makedirs(Settings().output_dir)

    model.name = 'LongitudinalMetricModel'
    logger.info('')
    logger.info(f"[ update method of the {estimator.name}  optimizer ]")

    start_time = time.time()
    estimator.update()
    estimator.write()
    end_time = time.time()
    logger.info(f">> Estimation took: {end_time-start_time}")

def main():
    path_data = 'ADNI_data/ADNI_t1'
    path_CAE = 'CVAE_3D_t1'
    xml_parameters = dfca.io.XmlParameters()
    xml_parameters._read_model_xml('model.xml')
    Settings().output_dir = 'output
    deformetrica = dfca.Deformetrica(output_dir=Settings().output_dir, verbosity=logger.level)
    estimate_longitudinal_auto_encoder_model(logger, path_data, path_CAE, xml_parameters=xml_parameters)

if __name__ == "__main__":
    main()