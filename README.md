# Longitudinal Variational Autoencoders 

This repo proposes a framework to evaluate longitudinal evolution of imaging data with a mixed effect model. 
The goal is to both recover the population-average trajectory and the individual variations to this 
mean trajectory for each individual in order to best reconstruct the trajectories. 

Based on the article [Progression models for imaging data using longitudinal autoencoders](https://hal.inria.fr/hal-03701632/document).

## Abstract

Disease progression models are crucial to understanding degenerative diseases. Mixed-effects models have been consistently used to model clinical assessments or biomarkers extracted from medical images, allowing missing data imputation and prediction at any timepoint. However, such progression models have seldom been used for entire medical images. In this work, a Variational Auto Encoder is coupled with a temporal linear mixed-effect model to learn a latent representation of the data such that individual trajectories follow straight lines over time and are characterised by a few interpretable parameters. A Monte Carlo estimator is devised to iteratively optimize the networks and the statistical model. We apply this method on a synthetic data set to illustrate the disentanglement between time dependant changes and inter-subjects variability, as well as the predictive capabilities of the method. We then apply it to 3D MRI and FDG-PET data from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) to recover well documented patterns of structural and metabolic alterations of the brain.

## Architecture

The network architectures are in `core/model_tools/neural_networks/networks_LVAE.py`. They are vanilla convolutional VAEs, coupled with a latent linear mixed-effect model.

![image](https://user-images.githubusercontent.com/42009161/184084018-dc7a6ef3-f6ec-4d69-86cb-292491eed4da.png)

## Sample results

Average progression of t1-MRI scans over the course of Alzheimer's Disease :

![image](https://user-images.githubusercontent.com/42009161/184084405-77f6d304-464a-415a-9e51-1c1c8fab47ab.png)

