# Riemannian metric learning

This repo proposes a framework to evaluate longitudinal evolution of scalar data with a mixed effect model. 
The goal is to both recover the average trajectory of the features and the individual variations to this 
mean trajectory for each individual in order to best reconstruct the trajectories. The goal is threesome : get a better
understanding of the **average evolution** of the features and allow to fit every patients to a personalize trajectory in order 
to perform **imputation** of missing data and **prediction** for future timepoints.


## Quick overview of the file tree

- `deformetrica` : main folder that is inherited from the  [deformetrica source code](https://www.deformetrica.org/) . This repo is based 
on their work as Riemmannian geometry tools were often already implemented.

- `api` : contains the deformetrica object that handles logger and launches the training functions.

- `core` : contains most of the content of this repo.
  - `estimator_tools` :  sampler used to get realisations of the individual parameters for the Metropolis
    Hastings Gibbs procedure.
  - `estimators` : the general classes of estimators used `mcmc_saem`, `gradient_ascent` and
  `scipy_minimize`. These are used to recover the fixed effects and random effects.
  - `model_tools` : Riemannian tools. It defines the `exponential` and `geodesic` objects and an object called
  `spatiotemporal_reference_frame` that contains the reference geodesic, the two exponentials (forward and backward) and all 
    the numerical solvers for Hamiltonian equations (for both parallel transport and exponentiation).
  - `models` : here there is some legacy code from deformetrica where a variety of models were used. The only one we use here 
  is the `longitudinal_metric_learning`. It wraps the dataset, all the individual parameters that are sampled in the MCMC and 
    all the fixed effects that are optimized in the EM phase.
  - `observations` : all the tools to load the dataset.
 - `in_out` : every function to load parameters or save things.
 - `launch` : all the functions that launch the fit procedures for the models in `models/`. Once again, we only use the
`estimate_longitudinal_metric_model`.
- `parametric` : all the data we use for various tests and all the scripts for our study.
- `support` : contains the statistical objects for various distributions that are used for sampling.

More specifically in `parametric/` there are 3 relevant files :
- `initialize_parametric_model` : first rough estimation of the parameters through a gradient descent
  in order to 'help' the MCMC algorithm. This is full of heuristic that are data-dependant because its only goal is to provide 
  the best possible starting point for the fit.
- `fit_parametric_model`: once the initialization is done, we launch the MCMC-SAEM procedure to fit the model.
- `personalize_parametric_model` : after the fit we may want to estimate the individual parameters for patients that were not in 
the training dataset, to evaluate the model's ability to generalize. This is a simple gradient descent on the individual parameters
  while keeping the fixed effects as they were learned during the fit.
  
## TODO

- [ ] Trim the file tree to only keep the relevant files
- [ ] Reorganise the train files to make a more consistent architecture
- [ ] Reorganise the way arguments are passed to the model because the `dataset.xml`/`optimization_parameters.xml`/`model.xml`is
super ugly
- [ ] Pass the amount of MCMC steps between each gradient descent steps as a meta-parameters instead of a fixed one
- [ ] Clean the intialize/personalize files, especially with the way arguments are handled. Also use the clean logger of deformetrica
for initialization and save the personalized ip in the correct format
- [ ] Delete all the debugging prints


## References

