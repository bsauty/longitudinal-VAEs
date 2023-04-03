# Tutorial

The `fit_longitudinal_auto_encoder.py` file is the main script that should be run. To run it, one needs

- A `data_dict` dictionnary that contains `data_dict['data']` tensor of dimension `N_{samples} x  W x L x D` will all the 3D images, a `data_dict['timepoints']` with all the ages of subjects at the times of visits, and `data_dict['label']` that contains the ID of the patients.
- A `model.xml` file that contains all the training parameters and paths to save the model. This is especially useful if one want's to load a pre-trained model and pass all the parameters as arguments.

In the `main` function, one should also give the path to the `data_dict` file and the `output` folder. 

If the convolutional network is not pre-trained, the training starts by training a regular VAE with the data, before adding the longitudinal structure later on, with the learned convolutional weights.