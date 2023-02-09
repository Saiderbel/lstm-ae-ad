
<div align="center">

# Anomaly Detection and Prediction explainability

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.10+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.8+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.1-89b8cd)](https://hydra.cc/)
</div>




## Overview

The goal of this project is to develop a machine learning model that can accurately identify anomalies in network logs for industrial control systems. To achieve this, the project employs an LSTM-Autoencoder model, which is a type of deep learning neural network architecture that is well-suited for time series data.

The project provides several pipelines for data processing, model training and testing, and prediction on new datasets.

The *new_model_version pipeline* allows for the creation of a new model version based on a provided dataset. This pipeline processes the dataset, trains the LSTM-Autoencoder model, and saves the trained model along with all necessary data and files for later use.

The *explain_with_model_version* pipeline can be used to make predictions on new datasets using a previously trained model version. This pipeline loads the trained model and uses it to make predictions on the new data, outputting the results as well as their explainability.

In addition to these main pipelines, the project also includes other utilities for data preprocessing, 
model evaluation, and visualization of results.


### For full documentation refer to *docs/_build/html/index.html*

## Setup

Install package requirements via::

      $ pip install -r requirements.txt


A script is provided to install conda, create a new conda environment and install the packages. For that run::

      $ chmod+x conda.sh init_env.sh
      $ ./conda.sh

restart terminal and run::

      $ ./init_env.sh

After this you should be able to activate the environment via::

      $ conda activate ad



## Usage

**Note**: The dataset is expected to be a ``.csv`` file with a name of this format:
   -  ``yyyy-mm-dd_hh-mm-ss_log-data.csv``
and have the following features:
   - 'Bytes (custom)', 'Destination IP', 'Destination Port', 'Event Name', 'Log Source', 'Magnitude', 'Source IP', 'Start Time'

Any additional or missing features will lead to failure. In order to fix that, a code update is required (preprocessing pipeline, setting the right number of features to the model).

It is possible to ignore one or many features by means of the ``features_to_ignore`` parameter.

To run any of the provided pipelines one can either specify the pipeline configs in the respecitive config file in `configs/` or provide them as command line arguments.
For most of the pipelines it is recommended to use the config files and then run the pipeline in project root::

      $ python pipeline_name.py





*new_model_version pipeline* and *explain_with_model_version* represent the core pipelines for a straightforward use of the package.

*new_model_version pipeline*: this pipeline takes as arguments a dataset path ``dataset_path``, a set of features ``features_to_ignore`` one wishes
to ignore and the number ``gpus`` of gpus one wishes to use (0 for cpu), and does the following:

1. Preprocess the provided dataset
2. Save the preprocessed dataset as well as feature encoders/transformers to use for predicting on other datasets
3. Train a model version based on the dataset
4. Save model and the set of training mean absolute errors (used later to define anomaly thresholds)
5. Generate config file to predict on other datasets using this model version

To run the pipeline:

      $ python new_model_version.py dataset_path="path/to/set" 'features_to_ignore=["feat_name", .. ]' gpus=<num_of_gpus>



note that passing the ``features_to_ignore attribute`` is done within two single quotes ' '. If given solely a dataset filename
we assume the file is situatued in the *data/*  folder.

*explain_with_model_version*: this pipeline takes as arguments a dataset path ``dataset_to_explain_path``, a model version name ``model_version`` one wishes
to use, and a float that represent the benign fraction of the training dataset and does the following
1. Preprocess the provided dataset using the feature processors of the provided model version and save it
2. Use the provided model to predict anomalies in the dataset
3. Explain model predictions.
4. Generate a notebook that runs out-of-the-box to provide an overview of the detected anomalies as well as their respective explanations.


To run the pipeline:

      $ python explain_with_model_version.py dataset_to_explain_path="/path/to/set" model_version=<model_version> outlier_threshold=0.99


If given solely a dataset filename we assume the file is situatued in the *data/* folder.

Running this pipeline triggers the following warning ``Warning: unrecognized nn.Module: RNN``. This is due to the fact that LSTMs and RNNs are not yet supported by shap.
It is to note that explaining large datasets takes a bit long since the computations don't run on the gpu.

The authors quoted:
   "RNNs aren't yet supported for the PyTorch DeepExplainer (A warning pops up to let you know which modules aren't supported yet: Warning: unrecognized nn.Module: RNN).
   In this case, the explainer assumes the module is linear, and makes no change to the gradient.
   Since RNNs contain nonlinearities, this is probably contributing to the problem.

   Adding RNN support for the PyTorch DeepExplainer is definitely planned, but is a little tricky because - since
   a lot of stuff is hidden in PyTorch's C++ backend - RNN layers can't be broken down into their simpler operations."


## Authors 
Project implemented and maintained by Mohamed Said Derbel, under the supervision of Christian Lübbeln, Holger Kinkelin and Lars Wüstrich.