# Anomaly Detection and Prediction explainability

## Overview

Implementation of an LSTM-Autoencoder based neural network for Anomaly detection in Industrial Conntrol Systems (ICS) network logs. 
The output of the model is then explained using the [shap](https://github.com/slundberg/shap) library 


## How to install dependencies

To install them, run:

```
pip install -r src/requirements.txt
```

## How to retrain the model

First, specify the training configuration in configs/train.yaml and it's children config files and then run the following
```
python train.py
```

## How to explain the model's output
The explainer pipeline is a preprocessing-prediction-explaining pipeline. First, one has to specify the configs in configs/explain.yaml and its children config files. The pipeline then generates an output structure for all discovered anomalies and a jupyter notebook to help navigate between these.
To run this pipeline, execute:
```
python explain.py
```

## Authors 
Project implemented and maintained by Mohamed Said Derbel, under the supervision of:
Christian Lübbeln, Holger Kinkelin and Lars Wüstrich 