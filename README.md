# Energy Forecasting with Deep Learning using MLDM
author: Andrew Mendez, andrew.mendez@hpe.com

Version: 0.0.1

Date: 3.14.23

# Background
This tutorial introduces the process of training a deep learning model for time series forecasting, specifically forecasting energy demand. Time series forecasting is a crucial aspect of many domains, including energy management, where accurate predictions can lead to efficient energy use and help in decision-making processes.

# Dataset
The dataset used in this example contains various energy generation and consumption data. We focus on the 'generation fossil gas' feature to predict future energy demand.
The dataset comprises 4 years of data on electrical consumption, generation, pricing, and weather for Spain. It includes detailed hourly data, allowing for fine-grained analysis and forecasting of energy demand and supply dynamics. More details about the dataset can be seen [here](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)
# Model
We employ a Temporal Convolutional Network (TCN) model, known for its effectiveness in handling sequence data like time series.

The TCN model uses a series of dilated convolutional layers that allow the network to have an exponentially increasing receptive field. This means the model can incorporate information from points further back in the time series without a proportional increase in computational complexity. The architecture is particularly suited for time series data because it can effectively capture long-term dependencies and patterns. Key features include causal convolutions (ensuring predictions at time t are only dependent on data from times t' <= t) and residual connections (helping with training deep networks by allowing gradients to flow through the network's layers more effectively).

# Interactive Notebook Demo steps
* Run notebook `tutorial_pt1.ipynb` to run time series code without pachyderm
* Run notebook `pt2_mldm_tutorial.ipynb` to run time series modeling with MLDM/Pachyderm

# Manually Deploy the Pachyderm Pipeline

## Create Pachyderm repos:

* `pachctl create repo data` 
* `pachctl create repo code` 
* `pachctl create repo model`

## Upload data to repos
* `pachctl put file model@master: -r -f models/TCN_model.pt`
* `pachctl put file model@master: -r -f models/TCN_model.pt.ckpt`
* `pachctl put file data@master: -r -f data/energy_dataset.csv`
* `pachctl put file data@master: -r -f data/energy_dataset.csv`
* `pachctl put file code@master: -r -f code/`

## Create file called process.yaml:
```yaml
pipeline:
    name: 'process'
description: 'Extract content in xml files to a csv file'
input:
    cross:
        - pfs: 
            repo: 'data'
            branch: 'master'
            glob: '/'
        - pfs: 
            repo: 'code'
            branch: 'master'
            glob: '/'
transform:
    image: mendeza/python38_process:0.2
    cmd: 
        - '/bin/sh'
    stdin: 
    # - "while :; do echo 'Hello'; sleep 5 ; done"
    - 'pip install darts;'
```

## Run command to deploy the preprocessing pipeline:
* `pachctl create pipeline -f process.yaml`

## Create a yaml file named train.yaml:
```yaml
pipeline:
    name: 'train'
description: 'Extract content in xml files to a csv file'
input:
    cross:
        - pfs: 
            repo: 'process'
            branch: 'master'
            glob: '/'
        - pfs: 
            repo: 'code'
            branch: 'master'
            glob: '/'
        - pfs: 
            repo: 'model'
            branch: 'master'
            glob: '/'
transform:
    image: mendeza/python38_process:0.2
    cmd: 
        - '/bin/sh'
    stdin: 
    # - "while :; do echo 'Hello'; sleep 5 ; done"
    - 'pip install darts'
    - 'bash /pfs/code/code/scripts/train.sh'
autoscaling: False
```
## Run command the command to deploy the training pipeline:
* `pachctl create pipeline -f train.yaml`

## Create yaml file named val.yaml:
```yaml
pipeline:
    name: 'val'
description: 'Extract content in xml files to a csv file'
input:
    cross:
        - pfs: 
            repo: 'process'
            branch: 'master'
            glob: '/'
        - pfs: 
            repo: 'train'
            branch: 'master'
            glob: '/'
        - pfs: 
            repo: 'code'
            branch: 'master'
            glob: '/'
transform:
    image: mendeza/python38_process:0.2
    cmd: 
        - '/bin/sh'
    stdin:
    # - "while :; do echo 'Hello'; sleep 5 ; done"
    - 'pip install darts'
    - 'bash /pfs/code/code/scripts/val.sh'
autoscaling: False
```

## Run command the command to deploy the validation pipeline: 
* `pachctl create pipeline -f val.yaml`

## Run the step to show how uploading more data can automatically retigger the training and validation end-to-end:

* `pachctl put file data@master: -r -f data/energy_dataset2.csv`

## Clean up demo:
```bash
pachctl delete pipeline val
pachctl delete pipeline train
pachctl delete pipeline process
pachctl delete repo data
pachctl delete repo code
pachctl delete repo model
```