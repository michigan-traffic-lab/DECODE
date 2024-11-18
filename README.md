# DECODE: Domain-aware Continual Domain Expansion for Motion Prediction
This is prediction module for domain-awared trajectory prediction and continual learning

<img src="visualization/framework.png" width="800" alt="Decode framework">

## Installation 

### Pre-requirement 
Create a conda environment and activate it
```
conda create -n DECODE python=3.8
conda activate DECODE
```
### Install pytorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
### Install ScenarioNet: https://scenarionet.readthedocs.io/en/latest/install.html
### Install Unitraj: https://github.com/vita-epfl/UniTraj/tree/main?tab=readme-ov-file

### Install required packages 
```
pip install waymo-open-dataset-tf-2-11-0==1.5.2
pip install lightning==2.0.5 matplotlib==3.6.1

```
### Continual Train MTR model with DECODE strategy:
#### Domain 1
```
python continual_train method=DECODEMTR domain=rounD
```
#### Domain 2
```
python continual_train method=DECODEMTR domain=highD
```
#### Domain 3
```
python continual_train method=DECODEMTR domain=inD
```

### Evaluate trained MTR model with DECODE strategy:
```
python model_evaluation method=DECODEMTR
```

### Main results
<img src="visualization/domain_expansion.png" width="800" alt="Main results">