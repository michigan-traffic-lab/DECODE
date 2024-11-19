# DECODE: Domain-aware Continual Domain Expansion for Motion Prediction

## I. Introduction

### About

Welcome to the official GitHub repository for our paper on the DECODE framework, a novel continual learning framework for motion prediction. This repository contains all the necessary code and resources to replicate the results and experiments presented in our study. It is developed by the [Michigan Traffic Lab](https://traffic.engin.umich.edu/) at the University of Michigan.

### Features

The code in this repository provides the following features:
- Data processing for several motion datasets following ScenarioNet definition.
- Continual learning of a motion prediction model using the DECODE framework as shown below
<img src="docs/visuals/framework.png" width="800" alt="Decode framework">

### Code Structure

The structure of the code is as follows:

```
DECODE
├── configs # This folder contains all configuation files which are managed by Hydra
│   ├── config.yaml
│   ├── method
│   │   ├── DECODEMTR.yaml
│   │   ├── ...
│   │   └── preMTR.yaml
│   ├── method
│   │   ├── rounD.yaml
│   │   ├── highD.yaml
│   │   └── inD.yaml
├── domain_expansion # This folder contains main functions
│   ├── dataset # This folder contains data preprocessing functions
│   │   ├── converter.py
│   │   └── data_process.py
│   ├── continual # This folder contains continual learning models
│   │   ├── hyperMTR.py
│   │   ├── erMTR.py
│   │   └── ewcMTR.py
│   ├── detector # This folder contains domain awareness functions
│   │   ├── flow.py
│   │   └── lgmm.py
│   ├── model # This folder contains the specific motion prediction model
│   │   ├── mtr
│   ├── hypermodel # This folder contains hypernetwork-related functions
│   ├── natural # This folder contains deep Bayesian uncertainty estimation functions
│   └── utils
├── scripts # This folder contains executable scripts and main functions
│   ├── continual_train.py
│   ├── convert_data.py
├── data # [Optional] Store raw and processed data here 
└── resutls # [Optional] Store model checkpoints here
```

## II. Installation and Environment Configuration

### Pre-requirements 

0. We strongly suggest users to create a conda environment to install all dependencies.
```bash
conda create -n DECODE python=3.8
conda activate DECODE
```
1. Install ScenarioNet: https://scenarionet.readthedocs.io/en/latest/install.html
2. Install Unitraj: https://github.com/vita-epfl/UniTraj/tree/main?tab=readme-ov-file

### Installation and configuration

```bash
git clone https://github.com/michigan-traffic-lab/DECODE.git
cd DECODE
pip install -r requirements.txt
```

## III. Dataset

## IV. Usage

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

## V. Main results
<img src="docs/visuals/domain_expansion.png" width="800" alt="Main results">

## VI. Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

## VII. License

This project is licensed under the [GNU Affero General Public License v3.0]. Please refer to LICENSE for more details.

## VIII. Third-Party Attribution

This project includes code and content adapted from the following sources:

1. **MIT-Licensed Code**  
   Adapted from [Project Name](<URL>).  
   Licensed under the MIT License.  

2. **CC BY-NC 4.0 Content**  
   Adapted from *Original Work Title* by *Author Name*, available at [Original URL].  
   Licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).  

See the `NOTICE` file for more details.


## VIII. Developers

- Boqi Li (boqili@umich.edu)

## IX. Contact

- Henry Liu (henryliu@umich.edu)
