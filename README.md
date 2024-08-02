<div align="center">

# LDM_res: a Latent Diffusion Model for meteorological downscaling

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2406.13627)
<!-- [![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https:) -->

</div>

## Description

LDM_res is a downscaling model based on a Latent Diffusion Model (LDM, used by e.g. [Stable Diffusion](https://github.com/CompVis/stable-diffusion)), developed to downscale meteorological variables from ERA5 reanalyses.

We trained and tested LDM_res to produce 2-km fields of 2-m temperature and 10-m wind speed horizontal components starting from a list of predictors from ERA5 (interpolated @16km). The high-resolution reference truth data are provided by a dynamical downscaling performed with COSMO\_CLM ([VHR-REA IT](https://www.mdpi.com/2306-5729/6/8/88)). The model and its performance are presented and discussed in this [arXiv preprint](https://arxiv.org/abs/2406.13627).

This repository contains the code for testing and training LDM_res, all the baselines presented in the paper, and the code used to generate all the paper Figures.

A GPU is recommended for both testing and training LDM_res.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# install pytorch according to instructions
# https://pytorch.org/get-started/

# create virtual env
python3 -m venv .venv
source .venv/bin/activate

# install requirements
pip install -r requirements.txt
```

## Get training and testing data

Zenodo hosts:
1. a [sample dataset](https://zenodo.org/records/12934521) [45GB], that you can use to perform a sample training of the models
2. the [full dataset](https://zenodo.org/records/12944960) [330GB], divided into 7 Zenodo repository, that you can use to reproduce our experiments
3. the [pretrained models](https://zenodo.org/records/12941117) (UNET, GAN, VAE_res and LDM_res) and their main outputs [12GB], that you can use to test the models and to reproduce the paper figures.

You can download the sample dataset by running the `download_sample_dataset.sh` script in [data/](data/):
```bash
cd data

# download and unzip the sample dataset
bash download_sample_dataset.sh
```

You can download the full dataset by running the `download_full_dataset.sh` script in [data/](data/):
```bash
cd data

# download and unzip the full dataset
bash download_full_dataset.sh
```

You can download the pretrained models by running the `download_pretrained_models.sh` script in  [pretrained_models/](pretrained_models/):
```bash
cd data

# download the pretrained models
bash download_pretrained_models.sh
```

## How to test the pretrained models in inference

The notebook [notebooks/models_inference.ipynb](notebooks/models_inference.ipynb) guides you through the setup and test of the LDM_res model and its baselines. To use this notebook you only need to download the pretrained models Zenodo repo. The notebook also saves model results for later plotting.

You can plot snapshots of the model results with the notebook [notebooks/Fig_snapshots.ipynb](notebooks/Fig_snapshots.ipynb) (as they are presented in the paper), adjusting the paths.

You can recreate all the figures presented in the paper using all the [notebooks/Fig_*.ipynb](notebooks/) notebooks, reading results files included in [pretrained_models/outputs](pretrained_models/outputs), after downloading from the Zenodo repository.

## How to train the models

If you like to train the models you can run the following command, which rely on the configurations set in the files within [configs/](configs/). To do this you need to download either the sample dataset or the whole dataset from Zenodo.

Train any model with a chosen experiment configuration from [configs/experiment/](configs/experiment/).
 - Available ModelNames are UNET, GAN, VAE_res, LDM_res.
 - Available VarName are 2mT and UV.
To train VAE_res you need to provide a UNET checkpoint and to train LDM_res you need to provide a UNET checkpoint and a VAE_res checkpoint: by default configuration files point to [pretrained_models/](pretrained_models/), reading and using pretrained models checkpoints. If you didn't download the pretrained models from Zenodo, adjust the checkpoint configurations in [configs/experiment/downscaling_VAE_res_**.yaml](configs/experiment/) and in [configs/experiment/downscaling_LDM_res_**.yaml](configs/experiment/) to point to your UNET and VAE_res checkpoints.

```bash
python src/train.py experiment=downscaling_ModelName_VarName
```


