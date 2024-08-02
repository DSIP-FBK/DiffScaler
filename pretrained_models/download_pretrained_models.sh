#!/bin/bash

# download data
wget "https://zenodo.org/records/12941117/files/pretrained_models.zip?download=1" -O pretrained_models.zip

# unzip data
unzip pretrained_models -d ./
rm pretrained_models.zip