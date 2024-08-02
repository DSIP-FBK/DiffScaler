#!/bin/bash

# download data
wget "https://zenodo.org/records/12934521/files/sample_data.zip?download=1" -O sample_dataset.zip

# unzip data
unzip sample_dataset -d ./
rm sample_dataset.zip