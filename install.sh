#!/bin/bash

echo Creating environment
conda create --name flower python=3.8 

echo Activating environment
conda activate flower

echo Fetching nesscessary packages
python -m pip install -U --pre flwr[simulation]
conda install pandas matplotlib jupyter
pip install -U scikit-learn statsmodels plotly mlxtend charset-normalizer==2.1.1
pip3 install torch torchvision torchaudio