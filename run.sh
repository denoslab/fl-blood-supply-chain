#!/bin/bash

echo Activating environment
conda activate flower

echo Starting Centralized Training
python ./flower/central.py

echo Starting Federated Learning Simulation
python ./flower/fl_simulation.py

echo Plotting Evaluation Metrics
python ./flower/saved_model_eval.py