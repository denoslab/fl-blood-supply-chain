import torch

# Model params 
N_FEATURES = 11
INPUT_DIM = N_FEATURES
OUTPUT_DIM = 1
HIDDEN_DIM = 64
LAYER_DIM = 3
BATCH_SIZE = 64
DROPOUT = 0.2
EPOCH = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# Models chosen from rnn, lstm 
MODEL = "lstm"
MODEL_PARAMS = {"input_dim": INPUT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "layer_dim": LAYER_DIM,
                "output_dim": OUTPUT_DIM,
                "dropout_prob": DROPOUT}


# FL Settings
ROUND = 7
NUM_CLIENTS = 5 

# Balanced Datasets
DATASETS = ["./arima/C1.csv", "./arima/C2.csv", "./arima/C3.csv", "./arima/C4.csv", "./arima/C5.csv"]
DATASETS = DATASETS[:NUM_CLIENTS]
