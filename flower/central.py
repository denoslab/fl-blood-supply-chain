from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import numpy as np
from preprocessing import preprocessing, preprocessing_centralized
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings("ignore")   

from myconstants import *
from ml_models import get_model
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def train(net, train_loader, epochs):
    """Train the network on the training set."""
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss(reduction="mean")

    def train_step(x, y):
        net.train()

        #make prediction
        yhat = net(x)

        #computes loss
        loss = loss_fn(y, yhat)

        #computes gradients
        loss.backward()

        #update params
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    for epoch in range(epochs):
        batch_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.view([BATCH_SIZE, -1, N_FEATURES]).to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            loss = train_step(x_batch, y_batch)
            batch_losses.append(loss)
        training_loss = np.mean(batch_losses)
        if epoch + 1 % 10 == 0:
            print(f"[{epoch}/{epochs}] Training loss: {training_loss: .4f}")
        

def test(net, testloader, X_test, scaler):
    """Validate the network on the entire test set."""
    with torch.no_grad():
        predictions = []
        values = []
        for x_test, y_test in testloader:
            x_test = x_test.view([BATCH_SIZE, -1, N_FEATURES]).to(DEVICE)
            y_test = y_test.to(DEVICE)
            net.eval()
            yhat = net(x_test)
            predictions.append(yhat.cpu().numpy())
            values.append(y_test.cpu().numpy())

    def inverse_transform(scaler, df, columns):
        for col in columns:
            df[col] = scaler.inverse_transform(df[col])
        return df

    def format_predictions(predictions, values, df_test, scaler):
        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()
        df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
        df_result = df_result.sort_index()
        df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
        return df_result
    
    df_result = format_predictions(predictions, values, X_test, scaler)

    return df_result


def load_data(filepath):
    """Load hospital data"""
    
    X_train_arr,  X_test_arr, y_train_arr, y_test_arr, X_test, scaler = preprocessing(filepath)
    batch_size = BATCH_SIZE

    train_features = torch.Tensor(X_train_arr).to(DEVICE)
    train_targets = torch.Tensor(y_train_arr).to(DEVICE)

    test_features = torch.Tensor(X_test_arr).to(DEVICE)
    test_targets = torch.Tensor(y_test_arr).to(DEVICE)

    train = TensorDataset(train_features, train_targets)
    test = TensorDataset(test_features, test_targets)

    # kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    num_examples = {"trainset" : len(X_train_arr), "testset" : len(X_test_arr)}
    num_features = X_train_arr.shape[1]
    return train_loader, test_loader, num_examples, num_features, X_test, scaler

def load_centralized_data(filepaths):
    
    X_train_arr,  X_test_arr, y_train_arr, y_test_arr, X_test, scaler = preprocessing_centralized(filepaths)
    batch_size = BATCH_SIZE

    train_features = torch.Tensor(X_train_arr).to(DEVICE)
    train_targets = torch.Tensor(y_train_arr).to(DEVICE)

    test_features = torch.Tensor(X_test_arr).to(DEVICE)
    test_targets = torch.Tensor(y_test_arr).to(DEVICE)

    train = TensorDataset(train_features, train_targets)
    test = TensorDataset(test_features, test_targets)

    # kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    num_examples = {"trainset" : len(X_train_arr), "testset" : len(X_test_arr)}
    num_features = X_train_arr.shape[1]
    return train_loader, test_loader, num_examples, num_features, X_test, scaler


def main():
    datasets = DATASETS
    print("Centralized training")
    print("Loading data")
    for i in range(len(datasets)):
        trainloader, testloader, num_examples, num_features, X_test, scaler = load_data(datasets[i])
        model = get_model(MODEL, MODEL_PARAMS).to(DEVICE)
        model.eval()
        print("Start training on " + MODEL)
        train(net=model, train_loader=trainloader, epochs=EPOCH)
        print("Evaluate model on " + MODEL)
        df_result = test(net=model, testloader = testloader, X_test = X_test, scaler = scaler)
        print("R2 score: ", r2_score(df_result.value, df_result.prediction))
        print("RMSE: ", mean_squared_error(df_result.value, df_result.prediction, squared=False))
        torch.save(model.state_dict(), './flower/savedmodels/local' + str(i+1) + '.pt')

    trainloader, testloader, num_examples, num_features, X_test, scaler = load_centralized_data(datasets)
    model = get_model(MODEL, MODEL_PARAMS).to(DEVICE)
    model.eval()
    print("Start training on " + MODEL)
    train(net=model, train_loader=trainloader, epochs=EPOCH)
    print("Evaluate model on " + MODEL)
    df_result = test(net=model, testloader = testloader, X_test = X_test, scaler = scaler)
    print("R2 score: ", r2_score(df_result.value, df_result.prediction))
    print("RMSE: ", mean_squared_error(df_result.value, df_result.prediction, squared=False))
    torch.save(model.state_dict(), './flower/savedmodels/centralized.pt')

if __name__ == "__main__":
    main()