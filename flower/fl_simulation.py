from collections import OrderedDict
from typing import List
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
import numpy as np
import flwr as fl
from preprocessing import preprocessing
from sklearn.metrics import mean_squared_error
from myconstants import *
from ml_models import get_model

# Loading in data
def load_data(batch_size: int):
    """Load hospital data"""

    datasets = DATASETS
    train_loaders = []
    test_loaders = []
    nums_examples = []
    nums_features = []
    X_tests = []
    scalers = []

    for path in datasets:
        X_train_arr,  X_test_arr, y_train_arr, y_test_arr, X_test, scaler = preprocessing(path)
        train_features = torch.Tensor(X_train_arr).to(DEVICE)
        train_targets = torch.Tensor(y_train_arr).to(DEVICE)

        test_features = torch.Tensor(X_test_arr).to(DEVICE)
        test_targets = torch.Tensor(y_test_arr).to(DEVICE)

        train = TensorDataset(train_features, train_targets)
        test = TensorDataset(test_features, test_targets)

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
        num_examples = {"trainset" : len(X_train_arr), "testset" : len(X_test_arr)}
        num_features = X_train_arr.shape[1]

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
        nums_examples.append(num_examples)
        nums_features.append(num_features)
        X_tests.append(X_test)
        scalers.append(scaler)

    return train_loaders, test_loaders, nums_examples, nums_features, X_tests, scalers

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

    for _ in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.view([BATCH_SIZE, -1, N_FEATURES]).to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            loss = train_step(x_batch, y_batch)


def test(net, testloader, X_test, scaler):
    """Validate the network on the entire test set."""
    loss = 0
    criteron = torch.nn.MSELoss(reduction="mean")

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
            loss += criteron(yhat, y_test)

    df_result = format_predictions(predictions, values, X_test, scaler)
    
    rmse = mean_squared_error(df_result.value, df_result.prediction, squared=False)
    return loss, rmse

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


trainloaders, testloaders, nums_examples, nums_features, X_tests, scalers = load_data(batch_size=BATCH_SIZE)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class Client(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, testloader, num_examples, num_features, X_test, scaler):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.num_features = num_features
        self.X_test = X_test
        self.scaler = scaler
    
    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=EPOCH)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, rmse = test(self.net, self.testloader, self.X_test, self.scaler)
        print("loss ", loss)
        print("rmse ", rmse)
        return float(loss), self.num_examples["testset"], {"rmse": float(rmse)}

def client_fn(cid) -> Client:
    net = get_model(MODEL, MODEL_PARAMS).to(DEVICE)
    trainloader = trainloaders[int(cid)]
    testloader = testloaders[int(cid)]
    num_examples = nums_examples[int(cid)]
    num_features = nums_features[int(cid)]
    X_test = X_tests[int(cid)]
    scaler = scalers[int(cid)]

    return Client(cid, net, trainloader, testloader, num_examples, num_features, X_test, scaler)

client_resources = None
if DEVICE.type == "cuda":
  client_resources = {"num_gpus": 1}

# FedAVG/FedProx algorithm
class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"./flower/savedmodels/round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation rmse using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh rmse of each client by number of examples used
        rmses = [r.metrics["rmse"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_rmse = sum(rmses) / sum(examples)
        print(f"Round {server_round} rmse aggregated from client results: {aggregated_rmse}")

        # Return aggregated loss and metrics (i.e., aggregated rmse)
        return aggregated_loss, {"rmse": aggregated_rmse}

# Create strategy
if __name__ == "__main__":
    # strategy = CustomStrategy(proximal_mu=1)
    strategy = CustomStrategy()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=ROUND),
        client_resources=client_resources,
        strategy = strategy,
        ray_init_args = {"include_dashboard": False}
    )