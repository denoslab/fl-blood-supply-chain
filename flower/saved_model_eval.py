from collections import OrderedDict
import sys
from typing import List, Optional, Tuple
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from fl_simulation import test, load_data, inverse_transform, format_predictions

import plotly
import plotly.graph_objs as go

from myconstants import *
from ml_models import get_model

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    
    model = get_model(MODEL, MODEL_PARAMS)
    keys = []
    for k in model.state_dict().keys():
        keys.append(k)

    print("Loading data\n")
    trainloaders, testloaders, nums_examples, nums_features, X_tests, scalers = load_data(1)
    print(len(testloaders[0]))
    df_results = []

    print("Loading saved model\n")
    for i in range(len(testloaders)):
        result_per_dataset = []
        for round in range(1, ROUND + 1):
            data = dict(np.load(f'./flower/savedmodels/round-{round}-weights.npz'))
            npz_keys = []
            for k in data.keys():
                npz_keys.append(k)

            list_of_typles = [(keys[i], torch.FloatTensor(data[npz_keys[i]])) for i in range(len(npz_keys))]
            new_dict = OrderedDict(list_of_typles)

            model.load_state_dict(new_dict)

            predictions, values = evaluate(model, testloaders[i], batch_size=1, n_features=nums_features[i])

            # adding mean shifting             
            # mean_difference = np.mean(values) - np.mean(predictions)
            # predictions = [x + mean_difference for x in predictions]

            result  = format_predictions(predictions, values, X_tests[i], scalers[i])
            result_per_dataset.append(result)
        
    
        model.load_state_dict(torch.load('./flower/savedmodels/local' + str(i+1)+'.pt'))
        predictions, values = evaluate(model, testloaders[i], batch_size=1, n_features=nums_features[i])
        
        result  = format_predictions(predictions, values, X_tests[i], scalers[i])
        result_per_dataset.append(result)

        model.load_state_dict(torch.load('./flower/savedmodels/centralized.pt'))
        predictions, values = evaluate(model, testloaders[i], batch_size=1, n_features=nums_features[i])
        mean_difference = np.mean(values) - np.mean(predictions)
        predictions = [x + mean_difference for x in predictions]

        result  = format_predictions(predictions, values, X_tests[i], scalers[i])
        result_per_dataset.append(result)
        df_results.append(result_per_dataset)

    plot_predictions(df_results)
    print("Results plotted to /flower/evaluation/DASHBOARD.html")

def evaluate(model, testloader, batch_size = 1, n_features = 1):
    with torch.no_grad():
        predictions = []
        values = []
        for x_test, y_test in testloader:
            x_test = x_test.view([batch_size, -1, n_features]).to(DEVICE)
            y_test = y_test.to(DEVICE)
            model.eval()
            yhat = model(x_test)
            predictions.append(yhat.cpu().numpy())
            values.append(y_test.cpu().numpy())
    return predictions, values

def plot_predictions(df_results):

    html_graphs=open("./flower/evaluation/DASHBOARD.html",'w')
    html_graphs.write("<html><head></head><body>"+"\n")

    for i in progressbar(range(len(df_results)), "Plotting results and metrics"):
        df_result = df_results[i]
        data = []

        value = go.Scatter(
            x=df_result[0].index,
            y=df_result[0].value,
            mode="lines",
            name="Actual Number of Transfused Units",
            marker=dict(),
            text=df_result[0].index,
            line=dict(color="rgba(0,0,0, 0.3)"),
        )
        data.append(value)
        colors = ['rgb(64,64,64)', 'rgb(153,0,153)', 'rgb(0,0,153)', 'rgb(0,153,153)', 'rgb(0,153,0)', 'rgb(153, 153, 0)', 'rgb(153,0,0)', 'rgb(153,25,0)', 'rgb(153,50,0)' , 'rgb(153,100,0)']
        print(len(df_result[ROUND-1].prediction))

        prediction = go.Scatter(
            x=df_result[ROUND-1].index,
            y=df_result[ROUND-1].prediction,
            mode="lines",
            line=dict(dash= "dot", color=colors[ROUND-1]),
            name='Federated',
            marker=dict(),
            text=df_result[ROUND-1].index,
            opacity=0.8,
        )
        data.append(prediction)
        
        prediction = go.Scatter(
            x=df_result[-2].index,
            y=df_result[-2].prediction,
            mode="lines",
            line={"dash": "dot"},
            name='Local',
            marker=dict(),
            text=df_result[-2].index,
            opacity=0.8,
        )
        data.append(prediction)

        prediction = go.Scatter(
            x=df_result[-1].index,
            y=df_result[-1].prediction,
            mode="lines",
            line={"dash": "dot"},
            name='Centralized',
            marker=dict(),
            text=df_result[-1].index,
            opacity=0.8,
        )
        data.append(prediction)

        layout = dict(
            # title=dict(text = "Predictions vs Actual Values", font=dict(size=25)),
            xaxis=dict(title="Time", ticklen=50, zeroline=False),
            yaxis=dict(title="Number of Transfused Units", ticklen=50, zeroline=False),
            font=dict(
                size=17,
            )
        )

        fig = go.Figure(data = data, layout=layout)
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.7)"),
            width=600,
            height=600,
        )
        fig.write_image("./flower/evaluation/predictions"+ str(i+1) + ".pdf")
        plotly.offline.plot(fig, filename='./flower/evaluation/Chart_'+str(i+1)+'.html',auto_open=False)
        html_graphs.write("  <object data=\""+'Chart_'+str(i+1)+'.html'+"\" width=\"650\" height=\"650\"></object>"+"\n")
        
        eval_metrics = get_evaluation_metrics(df_result)
        metric_data = []
        rounds = [j for j in range(1, ROUND+1)]
        colors = {
            "RMSE": "red",
            "MAE": "blue",
            "MAPE": "magenta", 
            "SMAPE": "goldenrod"
        }
        for k, v in eval_metrics.items():
            metric = go.Scatter(
                x = rounds,
                y = v,
                mode= "lines",
                line=dict(color = colors[k]),
                name = "Federated " + k,
                marker=dict(),
                opacity = 0.8
            )
            metric_data.append(metric)
        for k, v in eval_metrics.items():
            metric = go.Scatter(
                x = rounds,
                y = [v[-2] for _ in range(len(rounds))],
                mode = "lines",
                line = {"dash": "dot", "color": colors[k]},
                name = "Local " + k,
                marker = dict(),
                opacity = 0.5 
            )
            metric_data.append(metric)
        for k, v in eval_metrics.items():
            metric = go.Scatter(
                x = rounds,
                y = [v[-1] for _ in range(len(rounds))],
                mode = "lines",
                line = {"dash": "dash", "color": colors[k]},
                name = "Centralized " + k,
                marker = dict(),
                opacity = 0.5 
            )
            metric_data.append(metric)
        layout = dict(
            # title=dict(text="Evaluation metrics", font=dict(size=25)),
            xaxis = dict(title = "Round"),
            yaxis = dict(title = "Error"),
            font=dict(
                size=17,
            )
        )

        fig = go.Figure(data=metric_data, layout=layout)
        fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.7)"),
            width=600,
            height=600,
        )
        fig.write_image("./flower/evaluation/metric"+ str(i+1) + ".pdf")
        plotly.offline.plot(fig, filename='./flower/evaluation/Chart_metric'+str(i+1)+'.html',auto_open=False)
        html_graphs.write("  <object data=\""+'Chart_metric'+str(i+1)+'.html'+"\" width=\"650\" height=\"650\"></object>"+"\n")
        
    html_graphs.write("</body></html>")

def get_evaluation_metrics(df_result):
    rmse = []
    mae = []
    mape = []
    smape = []
    
    for round in range(len(df_result)):
        # print(df_result[round].value)
        # if 0 in df_result[round]["value"].unique():
        #     print("there is a zero")
        #     df_result[round]["value"] +=1 
        #     df_result[round]["prediction"] +=1
        rmse.append(mean_squared_error(df_result[round].value, df_result[round].prediction,squared=False))
        mae.append(mean_absolute_error(df_result[round].value, df_result[round].prediction))
        mape.append(mean_absolute_percentage_error(df_result[round].value, df_result[round].prediction)*100)
        smape.append(symmetric_mean_absolute_percentage_error(df_result[round].value, df_result[round].prediction))

    return {"RMSE":rmse, "MAE": mae, "MAPE": mape, "SMAPE": smape}
    # return {"rmse": rmse, "mae": mae, "smape": smape}

def symmetric_mean_absolute_percentage_error(actual, predicted) -> float:
  
    if not all([isinstance(actual, np.ndarray), 
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual), np.array(predicted)
  
    return round(
        np.mean(
            np.abs(predicted - actual) / 
            ((np.abs(predicted) + np.abs(actual))/2)
        )*100, 2
    )

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

if __name__ == "__main__":
    main()