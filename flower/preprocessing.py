from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.utils import shuffle
from myconstants import *

def preprocessing(filepath: str):
        
    # Loading in data
    df = pd.read_csv(filepath, index_col = [1])
    df.index = pd.to_datetime(df.index)
    df.drop(df.filter(regex="Unname"),axis=1, inplace=True)

    def generate_time_lags(df, n_lags):
        df_n = df.copy()
        for n in range(1, n_lags + 1):
            df_n[f"lag{n}"] = df_n["Transfused"].shift(n)
        df_n = df_n.iloc[n_lags:]
        return df_n
        
    input_dim = N_FEATURES - 4

    df_generated = generate_time_lags(df, input_dim)

    df_features = ( df_generated
                    .assign(day = df_generated.index.day)
                    .assign(month = df_generated.index.month)
                    .assign(day_of_week = df_generated.index.dayofweek)
                    .assign(week_of_year = pd.Index(df_generated.index.isocalendar().week)))

    # def onehot_encode_pd(df, cols):
    #     for col in cols:
    #         dummies = pd.get_dummies(df[col], prefix=col)
        
    #     return pd.concat([df, dummies], axis=1)

    # df_features = onehot_encode_pd(df_features, ["month","day","day_of_week","week_of_year"])

    # def generate_cyclical_features(df, col_name, period, start_num=0):
    #     kwargs = {
    #         f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
    #         f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
    #             }
    #     return df.assign(**kwargs).drop(columns=[col_name])

    # df_features = generate_cyclical_features(df_features, 'day_of_week', 7, 0)

    df_features.drop(columns = ['Location'], inplace=True)

    # df_features = df_features.sample(frac = 1)

    # Train test split

    def feature_label_split(df, target_col):
        y = df[[target_col]]
        X = df.drop(columns=[target_col])
        return X, y

    def train_val_test_split(df, target_col, test_ratio):
        X, y = feature_label_split(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
        X_train, y_train = shuffle(X_train, y_train)
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_val_test_split(df_features, 'Transfused', 0.2)

    def get_scaler(scaler):
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()

    scaler = get_scaler('minmax')
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train)
    y_test_arr = scaler.transform(y_test)
    
    return X_train_arr, X_test_arr, y_train_arr, y_test_arr, X_test, scaler

def preprocessing_centralized(filepaths):
            
    # Loading in data
    df = pd.read_csv(filepaths[0], index_col = [1])
    for i in range(1, len(filepaths)):
        df2 = pd.read_csv(filepaths[i])
        df.append(df2, ignore_index = True)
    df.index = pd.to_datetime(df.index)
    df.drop(df.filter(regex="Unname"),axis=1, inplace=True)

    def generate_time_lags(df, n_lags):
        df_n = df.copy()
        for n in range(1, n_lags + 1):
            df_n[f"lag{n}"] = df_n["Transfused"].shift(n)
        df_n = df_n.iloc[n_lags:]
        return df_n
        
    input_dim = N_FEATURES - 4

    df_generated = generate_time_lags(df, input_dim)

    df_features = ( df_generated
                    .assign(day = df_generated.index.day)
                    .assign(month = df_generated.index.month)
                    .assign(day_of_week = df_generated.index.dayofweek)
                    .assign(week_of_year = pd.Index(df_generated.index.isocalendar().week)))

    # def onehot_encode_pd(df, cols):
    #     for col in cols:
    #         dummies = pd.get_dummies(df[col], prefix=col)
        
    #     return pd.concat([df, dummies], axis=1)

    # df_features = onehot_encode_pd(df_features, ["month","day","day_of_week","week_of_year"])

    # def generate_cyclical_features(df, col_name, period, start_num=0):
    #     kwargs = {
    #         f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
    #         f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
    #             }
    #     return df.assign(**kwargs).drop(columns=[col_name])

    # df_features = generate_cyclical_features(df_features, 'day_of_week', 7, 0)

    df_features.drop(columns = ['Location'], inplace=True)

    # Train test split

    def feature_label_split(df, target_col):
        y = df[[target_col]]
        X = df.drop(columns=[target_col])
        return X, y

    def train_val_test_split(df, target_col, test_ratio):
        X, y = feature_label_split(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
        X_train, y_train = shuffle(X_train, y_train)
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_val_test_split(df_features, 'Transfused', 0.2)

    def get_scaler(scaler):
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()

    scaler = get_scaler('minmax')
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train)
    y_test_arr = scaler.transform(y_test)
    
    return X_train_arr, X_test_arr, y_train_arr, y_test_arr, X_test, scaler
