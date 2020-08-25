from typing import List, Tuple 
from collections import namedtuple
import numpy as np
import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence


def extract_features(input_df: pd.DataFrame) -> torch.Tensor:
    
    with np.errstate(all='ignore'):
        case_stats = np.log10(1.1+input_df[["num_cases", 
                                            "num_days", 
                                            "new_cases"]]).fillna(-1)
        
    weekend = (((input_df["date"].dt.dayofweek == 5) | 
                (input_df["date"].dt.dayofweek == 6)).
               astype(int).values[:,np.newaxis])
    
    
    geo = input_df[["Lat", "Long_"]].values


    features = np.hstack((case_stats, geo, weekend))
    
    return torch.tensor(features, dtype = torch.float32)



def forecast_set(input_df, forecast_horizon = 1, 
                 feature_extractor = extract_featuers) -> Tuple[torch.Tensor, 
                                                                torch.Tensor]:
    features = feature_extractor(input_df)
    X = features[:-forecast_horizon,:]
    y = torch.tensor(np.log10(1+ input_df[["num_cases"]].values)[forecast_horizon:,:],
                     dtype = torch.float32)
    
    X.rename_("day", "features")
    y.rename_("day", None)   
    
    return X, y
      


def windowed_forecast_set(input_df, window_len = 10, 
                          forecast_horizon = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    X, y = forecast_set(input_df, forecast_horizon)
    nrows, nfeats = X.shape
    N = nrows//window_len
    # reshaping named tensors is not supported so we drop names    
    X, y = X.rename(None), y.rename(None)

    X_windowed = X[-N*window_len:,:].reshape(-1, nfeats, window_len)
    y_windowed = y[-N*window_len:,:].reshape(-1, y.shape[1], window_len)
    
    X_windowed.rename_("window", "features", "day")
    y_windowed.rename_("window", "features", "day")  
    
    return X_windowed, y_windowed
    


def gen_forecasting_collate(f):
    ''''''
    def forecasting_collator(batch, padding_value = -1) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        geo_ids = [item[0] for item in batch]
        features = []
        targets = []
        for item in batch:
            X, y = f(item[1])
            features.append(X.rename(None))
            targets.append(y.rename(None))

        features_padded = (pad_sequence(features, batch_first = True).
                           rename("batch", *X.names))
        targets_padded = (pad_sequence(targets, batch_first = True).
                          rename("batch", *y.names))


        Batch = namedtuple('Batch', ["id_", "features", "targets"])
        batch = Batch(id_ = geo_ids, 
                      features = features_padded,
                      targets = targets_padded)

        return batch
    
    
    return feature_collator

windowed_features_collate = gen_feature_collate(windowed_forecast_set)
features_collate = gen_feature_collate(forecast_set, 
                                       padding_value = -99)