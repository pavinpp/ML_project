

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob
import gcsfs
from tqdm.notebook import tqdm


def make_dir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
        
def open_dataset(file_path):
    """Flexible opener that can handle both local files (legacy) and cloud urls. IMPORTANT: For this to work the `file_path` must be provided without extension."""
    if 'gs://' in file_path:
        # For GCS zarr stores use gcsfs to create a mapper
        store = f"{file_path}.zarr"
        try:
            fs = gcsfs.GCSFileSystem()
            mapper = gcsfs.GCSMap(store, gcs=fs)
            ds = xr.open_zarr(mapper, consolidated=True)
        except Exception:
            # Fallback to xarray engine open_dataset where appropriate
            ds = xr.open_dataset(store, engine='zarr')
    else:
        path = f"{file_path}.nc"
        ds = xr.open_dataset(path)
        # record the file name in attributes for traceability
        try:
            ds.attrs['file_name'] = os.path.basename(path)
        except Exception:
            pass

    return ds
        
        
        
def prepare_predictor(data_sets, data_path,time_reindex=True):
    """
    Args:
        data_sets list(str): names of datasets
    """
        
    # Create training and testing arrays
    if isinstance(data_sets, str):
        data_sets = [data_sets]
        
    X_all      = []
    length_all = []
    
    for file in tqdm(data_sets):
        data = open_dataset(f"{data_path}inputs_{file}")
        X_all.append(data)
        length_all.append(len(data.time))
    
    X = xr.concat(X_all,dim='time')
    length_all = np.array(length_all)
    # X = xr.concat([xr.open_dataset(data_path + f"inputs_{file}.nc") for file in data_sets], dim='time')
    if time_reindex:
        X = X.assign_coords(time=np.arange(len(X.time)))

    return X, length_all

def prepare_predictand(data_sets,data_path,time_reindex=True):
    if isinstance(data_sets, str):
        data_sets = [data_sets]
        
    Y_all = []
    length_all = []
    
    for file in tqdm(data_sets):
        data = open_dataset(f"{data_path}outputs_{file}")
        Y_all.append(data)
        length_all.append(len(data.time))
    
    length_all = np.array(length_all)
    Y = xr.concat(Y_all, dim='time')

    # If there's a 'member' dimension (ensemble), collapse it
    if 'member' in Y.dims:
        Y = Y.mean('member')

    # Safely rename coordinates if present
    rename_map = {}
    if 'lon' in Y.dims or 'lon' in Y.coords:
        rename_map['lon'] = 'longitude'
    if 'lat' in Y.dims or 'lat' in Y.coords:
        rename_map['lat'] = 'latitude'
    if rename_map:
        try:
            Y = Y.rename(rename_map)
        except Exception:
            pass

    # Transpose only if the target dims exist
    target_dims = ['time', 'latitude', 'longitude']
    if all(d in Y.dims for d in target_dims):
        Y = Y.transpose(*target_dims)

    # Drop 'quantile' if it's present (coordinate or variable)
    try:
        if 'quantile' in Y.coords or 'quantile' in Y.data_vars or 'quantile' in Y.dims:
            # prefer drop_vars for variables/coords; wrap in try for safety
            try:
                Y = Y.drop_vars('quantile')
            except Exception:
                Y = Y.drop(labels='quantile', dim='time') if 'time' in Y.dims else Y
    except Exception:
        pass
    if time_reindex:
        Y = Y.assign_coords(time=np.arange(len(Y.time)))
    
    return Y, length_all


def get_rmse(truth, pred):
    weights = np.cos(np.deg2rad(truth.latitude))
    return np.sqrt(((truth-pred)**2).weighted(weights).mean(['latitude', 'longitude'])).data.mean()

def plot_history(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.show()
    
# Utilities for normalizing the input data
def normalize(data, var, meanstd_dict):
    mean = meanstd_dict[var][0]
    std = meanstd_dict[var][1]
    return (data - mean)/std

def mean_std_plot(data,color,label,ax):
    
    mean = data.mean(['latitude','longitude'])
    std  = data.std(['latitude','longitude'])
    yr   = data.time.values

    ax.plot(yr,mean,color=color,label=label,linewidth=4)
    ax.fill_between(yr,mean+std,mean-std,facecolor=color,alpha=0.4)
    
    return yr, mean
