import numpy as np
import pandas as pd
import xarray as xr
import dask
import os, sys
import glob
import zarr
from joblib import Parallel, delayed
import os
import dask.array as da
import matplotlib.pyplot as plt
import dask.distributed as dd
import time
import gc

sys.path.append('/')
from libraries import *

dates = pd.date_range(start='2011-01-01T00', end='2020-12-31T23', freq='h')
wind_speed_zarr_store = '/media/harish/External_3/CERRA_wind_profiles_and_Chebyshev_coefficients/CERRA_height_level_winds.zarr'
Cheybshev_zarr_store = '/data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/CERRA_Chebyshev_coefficients.zarr'

def write_chunk(ds_chunk, zarr_store, region):
    """
    Function to write a single chunk to the Zarr store.
    """
    ds_chunk.to_zarr(zarr_store, region=region, mode="r+")

def write_to_zarr_parallel(ds, zarr_store, n_jobs=os.cpu_count()):
    """
    Writes the dataset to the Zarr store in parallel using joblib.
    """
    # Determine the time and height indices
    time_indices_monthly = np.searchsorted(dates.values, ds.time.values)
    start,end = time_indices_monthly[0],time_indices_monthly[-1]
    batch_size = 24

    # List to store all tasks
    tasks = []

    # Iterate over time indices in batches
    for t_idx in range(start, end + 1, batch_size):
        # Calculate the batch range (start to end within bounds)
        batch_end = min(t_idx + batch_size, end + 1)

        # Define the region for this batch
        region = {
            "time": slice(t_idx, batch_end),
        }

        # Select the batch of data
        ds_chunk = (
            ds.sel(time=dates[t_idx:batch_end])
            .to_dataset(name="Chebyshev_coefficients")
        ).drop(['time'])
        # Add the task to the task list
        tasks.append(delayed(write_chunk)(ds_chunk, zarr_store, region))

    # Run all tasks in parallel
    with Parallel(n_jobs=n_jobs, verbose=10) as parallel:
        parallel(tasks)

if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    # == starting dask client ==
    print("Starting parallel computing...")
    cluster = dd.LocalCluster(n_workers=64,threads_per_worker=8,memory_limit='8GB',dashboard_address='8787')
    # Connect to the cluster
    client = dd.Client(cluster)
    print(client)

    s_time = time.time()
    try:
        print(f"Processing {year}-{month:02d}...")
        ds = xr.open_zarr(wind_speed_zarr_store).wind_speed.sel(time=(f'{year}-{month:02d}'))
        print(ds.time.values[0],ds.time.values[-1])
        # Compute the Chebyshev coefficients
        ds_chebyshev = chebyshev_vec(ds, dim="heightAboveGround").load()
        # save the Chebyshev coefficients to the Zarr store
        write_to_zarr_parallel(ds_chebyshev, Cheybshev_zarr_store, n_jobs=128)
        print(f"Successfully processed {year}-{month:02d}.")
        del ds  # Explicitly delete dataset
    
    except Exception as e:
        print(f"Unexpected error in reading {year}-{month:02d}: {e}")

    gc.collect()  # Force garbage collection
    e_time = time.time()
    print('Time taken:',e_time-s_time)

    # == closing dask client ==
    client.close()
    cluster.close()