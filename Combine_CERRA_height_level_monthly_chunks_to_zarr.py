import numpy as np
import pandas as pd
import xarray as xr
import dask
import os, sys
import glob
import zarr
from joblib import Parallel, delayed
import os
import dask.distributed as dd

sys.path.append('/')
from libraries import *

dates = pd.date_range(start='2011-01-01T00', end='2020-12-31T23', freq='h')
zarr_store = '/media/harish/External_3/CERRA_wind_profiles_and_Chebyshev_coefficients/CERRA_height_level_winds.zarr'

def preprocess(ds):
    ds['time'] = ds['valid_time']
    ds = ds.drop(['valid_time', 'step', 'latitude','longitude'])
    return ds
def preprocess_2(ds):
    '''
    This script process the remaining height level forecast data
    '''
    ds = ds.rename({'valid_time':'time'})
    ds = ds.drop(['expver','latitude','longitude'])
    return ds

def open_file(file_path,var, preprocess_fn=None):
    chunks = {'x': 256, 'y': 256}
    try:
        ds = xr.open_dataset(file_path, chunks=chunks)[f'{var}']
        if preprocess_fn:
            ds = preprocess_fn(ds)
        return ds
    except Exception as e:
        print(f"Error opening file: {file_path}: {e}")
        return None
    
def read_monthly_data(year, month):

    # File paths
    file_paths = {
        "ds_10m": f"/data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/temp/CERRA_ws10/{year}/CERRA_{year}_{month}.nc",
        "ds_10m_1": f"/data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/temp/CERRA_ws10_step1/{year}/CERRA_{year}_{month}.nc",
        "ds_10m_2": f"/data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/temp/CERRA_ws10_step2/{year}/CERRA_{year}_{month}.nc",
        "ds_100m": f"/data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/temp/CERRA_ws100/{year}/CERRA_gridded_100_m_wind_{year}_{month}.nc",
        "ds_150m": f"/data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/temp/CERRA_ws150/{year}/CERRA_gridded_150_m_wind_{year}_{month}.nc",
        "ds_100_150m_1": f"/data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/temp/CERRA_ws_100_150_step1/{year}/CERRA_gridded_wind_{year}_{month}_1.nc",
        "ds_100_150m_2": f"/data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/temp/CERRA_ws_100_150_step2/{year}/CERRA_gridded_wind_{year}_{month}_2.nc",
        "ds_remaining_height": f"/data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/temp/CERRA_ws_15_30_50_75_200_250_300_400_500/{year}/CERRA_gridded_15_30_50_75_200_250_300_400_500_wind_{year}_{month}.nc",
        "ds_remaining_height_1": f"/data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/temp/CERRA_ws_15_30_50_75_200_250_300_400_500_step1/{year}/CERRA_gridded_15_30_50_75_200_250_300_400_500_wind_{year}_{month}_1.nc",
        "ds_remaining_height_2": f"/data/harish/CERRA_wind_profiles_and_Chebyshev_coefficients/temp/CERRA_ws_15_30_50_75_200_250_300_400_500_step2/{year}/CERRA_gridded_15_30_50_75_200_250_300_400_500_wind_{year}_{month}_2.nc",
    }

    # Open datasets
    datasets = {
        "ds_10m": open_file(file_paths["ds_10m"],'si10', preprocess),
        "ds_10m_1": open_file(file_paths["ds_10m_1"],'si10', preprocess),
        "ds_10m_2": open_file(file_paths["ds_10m_2"],'si10', preprocess),
        "ds_100m": open_file(file_paths["ds_100m"],'ws', preprocess),
        "ds_150m": open_file(file_paths["ds_150m"],'ws', preprocess),
        "ds_100_150m_1": open_file(file_paths["ds_100_150m_1"],'ws', preprocess),
        "ds_100_150m_2": open_file(file_paths["ds_100_150m_2"],'ws', preprocess),
        "ds_remaining_height": open_file(file_paths["ds_remaining_height"],'ws', preprocess_2) if (year == 2015 and month == 12) else open_file(file_paths["ds_remaining_height"],'ws', preprocess),
        "ds_remaining_height_1": open_file(file_paths["ds_remaining_height_1"],'ws', preprocess_2),
        "ds_remaining_height_2": open_file(file_paths["ds_remaining_height_2"],'ws', preprocess_2),
    }
    
    # Identify problematic files
    problematic_files = [key for key, ds in datasets.items() if ds is None]
    if problematic_files:
        print(f"Skipping {year}-{month} due to the following file issues:")
        for key in problematic_files:
            print(f"  - {file_paths[key]}")
        return None

    # Concatenate datasets
    ws_10m = xr.concat([datasets["ds_10m"], datasets["ds_10m_1"], datasets["ds_10m_2"]], dim='time')
    ws_100_150m = xr.concat([datasets["ds_100m"], datasets["ds_150m"]],dim='heightAboveGround')
    ws_100_150m = xr.concat([ws_100_150m,datasets["ds_100_150m_1"], datasets["ds_100_150m_2"]], dim='time')
    ws_remaining_height = xr.concat([datasets["ds_remaining_height"], datasets["ds_remaining_height_1"], datasets["ds_remaining_height_2"]], dim='time')

    ws_monthly = xr.concat([ws_10m, ws_100_150m, ws_remaining_height], dim='heightAboveGround')
    ws_monthly = ws_monthly.sortby('heightAboveGround')
    ws_monthly = ws_monthly.sortby('time')
    ws_monthly = ws_monthly.chunk({'time': 24, 'heightAboveGround': -1})
    return ws_monthly

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
            .to_dataset(name="wind_speed")
        ).drop(['time','heightAboveGround'])

        # Add the task to the task list
        tasks.append(delayed(write_chunk)(ds_chunk, zarr_store, region))

    # Run all tasks in parallel
    with Parallel(n_jobs=n_jobs, verbose=10) as parallel:
        parallel(tasks)

import psutil
import dask.distributed as dd
def auto_configure_dask():
    total_memory = psutil.virtual_memory().total // (1024**3)  # Convert bytes to GB
    total_cores = os.cpu_count()

    # Configure based on workload type
    workload_type = "CPU"  # Change to "IO" for I/O-bound workloads

    if workload_type == "CPU":
        n_workers = total_cores
        threads_per_worker = 1
    elif workload_type == "IO":
        n_workers = total_cores // 2
        threads_per_worker = 2

    memory_limit = f"{total_memory // n_workers}GB"

    return dd.LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker,
                           memory_limit=memory_limit, dashboard_address='8787')


if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    '''
    # == starting dask client ==
    print("Starting parallel computing...")

    cluster = dd.LocalCluster(n_workers=12,threads_per_worker=1,memory_limit='12GB',dashboard_address='8787')
    # Connect to the cluster
    client = dd.Client(cluster)
    print(client)
    '''
    '''
    #cluster = auto_configure_dask()
    #print(cluster)
    '''
    s_time = time.time()
    try:
        print(f"Processing {year}-{month:02d}...")
        ds = read_monthly_data(year, month)
        if ds is None:
            print(f"Skipping {year}-{month:02d} due to missing or corrupted files.")
        else:
            # Uncomment and replace with your actual Zarr writing function
            ds = ds.load()
            write_to_zarr_parallel(ds, zarr_store)
            print(f"Successfully processed {year}-{month:02d}.")
        
        ds.close()
        del ds  # Explicitly delete dataset
    except Exception as e:
        print(f"Unexpected error in reading {year}-{month:02d}: {e}")
    import gc
    gc.collect()  # Force garbage collection
    e_time = time.time()
    print('Time taken:',e_time-s_time)

    # == closing dask client ==
    #client.close()
    #cluster.close()
