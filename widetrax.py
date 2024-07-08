#Imports

import os
import xarray as xr
import numpy as np
import pyinterp
import pyinterp.fill as fill
import scipy.signal as signal

import re
from datetime import datetime

from collections import defaultdict

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

# =============================================================================
# extract_xarray_in_region
# =============================================================================


def extract_xarray_in_region(directory, area):
    """
    Input:
    directory (str): Path to directory containing NetCDF files.
    area (list): List with boundaries of region of interest [longitude_min, latitude_min, longitude_max, latitude_max].
    

    Output:
    dict: Dictionary containing the xarray.Dataset for the region.
    """
    
    datasets = {} 
    i = 0 
    variables_to_load = ["ssha", "mdt", "latitude", "longitude"]
    files_in_dir = os.listdir(directory)

    for filename in files_in_dir:
        file_path = os.path.join(directory, filename)
        
        
        ds_tmp = xr.open_dataset(file_path, chunks={})
        variables_to_drop = [var for var in ds_tmp.variables if var not in variables_to_load]
        ds_tmp.close()
        del ds_tmp
        
        # Open the file (lazy loading) excluding unnecessary variables
        ds = xr.open_dataset(file_path, chunks={}, drop_variables=variables_to_drop)
        
        if ds:  
            selection = (
                (ds.longitude > area[0]) &
                (ds.longitude < area[2]) &
                (ds.latitude > area[1]) &
                (ds.latitude < area[3])
            )
            
            selection = selection.compute()
            
            ds_area = ds.where(selection, drop=True)  
            ds.close()

            if ds_area['latitude'].size > 0:  
                datasets[i] = ds_area
                i += 1
                
            ds_area.close()

    return datasets


# =============================================================================
# count_observations
# =============================================================================


def count_observations(datasets, area, resolution):
    """
    Input:
    datasets (dict): Dictionary containing xarray.Dataset 
    area (list): List with region of interest limits [longitude_min, latitude_min, longitude_max, latitude_max].
    resolution (float): Grid resolution.

    Output:
    np.ndarray: Array containing the number of observations per pixel.
    """
    lon_min,lat_min,lon_max,lat_max = area
    
    #Define the grid
    lon_grid = np.arange(lon_min, lon_max, resolution)
    lat_grid = np.arange(lat_min, lat_max, resolution)

    # Create an array filled with zeros
    obs_count = np.full((len(lat_grid), len(lon_grid)), 0)

    # Iterating on xarrays in datasets dictionnary
    for key in range(len(datasets)):
        one_dtset = datasets[key].compute()

        #Iterating on rows/columns
        for j in range(one_dtset.ssha.shape[0]):  
            for k in range(one_dtset.ssha.shape[1]):
                valeur = one_dtset.ssha[j, k]
                
                if not np.isnan(valeur):
                    lt = float(one_dtset.latitude[j, k])
                    ln = float(one_dtset.longitude[j, k])
                    
                    # Find the corresponding indexes in the grid
                    lon_index = int((ln - lon_min) / resolution)
                    lat_index = int((lt - lat_min) / resolution)
                    
                    # Increment the corresponding element in obs_count
                    obs_count[lat_index, lon_index] += 1
    
    one_dtset.close()
    del(one_dtset)
    
    return obs_count


# =============================================================================
# fill_nan
# =============================================================================


def fill_nan(datasets):
   
    """
    Fills in missing values (NaN) in each xarray.Dataset using Gauss-Seidel method.

    Inputs:
        datasets (Dict): Dictionary containing xarray.Dataset 

    Outputs:
        has_converged (bool): Indicates whether the method has converged
        filled_dataset (Dict): Dictionary containing xarray.Dataset (with missing values filled in)
    """

    filled_datasets = {}
    
    for key in range(len(datasets)):
        
        data = datasets[key]
        
        latitudes = data['latitude'].values
        longitudes = data['longitude'].values
        ssha_values = data['ssha'].values
        
        # Replace all values in columns 33 and 34 with NaN (the two columns in the middle)
        if ssha_values.shape[1]>=34:
            ssha_values[:, 33:35] = np.nan

        # Identify columns entirely NaN
        nan_columns = np.all(np.isnan(ssha_values), axis=0)

        # fill NaN data
        x_axis = pyinterp.core.Axis(longitudes[0, :], is_circle=True)  
        y_axis = pyinterp.core.Axis(latitudes[:, 0], is_circle=False)  
        grid = pyinterp.Grid2D(y_axis, x_axis, ssha_values)
        has_converged, filled_ssha_values = fill.gauss_seidel(grid, num_threads=16)

        # Restore columns entirely NaN
        filled_ssha_values[:, nan_columns] = np.nan

        # Create a new dataset with filled values and add it to the dictionary
        filled_data = data.copy()
        filled_data['ssha'] = (('num_lines', 'num_pixels'), filled_ssha_values)
        filled_datasets[key] = filled_data
        
    return has_converged, filled_datasets


# =============================================================================
# retrieve_segments
# =============================================================================


# to be included in the function retrive_segments
#for key in range(len(filled_datasets)):
#    filled_datasets[key]['ssh'] = filled_datasets[key]['ssha'] + filled_datasets[key]['mdt']
    
def retrieve_segments(datasets):
    """
    Input:
    - datasets (dict): Dictionary containing xarray.Datasets.

    Output:
    - dict: New dictionary containing numpy.arrays.
    """
    segments_dict = {}
    counter = 0
    
    #Calcul de SSH = ssha + mdt
    for ky in range(len(datasets)):
        datasets[ky]['ssh'] = datasets[ky]['ssha'] + datasets[ky]['mdt']
    
    for key, dataset in datasets.items():
        for col in range(dataset.dims['num_pixels']):
            # Extract data for one column
            col_data = dataset.isel(num_pixels=col)
            
            # Delete coords to avoid duplicates and variables we don't need
            col_dataset = col_data.drop_vars(['latitude', 'longitude','ssha', 'mdt'])
            
            # Check whether the column is entirely NaN
            if not np.all(np.isnan(col_dataset['ssh'])):
                segment_data = col_dataset.to_array().values.squeeze()
                
                # Remove NaN values at the beginning
                while len(segment_data) > 0 and np.isnan(segment_data[0]):
                    segment_data = segment_data[1:]

                # Remove NaN values at the end
                while len(segment_data) > 0 and np.isnan(segment_data[-1]):
                    segment_data = segment_data[:-1]
                
                segments_dict[counter] = segment_data
                counter += 1
                
    return segments_dict


# =============================================================================
# calculate_psd
# =============================================================================


def calculate_psd(segments_dict):
    """

    Input:
    segments_dict (dict): 

    Output:
    Two dictionaries containing PSDs and associated frequencies.
    
    """
    psd_dict = {}
    freqs_dict = {}
    new_key = 0
    fs = 3.25 #sampling frequency = sat speed/distance between each two values(6.5/2)

    for key, segment_data in segments_dict.items():
        if len(segment_data) > 199:  # Check segment length
            freqs, psd = signal.welch(segment_data, fs=fs, nperseg=len(segment_data), noverlap=0)
            psd_dict[new_key] = psd
            freqs_dict[new_key] = freqs
            new_key += 1

    return psd_dict, freqs_dict


# =============================================================================
# check_directory
# =============================================================================


def check_directory(database_path, start_date_str, end_date_str):
    # Regex pattern to match folder names like cyc_001, cyc_002, etc.
    folder_pattern = re.compile(r'cycle_\d{3}')
    # Regex pattern to match the date in the file name SWOT........_YYYYMMDD...
    file_pattern = re.compile(r'SWOT.*_(\d{8})T.*\.nc')
    
    # Convert date strings to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')
    
    # List to store the names of folders that meet the criteria
    matching_folders = []
    
    try:
        # List all items in the database directory
        items = os.listdir(database_path)
        
        for item in items:
            folder_path = os.path.join(database_path, item)
            
            # Check if the item is a directory and matches the pattern
            if os.path.isdir(folder_path) and folder_pattern.match(item):
                netcdf_files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
                for nc_file in netcdf_files:
                    match = file_pattern.search(nc_file)
                    if match:
                        file_date_str = match.group(1)
                        file_date = datetime.strptime(file_date_str, '%Y%m%d')
                        
                        if start_date <= file_date <= end_date:
                            matching_folders.append(item)
                            break  # Stop checking files in this folder once a match is found
        
        return matching_folders
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    
# =============================================================================
# extract_xarrays_by_time
# =============================================================================


def extract_xarrays_by_time(database_path, start_date_str, end_date_str,area):
    
    matching_folders = check_directory(database_path, start_date_str, end_date_str)
    combined_datasets_dict = defaultdict(list)
    current_key = 0
    
    for folder in matching_folders:
        folder_path = os.path.join(database_path, folder)
        result_dict = extract_xarray_in_region(folder_path,area)
    
        # Add entries to the combined_dict with sequential keys
        for value in result_dict.values():
            combined_datasets_dict[current_key] = value
            current_key += 1
            
            
    # Convert defaultdict back to dict
    combined_datasets_dict = dict(combined_datasets_dict)
    
    return combined_datasets_dict


# =============================================================================
# extract_xarrays_by_time
# =============================================================================


def plot_obs_count(ax, obs_count, area,title=None):
    """
    Plots the observation count per bin.

    Parameters:
    - ax: Matplotlib Axes instance
    - obs_count: 2D array of observation counts
    - area: list
    """
    lon_min,lat_min,lon_max,lat_max = area
    
    #if lon_min > 180:
    #    lon_min = lon_min - 360
        
    #if lon_max > 180:
    #    lon_max = lon_max - 360
        
        
    ax.add_feature(cfeature.LAND, facecolor='gray')
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Create a mask for zero values to present the continent in its color
    obs_count_masked = np.ma.masked_where(obs_count == 0, obs_count)
    
    im = ax.imshow(obs_count_masked, extent=[lon_min, lon_max, lat_min, lat_max], 
                   origin='lower', cmap="jet", transform=ccrs.PlateCarree(), 
                   vmin=0, vmax=500)
    
    
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    
    plt.colorbar(im, ax=ax, label='Number of observations per bin',shrink=0.5)
    
    # Set the title if provided
    if title:
        ax.set_title(title, fontsize=15, fontweight='bold', color='black')

        
# =============================================================================
# psd_mean_and_freq
# =============================================================================        
 
    
def psd_mean_and_freq(psd_dict, freqs_dict):
    max_length = max(len(array) for array in psd_dict.values())

    # Initialize 2 arrays.
    psd_sum = np.zeros(max_length)
    counts = np.zeros(max_length)

    for array in psd_dict.values():
        # Adjust the array to the maximum size by adding NaNs
        padded_array = np.full(max_length, np.nan)
        padded_array[:len(array)] = array
        
        # Calculate the sum of the PSD values, ignoring the NaNs
        psd_sum = np.nansum([psd_sum, padded_array], axis=0)
        
        # Count the non-NaN values
        counts = np.nansum([counts, ~np.isnan(padded_array)], axis=0)

    # Calculate the PSD mean (np.array)
    psd_mean = psd_sum / counts

    # take the frequency of the longest column
    for k, array in freqs_dict.items():
        if len(array) == max_length:
            freqs_mean = array
            break
               
    return psd_mean, freqs_mean        

