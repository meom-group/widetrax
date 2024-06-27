#Imports

import os
import xarray as xr
import numpy as np
import pyinterp
import pyinterp.fill as fill
import scipy.signal as signal


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

####

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

####

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

####


# to be included in the function retrive_segments
for key in range(len(filled_datasets)):
    filled_datasets[key]['ssh'] = filled_datasets[key]['ssha'] + filled_datasets[key]['mdt']
    
def retrieve_segments(datasets):
    """
    Input:
    - datasets (dict): Dictionary containing xarray.Datasets.

    Output:
    - dict: New dictionary containing numpy.arrays.
    """
    segments_dict = {}
    counter = 0

    for key, dataset in datasets.items():
        for col in range(dataset.dims['num_pixels']):
            # Extract data for one column
            col_data = dataset.isel(num_pixels=col)
                       
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

####

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