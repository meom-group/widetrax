import numpy as np
import xarray as xr
import pyinterp
import pyinterp.fill as fill
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timedelta
import os
import re
import sys

# =============================================================================
# filtre_donnees
# =============================================================================

def filtre_donnees(donnees, seuil_min, seuil_max, type_filtre="passe-bas"):
    """
    Filters the data based on the specified thresholds and the chosen filter type.

    This function applies a filter to the provided data, either in low-pass or high-pass mode. Data outside the thresholds will be excluded
    
    Parameters
    -----------
    param donnees : str
        Path to the directory containing the NetCDF files
    param seuil_min : list
        List with the boundaries of the region of interest [longitude_min, latitude_min, longitude_max, latitude_max]
    param seuil_max : str
        the max seuil
    param type_filtre : int
        just for test
    
    Returns
    --------
    donnees_filtrees : Dict
        Dictionary containing the xarray.Datasets for the region
        
    """
    if type_filtre not in ["passe-bas", "passe-haut"]:
        raise ValueError("Le type de filtre doit être 'passe-bas' ou 'passe-haut'.")
    
    if not isinstance(donnees, (list, np.ndarray)):
        raise TypeError("Les données doivent être une liste ou un tableau numpy.")
    
    if type_filtre == "passe-bas":
        # Garder uniquement les valeurs en dessous du seuil_max
        donnees_filtrees = [x for x in donnees if x <= seuil_max]
    else:  # Passe-haut
        # Garder uniquement les valeurs au-dessus du seuil_min
        donnees_filtrees = [x for x in donnees if x >= seuil_min]
    
    return donnees_filtrees

# =============================================================================
# count_observations
# =============================================================================

def count_observations(datasets, area, resolution):
    """
    Calculates the number of available observations per bin in the region of interest.
    
    
    Parameters
    ------------
    datasets : Dict
        Dictionary containing xarray.Datasets
    area : list
        List with the boundaries of the region of interest [longitude_min, latitude_min, longitude_max, latitude_max]
    resolution : float
        Grid resolution

    Returns
    ---------
    obs_count : np.ndarray
        Array containing the number of observations per pixel
    
    """
    lon_min, lat_min, lon_max, lat_max = area

    # Define the grid
    if lon_min > lon_max:  # if for example lon_min est 2W=358 et lon max 5E=5
        lon_grid = np.concatenate([np.arange(lon_min, 360, resolution),
                                   np.arange(0, lon_max, resolution)])
    else:
        lon_grid = np.arange(lon_min, lon_max, resolution)

    # for latitudes
    lat_grid = np.arange(lat_min, lat_max, resolution)

    # Create an array filled with zeros
    obs_count = np.full((len(lat_grid), len(lon_grid)), 0)

    # Iterating on xarrays in datasets dictionnary
    for key in range(len(datasets)):
        # print(f"{key}/{len(datasets)}")

        one_dtset = datasets[key].compute()

        # Iterating on rows/columns
        for j in range(one_dtset.ssha.shape[0]):
            for k in range(one_dtset.ssha.shape[1]):
                valeur = one_dtset.ssha[j, k]
                if not np.isnan(valeur):
                    lt = float(one_dtset.latitude[j, k])
                    ln = float(one_dtset.longitude[j, k])

                    # Find the corresponding indexes in the grid
                    lon_index = int((ln - lon_min) / resolution)
                    lat_index = int((lt - lat_min) / resolution)

                    if lon_index < 0:
                        lon_index = int((ln - (lon_min - 360)) / resolution)

                    # Increment the corresponding element in obs_count
                    obs_count[lat_index, lon_index] += 1

    one_dtset.close()
    del (one_dtset)

    return obs_count

# =============================================================================
# extract_xarray_in_region
# =============================================================================

def extract_xarray_in_region(directory, area):
    """
    Extracts xarray datasets from SWOT NetCDF data for a specific region
    

    Parameters
    ------------
    directory : str
        Path to the directory containing the NetCDF files
    area : list
        List with the boundaries of the region of interest [longitude_min, latitude_min, longitude_max, latitude_max]
    

    Returns
    ---------
    datasets : Dict
        Dictionary containing the xarray.Datasets for the region
   
    """

    lon_min, lat_min, lon_max, lat_max = area
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
            if lon_min < lon_max:
                selection = (

                        (ds['latitude'] >= lat_min) &
                        (ds['latitude'] <= lat_max) &
                        (ds['longitude'] >= lon_min) &
                        (ds['longitude'] <= lon_max)
                )

            else:
                selection = (

                        (ds['latitude'] >= lat_min) &
                        (ds['latitude'] <= lat_max) &
                        (((ds['longitude'] >= lon_min) & (ds['longitude'] <= 360)) | (ds['longitude'] <= lon_max))

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
# fill_nan
# =============================================================================

def fill_nan(datasets, varname: str = "ssha"):
    """
    Fills in missing values (NaN) in each xarray.Dataset using Gauss-Seidel method.

    Parameters
    ------------
    datasets: Dict
        Dictionary containing xarray.Datasets
    varname: str, optional
        Variable name to fill in missing values.
        Defaults to "ssha"

    Returns
    ---------
    has_converged: bool
        Indicates whether the method has converged, returns True if the method has converged, otherwise returns False
    filled_dataset: Dict
        Dictionary containing xarray.Datasets (with missing values filled in)
    """
    has_converged = True
    filled_datasets = {}
    new_key = 0
    for key in range(len(datasets)):

        # Selects only the variable of interest
        data = datasets[key][[varname]]

        latitudes = data["latitude"].values
        longitudes = data["longitude"].values
        values = data[varname].values

        if longitudes.size > 0 and latitudes.size > 0:

            # Replace all values in columns 33 and 34 with NaN (the two middle columns)
            if values.shape[1] >= 34:
                values[:, 33:35] = np.nan

            # Identify columns entirely NaN
            nan_columns = np.all(np.isnan(values), axis=0)

            # fill NaN data
            x_axis = pyinterp.core.Axis(longitudes[0, :], is_circle=True)
            y_axis = pyinterp.core.Axis(latitudes[:, 0], is_circle=False)
            grid = pyinterp.Grid2D(y_axis, x_axis, values)
            _has_converged, filled_values = fill.gauss_seidel(grid, num_threads=16)
            has_converged &= _has_converged

            # Restore columns entirely NaN
            filled_values[:, nan_columns] = np.nan

            # Add filled values to the output dictionary
            datasets[key][varname] = (("num_lines", "num_pixels"), filled_values)
            filled_datasets[new_key] = datasets[key]
            new_key = new_key + 1
        else:
            print(f"Size of longitudes/latitudes is zero for dict number {key}")

    return has_converged, filled_datasets

# =============================================================================
# check_directory
# =============================================================================

def check_directory(database_path, start_date_str, end_date_str):
    """
    
    Scans the folders in the `database_path` directory, identifies the folders 
    containing NetCDF files whose dates are between `start_date_str` and `end_date_str`, 
    and returns a list of these folder names.

    
    Parameters
    ------------
    database_path : str
        Path to the `database` directory
    start_date_str : str
        Start date in 'YYYYMMDD' format
    end_date_str : str
        End date in 'YYYYMMDD' format
    
    Returns
    ---------
    
    matching_folders : list
        List of folder names containing NetCDF files within the specified date range
        If an error occurs, an error message is printed and an empty list is returned.
        
    """

    # Regex pattern to match folder names like cycle_001, cycle_002, etc.
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

