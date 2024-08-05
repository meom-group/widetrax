#Imports

import os
import xarray as xr
import numpy as np
import pyinterp
import pyinterp.fill as fill
import scipy.signal as signal

import re
from datetime import datetime

import zarr

from collections import defaultdict

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

from matplotlib.colors import ListedColormap

from datetime import datetime, timedelta





# =============================================================================
# extract_xarray_in_region
# =============================================================================


def extract_xarray_in_region(directory, area):
    
    """
    Extracts xarray datasets from SWOT NetCDF data for a specific region
    

    Parameters
    ----------
    directory : str
        Path to the directory containing the NetCDF files
    area : list
        List with the boundaries of the region of interest [longitude_min, latitude_min, longitude_max, latitude_max]
    

    Returns
    -------
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
            if lon_min < lon_max :
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
# count_observations
# =============================================================================


def count_observations(datasets, area, resolution):
    """
    Calculates the number of available observations per bin in the region of interest.
    
    
    Parameters
    ----------
    datasets : Dict
        Dictionary containing xarray.Datasets
    area : list
        List with the boundaries of the region of interest [longitude_min, latitude_min, longitude_max, latitude_max]
    resolution : float
        Grid resolution

    Returns
    -------
    obs_count : np.ndarray
        Array containing the number of observations per pixel
    
    """
    lon_min,lat_min,lon_max,lat_max = area
    

    #Define the grid
    if lon_min > lon_max : # if for example lon_min est 2W=358 et lon max 5E=5 
        lon_grid = np.concatenate([np.arange(lon_min, 360, resolution),
                                  np.arange(0, lon_max, resolution)])
    else :
        lon_grid = np.arange(lon_min, lon_max, resolution)
    
    #for latitudes    
    lat_grid = np.arange(lat_min, lat_max, resolution)

    # Create an array filled with zeros
    obs_count = np.full((len(lat_grid), len(lon_grid)), 0)

    # Iterating on xarrays in datasets dictionnary
    for key in range(len(datasets)):
        print(f"{key}/{len(datasets)}")
    
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

                    if lon_index < 0 :
                        lon_index = int((ln - (lon_min - 360)) / resolution)

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

    Parameters
    ----------
    datasets : Dict
        Dictionary containing xarray.Datasets
    

    Returns
    -------
    has_converged : bool
        Indicates whether the method has converged, returns True if the method has converged, otherwise returns False
    filled_dataset : Dict
        Dictionary containing xarray.Datasets (with missing values filled in)

    """

    filled_datasets = {}
    new_key = 0
    for key in range(len(datasets)):
        
        data = datasets[key]
        
        latitudes = data['latitude'].values
        longitudes = data['longitude'].values
        ssha_values = data['ssha'].values
        
        if longitudes.size > 0 and latitudes.size > 0:
        
            # Replace all values in columns 33 and 34 with NaN (the two middle columns)
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
            filled_datasets[new_key] = filled_data
            new_key = new_key + 1
        else : 
            print(f"Size of longitudes/latitudes is zero for dict number {key}")
        
    return has_converged, filled_datasets


# =============================================================================
# retrieve_segments
# =============================================================================

    
def retrieve_segments(datasets,FileType):
    """
    Extracts segments from xarray.datasets
    
    Parameters
    ----------
    datasets : Dict
        Dictionary containing xarray.Datasets
    FileType : str
        The file type used to calculate datasets, "NetCDF" or "Zarr"

    Returns
    -------
    segments_dict : Dict
        Dictionary containing segments (numpy.arrays).
    

    """
    segments_dict = {}
    counter = 0
    
    #Calculation of Sea Surface Height (SSH) : SSH = SSHA + MDT
    for ky in range(len(datasets)):
        datasets[ky]['ssh'] = datasets[ky]['ssha'] + datasets[ky]['mdt']
    
    for key, dataset in datasets.items():
        print(f"starting processing dict number {key} among {len(datasets)}")
        for col in range(dataset.dims['num_pixels']):
            # Extract data for one column
            col_data = dataset.isel(num_pixels=col)
            
            # Delete coords to avoid duplicates and variables we don't need
            if FileType == "NetCDF":
                col_dataset = col_data.drop_vars(['latitude', 'longitude','ssha', 'mdt'])
                
            elif FileType == "Zarr":
                col_dataset = col_data.drop_vars(['latitude', 'longitude','ssha', 'mdt',
                                                  'cycle_number','duacs_land_sea_mask','pass_number'])
            else:
                print("The specified format is not supported")
            
            
            # Check whether the column is entirely NaN
            if not np.all(np.isnan(col_dataset['ssh'])):
                
                segment_data = col_dataset.to_array().values.squeeze()
                
                # Remove NaN values at the beginning
                while len(segment_data) > 0 and np.isnan(segment_data[0]):
                    segment_data = segment_data[1:]

                # Remove NaN values at the end
                while len(segment_data) > 0 and np.isnan(segment_data[-1]):
                    segment_data = segment_data[:-1]
                
                #pour vérifier s'il reste pas des nan correspondants aux iles ou continent apres l'interpolation
                if not np.isnan(segment_data).any():
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
# plot_obs_count
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
    
    if lon_min > 180:
        lon_min = lon_min - 360
        
    if lon_max > 180:
        lon_max = lon_max - 360
        
        
    ax.add_feature(cfeature.LAND, facecolor='gray')
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Create a mask for zero values to present the continent in its color
    obs_count_masked = np.ma.masked_where(obs_count == 0, obs_count)
    
    im = ax.imshow(obs_count_masked, extent=[lon_min, lon_max, lat_min, lat_max], 
                   origin='lower', cmap="jet", transform=ccrs.PlateCarree(), 
                   vmin=0, vmax=obs_count.max())
    
    
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


# =============================================================================
# plot_psd
# ============================================================================= 


def plot_psd(ax,freqs, psd1,psd2=None,title=None,psd1_label=None,psd2_label=None):
    """
    
    """
   
    if len(freqs) == 0 or len(psd1) == 0 :
        print("Empty data provided.")
        return
    
    if psd1_label :
        ax.loglog(freqs,psd1,label=psd1_label,color='red')
    else:
        ax.loglog(freqs,psd1,label="psd1",color='red')
        
    if psd2 is not None and len(psd2) > 0: 
        if psd2_label :   
            ax.loglog(freqs,psd2,label=psd2_label,color='green')
        else:
            ax.loglog(freqs,psd2,label="psd2",color='blue')

    ax.set_ylim(1e-7, 1e2)
    ax.set_ylabel('PSD [m²/(cy/km)]', fontsize=8, fontweight='bold', color='black')                

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('Wavenumber [cy/km]', fontsize=8, fontweight='bold', color='black')


    #filters out zero frequencies 
    non_zero_freqs = freqs[freqs != 0]
    wavelengths_km = 1 / non_zero_freqs

    # k^-2 & k^-5
    k_2 = non_zero_freqs**-2 * (psd1[0] / (non_zero_freqs[0]**-2)) # Scale according to the first PSD value
    k_5 = non_zero_freqs**-5 * (psd1[0] / (non_zero_freqs[0]**-5))# 

    ax.loglog(non_zero_freqs, k_2*30, 'k--', label='$k^{-2}$') #30 to fix the line
    ax.loglog(non_zero_freqs, k_5*1e6, 'b--', label='$k^{-5}$') #1e6 to fix  the line

    # 
    ax2 = ax.secondary_xaxis('bottom', functions=(lambda x: 1/x, lambda x: 1/x))
    ax2.set_xlabel('Wave-length [km]', fontsize=8, fontweight='bold', color='black')

    wavelength_ticks = np.array([5,10, 25, 50, 100, 200, 500, 1000])
    ax2.set_xticks(wavelength_ticks)
    ax2.set_xticklabels(wavelength_ticks)
    ax.grid(True, which='both')
    ax.legend()
    
    # Set the title if provided
    if title:
        ax.set_title(title, fontsize=15, fontweight='bold', color='black')


# =============================================================================
# read_zarr_to_xarray_dict
# =============================================================================         
        
        
# Fonction pour lire les fichiers Zarr pour une plage de dates spécifique
def read_zarr_to_xarray_dict(base_directory, area,start_date_str, end_date_str,variables_to_keep):
    
    """
    Entrées :
    - base_directory : str
        Chemin vers le répertoire de base contenant les données Zarr organisées par mois et par jour.
    - start_date_str : str
        Date de début au format 'YYYYMMDD'.
    - end_date_str : str
        Date de fin au format 'YYYYMMDD'.
    - variables_to_keep : list of str
        Liste des variables à conserver dans chaque xarray.Dataset.
        
    Sorties :
    - datasets_dict : dict
        Dictionnaire contenant les xarray.Dataset résultants.   
    """
    
    lon_min = area[0]
    lon_max = area[2]
    lat_min = area[1]
    lat_max = area[3]
    
    #le nombre de fichiers
    nfiles = 0
    
    datasets_dict = {}
    index = 0
    
    # Convertir les chaînes de date en objets datetime
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')

    # Parcourir chaque jour entre les dates de début et de fin
    current_date = start_date
    while current_date <= end_date:
        month = current_date.strftime('%m')
        day = current_date.strftime('%d')
        

        month_directory = os.path.join(base_directory, f'month={month}')
        if os.path.exists(month_directory):
            day_directory = os.path.join(month_directory, f'day={day}')
            if os.path.exists(day_directory):
                
                zarr_ds = xr.open_zarr(day_directory)
                
                # Garder uniquement les variables spécifiées
                zarr_ds = zarr_ds[variables_to_keep]
                
                coord_vars = ['latitude', 'longitude']
                zarr_ds = zarr_ds.set_coords(coord_vars)

                if lon_min > lon_max :
                    
                    selection = (

                    (zarr_ds['latitude'] >= lat_min) &
                    (zarr_ds['latitude'] <= lat_max) &
                    (((zarr_ds['longitude'] >= lon_min) & (zarr_ds['longitude'] <= 360)) | (zarr_ds['longitude'] <= lon_max))
                            )
                else:
                    selection = (

                    (zarr_ds['latitude'] >= lat_min) &
                    (zarr_ds['latitude'] <= lat_max) &
                    (zarr_ds['longitude'] >= lon_min) & 
                    (zarr_ds['longitude'] <= lon_max)
                            )
            
                selection = selection.compute()
            
                zarr_ds = zarr_ds.where(selection, drop=True)  
                
                #créer la variable ssha
                zarr_ds['ssha'] =  zarr_ds.duacs_ssha_karin_2_calibrated.where(zarr_ds.duacs_editing_flag == 0)
                zarr_ds = zarr_ds.drop_vars('duacs_ssha_karin_2_calibrated')
                zarr_ds = zarr_ds.drop_vars('duacs_editing_flag')
                
                #créer la variable mdt
                zarr_ds['mdt'] =  zarr_ds.cvl_mean_dynamic_topography_cnes_cls_22
                zarr_ds = zarr_ds.drop_vars('cvl_mean_dynamic_topography_cnes_cls_22')
                
                datasets_dict[index] = zarr_ds
                
                index +=1
                nfiles +=1

            else:
                print(f'The directory for day{day} does not exist in month {month}')
        else:
            print(f'The directory for month{month} does not exist')
        
        # Passer au jour suivant
        current_date += timedelta(days=1)
        
    return datasets_dict     



# =============================================================================
# split_dsets_based_cnum
# =============================================================================  



def split_dsets_based_cnum(datasets_dict):
    
    splited_dict = {}
    index = 0
    
    for key in range(len(datasets_dict)):
        # Check if the condition is met (you can define your own condition here)
        if len(np.unique(datasets_dict[key].cycle_number.compute())) >= 2 :
            #recuperer le num de cycle
            long_cycle = len(np.unique(datasets_dict[key].cycle_number.compute()))
            #print(f"la longuer de cycle number est {long_cycle}")
            
            for i in np.arange(long_cycle-1) :
                cycle = np.unique(datasets_dict[key].cycle_number.compute())[i]
                ds_cycle = datasets_dict[key].where(datasets_dict[key]['cycle_number'].compute() == cycle.astype(float), drop=True)
               
                #condition séparation sur les pass number
                if len(np.unique(ds_cycle.pass_number.compute())) >= 2:
                    
                    long_pass = len(np.unique(ds_cycle.pass_number.compute()))
                    #print(f"la longuer de pass number est {long_pass}")
                    
                    for i in np.arange(long_pass-1) :
                        passs = np.unique(ds_cycle.pass_number.compute())[i]
                        ds_cyclepass = ds_cycle.where(ds_cycle.pass_number.compute() == passs.astype(float), drop=True)
    
                
                        # Add the new datasets to the output dictionary with sequential keys
                        splited_dict[index] = ds_cyclepass
                        index += 1
        else:
            # If the dataset doesn't meet the condition, add it to the output dictionary as is
            splited_dict[index] = datasets_dict[key]
            index += 1
    
    return splited_dict






