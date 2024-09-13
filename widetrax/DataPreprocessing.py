import numpy as np

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
