from collections import defaultdict
from datetime import datetime, timedelta
import os
import re
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pyinterp
import pyinterp.fill as fill
import xarray as xr

# =============================================================================
# ma_fonctiontest
# =============================================================================

def ma_fonctiontest():
    """
    Description de la fonction.

    :return: Valeur de retour
    """
    pass

# =============================================================================
# extract_xarray_in_region
# =============================================================================

def extract_xarray_in_region(directory, area):
    """
    Extracts xarray datasets from SWOT NetCDF data for a specific region
    :directory: (str) Path to the directory containing the NetCDF files.
    :area: (list) List with the boundaries of the region of interest [longitude_min, latitude_min, longitude_max, latitude_max]

    :Return:
    :datasets: (Dict) Dictionary containing the xarray.Datasets for the region
         
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
# function_test2
# =============================================================================

def function_test2(donnees, seuil_min, seuil_max, type_filtre="passe-bas"):
    
    """
    Filtre les données en fonction des seuils spécifiés et du type de filtre choisi.

    Cette fonction applique un filtre sur les données fournies, soit en mode passe-bas, soit en mode passe-haut.
    Les données en dehors des seuils seront exclues.

    :param donnees: (list ou ndarray) Liste ou tableau des données à filtrer.
    :param seuil_min: (float) Le seuil minimal pour le filtrage des données.
    :param seuil_max: (float) Le seuil maximal pour le filtrage des données.
    :param type_filtre: (str, optionnel) Le type de filtre à appliquer, peut être "passe-bas" ou "passe-haut". Par défaut, "passe-bas".
    
    :return: (list ou ndarray) Les données filtrées.
    
    :raises ValueError: Si le type de filtre n'est ni "passe-bas" ni "passe-haut".
    :raises TypeError: Si le type des données d'entrée n'est pas une liste ou un tableau numpy.
    
    :example:

    >>> donnees = [0.5, 1.5, 2.0, 3.0, 4.5, 5.0]
    >>> filtre_donnees(donnees, seuil_min=1.0, seuil_max=4.0, type_filtre="passe-bas")
    [0.5, 1.5, 2.0, 3.0]
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


