#Imports

import os
import xarray as xr

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


