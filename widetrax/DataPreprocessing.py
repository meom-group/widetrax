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
# retrieve_segments
# =============================================================================

def retrieve_segmentest(col, dataset, varname):
    col_data = dataset.isel(num_pixels=col)

    if not np.all(np.isnan(col_data[varname])):
        segment_data = col_data[varname].values.squeeze()

        finite_idx = np.flatnonzero(np.isfinite(segment_data))
        segment_data = segment_data[finite_idx.min():finite_idx.max()]

        if not np.isnan(segment_data).any():
            return segment_data

    return None


def retrieve_segmentsest(datasets, varname: str = "ssha"):
    """
    Extracts segments from xarray.datasets
    
    Parameters
    ------------
    datasets: Dict
        Dictionary containing xarray.Datasets
    varname: str, optional
        Variable name for which PSD will be computed.
        Defaults to "ssha"

    Returns
    ---------
    segments_dict: Dict
        Dictionary containing segments (numpy.arrays)
    """
    segments_dict = {}
    counter = 0

    for key, dataset in datasets.items():
        print(f"Processing dataset {key + 1} among {len(datasets)}")

        ds = dataset[[varname]]

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(retrieve_segmentest, col, ds, varname) for col in range(ds.sizes["num_pixels"])
            ]

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    segments_dict[counter] = result
                    counter += 1

    return segments_dict
