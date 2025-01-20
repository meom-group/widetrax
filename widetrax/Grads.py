from widetrax import DataPreprocessing as dp
import jax.numpy as jnp
import jaxparrow as jxr
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
from matplotlib.ticker import MaxNLocator


# =============================================================================
# compute_vorticity_1_ds
# =============================================================================


def compute_vorticity_1_ds(dataset, variable_name="ssha", mask=None):
    """
    Compute the vorticity from a single xarray dataset.

    Parameters
    -----------
    dataset : xarray.Dataset
        Dataset containing geophysical variables like `ssha`, `ssh`, or `ssh_filtered`.
    variable_name : str
        The name of the variable to use for calculation. Default is "ssha".
    mask : np.ndarray, optional 
        Mask array indicating zones to exclude (1 for land, 0 for ocean).
        If None, a zero array is used.


    Returns
    --------
    vorticity_f : np.ndarray
        2D array of vorticity.
    """
    
    if variable_name not in dataset:
        raise ValueError(f"The dataset does not contain the variable '{variable_name}'.")

    # Handle the case where the variable is already "ssh"
    if variable_name != "ssh":
        dataset["ssh"] = dataset[variable_name] + dataset.get("mdt", 0)  # Use "mdt" if it exists, else assume 0
    else:
        dataset["ssh"] = dataset[variable_name]

    lat_t = jnp.array(dataset.latitude.values)
    lon_t = jnp.array(dataset.longitude.values)
    ssh_t = jnp.array(dataset.ssh.values)

    # Compute spatial steps dx and dy
    dx, dy = jxr.tools.geometry.compute_spatial_step(lat_t, lon_t)

    # Compute geostrophic velocities
    u_geos_u, v_geos_v, *_ = jxr.geostrophy(ssh_t, lat_t, lon_t, return_grids=True)

    # Initialize mask if not provided
    if mask is None:
        mask = np.zeros_like(u_geos_u)

    # Compute derivatives
    dv_dx = jxr.tools.operators.derivative(v_geos_v, dx, mask=mask, axis=1, padding="right")  # ∂v/∂x
    du_dy = jxr.tools.operators.derivative(u_geos_u, dy, mask=mask, axis=0, padding="right")  # ∂u/∂y

    # Compute vorticity: ∂v/∂x - ∂u/∂y
    vorticity_f = dv_dx - du_dy

    return vorticity_f


# =============================================================================
# interpolate_to_f_grid
# =============================================================================

def interpolate_to_f_grid(var):
    """
    Interpolates a variable defined on the U or V grid to place it on the F grid (at the corners).

    Parameters
    ----------
    var : ndarray
        Variable defined on the U or V grid.


    Returns
    -------
    var_f : ndarray
        Variable interpolated onto the F grid.
    """
    
    var_f = np.zeros_like(var)

    # Interpolation of Internal Points (Inside the Grid)
    var_f[1:-1, 1:-1] = (var[:-2, :-2] + var[1:-1, :-2] + var[:-2, 1:-1] + var[1:-1, 1:-1]) / 4

    # Filling the Edges
    var_f[0, :] = var[0, :]  # Top Edge
    var_f[-1, :] = var[-1, :]  # Bottom Edge
    var_f[:, 0] = var[:, 0]  # Left Edge
    var_f[:, -1] = var[:, -1]  # Right Edge
    
    return var_f

# =============================================================================
# compute_strain_1_ds
# =============================================================================


def compute_strain_1_ds(dataset, variable_name="ssha", mask=None):
    """
    Compute the strain magnitude from a single xarray dataset.

    Parameters
    -----------
    
    dataset : xarray.Dataset
        Dataset containing geophysical variables like `ssha`, `ssh`, or `ssh_filtered`.
    variable_name : str
        The name of the variable to use for calculation. Default is "ssha".
    mask : np.ndarray, optional
        Mask array indicating zones to exclude (1 for land, 0 for ocean).
        If None, a zero array is used.


    Returns
    --------
    strain_f : np.ndarray
        2D array of strain magnitude.
        
    """
    if variable_name not in dataset:
        raise ValueError(f"The dataset does not contain the variable '{variable_name}'.")

    # Handle the case where the variable is already "ssh"
    if variable_name != "ssh":
        dataset["ssh"] = dataset[variable_name] + dataset.get("mdt", 0)  # Use "mdt" if it exists, else assume 0
    else:
        dataset["ssh"] = dataset[variable_name]

    lat_t = jnp.array(dataset.latitude.values)
    lon_t = jnp.array(dataset.longitude.values)
    ssh_t = jnp.array(dataset.ssh.values)

    # Compute spatial steps dx and dy
    dx, dy = jxr.tools.geometry.compute_spatial_step(lat_t, lon_t)

    # Compute geostrophic velocities
    u_geos_u, v_geos_v, *_ = jxr.geostrophy(ssh_t, lat_t, lon_t, return_grids=True)

    # Initialize mask if not provided
    if mask is None:
        mask = np.zeros_like(u_geos_u)

    # Compute derivatives
    du_dx = jxr.tools.operators.derivative(u_geos_u, dx, mask=mask, axis=1, padding="right")  # ∂u/∂x
    dv_dy = jxr.tools.operators.derivative(v_geos_v, dy, mask=mask, axis=0, padding="right")  # ∂v/∂y
    du_dy = jxr.tools.operators.derivative(u_geos_u, dy, mask=mask, axis=0, padding="right")  # ∂u/∂y
    dv_dx = jxr.tools.operators.derivative(v_geos_v, dx, mask=mask, axis=1, padding="right")  # ∂v/∂x

    # Interpolate derivatives to the F-grid
    du_dx_f = interpolate_to_f_grid(du_dx)
    dv_dy_f = interpolate_to_f_grid(dv_dy)
    du_dy_f = interpolate_to_f_grid(du_dy)
    dv_dx_f = interpolate_to_f_grid(dv_dx)

    # Compute strain magnitude on the F-grid
    strain_f = np.sqrt((du_dx_f - dv_dy_f) ** 2 + (du_dy_f + dv_dx_f) ** 2)

    return strain_f

# =============================================================================
# compute_vorticity
# =============================================================================


def compute_vorticity(datasets_dict, variable_name="ssha"):
    """
    Compute vorticity for multiple xarray datasets.

    Parameters
    -----------
    datasets_dict : dict
        A dictionary of xarray datasets, where each key is a dataset identifier, and each value is an xarray dataset.

    variable_name : str, optional (default: "ssha")
        The name of the variable to compute vorticity for.

    Returns
    --------
    vorticity_2D : dict
        A dictionary containing the 2D vorticity arrays for each dataset.

    """

    vorticity_2D = {}


    for ds_num, dataset in datasets_dict.items():
        if variable_name not in dataset:
            raise ValueError(f"The dataset '{ds_num}' does not contain the variable '{variable_name}'.")

        if dataset[variable_name].size > 1:  # Skip datasets with shape (1,1)
            # Compute vorticity
            vorticity = compute_vorticity_1_ds(dataset, variable_name)

            # Save 2D arrays in dictionaries
            vorticity_2D[ds_num] = vorticity

        else:
            print(f"Dataset '{ds_num}' is empty or too small to process.")

    return vorticity_2D


# =============================================================================
# compute_strain
# =============================================================================

def compute_strain(datasets_dict, variable_name="ssha"):
    """
    Compute strain for multiple xarray datasets.

    Parameters
    -----------
    datasets_dict : dict
        A dictionary of xarray datasets, where each key is a dataset identifier, and each value is an xarray dataset.

    variable_name : str, optional (default: "ssha")
        The name of the variable to compute strain for.

    Returns
    --------
    strain_2D : dict
        A dictionary containing the 2D strain arrays for each dataset.

    """
    
    strain_2D = {}


    for ds_num, dataset in datasets_dict.items():
        if variable_name not in dataset:
            raise ValueError(f"The dataset '{ds_num}' does not contain the variable '{variable_name}'.")

        if dataset[variable_name].size > 1:  # Skip datasets with shape (1,1)
            # Compute strain
            strain = compute_strain_1_ds(dataset, variable_name)

            # Save 2D arrays in dictionaries
            strain_2D[ds_num] = strain
        else:
            print(f"Dataset '{ds_num}' is empty or too small to process.")

    return strain_2D

# =============================================================================
# process_2D_to_1D
# =============================================================================


def process_2D_to_1D(vorticity_2D, strain_2D):
    """
    Process dictionaries of 2D arrays for vorticity and strain, converting them to aligned 1D arrays.

    Parameters
    ----------
    vorticity_2D : dict
        Dictionary containing 2D arrays of vorticity for each dataset (key: dataset identifier).

    strain_2D : dict
        Dictionary containing 2D arrays of strain for each dataset (key: dataset identifier).

    Returns
    -------
    vorticity_1D : ndarray
        A single 1D array of aligned vorticity values across all datasets.

    strain_1D : ndarray
        A single 1D array of aligned strain values across all datasets.
    """
    vorticity_1D = []
    strain_1D = []

    for key in vorticity_2D:
        # Flatten the 2D arrays
        vorticity_data = vorticity_2D[key].flatten()
        strain_data = strain_2D[key].flatten()
        
        # Create a mask to filter valid data
        valid_mask = ~np.isnan(vorticity_data) & ~np.isnan(strain_data)
        
        # Apply the mask to both arrays
        vorticity_1D.append(vorticity_data[valid_mask])
        strain_1D.append(strain_data[valid_mask])
    
    # Concatenate all the arrays into a single 1D array
    vorticity_1D = np.concatenate(vorticity_1D)
    strain_1D = np.concatenate(strain_1D)
    
    return vorticity_1D, strain_1D



# =============================================================================
# histo_vorticity_strain
# =============================================================================


def histo_vorticity_strain(final_vorti_data, final_strain_data):
    """
    Plot a 2D histogram of vorticity and strain data.

    Parameters
    ----------
    
    final_vorti_data: ndarray
        1D array of vorticity data.
        
    final_strain_data: ndarray
        1D array of strain data.

    Returns
    --------
    Displays the plot and saves it as a PNG file.
    
    """
    #
    plt.figure(figsize=(8, 4))

    # Calculate M as the absolute maximum of x and y
    M = max(np.max(np.abs(final_vorti_data)), np.max(final_strain_data))

    # compute 2D histo
    hist, xedges, yedges = np.histogram2d(final_vorti_data, final_strain_data, bins=100)

    # Use percentiles to dynamically define vmin and vmax
    vmin = np.percentile(hist[hist > 0], 1)  # 1st percentile
    vmax = np.percentile(hist, 99)  # 99e percentile

    # Create the histogram with a logarithmic scale
    hist2d = plt.hist2d(final_vorti_data, final_strain_data, bins=200, cmap='Reds',
                        norm=colors.LogNorm(vmin=vmin, vmax=vmax))

    # Contours: display only where the density is significant
    min_density = 10
    masked_hist = np.ma.masked_where(hist < min_density, hist)
    plt.contour(masked_hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                linewidths=0.3, colors='k')

    # 
    x_vals = np.linspace(-M, M, 100) 
    # Line x = y
    plt.plot(x_vals, x_vals, 'gray', linestyle='--', label=r'$x = y$')  

    # Line x = -y (representing |x| = y for negative values)
    plt.plot(x_vals, np.abs(x_vals), 'gray', linestyle='--', label=r'$|x| = y$')

    # Set the axis limits
    M_ = 0.3 * M
    plt.xlim(-M_, M_)
    plt.ylim(0, M_)

    #
    plt.xlabel(' $\zeta / f_0 $', fontsize=12)
    plt.ylabel(r'$\sigma / |f_0|$', fontsize=12)

    # 
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(5))  # Limit to 5 labels on the X-axis
    ax.yaxis.set_major_locator(MaxNLocator(5))  #Limit to 5 labels on the Y-axis

    # colorbar
    cbar = plt.colorbar(hist2d[3])
    cbar.set_label('Counts')
    cbar.update_ticks()

    plt.grid()
    
    plt.title('Histogram of Vorticity vs Strain')

    
    plt.savefig("hito_vortistrain_ssha.png", dpi=300)
    plt.show()
    
    