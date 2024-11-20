from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.signal as signal


# =============================================================================
# retrieve_segments
# =============================================================================

def retrieve_segments(datasets,FileType,namevar=None):
    """
    Extracts segments from xarray.datasets
    
    Parameters
    ----------
    datasets : Dict
        Dictionary containing xarray.Datasets
    FileType : str
        The file type used to calculate datasets, "NetCDF" or "Zarr"
    namevar : str, optional
        The variable name used to calculate PSD.
        Default is 'ssha'.
        
    Returns
    -------
    segments_dict : Dict
        Dictionary containing segments (numpy.arrays).
    

    """
    segments_dict = {}
    counter = 0
    
    #Calculation of Sea Surface Height (SSH) : SSH = SSHA + MDT
    for ky in range(len(datasets)):
        datasets[ky]['ssh'] = datasets[ky][namevar] + datasets[ky]['mdt']
    
    for key, dataset in datasets.items():
        #print(f"starting processing dict number {key} among {len(datasets)}")
        for col in range(dataset.dims['num_pixels']):
            # Extract data for one column
            col_data = dataset.isel(num_pixels=col)
            
            # Drop unnecessary variables based on FileType
            drop_vars = ['latitude', 'longitude', namevar, 'mdt']
            if FileType == "Zarr":
                drop_vars.extend(['cycle_number', 'duacs_land_sea_mask', 'pass_number'])
            elif FileType != "NetCDF":
                raise ValueError(f"Unsupported FileType: {FileType}")
            
            col_dataset = col_data.drop_vars(drop_vars, errors="ignore")
            
            
            # Extract and clean segment data
            ssh_data = col_dataset['ssh'].values
            if np.any(~np.isnan(ssh_data)):
                segment_data = ssh_data[~np.isnan(ssh_data)]  # Remove NaNs directly
                
                # Remove NaN values at the beginning
                while len(segment_data) > 0 and np.isnan(segment_data[0].item() if np.ndim(segment_data[0]) else segment_data[0]):
                    segment_data = segment_data[1:]
                
                # Remove NaN values at the end
                while len(segment_data) > 0 and np.isnan(segment_data[-1].item() if np.ndim(segment_data[-1]) else segment_data[-1]):
                    segment_data = segment_data[:-1]
                
                # Store segment if it has no remaining NaNs
                if not np.isnan(segment_data).any():
                    segments_dict[counter] = segment_data
                    counter += 1
                
    return segments_dict


# =============================================================================
# calculate_psd
# =============================================================================

def calculate_segment_psd(segment_data, fs):
    if len(segment_data) > 120:  # Check segment length
        return signal.welch(segment_data, fs=fs, nperseg=len(segment_data), noverlap=0)


def calculate_psd(segments_dict):
    """
    Computes the power spectral density (PSD)
    
    Parameters
    ------------
    segments_dict: Dict
        Dictionary containing segments (numpy.arrays)

    Returns
    ---------
    psd_dict: Dict
        Dictionary containing PSDs for each segment
    freqs_dict: Dict
        Dictionary containing the associated frequencies
    """
    fs = 0.5  # maybe it could be an argument?

    psd_dict = {}
    freqs_dict = {}
    counter = 0

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calculate_segment_psd, segment_data, fs) for segment_data in segments_dict.values()]

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                freqs, psd = result
                psd_dict[counter] = psd
                freqs_dict[counter] = freqs
                counter += 1

    return psd_dict, freqs_dict


# =============================================================================
# psd_mean_and_freq
# =============================================================================        


def psd_mean_and_freq(psd_dict, freqs_dict):
    """
    Calculate the mean of the Power Spectral Densities (PSD) for each frequency point using the values from a PSD
    dictionary.

    Parameters
    ------------
    psd_dict: Dict
        A dictionary of numpy arrays containing the Power Spectral Densities (PSD) for segments.
        The arrays can be of different lengths.
    freqs_dict: Dict
        A dictionary containing the corresponding frequencies for each segment (numpy array)

    Returns
    ---------
    psd_mean: np.ndarray
        A numpy array containing the mean of the Power Spectral Densities (PSD) for each frequency point.
        The length of this array is equal to the maximum length of the arrays in psd_dict.
    freqs_mean: np.ndarray
        A numpy array containing frequency values of the longest column in freqs_dict
    """
    max_length = 0
    freqs_mean = None
    for k, array in psd_dict.items():
        if array.size > max_length:
            max_length = array.size
            freqs_mean = freqs_dict[k]

    psd_sum = np.zeros(max_length)
    psd_counts = np.zeros(max_length)
    for array in psd_dict.values():
        psd_sum[:array.size] += array
        psd_counts[:array.size] += 1
    psd_mean = psd_sum / psd_counts

    return psd_mean, freqs_mean


# =============================================================================
# plot_psd
# ============================================================================= 

def plot_psd(ax, freqs, psds, unit, psd_labels, title=None):
    """
    Plots the Power Spectral Density (PSD) on a logarithmic scale.
    
    The function plots the PSD on the given axes, `ax`. The plot is on a logarithmic scale 
    for both x and y axes. The function can handle one or two PSD arrays. If provided, 
    it adds labels and a title to the plot. It also includes reference lines for $k^{-2}$ 
    and $k^{-5}$ slopes.

    The function adds grid lines, legends, and adjusts axis labels and ticks for better readability.

    If `freqs` or `psd1` are empty, the function prints an error message and returns 
    without plotting.
    
    Parameters
    ------------
    ax: matplotlib.axes.Axes
        The axes on which to plot the PSD. 
    freqs: np.ndarray
        A numpy array containing frequency values.
    psds: [np.ndarray, ...] | np.ndarray
        A list of numpy array or a single numpy array of PSD values corresponding to `freqs`.
    unit: str
        Unit of the physical quantity for which the PSD was computed.
    psd_labels: [str, ...] | str, optional
        A list of labels or a single label for the PDS array(s).
    title: str, optional
        Title of the plot.
    """
    if isinstance(psds, np.ndarray):
        psds = [psds]
        psd_labels = [psd_labels]
    if psd_labels is None:
        psd_labels = [None] * len(psds)

    for psd, label in zip(psds, psd_labels):
        ax.loglog(freqs, psd, label=label)

    ax.set_ylim(1e-7, 1e2)
    ax.set_ylabel(f"PSD [${unit}/(cy/km)$]", fontsize=8, fontweight="bold", color="black")

    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Wavenumber [$cy/km$]", fontsize=8, fontweight="bold", color="black")

    # filters out zero frequencies
    non_zero_freqs = freqs[freqs != 0]

    # k^-2 & k^-5
    k_2 = non_zero_freqs ** -2 * (psds[0][0] / (non_zero_freqs[0] ** -2))  # Scale according to the first PSD value
    k_5 = non_zero_freqs ** -5 * (psds[0][0] / (non_zero_freqs[0] ** -5))

    ax.loglog(non_zero_freqs, k_2*30, "k--", label="$k^{-2}$")  # 30 to fix the line
    ax.loglog(non_zero_freqs, k_5*1e6, "b--", label="$k^{-5}$")  # 1e6 to fix the line

    ax2 = ax.secondary_xaxis("bottom", functions=(lambda x: 1 / x, lambda x: 1 / x))
    ax2.set_xlabel("Wave-length [$km$]", fontsize=8, fontweight="bold", color="black")

    wavelength_ticks = np.array([5, 10, 25, 50, 100, 200, 500, 1000])
    ax2.set_xticks(wavelength_ticks)
    ax2.set_xticklabels(wavelength_ticks)
    ax.grid(True, which="both")
    ax.legend()

    # Set the title if provided
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", color="black")
