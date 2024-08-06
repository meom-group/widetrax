#Imports

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


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
                
                # Verify if any NaN values remain for islands or continents after interpolation
                if not np.isnan(segment_data).any():
                    segments_dict[counter] = segment_data
                    counter += 1
                
    return segments_dict


# =============================================================================
# calculate_psd
# =============================================================================


def calculate_psd(segments_dict):
    """
    Computes the power spectral density (PSD)
    
    Parameters
    ----------
    segments_dict : Dict
        Dictionary containing segments (numpy.arrays)
    
    
    Returns
    -------
    
    psd_dict : Dict
        Dictionary containing PSDs for each segment
    freqs_dict : Dict
        Dictionary containing the associated frequencies
    
    
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
# psd_mean_and_freq
# =============================================================================        
 
    
def psd_mean_and_freq(psd_dict, freqs_dict):
    """
    Calculate the mean of the Power Spectral Densities (PSD) for each frequency point using the values from a PSD dictionary.

    Parameters
    ----------
    psd_dict : Dict
        A dictionary of numpy arrays containing the Power Spectral Densities (PSD) for segments. The arrays can be of different lengths.
    
    freqs_dict : Dict
        A dictionary containing the corresponding frequencies for each segment (numpy array)

    Returns
    -------
    psd_mean : np.ndarray
        A numpy array containing the mean of the Power Spectral Densities (PSD) for each frequency point. The length of this array is equal to the maximum length of the arrays in psd_dict.
    
    freqs_mean : np.ndarray 
        A numpy array containing frequency values of the longest column in freqs_dict
    
    """
    
    
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
    Plots the Power Spectral Density (PSD) on a logarithmic scale.
    
    The function plots the PSD on the given axes, `ax`. The plot is on a logarithmic scale 
    for both x and y axes. The function can handle one or two PSD arrays. If provided, 
    it adds labels and a title to the plot. It also includes reference lines for $k^{-2}$ 
    and $k^{-5}$ slopes.

    The function adds grid lines, legends, and adjusts axis labels and ticks for better readability.

    If `freqs` or `psd1` are empty, the function prints an error message and returns 
    without plotting.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the PSD. 
    freqs : np.ndarray
        A numpy array containing frequency values.
    psd1 : np.ndarray
        A numpy array of PSD values corresponding to `freqs`for the first PSD array.
    psd2 : np.ndarray, optional
        Array of PSD values corresponding to `freqs` for the second PSD array, used for comparison.
    title : str, optional
        Title of the plot.
    psd1_label : str, optional
        Label for the first PSD array. Default is "psd1".
    psd2_label : str, optional
        Label for the second PSD array. Default is "psd2".
    

    Returns
    -------
    - None

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



