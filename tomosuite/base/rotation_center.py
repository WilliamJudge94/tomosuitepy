import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tomosuite.base.common import load_extracted_prj

def obtain_prj_sinograms(prj_data):
    """Also found in tomosuite.easy_networks.deepfillv2.data_prep """
    rows = np.arange(0, np.shape(prj_data)[1])
    sino = prj_data[:, rows]
    
    sino = np.asarray(sino)
    
    data = []
    for index in tqdm(range(0, np.shape(sino)[1]), desc='Obtaining Sino'):
        data.append(sino[:, index, :])
        
    return np.asarray(data)


def obtain_rotation_center(basedir, pixel_shift_range, sino_idx=0, log_multiplier=40, plot=False, figsize=(15, 5), number2zero=None):
    """Plots a figure to help Users determine the proper rotation center. Applied a Fourier Transform to the sinogram, shifts the sinogram left and right,
    and plots the summed values of the results. The lowest value is the rotation center.
    
    Parameters
    ----------
    basedir : str
        Path to the project.
    
    pixel_shift_range : int
        Shift the sinogram left and right by this many pixels.
        
    sino_idx : int
        The index of the sinogram to use.
        
    log_multiplier : float
        A number to be multipled by the log of the np.abs() of the FFT
        
    plot : bool
        If True, plot the results to the User.
        
    figsize : list
        To be passed into plt.figure()
        
    number2zero : int
        If number2zero != None then zero out this many projections from the start and end of the data files.
        
    Returns:
    A figure or raw_center_of_image, y_values4plot, minimum_idx_finder, x_values4plot
    """
    
    data = load_extracted_prj(basedir)
    
    data_shape = np.shape(data)
    
    if number2zero != None:
    
        data[:number2zero] = np.zeros((number2zero, data_shape[1], data_shape[2]))
        data[-number2zero:] = np.zeros((number2zero, data_shape[1], data_shape[2]))
    
    center_of_image = data.shape[2]/2
    
    sino = obtain_prj_sinograms(data)
    
    store = []
    
    ranger = np.arange(-pixel_shift_range, pixel_shift_range+1, 1)
    
    for i in tqdm(ranger, desc='Checking Sinogram Offset'):
        
        if i != 0:

            og_sino = sino[sino_idx].copy()
            flip_sino1 = np.fliplr(og_sino.copy())
            flip_sino1 = np.roll(flip_sino1, i)
            full_360 = np.vstack((og_sino, flip_sino1))

            full_360 = full_360[:, pixel_shift_range:-pixel_shift_range]

            f = np.fft.fft2(full_360)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = log_multiplier*np.log(np.abs(fshift) + 0.001)
        
            store.append(np.sum(magnitude_spectrum))
        else:
            store.append(store[-1])
        
    store = np.asarray(store)
    finder = np.where(store == np.nanmin(store))
        
    if plot:
        plt.figure(figsize=figsize)
        plt.plot(ranger, store)
        plt.xlabel(f'Pixels away from the absolute center of {center_of_image} - This script assumes you have input a 0-180 degree scan')
        plt.ylabel('Sum of FFT for a given pixel shift')
        plt.axvline(ranger[finder[0]], color='k')
        plt.title(f'New Rotation Center Is: {center_of_image + ranger[finder[0]] }')
        
    else:
        return center_of_image, store, finder, ranger