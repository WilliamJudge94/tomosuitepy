import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tomosuite.base.common import load_extracted_prj
from ipywidgets import interact, interactive, fixed, widgets

def obtain_prj_sinograms(prj_data):
    """Also found in tomosuite.easy_networks.deepfillv2.data_prep """
    rows = np.arange(0, np.shape(prj_data)[1])
    sino = prj_data[:, rows]
    
    sino = np.asarray(sino)
    
    data = []
    for index in tqdm(range(0, np.shape(sino)[1]), desc='Obtaining Sino'):
        data.append(sino[:, index, :])
        
    return np.asarray(data)



def plot_sino(sino_data, idx, og_center, min_idx_val):
    idx = idx + min_idx_val
    plt.figure(figsize=(15, 15))
    plt.imshow(sino_data[idx][0])
    lens = len(sino_data)
    
    range_vals = np.arange(lens)
    range_vals -= int((lens-1)/2)
    
    plt.title(f"Shifts for ---- {og_center} + {range_vals[idx]} = {og_center + range_vals[idx]}")
    


def obtain_rotation_center(basedir, pixel_shift_range, sino_idx=0, log_multiplier=40, plot=False, figsize=(15, 5), number2zero=None, crop_sinogram=0, return_vals=False):
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
    
    #if number2zero != None:
    
        #data[:number2zero] = np.zeros((number2zero, data_shape[1], data_shape[2]))
        #data[-number2zero:] = np.zeros((number2zero, data_shape[1], data_shape[2]))
    
    center_of_image = data.shape[2]/2
    
    sino = obtain_prj_sinograms(data)
    
    store = []
    store_sino = []
    
    ranger = np.arange(-pixel_shift_range, pixel_shift_range+1, 1)
    
    for i in tqdm(ranger, desc='Checking Sinogram Offset'):
        


        og_sino = sino[sino_idx].copy()
        flip_sino1 = np.fliplr(og_sino.copy())
        flip_sino1 = np.roll(flip_sino1, i)
        full_360 = np.vstack((og_sino, flip_sino1))

        full_360 = full_360[:, (pixel_shift_range+1+crop_sinogram):-(pixel_shift_range+1+crop_sinogram)]

        full_360_shape = np.shape(full_360)

        if number2zero != None:
            zeros = np.zeros((number2zero*2, full_360_shape[1]))
            halfs = int(full_360_shape[0]/2)
            full_360[halfs-number2zero:halfs+number2zero] = zeros
            full_360 = full_360[number2zero:-number2zero]

        f = np.fft.fft2(full_360)
        fshift = np.fft.fftshift(f)

        magnitude_spectrum = log_multiplier*np.log(np.abs(fshift) + 0.001)
        
        store_sino.append([full_360, magnitude_spectrum])
        
        if i != 0:
            store.append(np.sum(magnitude_spectrum))
        else:
            store.append((store[-1] - (store[-1]*1e-6)))
        
    store = np.asarray(store)
    finder = np.where(store == np.nanmin(store))
        
    if plot:
        plt.figure(figsize=figsize)
        plt.plot(ranger, store)
        plt.xlabel(f'Pixels away from the absolute center of {center_of_image} - This script assumes you have input a 0-180 degree scan')
        plt.ylabel('Sum of FFT for a given pixel shift')
        plt.axvline(ranger[finder[0]], color='k')
        plt.title(f'New Rotation Center Is: {center_of_image + ranger[finder[0]] }')
        plt.show()

        sliders = widgets.IntSlider(value=ranger[finder[0]], min=-pixel_shift_range, max=pixel_shift_range)
        interact(plot_sino, sino_data=fixed(store_sino), idx=sliders, og_center=fixed(center_of_image), min_idx_val=fixed(pixel_shift_range))
        
        #plt.figure(figsize=(15, 15))
        #plt.title('Sinogram Related To The Best Alignment')
        #plt.imshow(store_sino[finder[0][0]][0])
        #plt.show()
        
    if return_vals:
        return (center_of_image, ranger[finder[0]]), store, finder, ranger, store_sino