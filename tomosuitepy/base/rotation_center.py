import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import median_filter
from tomosuitepy.base.common import load_extracted_prj
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


def plot_sino(sino_data, idx, og_center, min_idx_val, ranger, store, center_of_image, finder):

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :-1])
    ax2 = fig.add_subplot(gs[0:, -1])

    idx = idx + min_idx_val
    lens = len(sino_data)

    range_vals = np.arange(lens)
    range_vals -= int((lens-1)/2)

    # Math for quad fit

    model = np.poly1d(np.polyfit(ranger, store, 2))
    polyline = np.linspace(ranger[0], ranger[-1], len(ranger))
    finds2 = np.where(model(polyline) == np.min(model(polyline)))

    ax1.plot(ranger, store)
    ax1.plot(polyline, model(polyline))
    ax1.set_xlabel(
        f'Pixels away from the absolute center of {center_of_image} - This script assumes you have input a 0-180 degree scan')
    ax1.set_ylabel('Sum of FFT for a given pixel shift')
    ax1.axvline(range_vals[idx], color='k')
    ax1.axvline(ranger[finder[0]], color='C0')
    ax1.axvline(ranger[finds2[0]], color='C1')
    ax1.set_title(
        f"Algo Min Rot Center Is: {center_of_image + ranger[finder[0]] } --- PolyFit Min Rot Center Is: {center_of_image + ranger[finds2[0]] } --- User's Selected Rot Center is: {og_center + range_vals[idx]}")

    ax2.imshow(sino_data[idx][0])

    ax2.set_title(
        f"Shifts for ---- {og_center} + {range_vals[idx]} = {og_center + range_vals[idx]}")


def obtain_rotation_center(basedir, pixel_shift_range, sino_idx=0, log_multiplier=40,
                           number2zero=None, crop_sinogram=0, med_filter=False, min_val=0):
    """
    Plots a figure to help Users determine the proper rotation center.

    Applied a Fourier Transform to the sinogram, shifts the sinogram left and right,
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

    number2zero : int
        If number2zero != None then zero out this many projections from the start and end of the data files.

    crop_sinogram : int
        The amount of pixles on the left and right to crop out.

    med_filter : bool
        If True apply a (3, 3) median filter to data.

    min_val : int
        The minimum value for the sinogram. All sinograms are scaled from 0.0 - 1.0.

    Returns:
    Fig
        An interactive figure.
    """

    data = load_extracted_prj(basedir)

    data_shape = np.shape(data)

    center_of_image = data.shape[2]/2

    sino = obtain_prj_sinograms(data)

    store = []
    store_sino = []

    ranger = np.arange(-pixel_shift_range, pixel_shift_range+1, 1)

    for i in tqdm(ranger, desc='Checking Sinogram Offset'):

        og_sino = sino[sino_idx].copy()
        og_sino = og_sino[:, ~np.isnan(og_sino).any(axis=0)]
        og_sino = og_sino[:, ~np.isinf(og_sino).any(axis=0)]
        og_sino /= np.max(og_sino)
        og_sino[og_sino < min_val] = 0

        if med_filter:
            og_sino = median_filter(og_sino, size=(3, 3))

        flip_sino1 = np.fliplr(og_sino.copy())
        flip_sino1 = np.roll(flip_sino1, i)
        full_360 = np.vstack((og_sino, flip_sino1))

        full_360 = full_360[:, (pixel_shift_range+1+crop_sinogram)                            :-(pixel_shift_range+1+crop_sinogram)]

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

    sliders = widgets.IntSlider(
        value=ranger[finder[0]], min=-pixel_shift_range, max=pixel_shift_range)

    interact(plot_sino, sino_data=fixed(store_sino),
             idx=sliders, og_center=fixed(center_of_image),
             min_idx_val=fixed(pixel_shift_range),  ranger=fixed(ranger),
             store=fixed(store), center_of_image=fixed(center_of_image),
             finder=fixed(finder))

    return store_sino, sino, data, full_360
