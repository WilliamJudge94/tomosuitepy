from ...base.common import loading_tiff_prj
import tifffile as tif
import numpy as np
from progressbar import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from ...easy_networks.tomogan.data_prep import format_data_tomogan, setup_data_tomogan


def start_pbar(prj):
    widgets = ['Progress: ', Percentage(), ' ', Bar(marker='-', left='[', right=']'),
               ' ', Timer(), '  ', ETA(), ' ', FileTransferSpeed()]  # see docs for other options
    pbar = ProgressBar(widgets=widgets, maxval=len(prj))
    pbar_val = 0
    pbar.start()
    return pbar


def add_poisson_noise(prj, variable):

    pbar = start_pbar(prj)

    vect_func = np.vectorize(add_train_noise, otypes=[list], excluded=[
                             'array', 'nothing', 'pbar'])
    varss = np.arange(len(prj))
    varss2 = variable * np.ones(len(varss))

    out = vect_func(varss, varss2, array=prj, pbar=pbar, nothing=False)

    new_out = []
    for value in out:
        new_out.append(value[0])

    return np.asarray(new_out)


def add_train_noise(idx, VARIABILITY, array, pbar, nothing=False):
    """Add random poisson noise to an image

    Parameters
    ----------
    idx : int
        the index of the array the User would like to add noise to

    VARIABILITY : int
        a value passed to the np.poisson scaling
    array : np.array
        2D numpy image

    pbar : progressbar
        the progressbar to let the User know something is working

    nothing : bool
        if the User would like to return the input value with no noise added
    Returns
    -------
    An array with poisson noise added or the exact same array. Depends on the value of 'nothing'
    """

    img = array[idx]

    pbar.update(idx)

    if nothing:
        return [img, idx]

    mins = np.min(img)
    maxs = np.max(img)

    PEAK = VARIABILITY

    img = np.random.poisson(img / maxs * PEAK) / PEAK * maxs

    return np.asarray([img, idx], dtype=object)


def fake_noise_test(basedir, noise=125, image_step=20, plot=True, idx=0, figsize=(10, 10), clim=None):
    """A way to quickly see what the noise value does to the data.

    Parameters
    ----------
    basedir : str
        the path to the experiment

    noise : int
        parameter used to create the poisson noise

    image_step : int
        take every image_step image

    plot : bool
        if True the idx value will be plotted

    idx : int
        the noise_image index to plot

    figsize : tup
        figsize passed into plt.figure()

    Returns
    -------
    The noisy data
    """

    data = loading_tiff_prj(f'{basedir}extracted/projections/')
    data = data[::image_step]
    noise_data = add_poisson_noise(data, noise)

    if plot:
        plt.figure(figsize=figsize)
        im = plt.imshow(data[idx])
        plt.title('Extracted PRJ')
        if clim is not None:
            plt.clim(clim)
        plt.colorbar(im)
        plt.show()

        plt.figure(figsize=figsize)
        plt.imshow(noise_data[idx])
        if clim is not None:
            plt.clim(clim)
        plt.title('Noisy PRJ')
        plt.colorbar(im)
        plt.show()

        plt.figure(figsize=figsize)
        plt.title('Difference')
        image = np.subtract(data[idx], noise_data[idx])
        plt.imshow(image)
        plt.colorbar()
        plt.title((image.min(), image.max()))
        plt.show()

    if not plot:
        return data[idx], noise_data[idx]


def setup_fake_noise_train(basedir, noise=125, interval=5, dtype=np.float32):
    """Prepares data for TomoGAN. Please use tomosuite.noise_test_tomogan to set the correct noise value.

    Parameters
    ----------
    basedir : str
        the path to the current experiment

    nosie : int
        integer value to be passed into the add_poisson_noise() function.
        Use tomosuite.noise_test_tomogan() to find correct noise level

    Returns
    -------
    Nothing
    """

    data = loading_tiff_prj(f'{basedir}extracted/projections/')
    noise_data = add_poisson_noise(data, noise)

    xtrain, ytrain, xtest, ytest = format_data_tomogan(clean_data=data,
                                                       noisy_data=noise_data,
                                                       interval=interval,
                                                       dtype=dtype)

    setup_data_tomogan(basedir=basedir,
                       xtrain=xtrain,
                       ytrain=ytrain,
                       xtest=xtest,
                       ytest=ytest,
                       types='noise')


def setup_fake_noise_predict(basedir):
    """Obtain the data to be passed into the TomoGAN predict function

    Parameters
    ----------
    basedir : str
        the path to the current project

    Returns
    -------
    The current projection data as a np.array
    """
    data = loading_tiff_prj(f'{basedir}extracted/projections/')
    return np.asarray(data)
