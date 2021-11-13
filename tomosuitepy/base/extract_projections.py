import os
import sys
import time
import tomopy
import pathlib
import dxchange
import functools
import numpy as np
from tqdm import tqdm
import tifffile as tif
import tifffile as tif
from pympler import muppy, summary
from ..base.common import save_metadata


def median_filter_nonfinite(data, size=3, verbose=False):
    """
    Remove nonfinite values from a 3D array using an in-place 2D median filter.

    The 2D selective median filter is applied along the last two axes of
    the array.

    Parameters
    ----------
    data : ndarray
        The 3D array of data with nonfinite values in it.
    size : int, optional
        The size of the filter.
    verbose : bool
        If True then a step name is printed to the User.

    Returns
    -------
    ndarray
        The corrected 3D array with all nonfinite values removed based upon the local
        median value defined by the kernel size.

    Raises
    ------
    ValueError
        If the filter comes across a kernel only containing non-finite values a ValueError
        is raised for the user to increase their kernel size to avoid replacing the current
        pixel with 0.
    """
    # Defining a callback function if None is provided
    if verbose:
        print('\n** Removing Bad Values')

    # Iterating throug each projection to save on RAM
    for projection in tqdm(data):
        nonfinite_idx = np.nonzero(~np.isfinite(projection))

        # Iterating through each bad value and replace it with finite median
        for x_idx, y_idx in zip(*nonfinite_idx):

            # Determining the lower and upper bounds for kernel
            x_lower = max(0, x_idx - (size // 2))
            x_higher = min(data.shape[1], x_idx + (size // 2) + 1)
            y_lower = max(0, y_idx - (size // 2))
            y_higher = min(data.shape[2], y_idx + (size // 2) + 1)

            # Extracting kernel data and fining finite median
            kernel_cropped_data = projection[x_lower:x_higher,
                                             y_lower:y_higher]

            if len(kernel_cropped_data[np.isfinite(kernel_cropped_data)]) == 0:
                raise ValueError("Found kernel containing only non-finite values.\
                                 Please increase kernel size")

            median_corrected_data = np.median(
                kernel_cropped_data[np.isfinite(kernel_cropped_data)])

            # Replacing bad data with finite median
            projection[x_idx, y_idx] = median_corrected_data

    return data


def cache_clearing_downsample(data, binning):
    data = tomopy.downsample(data, level=binning)
    data = tomopy.downsample(data, level=binning, axis=1)
    return data


def save_prj_ds_chunk(data, iteration, path):
    """
    Saving data to iterated numpy files.

    Parameters
    ----------
    data : nd.array
        A data array to save
    iteration : int
        The iteration number for that data chunk.
    path : str
        The path where to save the file to.

    Returns
    -------
    None
        Saves chunked data to
        {path}/tomsuitepy_downsample_save_it_{str(iteration).zfill(4)}.npy
    """
    save_path = f'{path}/tomsuitepy_downsample_save_it_{str(iteration).zfill(4)}.npy'
    try:
        np.save(save_path, data)
    except Exception as ex:
        raise ValueError(
            f"Unable to save data - path={save_path}\nError - {ex}")


def load_prjs_norm_chunk(iterations, path, img_shape, dtypes):
    """
    Loading saved data chunks for normalization.

    Parameters
    ----------
    iterations : int
        The max amount of iterations to load.
    path : str
        The path to the save chunks.
    img_shape : list
        The shape of all chunked data
    dtypes : dtype
        The data type to load data into

    Returns
    -------
    nd.array
        The complete data - unchunked.
    """

    chunk_amount = img_shape[0]

    prj_store = np.zeros(
        (chunk_amount * iterations, img_shape[1], img_shape[2]))
    prj_store = prj_store.astype(dtypes)

    for idx in tqdm(range(0, iterations), desc='Reading Chunked Data'):
        first = idx * chunk_amount
        second = (idx + 1) * chunk_amount
        var = np.load(
            f'{path}/tomsuitepy_downsample_save_it_{str(idx).zfill(4)}.npy')

        prj_store[first: second] = var.copy()
        del var

    return prj_store


def load_prj_ds_chunk(iterations, path):
    """
    Loading chunked data for the downsampling.

    Parameters
    ----------
    iterations : int
        The total amount of iterations to load.
    path : str
        The path where the chunked data is saved.

    Returns
    -------
    nd. array
        The unchunked data.
    """

    data = []

    for it in tqdm(range(0, iterations), desc='Reading Chunked Data'):
        data.append(
            np.load(f'{path}/tomsuitepy_downsample_save_it_{str(it).zfill(4)}.npy'))
        #print(f'loading - {path}/tomsuitepy_downsample_save_it_{str(it).zfill(4)}.npy')

    data = np.asarray(data)
    data = np.concatenate(data)
    return data


def remove_saved_prj_ds_chunk(iterations, path):
    """
    Deletes the saved chunked data for downsampling.

    Parameters
    ----------
    iterations : int
        The max amount of iterations to remove.
    path : str
        The path where the chunked data is located.

    Returns
    -------
    None
        Deletes the chunked data from storage. 
    """
    for it in range(0, iterations):
        #print(f'removing - {path}/tomsuitepy_downsample_save_it_{str(it).zfill(4)}.npy')
        os.remove(
            f'{path}/tomsuitepy_downsample_save_it_{str(it).zfill(4)}.npy')


def flat_roll_func(flat, flat_roll):
    """
    Rolls the flat-fild images left or right.

    Parameters
    ----------
    flat : nd.array
        The data array for the flat-field images.
    flat_roll : int
        How many pixles the User wants to roll the image by.

    Returns
    -------
    nd.array
        The flat field images rolled.
    """
    if flat_roll != None:
        flat = np.roll(flat, flat_roll, axis=2)
    return flat


def outlier_diff_func(prj, flat, outlier_diff, outlier_size, verbose):
    """
    Containerizing data-prep function tomopy.misc.corr.remove_outlier.

    Parameters
    ----------
    prj : nd.array
        An array containing the projection data.
    flat : nd. array
        An array containing the flat-field data.
    outlier_diff : float
        Value to be passed into tomopy.misc.corr.remove_outlier.
    outlier_size : float
        Value to be passed into tomopy.misc.corr.remove_outlier.
    verbose : bool
        If True prints out current step name.

    Returns
    -------
    nd. array, nd.array
        tomopy.misc.corr.remove_outlier corrected projections and flat-field data. 
    """
    if outlier_diff != None and outlier_size != None:
        if verbose:
            print('\n** Remove Outliers')
        prj = tomopy.misc.corr.remove_outlier(
            prj, outlier_diff, size=outlier_size, axis=0)
        flat = tomopy.misc.corr.remove_outlier(
            flat, outlier_diff, size=outlier_size, axis=0)
    return prj, flat


def flat_field_corr_func(prj, flat, dark, chunking_size, normalize_ncore, muppy_amount, dtype, verbose):
    """
    Containerizing data-prep function tomopy.normalize.

    Parameters
    ----------
    prj : nd.array
        An array containing the projection data.
    flat : nd.array
        An array containing the flat-field data.
    datk : nd.array
        An array containing the dark-field data.
    chunking_size : int
        The amount of data chunks to iterate through.
    normalize_ncore : int
        The amount of cores to use for normalization.
    muppy_ammount : int
        The amount of global variables to check - rests RAM usage.
    dtype : data-type
        The data type to load data into.
    verbose : bool
        If True prints out current step name.

    Returns
    -------
    nd.array, int, list
        The corrected projection data, iteration number, and total projection_chunk_shape.
    """
    iteration = 0
    prj_chunk_shape = 0

    if verbose:
        print('\n** Flat field correction')

    if chunking_size > 1:

        path_chunker = pathlib.Path('.').absolute()

        for prj_ds_chunk in tqdm(np.array_split(prj, chunking_size), desc='Flat Field Correction - Chunked'):

            prj_ds_chunk = tomopy.normalize(
                prj_ds_chunk, flat, dark, ncore=normalize_ncore)
            prj_chunk_shape = np.shape(prj_ds_chunk)
            save_prj_ds_chunk(prj_ds_chunk, iteration, path_chunker)
            iteration += 1
            del prj_ds_chunk
            all_objects = muppy.get_objects()[:muppy_amount]
            sum1 = summary.summarize(all_objects)
            time.sleep(1)

        del prj

        prj = 0

        all_objects = muppy.get_objects()[:muppy_amount]
        sum1 = summary.summarize(all_objects)

    else:
        prj = tomopy.normalize(prj, flat, dark, ncore=normalize_ncore)

    return prj, iteration, prj_chunk_shape


def flat_field_load_func(iteration, prj_chunk_shape, dtype, muppy_amount):
    """
    Containerizing loading chunked data from tomopy.normalize.

    Parameters
    ----------
    iteration : int
        The max iterations to import.
    prj_chunk_shape : list
        The shape of the total chunked data.
    dtype : dtype, str
        The dtype to import the data as.
    muppy_ammount : int
        The amount of values to check for muppy.
        RAM reset for tomopy.

    Returns
    -------
    nd.array
        The loaded chunked data from the flat field correction.
    """

    path_chunker = pathlib.Path('.').absolute()

    prj = load_prjs_norm_chunk(iteration, path_chunker, prj_chunk_shape, dtype)
    remove_saved_prj_ds_chunk(iteration, path_chunker)

    all_objects = muppy.get_objects()[:muppy_amount]
    sum1 = summary.summarize(all_objects)

    return prj


def bkg_norm_func(bkg_norm, prj, chunking_size, air):
    """
    Containerizing the tomopy.normalize.normalize_bg()

    Parameters
    ----------
    bkg_norm : bool
        Left over from update.
    prj : nd.array
        The data array one would like to apply the normalization to.
    chunking_size : int
        The amount of data chunks to save the data as.
    air : int
        The air value passed into the normalization.

    Returns
    -------
    nd.array
        The background normalized projections.
    """

    prj_chunks = []
    for prj_chunk in tqdm(np.array_split(prj, chunking_size), desc='Bkg Normalize'):
        prj_tmp = tomopy.prep.normalize.normalize_bg(
            prj_chunk, air=air, ncore=1)
        prj_chunks.append(prj_tmp.copy())
        del prj_tmp

    prj = np.concatenate(prj_chunks)

    return prj


def correct_norma_extremes_func(correct_norma_extremes, verbose, prj):
    """
    Makes sure no values are less than or equal to zero in the array.

    Parameters
    ----------
    correct_norma_extremes : bool
        Continue with this procedure if this is true.
    verbose : bool
        If verbose then print current operation.
    prj : nd.array
        The data array to apply the correction to.

    Returns
    -------
    nd.array
        The corrected array.
    """
    if correct_norma_extremes:
        if verbose:
            print(
                f'\n** Normalization pre-log correction - Min: {prj.min()} - Max: {prj.max()}')

        prj += np.abs(np.nanmin(prj))
        prj += np.nanmax(prj) * 0.0001

        if verbose:
            print(
                f'\n** After Normalization Min: {prj.min()} - Max: {prj.max()}')

    return prj


def minus_log_func(minus_log, verbose, prj, muppy_amount, chunking_size):
    """
    Containerized the tomopy.minus_log() function.

    Parameters
    ----------
    minus_log : bool
        If True then continue with the operation.
    verbose : bool
        If true then print out operation name.
    prj : nd.array
        The data the User wants to apply the minus log to.
    muppy_amount : int
        The amount of local variables to load to reset RAM usage.
    chunking_size : int
        The amount of data chunks to create.

    Returns
    -------
    nd.array
        The array that has had a minus log applied to it.
    """

    iteration = 0

    if minus_log:
        if verbose:
            print('\n** Applying minus log')

        try:
            all_objects = muppy.get_objects()[:muppy_amount]
            sum1 = summary.summarize(all_objects)

        except Exception as ex:
            raise ValueError(
                f"\n** Failed to initiate muppy RAM collection - Error: {ex}")

        if chunking_size > 1:

            path_chunker = pathlib.Path('.').absolute()

            for prj_ds_chunk in tqdm(np.array_split(prj, chunking_size), desc='Applying Minus Log'):

                prj_ds_chunk = tomopy.minus_log(prj_ds_chunk)
                prj_chunk_shape = np.shape(prj_ds_chunk)
                save_prj_ds_chunk(prj_ds_chunk, iteration, path_chunker)
                iteration += 1
                del prj_ds_chunk
                all_objects = muppy.get_objects()[:muppy_amount]
                sum1 = summary.summarize(all_objects)
                time.sleep(1)

        else:
            prj = tomopy.minus_log(prj)
            prj_chunk_shape = np.shape(prj)

    return prj, iteration, prj_chunk_shape


def minus_log_load_func(iteration, muppy_amount, img_shape, dtype):
    """
    Loading chunked minus_log data.

    Parameters
    ----------
    iteration : int
        The max iteration to load.
    muppy_ammount : int
        The amount of local variables to load. Resets RAM usage.
    img_shape : list
        Total chunked data shape.
    dtype : dtype, str
        The dtype to load the data into.

    Returns
    -------
    nd.array
        The loaded minus_log data after it was saved from chunking.
    """

    path_chunker = pathlib.Path('.').absolute()

    prj = load_prjs_norm_chunk(iteration, path_chunker, img_shape, dtype)
    remove_saved_prj_ds_chunk(iteration, path_chunker)

    all_objects = muppy.get_objects()[:muppy_amount]
    sum1 = summary.summarize(all_objects)

    return prj


def neg_nan_inf_func(prj, verbose, remove_neg_vals, remove_nan_vals, remove_inf_vals, removal_val):
    """
    Containerizeing the neg/nan/inf correction.

    Parameters
    ----------
    prj : nd.array
        The data to correct.
    remove_neg_vals : bool
        If True then remove neg values.
    remove_nan_vals : bool
        If True then remove nan values.
    remove_inf_vals : bool
        If True then remove inf values.
    removal_val : float
        The value to replace these non-finite values with.

    Returns
    -------
    nd.array
        The corrected prj array.
    """
    if remove_neg_vals:
        if verbose:
            print('\n** Removing neg')
        prj = tomopy.misc.corr.remove_neg(prj, val=removal_val)

    if remove_nan_vals:
        if verbose:
            print('\n** Removing np.nan')
        prj = tomopy.misc.corr.remove_nan(prj, val=removal_val)

    if remove_inf_vals:
        if verbose:
            print('\n** Removing np.inf')

        prj[np.where(prj == np.inf)] = removal_val

    return prj


def force_positive_func(force_positive, verbose, prj):
    """
    Containerizing the force positive function.

    Forces all values in array to be positive values. 

    Parameters
    ----------
    force_positive : bool
        If True then complete operation.
    verbose : bool
        If True then print out operation name.
    prj : nd.array
        The data array to be corrected.

    Returns
    -------
    nd.array
        The array to correct.
    """
    if force_positive:
        if verbose:
            print('\n** Making positive numbers')

        # Force the projections to be >= 0
        if np.nanmin(prj) < 0:
            min1 = np.nanmin(prj)
            prj += np.abs(np.nanmin(prj))

            # Force min projection to be zero
            prj -= np.nanmin(prj)

            print(
                f'\n** Applying positive numbers - OG min: {min1} Corrected Min: {np.nanmin(prj)}')

    elif not force_positive:
        minimum_prj = np.min(prj)
        if minimum_prj < 0:
            print(f"Your lowest projection value is negative ({minimum_prj}). This may result in undesireable machine\
                learning outputs. To change this set force_positive=True.")

    return prj


def downsample_func(binning, verbose, muppy_amount, chunking_size, prj):
    """
    Containerizes the tomopy.downsample() function.

    Parameters
    ----------
    binning : int
        The power of 2 to bin data to.
    verbose : bool
        If True then print the name of operation.
    muppy_amount : int
        The amount of local variables to load. Resets RAM usage.
    chunking_size : int
        The amount of data chunks to create.
    prj : nd.array
        The data to downsample.

    Returns
    -------
    nd.array
        The downsampled data.
    """
    if binning > 0:
        if verbose:
            print('\n** Down sampling data')

        try:
            all_objects = muppy.get_objects()[:muppy_amount]
            sum1 = summary.summarize(all_objects)

        except Exception as ex:
            raise ValueError(
                f"\n** Failed to initiate muppy RAM collection - Error: {ex}")

        if chunking_size > 1:

            path_chunker = pathlib.Path('.').absolute()
            iteration = 0

            for prj_ds_chunk in tqdm(np.array_split(prj, chunking_size), desc='Downsampling Data'):

                prj_ds_chunk = cache_clearing_downsample(prj_ds_chunk, binning)
                save_prj_ds_chunk(prj_ds_chunk, iteration, path_chunker)
                iteration += 1
                del prj_ds_chunk
                all_objects = muppy.get_objects()[:muppy_amount]
                sum1 = summary.summarize(all_objects)
                time.sleep(1)

            prj = load_prj_ds_chunk(iteration, path_chunker)
            remove_saved_prj_ds_chunk(iteration, path_chunker)

        else:
            prj = tomopy.downsample(prj, level=binning)
            prj = tomopy.downsample(prj, level=binning, axis=1)

    return prj


def extract(datadir, fname, basedir,
            extraction_func=dxchange.read_aps_32id,
            binning=1,
            starting=0,
            chunking_size=10,
            normalize_ncore=None,
            minus_log=True,
            nan_inf_selective=False,
            kernel_selective=5,
            remove_neg_vals=True,
            removal_val=0.001,
            dtype='float32',
            muppy_amount=1000,
            overwrite=True,
            verbose=True,
            save=True,
            custom_dataprep=False,
            data=None,
            outlier_diff=None,
            outlier_size=None,
            bkg_norm=False,
            air=10,
            flat_roll=None,
            force_positive=False,
            remove_nan_vals=False,
            remove_inf_vals=False,
            correct_norma_extremes=False,
            ):
    """
    Extract projection files from file experimental file formats. Allows User to not apply corrections after normalization.

    Parameters
    ----------
    datadir : str
        the base path to the Users experimental files.

    fname : str
        The name of the file the User would like to extract projections from.

    basedir : str
        The path with starting file name for the files to be saved.
        Example: /home/user/folder/proj_ims creates proj_ims_0000.tif inside this folder.

    extraction_fun : func
        The function from dxchange the User would like to use to extract the
        prj, flat, and dark field projections.

    binning : int
        An integer used to determine the downsampling rate of the data.

    starting : int
        The starting digit for the proj_ims files name.

    chunking_size : int
        The background normalization is RAM intensive. This allows the User to
        chunk the normalization process.

    normalize_ncore : int
        The amount of cores to use during tomopy.normalize()       

    minus_log : bool
        Allow the program to apply a minus log to the data.

    nan_inf_selective : bool
        Tf True apply median blur only to nan and inf values.

    kernel_selective : int
        The kernel size for the nan_inf_selective opperation.

    remove_neg - nan - inf_values : bool
        If True it will remove theses values iwht the removal_val.

    removal_val : float
        Value to be passed into the remove neg, nan, inf, parameters.

    dtype : str
        The data type to save the data as.

    muppy_amount : int
        Amount of data to read from the RAM usage.

    overwrite : bool
        If True then the projections files will overwrite the designated folder.

    verbose : bool
        If True this will output print statements for each section of the extraction process.

    save : bool
        If False this will return the extracted projections to the user in the form of anumpy array.

    custom_dataprep : bool
        If True this allows the User to define dataprep functions in the reconstruction script.
        Output stops after the initial flat/dark field normalization.

    data : np.array
        array that has prj, flat, dark, theta = data if the User wants to load in their own data outside of dxchange.

    outlier_diff : int or None
        Expected difference value between outlier value and the median value of the array.
        If None this step is ignored.

    outlier_size : int or None
        Size of the median filter. If None this step is ignored.

    bkg_norm : bool
        If True then a background normalization is applied through TomoPy.

    air : int
        Number of pixels at each boundary to calculate the scaling factor. Used during the bkg_norm step.

    flat_roll : int or None
        Move the flat field over this many pixles before applying normalization. If None this step is skipped.

    force_positive : bool
        Force the data to be positive after all data prep but before downsampling.

    correct_norma_extremes : bool
        If True it will try to set the data to be positive values before applying the -log().

    Returns
    -------
    None, nd.array
        Only saves the corrected projection files to the designated folder.
        Unless save=False. Then it will return a numpy array with the projections.
    """

    # Saving MetaData
    metadata_dic = {}
    metadata_keys = [var for var in locals().keys() if '__' not in var]
    for metadata_key in metadata_keys:
        metadata_dic[metadata_key] = locals()[metadata_key]
    save_metadata(basedir, metadata_dic)

    # Determine the location of the hdf5 file
    fname = os.path.join(datadir, fname)
    basename, ext = os.path.splitext(fname)

    start_time = time.time()

    # extract the data from the file
    if verbose:
        print('\n** Reading data')

    if data is None:
        prj, flat, dark, theta = extraction_func(fname)
    else:
        prj, flat, dark, theta = data

    if len(prj) % chunking_size != 0:
        raise ValueError(
            f'Length of projections ({len(prj)}) is not divisible by chunking_size ({chunking_size}).')

    # Determine how many leading zeros there should be
    digits = len(str(len(prj)))

    flat = flat_roll_func(flat, flat_roll)

    # Allows the User to remove outliers
    prj, flat = outlier_diff_func(
        prj, flat, outlier_diff, outlier_size, verbose)

    # Normalized the projections - Hella Memory
    prj, iteration, prj_chunk_shape = flat_field_corr_func(prj, flat, dark,
                                                           chunking_size, normalize_ncore,
                                                           muppy_amount, dtype, verbose)

    if chunking_size > 1:
        del prj, flat, dark
        time.sleep(5)
        all_objects = muppy.get_objects()[:muppy_amount]
        sum1 = summary.summarize(all_objects)

        prj = flat_field_load_func(
            iteration, prj_chunk_shape, dtype, muppy_amount)

    all_objects = muppy.get_objects()[:muppy_amount]
    sum1 = summary.summarize(all_objects)
    time.sleep(2)

    if not custom_dataprep:

        # Apply a background normalization to the projections
        if bkg_norm:
            prj = bkg_norm_func(bkg_norm, prj, chunking_size, air)

        if correct_norma_extremes:
            prj = correct_norma_extremes_func(
                correct_norma_extremes, verbose, prj)

        if minus_log:
            prj, iteration, prj_chunk_shape = minus_log_func(
                minus_log, verbose, prj, muppy_amount, chunking_size)

        if chunking_size > 1:
            del prj
            all_objects = muppy.get_objects()[:muppy_amount]
            sum1 = summary.summarize(all_objects)
            time.sleep(2)
            prj = minus_log_load_func(
                iteration, muppy_amount, prj_chunk_shape, dtype)

        prj = neg_nan_inf_func(prj, verbose, remove_neg_vals,
                               remove_nan_vals, remove_inf_vals, removal_val)

        if nan_inf_selective:
            prj = median_filter_nonfinite(
                data=prj, size=kernel_selective, verbose=verbose)

    else:
        if verbose:
            print(
                '\n** Not applying data manipulation after tomopy.normalize - Except for downsampling')

    # Bin the data
    if binning > 0:
        prj = downsample_func(
            binning, verbose, muppy_amount, chunking_size, prj)

    if not custom_dataprep:
        if force_positive:
            prj = force_positive_func(force_positive, verbose, prj)

    if save:

        if verbose:
            print('\n** Dumping tiff\n')

        dxchange.write_tiff_stack(prj[:, :, :], fname=f'{basedir}extracted/projections/proj',
                                  dtype=dtype, axis=0, digit=digits, start=starting, overwrite=overwrite)

        np.save(f'{basedir}extracted/theta/theta.npy', np.asarray(theta))

        if verbose:
            print('   done in %0.3f min' % ((time.time() - start_time)/60))

        del prj
        del theta

        all_objects = muppy.get_objects()[:muppy_amount]
        sum1 = summary.summarize(all_objects)

    else:
        print('   done in %0.3f min' % ((time.time() - start_time)/60))
        return np.asarray(prj), np.asarray(theta)


def extract_phantom(datadir, fname, basedir, starting=0, dtype='float32', flat_roll=None, overwrite=True, verbose=True, save=True):
    """Extract projection files from file experimental file formats.
    Then apply normalization, minus_log, negative, nan, and infinity corrections.

    Parameters
    ----------
    datadir : str
        the base path to the Users experimental files.

    fname : str
        the name of the file the User would like to extract projections from.

    basedir : str
        the path with starting file name for the files to be saved.
        Example: /home/user/folder/proj_ims creates proj_ims_0000.tif inside this folder.


    starting : int
        the starting digit for the proj_ims files name.

    overwrite : bool
        if True then the projections files will overwrite the designated folder.

    verbose : bool
        if True this will output print statements for each section of the extraction process.

    save : bool
        if False this will return the extracted projections to the user.


    Returns
    -------
    Nothing. Only saves the corrected projection files to the designated folder.
    """

    fname = os.path.join(datadir, fname)
    basename, ext = os.path.splitext(fname)

    start_time = time.time()

    if verbose:
        print('\n** Reading data')
    prj, flat, dark, theta = extraction_func(fname)

    digits = len(str(len(prj)))

    if flat_roll != None:
        flat = np.roll(flat, flat_roll, axis=2)

    if verbose:
        print('\n** Flat field correction')
    prj = tomopy.normalize(prj, flat, dark)

    if verbose:
        print('\n** Applying minus log')
    prj = tomopy.minus_log(prj)

    if verbose:
        print('\n** Removing negative and np.nan')
    prj = tomopy.misc.corr.remove_neg(prj, val=0.001)
    prj = tomopy.misc.corr.remove_nan(prj, val=0.001)

    if verbose:
        print('\n** Removing inf')
    prj[np.where(prj == np.inf)] = 0.001

    if binning > 0:
        if verbose:
            print('\n** Down sampling data\n')
        prj = tomopy.downsample(prj, level=binning)
        prj = tomopy.downsample(prj, level=binning, axis=1)

    if save:

        if verbose:
            print('\n** Dumping tiff\n')

        dxchange.write_tiff_stack(prj[:, :, :], fname=f'{basedir}extracted/projections/proj',
                                  dtype=dtype, axis=0, digit=digits, start=starting, overwrite=overwrite)

        np.save(f'{basedir}extracted/theta/theta.npy', np.asarray(theta))

        if verbose:
            print('   done in %0.3f min' % ((time.time() - start_time)/60))

    else:
        return np.asarray(prj), np.asarray(theta)


def obtain_phantoms(files):

    data = []

    for idx, file in tqdm(enumerate(files), total=len(files), desc='Importing Data'):
        im = tif.imread(file)
        data.append(im)

    data = np.asarray(data)

    output_data = []

    for i in tqdm(range(data.shape[1]), desc="Saving Data"):
        output_data.append(data[:, i, :])

    return output_data


def save_phantom_data(basedir, prjs_dir, flat_field_dir, binning=0, starting=0,
                      dtype='float32', flat_roll=None, overwrite=True, verbose=True, save=True):
    start_time = time.time()

    prj_files = os.listdir(f"{prjs_dir}")
    prj_files = [f"{prjs_dir}{x}" for x in prj_files]

    flat_files = os.listdir(f"{flat_field_dir}")
    flat_files = [f"{flat_field_dir}{x}" for x in flat_files]

    prj = obtain_phantoms(prj_files)
    flat = obtain_phantoms(flat_files)
    dark = np.zeros(np.shape(flat))

    prj = np.asarray(prj)
    flat = np.asarray(flat)
    dark = np.asarray(dark)

    theta = tomopy.angles(prj.shape[0], 0, 180)

    digits = len(str(len(prj)))

    if flat_roll != None:
        flat = np.roll(flat, flat_roll, axis=2)

    if verbose:
        print('\n** Flat field correction')
    prj = tomopy.normalize(prj, flat, dark)

    if verbose:
        print('\n** Applying minus log')
    prj = tomopy.minus_log(prj)

    if verbose:
        print('\n** Removing np.nan')

    # Set nan values to the lowest value
    prj = tomopy.misc.corr.remove_nan(prj, val=np.nanmin(prj[prj != -np.inf]))

    if verbose:
        print('\n** Removing +inf')

    # Set infinity values to the lowest value
    prj[np.where(prj == np.inf)] = np.nanmin(prj[prj != -np.inf])

    if verbose:
        print('\n** Removing -inf')
    prj[np.where(prj == -np.inf)] = np.nanmin(prj[prj != -np.inf])

    if verbose:
        print('\n** Making positive numbers')

    # Force the projections to be >= 0
    if np.min(prj) < 0:
        prj += np.abs(np.min(prj))

    if binning > 0:
        if verbose:
            print('\n** Down sampling data\n')
        prj = tomopy.downsample(prj, level=binning)
        prj = tomopy.downsample(prj, level=binning, axis=1)

    if save:

        if verbose:
            print('\n** Dumping tiff\n')

        dxchange.write_tiff_stack(prj[:, :, :], fname=f'{basedir}extracted/projections/proj',
                                  dtype=dtype, axis=0, digit=digits, start=starting, overwrite=overwrite)

        np.save(f'{basedir}extracted/theta/theta.npy', np.asarray(theta))

        if verbose:
            print('   done in %0.3f min' % ((time.time() - start_time)/60))

    else:
        return np.asarray(prj), np.asarray(theta)
