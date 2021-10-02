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

muppy_amount = 100000

def cache_clearing_downsample(data, binning):
    data = tomopy.downsample(data, level=binning)
    data = tomopy.downsample(data, level=binning, axis=1)
    return data


def save_prj_ds_chunk(data, iteration, path):
    np.save(f'{path}/tomsuitepy_downsample_save_it_{str(iteration).zfill(4)}.npy', data)
    #print(f'saving - {path}/tomsuitepy_downsample_save_it_{str(iteration).zfill(4)}.npy')

    
    
def load_prjs_norm_chunk(iterations, path, img_shape, dtypes):
    
    chunk_amount = img_shape[0]
    
    prj_store = np.zeros((chunk_amount * iterations, img_shape[1], img_shape[2]))
    prj_store = prj_store.astype(dtypes)

    for idx in tqdm(range(0, iterations), desc='Reading Chunked Data'):
        first = idx * chunk_amount
        second = (idx + 1) * chunk_amount
        var = np.load(f'{path}/tomsuitepy_downsample_save_it_{str(idx).zfill(4)}.npy') 
        
        prj_store[first: second] = var.copy()
        del var
        
    return prj_store



def load_prj_ds_chunk(iterations, path):
    
    data = []

    for it in tqdm(range(0, iterations), desc='Reading Chunked Data'):
        data.append(np.load(f'{path}/tomsuitepy_downsample_save_it_{str(it).zfill(4)}.npy'))
        #print(f'loading - {path}/tomsuitepy_downsample_save_it_{str(it).zfill(4)}.npy')
        
    data = np.asarray(data)
    data = np.concatenate(data)
    return data


def remove_saved_prj_ds_chunk(iterations, path):
    for it in range(0, iterations):
        #print(f'removing - {path}/tomsuitepy_downsample_save_it_{str(it).zfill(4)}.npy')
        os.remove(f'{path}/tomsuitepy_downsample_save_it_{str(it).zfill(4)}.npy')
        

def flat_roll_func(flat, flat_roll):
    if flat_roll != None:
        flat = np.roll(flat, flat_roll, axis=2)
    return flat

def outlier_diff_func(prj, flat, outlier_diff, outlier_size, verbose):
    if outlier_diff != None and outlier_size != None:
        if verbose:
            print('\n** Remove Outliers')
        prj = tomopy.misc.corr.remove_outlier(prj, outlier_diff, size=outlier_size, axis=0)
        flat = tomopy.misc.corr.remove_outlier(flat, outlier_diff, size=outlier_size, axis=0)
    return prj, flat

def flat_field_corr_func(prj, flat, dark, chunking_size, normalize_ncore, muppy_amount, dtype, verbose):

    iteration = 0 
    prj_chunk_shape = 0

    if verbose:
        print('\n** Flat field correction')  
        
    if chunking_size > 1:

        path_chunker = pathlib.Path('.').absolute()

        for prj_ds_chunk in tqdm(np.array_split(prj, chunking_size), desc='Flat Field Correction - Chunked'):
            
            prj_ds_chunk = tomopy.normalize(prj_ds_chunk, flat, dark, ncore=normalize_ncore)
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

    path_chunker = pathlib.Path('.').absolute()

    prj = load_prjs_norm_chunk(iteration, path_chunker, prj_chunk_shape, dtype)
    remove_saved_prj_ds_chunk(iteration, path_chunker)
    
    all_objects = muppy.get_objects()[:muppy_amount]
    sum1 = summary.summarize(all_objects)

    return prj



def bkg_norm_func(bkg_norm, prj, chunking_size, air):

    if bkg_norm:

        prj_chunks = []
        for prj_chunk in tqdm(np.array_split(prj, chunking_size), desc='Bkg Normalize'):
            prj_tmp = tomopy.prep.normalize.normalize_bg(prj_chunk, air=air, ncore=1)
            prj_chunks.append(prj_tmp.copy())
            del prj_tmp

        prj = np.concatenate(prj_chunks)

    return prj

def correct_norma_extremes_func(correct_norma_extremes, verbose, prj):
    if correct_norma_extremes:
        if verbose:
            print(f'\n** Normalization pre-log correction - Min: {prj.min()} - Max: {prj.max()}')

        prj += np.abs(prj.min())
        prj += prj.max() * 0.0001

        if verbose:
            print(f'\n** After Normalization Min: {prj.min()} - Max: {prj.max()}')

    return prj

def minus_log_func(minus_log, verbose, prj, muppy_amount, chunking_size):

    if minus_log:
        if verbose:
            print('\n** Applying minus log')

        try:
            all_objects = muppy.get_objects()[:muppy_amount]
            sum1 = summary.summarize(all_objects)

        except Exception as ex:
            raise ValueError(f"\n** Failed to initiate muppy RAM collection - Error: {ex}")

        if chunking_size > 1:

            path_chunker = pathlib.Path('.').absolute()
            iteration = 0
            
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

    return prj, iteration, prj_chunk_shape


def minus_log_load_func(iteration, muppy_amount, img_shape, dtype):

    path_chunker = pathlib.Path('.').absolute()
l   
    prj = load_prjs_norm_chunk(iteration, path_chunker, img_shape, dtype)
    remove_saved_prj_ds_chunk(iteration, path_chunker)

    all_objects = muppy.get_objects()[:muppy_amount]
    sum1 = summary.summarize(all_objects)

    return prj


def neg_nan_inf_func(prj, verbose, remove_neg_vals, remove_nan_vals, remove_inf_vals, removal_val):

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

    if force_positive:
        if verbose:
            print('\n** Making positive numbers')

        # Force the projections to be >= 0
        if np.min(prj) < 0:
            prj += np.abs(np.min(prj))
            
    elif not force_positive:
        minimum_prj = np.min(prj)
        if minimum_prj < 0:
            print(f"Your lowest projection value is negative ({minimum_prj}). This may result in undesireable machine learning outputs. To change this set force_positive=True.")

    return prj

def downsample_func(binning, verbose, muppy_amount, chunking_size, prj):

    if binning>0:
        if verbose:
            print('\n** Down sampling data')

        try:
            all_objects = muppy.get_objects()[:muppy_amount]
            sum1 = summary.summarize(all_objects)

        except Exception as ex:
            raise ValueError(f"\n** Failed to initiate muppy RAM collection - Error: {ex}")

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

        
        
def pre_pre_process_prj(prj, flat, dark, flat_roll, outlier_diff,
                         outlier_size, verbose, chunking_size, normalize_ncore, dtype):
    
    # Lets the User roll the flat field image
    flat = flat_roll_func(flat, flat_roll)

    # Allows the User to remove outliers
    prj, flat = outlier_diff_func(prj, flat, outlier_diff, outlier_size, verbose)
        
    # Normalized the projections - Hella Memory
    prj = flat_field_corr_func(prj, flat, dark, chunking_size, normalize_ncore, muppy_amount, dtype)

def pre_process_prj(prj,
                    flat,
                    dark,
                    flat_roll,
                    outlier_diff,
                    outlier_size,
                    air,
                    custom_dataprep,
                    binning,
                    bkg_norm,
                    chunking_size,
                    verbose,
                    minus_log,
                    force_positive,
                    removal_val,
                    remove_neg_vals,
                    remove_nan_vals,
                    remove_inf_vals, 
                    correct_norma_extremes,
                    normalize_ncore, dtype):

    """Preprocesses the projections data to be saves as .tif images
    
    Parameters
    ----------
    All variables defined in the extract() function
    
    Returns
    -------
    Pre-processed projection data
    """
            
    # Trying to figure out how to incorporate this
    if not custom_dataprep:
        
        # Apply a background normalization to the projections
        prj = bkg_norm_func(bkg_norm, prj, chunking_size, air)

        prj = correct_norma_extremes_func(correct_norma_extremes, verbose, prj)
                
        prj = minus_log_func(minus_log, verbose, prj)
        
        prj = neg_nan_inf_func(prj, verbose, remove_neg_vals, remove_nan_vals, remove_inf_vals, removal_val)

        prj = force_positive_func(force_positive, verbose, prj)

    else:
        if verbose:
            print('\n** Not applying data manipulation after tomopy.normalize - Except for downsampling')


    # Bin the data
    prj = downsample_func(binning, verbose, muppy_amount, chunking_size, prj)
        
    return prj
    

def extract(datadir, fname, basedir,
            extraction_func=dxchange.read_aps_32id, binning=1,
            outlier_diff=None, air=10, outlier_size=None,
            starting=0, bkg_norm=False, chunking_size=10, force_positive=True, removal_val=0.001, 
            custom_dataprep=False, dtype='float32', flat_roll=None,
            overwrite=True, verbose=True, save=True, minus_log=True,
            remove_neg_vals=False, remove_nan_vals=False, remove_inf_vals=False,
            correct_norma_extremes=False, normalize_ncore=None):

            
    """Extract projection files from file experimental file formats. Allows User to not apply corrections after normalization.
    
    Parameters
    ----------
    datadir : str
        the base path to the Users experimental files.
    
    fname : str
        the name of the file the User would like to extract projections from.
        
    basedir : str
        the path with starting file name for the files to be saved. Example: /home/user/folder/proj_ims creates proj_ims_0000.tif inside this folder.
        
    extraction_fun : func
        the function from dxchange the User would like to use to extract the prj, flat, and dark field projections.
        
    binning : int
        an integer used to determine the downsampling rate of the data.
        
    outlier_diff : int or None
        Expected difference value between outlier value and the median value of the array. If None this step is ignored.
        
    outlier_size : int or None
        Size of the median filter. If None this step is ignored.
        
    starting : int
        the starting digit for the proj_ims files name.
        
    bkg_norm : bool
        If True then a background normalization is applied through TomoPy.
        
    air : int
        Number of pixels at each boundary to calculate the scaling factor. Used during the bkg_norm step.
        
    chunk_size4bkg : int
        The background normalization is RAM intensive. This allows the User to chunk the normalization process.
            
            
    custom_dataprep : bool
        if True this allows the User to define dataprep functions in the reconstruction script. Output stops after the initial flat/dark field normalization.
    
    dtype : str
        The data type to save the data as.
        
    force_positive : bool
        Force the data to be positive after all data prep but before downsampling
        
    removal_val : float
        Value to be passed into the remove neg, nan, inf, parameters
        
    flat_roll : int or None
        Move the flat field over this many pixles before applying normalization. If None this step is skipped.
        
    overwrite : bool
        if True then the projections files will overwrite the designated folder.
        
    verbose : bool
        if True this will output print statements for each section of the extraction process.
    
    save : bool
        if False this will return the extracted projections to the user in the form of a numpy array.
        
    minus_log : bool
        Allow the program to apply a minus log to the data.
        
    remove_neg - nan - inf_values : bool
        If True it will remove theses values iwht the removal_val
        
    correct_norma_extremes : bool
        If True it will try to set the data to be positive values before applying the -log()
        
    chunk_size4downsample : int
        The amount of chunking steps for downsampling.
    
    Returns
    -------
    Nothing. Only saves the corrected projection files to the designated folder. Unless save=False.
    Then it will return a numpy array with the projections.
    """

    # Determine the location of the hdf5 file
    fname = os.path.join(datadir, fname)
    basename, ext = os.path.splitext(fname)

    start_time = time.time()

    # extract the data from the file
    if verbose:
        print('\n** Reading data')
    prj, flat, dark, theta = extraction_func(fname)
    
    # Determine how many leading zeros there should be
    digits = len(str(len(prj)))

    flat = flat_roll_func(flat, flat_roll)

    # Allows the User to remove outliers
    prj, flat = outlier_diff_func(prj, flat, outlier_diff, outlier_size, verbose)
        
    # Normalized the projections - Hella Memory
    prj, iteration, prj_chunk_shape = flat_field_corr_func(prj, flat, dark,
                                                        chunking_size, normalize_ncore,
                                                        muppy_amount, dtype, verbose)


    if chunking_size > 1:
        del prj
        all_objects = muppy.get_objects()[:muppy_amount]
        sum1 = summary.summarize(all_objects)
        time.sleep(2)
        prj = flat_field_load_func(iteration, prj_chunk_shape, dtype, muppy_amount)


    all_objects = muppy.get_objects()[:muppy_amount]
    sum1 = summary.summarize(all_objects)
    time.sleep(2)

    prj = prj.astype(dtype)
    
    if not custom_dataprep:
        
        # Apply a background normalization to the projections
        prj = bkg_norm_func(bkg_norm, prj, chunking_size, air)

        prj = correct_norma_extremes_func(correct_norma_extremes, verbose, prj)

        prj, iteration, prj_chunk_shape = minus_log_func(minus_log, verbose, prj, muppy_amount, chunking_size)

        if chunking_size > 1:
            del prj
            all_objects = muppy.get_objects()[:muppy_amount]
            sum1 = summary.summarize(all_objects)
            time.sleep(2)
            prj = minus_log_load_func(iteration, muppy_amount, prj_chunk_shape, dtype)

        prj = neg_nan_inf_func(prj, verbose, remove_neg_vals, remove_nan_vals, remove_inf_vals, removal_val)

        prj = force_positive_func(force_positive, verbose, prj)

    else:
        if verbose:
            print('\n** Not applying data manipulation after tomopy.normalize - Except for downsampling')

    # Bin the data
    prj = downsample_func(binning, verbose, muppy_amount, chunking_size, prj)
        
    if save:

        if verbose:
            print('\n** Dumping tiff\n')

        dxchange.write_tiff_stack(prj[:,:,:], fname=f'{basedir}extracted/projections/proj',
                                  dtype=dtype, axis=0, digit=digits, start=starting, overwrite=overwrite)

        np.save(f'{basedir}extracted/theta/theta.npy', np.asarray(theta))

        if verbose:
            print('   done in %0.3f min' % ((time.time() - start_time)/60))
            
        del prj
        del theta
            
    else:
        print('   done in %0.3f min' % ((time.time() - start_time)/60))
        return np.asarray(prj), np.asarray(theta)

    
    
def extract_phantom(datadir, fname, basedir, starting=0, dtype='float32', flat_roll=None, overwrite=True, verbose=True, save=True):
    """Extract projection files from file experimental file formats. Then apply normalization, minus_log, negative, nan, and infinity corrections.
    
    Parameters
    ----------
    datadir : str
        the base path to the Users experimental files.
    
    fname : str
        the name of the file the User would like to extract projections from.
        
    basedir : str
        the path with starting file name for the files to be saved. Example: /home/user/folder/proj_ims creates proj_ims_0000.tif inside this folder.

        
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

    if binning>0:
        if verbose:
            print('\n** Down sampling data\n')
        prj = tomopy.downsample(prj, level=binning)
        prj = tomopy.downsample(prj, level=binning, axis=1)
        
    if save:

        if verbose:
            print('\n** Dumping tiff\n')

        dxchange.write_tiff_stack(prj[:,:,:], fname=f'{basedir}extracted/projections/proj',
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
        
        
    
def save_phantom_data(basedir, prjs_dir, flat_field_dir, binning=0, starting=0, dtype='float32', flat_roll=None, overwrite=True, verbose=True, save=True):
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


    if binning>0:
        if verbose:
            print('\n** Down sampling data\n')
        prj = tomopy.downsample(prj, level=binning)
        prj = tomopy.downsample(prj, level=binning, axis=1)
        
    if save:

        if verbose:
            print('\n** Dumping tiff\n')

        dxchange.write_tiff_stack(prj[:,:,:], fname=f'{basedir}extracted/projections/proj',
                                  dtype=dtype, axis=0, digit=digits, start=starting, overwrite=overwrite)

        np.save(f'{basedir}extracted/theta/theta.npy', np.asarray(theta))

        if verbose:
            print('   done in %0.3f min' % ((time.time() - start_time)/60))
            
    else:
        return np.asarray(prj), np.asarray(theta)    
    
