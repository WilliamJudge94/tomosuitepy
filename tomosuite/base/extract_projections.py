import os
import time
import tomopy
import dxchange
import numpy as np
from tqdm import tqdm
import tifffile as tif


def pre_process_prj(prj, flat, dark, flat_roll, outlier_diff, outlier_size, air,
                custom_dataprep, binning, bkg_norm, chunk_size4bkg, verbose, force_positive, removal_val):
    """Preprocesses the projections data to be saves as .tif images
    
    Parameters
    ----------
    All variables defined in the extract() function
    
    Returns
    -------
    Pre-processed projection data
    """
    

    # Lets the User roll the flat field image
    if flat_roll != None:
        flat = np.roll(flat, flat_roll, axis=2)
        
    # Allows the User to remove outliers
    if outlier_diff != None and outlier_size != None:
        if verbose:
            print('\n** Remove Outliers')

        prj = tomopy.misc.corr.remove_outlier(prj, outlier_diff, size=outlier_size, axis=0)
        flat = tomopy.misc.corr.remove_outlier(flat, outlier_diff, size=outlier_size, axis=0)
        
    # Normalized the projections
    if verbose:
        print('\n** Flat field correction')  
    prj = tomopy.normalize(prj, flat, dark)
    
    # Trying to figure out how to incorporate this
    if not custom_dataprep:
        # Work in Progress
        if False:
            z = 33
            eng = 60
            pxl = 0.65e-4
            rat = 1.25e-03
            zinger_level = 200
            
            prj_chunks = []
            
            for prj_chunk in tqdm(np.array_split(prj, chunk_size4bkg), desc='Retrieve Phase'):
                prj_tmp = tomopy.prep.phase.retrieve_phase(prj_chunk, pixel_size=pxl, dist=z, energy=eng, alpha=rat, pad=True, ncore=1)
                prj_chunks.append(prj_tmp.copy())
                del prj_tmp

            prj = np.concatenate(prj_chunks)
        
        # Apply a background normalization to the projections
        if bkg_norm:

            prj_chunks = []
            for prj_chunk in tqdm(np.array_split(prj, chunk_size4bkg), desc='Bkg Normalize'):
                prj_tmp = tomopy.prep.normalize.normalize_bg(prj_chunk, air=air, ncore=1)
                prj_chunks.append(prj_tmp.copy())
                del prj_tmp

            prj = np.concatenate(prj_chunks)
            
        if verbose:
            print('\n** Applying minus log')

        prj = tomopy.minus_log(prj)
        
        # Old version of correcting projections
        if False:
            # Set nan values to the lowest value
            prj = tomopy.misc.corr.remove_nan(prj, val=np.nanmin(prj[prj != -np.inf]))
            # Set + infinity values to the highest value
            prj[np.where(prj == np.inf)] = np.nanmax(prj[prj != np.inf])
            # Set - infinity values to the lowest value
            prj[np.where(prj == -np.inf)] = np.nanmin(prj[prj != -np.inf])
        
        if verbose:
            print('\n** Removing neg')
            
        prj = tomopy.misc.corr.remove_neg(prj, val=removal_val)
        
        if verbose:
            print('\n** Removing np.nan')
    
        prj = tomopy.misc.corr.remove_nan(prj, val=removal_val)

        if verbose:
            print('\n** Removing np.inf')
   
        prj[np.where(prj == np.inf)] = removal_val

        
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
        
    else:
        if verbose:
            print('\n** Not applying data manipulation after tomopy.normalize - Except for downsampling')

    # Bin the data
    if binning>0:
        if verbose:
            print('\n** Down sampling data\n')
        prj = tomopy.downsample(prj, level=binning)
        prj = tomopy.downsample(prj, level=binning, axis=1)
        
        
    return prj
    

def extract(datadir, fname, basedir,
            extraction_func=dxchange.read_aps_32id, binning=1,
            outlier_diff=None, air=10, outlier_size=None,
            starting=0, bkg_norm=False, chunk_size4bkg=10, force_positive=True, removal_val=0.001, 
            custom_dataprep=False, dtype='float32', flat_roll=None,
            overwrite=True, verbose=True, save=True):
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
        
    air : int
        Number of pixels at each boundary to calculate the scaling factor. Used during the outlier removal step.
        
    starting : int
        the starting digit for the proj_ims files name.
        
    bkg_norm : bool
        If True then a background normalization is applied through TomoPy.
        
    chunk_size4bkg : int
        The background normalization is RAM intensive. This allows the User to chunk the normalization process.
    
    custom_dataprep : bool
        if True this allows the User to define dataprep functions in the reconstruction script. Output stops after the initial flat/dark field normalization.
    
    dtype : str
        The data type to save the data as.
        
    flat_roll : int or None
        Move the flat field over this many pixles before applying normalization. If None this step is skipped.
        
    overwrite : bool
        if True then the projections files will overwrite the designated folder.
        
    verbose : bool
        if True this will output print statements for each section of the extraction process.
    
    save : bool
        if False this will return the extracted projections to the user in the form of a numpy array.
        
    
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

    prj = pre_process_prj(prj=prj, flat=flat, dark=dark, flat_roll=flat_roll,
                      outlier_diff=outlier_diff, outlier_size=outlier_size,
                      air=air, custom_dataprep=custom_dataprep, binning=binning,
                      bkg_norm=bkg_norm, chunk_size4bkg=chunk_size4bkg, verbose=verbose,
                      force_positive=force_positive, removal_val=removal_val)
        
    if save:

        if verbose:
            print('\n** Dumping tiff\n')

        dxchange.write_tiff_stack(prj[:,:,:], fname=f'{basedir}extracted/projections/proj',
                                  dtype=dtype, axis=0, digit=digits, start=starting, overwrite=overwrite)

        np.save(f'{basedir}extracted/theta/theta.npy', np.asarray(theta))

        if verbose:
            print('   done in %0.3f min' % ((time.time() - start_time)/60))
            
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
    
