import os
import time
import tomopy
import dxchange
import numpy as np

def extract(datadir, fname, basedir, extraction_func=dxchange.read_aps_32id, binning=1, starting=0, dtype='float32', flat_roll=None, overwrite=True, verbose=True, save=True):
    """Extract projection files from file experimental file formats. Then apply normalization, minus_log, negative, nan, and infinity corrections.
    
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
