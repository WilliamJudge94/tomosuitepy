import os
import pickle
import numpy as np
import tifffile as tif
import h5py
from tqdm import tqdm
import itk
from itkwidgets import view

def loading_tiff_prj(folder):
    """For a given folder return all the .tiff images in a numpy array
    
    Parameters
    ----------
    folder : str
        the path to the folder which contains the .tiff images
        
    Returns
    -------
    A numpy array with the loaded .tiff images
    """
    data = []
    
    files = sorted(os.listdir(folder))
    files = [f'{folder}{fil}' for fil in files]
    
    for file in tqdm(files, desc='Importing TIFF Projections'):
        if '.tiff' in file or '.tif' in file:
            data.append(tif.imread(file))
    return np.asarray(data)


def load_extracted_prj(basedir):
    """Allow the User to retrive the extracted projection images.
    
    Parameters
    ----------
    basedir : str
        the path for the expermental file
        
    Returns
    -------
    the projections as a np.ndarray()
    """
    data = loading_tiff_prj(f'{basedir}extracted/projections/')
    return data


def load_extracted_theta(basedir):
    """Allow the User to retrive the extracted theta angles.
    
    Parameters
    ----------
    basedir : str
        the path for the expermental file
        
    Returns
    -------
    the theta angles as a np.ndarray()
    """
    theta = np.load(f'{basedir}extracted/theta/theta.npy')
    return theta


def skip_lowdose(basedir):
    """Save the raw projection data to /low_dose/noise.py. Used for inpainting and artifact removal when noise removal is not carried out.
    
    Parameters
    ----------
    basedir : str
        the path to the project
        
    Returns
    -------
    Nothing. Saves raw projection files into an easy to ready numpy file. To be used for inpainting and arifact removal
    when the User has no need to perform noise removal.
    """

    data = loading_tiff_prj(f'{basedir}extracted/projections/')
    np.save(f'{basedir}tomogan/noise_exp_data.npy', data)
    
    
def h5create_file(loc, name):
    """Creates hdf5 file
    
    Parameters
    ---------
    loc: (str)
        the location of the hdf5 file
    name: (str)
        the name of the hdf5 file WITHOUT .h5 at the end
        
    Returns
    -------
    Nothing
    """

    with h5py.File('{}/{}.h5'.format(loc, name), "w") as f:
        pass
    
    
def h5grab_data(file, data_loc):
    """Returns the data stored in the user defined group
    
    Parameters
    ----------
    file (str):
        the user defined hdf5 file
    data_loc (str):
        the group the user would like to pull data from
        
    Returns
    -------
    the data stored int the user defined location
    """
    with h5py.File(file, 'r') as hdf:
        data = hdf.get(data_loc)
        data = np.array(data)

    return data


def h5create_dataset(file, ds_path, ds_data):
    """Creates a dataset in the user defined group with data equal to the user defined data
    
    Parameters
    ----------
    file: (str)
        the user defined hdf5 file
    ds_path: (str)
        the group path to the dataset inside the hdf5 file
    ds_data: (nd.array)
        a numpy array the user would like to store
        
    Returns
    -------
    Nothing
    """
    with h5py.File(file, 'a') as hdf:
        hdf.create_dataset(ds_path, data=ds_data)
        
        
def h5group_list(file, group_name='base'):
    """Displays all group members for a user defined group
    
    Parameters
    ----------
    file (str)
        the path to the hdf5 file
    group_name (str)
        the path to the group the user wants the Keys for. Set to 'base' if you want the top most group
        
    Returns
    -------
    a list of all the subgroups inside the user defined group
    """
    # Parenthesis are needed - keep them
    with h5py.File(file, 'r') as hdf:
        if group_name == 'base':
            return (list(hdf.items()))
        else:
            g1=hdf.get(group_name)
            return (list(g1.items()))
        
        
        
def interactive_data_viewer():
    print("import itk")
    print("from itkwidgets import view")
    print("view(slcs)")


def save_metadata(basedir, meta_data2save, meta_data_type='extracted'):
    if meta_data_type == 'recon':
        meta_data_type = ''
    with open(f'{basedir}{meta_data_type}/meta_data.pkl', 'wb') as f:
        pickle.dump( meta_data2save, f, pickle.HIGHEST_PROTOCOL)

def load_metadata(basedir, meta_data_type='extracted'):
    if meta_data_type == 'recon':
        meta_data_type = ''
    with open(f'{basedir}{meta_data_type}/meta_data.pkl', 'rb') as f:
        return pickle.load(f)
