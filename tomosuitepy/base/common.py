import os
import pickle
import numpy as np
import tifffile as tif
import h5py
from tqdm import tqdm
import itk
from itkwidgets import view

def get_projection_shape(basedir):
    """
    Get a single extracted projection shape.
    
    Parameters
    ----------
    basedir : str
        The path to the project
        
    Returns
    -------
    nd.array
        The shape of the projection.
    """
    files = sorted(os.listdir(f"{basedir}extracted/projections/"))
    image = tif.imread(f"{basedir}extracted/projections/{files[0]}")
    return np.shape(image)

def loading_tiff_prj(folder):
    """
    For a given folder return all the .tiff images in a numpy array.

    Parameters
    ----------
    folder : str
        The path to the folder which contains the .tiff images.

    Returns
    -------
    nd.array
        A numpy array with the loaded .tiff images.
    """
    data = []

    files = sorted(os.listdir(folder))
    files = [f'{folder}{fil}' for fil in files]

    for file in tqdm(files, desc='Importing TIFF Projections'):
        if '.tiff' in file or '.tif' in file:
            data.append(tif.imread(file))
    return np.asarray(data)


def load_extracted_prj(basedir):
    """
    Load the extracted data completed by TomoSuitePY.

    Parameters
    ----------
    basedir : str
        The path for the expermental file.

    Returns
    -------
    nd.array
        Array containing the extracted projection.
    """
    data = loading_tiff_prj(f'{basedir}extracted/projections/')
    return data


def load_extracted_theta(basedir):
    """
    Allow the User to retrive the extracted theta angles.

    Parameters
    ----------
    basedir : str
        The path for the expermental file.

    Returns
    -------
    nd.array
        The theta angles for the extracted projections.
    """
    theta = np.load(f'{basedir}extracted/theta/theta.npy')
    return theta


def skip_lowdose(basedir):
    """
    Allow the user to skip low-dose machine learning for de-wedge machine learning.

    Save the raw projection data to /low_dose/noise.py. This function is used for
    ML inpainting and artifact removal when noise removal is not carried out.

    Parameters
    ----------
    basedir : str
        the path to the project

    Returns
    -------
    None
        Saves raw projection files into an easy to read
        numpy file (/basedir/low_dose/noise.py).
        To be used for inpainting and arifact removal
        when the User has no need to perform noise removal.
    """

    data = loading_tiff_prj(f'{basedir}extracted/projections/')
    np.save(f'{basedir}tomogan/noise_exp_data.npy', data)


def h5create_file(loc, name):
    """
    Creates hdf5 file

    Parameters
    ---------
    loc : str
        The location of the hdf5 file.
    name : str
        The name of the hdf5 file WITHOUT .h5 at the end.

    Returns
    -------
    None
        Nothing is returned. An h5py file is created.
    """

    with h5py.File('{}/{}.h5'.format(loc, name), "w") as f:
        pass


def h5grab_data(file, data_loc):
    """
    Returns the data stored in the user defined group in h5py file.

    Parameters
    ----------
    file : str
        The user defined hdf5 file.
    data_loc : str
        The group the user would like to pull data from.

    Returns
    -------
    arb
        The data stored int the user defined location in the designated
        h5py file.
    """
    with h5py.File(file, 'r') as hdf:
        data = hdf.get(data_loc)
        data = np.array(data)

    return data


def h5create_dataset(file, ds_path, ds_data):
    """
    Creates a dataset in the user defined group
    with data equal to the user defined data.

    Parameters
    ----------
    file : str
        The user defined hdf5 file.
    ds_path : str
        The group path to the dataset inside the hdf5 file.
    ds_data : nd.array
        a numpy array the user would like to store

    Returns
    -------
    None
        Nothing. The data is stored in the designated h5py file.
    """
    with h5py.File(file, 'a') as hdf:
        hdf.create_dataset(ds_path, data=ds_data)


def h5group_list(file, group_name='base'):
    """
    Displays all group members for a user defined group.

    Parameters
    ----------
    file : str
        The path to the hdf5 file.
    group_name : str
        The path to the group the user wants the Keys for.
        Set to 'base' if you want the top most group.

    Returns
    -------
    list
        A list of all the subgroups inside the user defined group.
    """
    # Parenthesis are needed - keep them
    with h5py.File(file, 'r') as hdf:
        if group_name == 'base':
            return (list(hdf.items()))
        else:
            g1 = hdf.get(group_name)
            return (list(g1.items()))


def h5delete_file(file):
    """
    Deletes the file set by the user

    Parameters
    ----------
    file : str
        the full location of the hdf5 file the user would like to delete

    Returns
    -------
    None
        removes an h5 file.
    """
    os.remove(file)


def interactive_data_viewer():
    """
    Show the commands used for viewing 3D volumes in an interactive way.

    Parameters
    ----------
    None : None
        There are no parameters

    Returns
    -------
    None
        Prints commands used for viewing 3D volumes in jupyter notebooks.
    """
    print("import itk")
    print("from itkwidgets import view")
    print("view(slcs)")


def save_metadata(basedir, meta_data2save, meta_data_type='extracted'):
    """
    Saving function metadata calls to a pkl file.

    Parameters
    ----------
    basedir : str
        The path to the current project.
    meta_data2save : dic
        A dictionary with all variables to save
    medta_data_type : str
        Designation on where to save meta data.
        Either 'extracted' or 'recon'.

    Returns
    -------
    None
        Saves function meta data to a pkl file. 
    """

    if meta_data_type == 'recon':
        meta_data_type = ''
    elif meta_data_type == 'extracted':
        meta_data_type = 'extracted/'
    with open(f'{basedir}{meta_data_type}meta_data.pkl', 'wb') as f:
        pickle.dump(meta_data2save, f, pickle.HIGHEST_PROTOCOL)


def load_metadata(basedir, meta_data_type='extracted'):
    """
    Loading function metadata from pkl file.

    Parameters
    ----------
    basedir : str
        The path to the current project.
    medta_data_type : str
        Designation on which meta data to load.
        Either 'extracted' or 'recon'.

    Returns
    -------
    dic
        A dictionary with all variables and values
        run during extraction or reconstruction.
    """
    if meta_data_type == 'recon':
        meta_data_type = ''
    elif meta_data_type == 'extracted':
        meta_data_type = 'extracted/'
    with open(f'{basedir}{meta_data_type}meta_data.pkl', 'rb') as f:
        return pickle.load(f)