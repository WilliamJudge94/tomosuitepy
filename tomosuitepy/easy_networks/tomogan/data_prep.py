import numpy as np
from tqdm import tqdm
from base.common import h5create_file, h5create_dataset

def format_data_tomogan(clean_data, noisy_data, interval=5, dtype=np.float32):
    """
    Formats data for TomoGAN network.

    Parameters
    ----------
    clean_data : np.array
        a numpy array of grey-scale images with shape (num_ims, im_x_dim, im_y_dim)

    noisy_data : np.array
        a numpy array of grey-scale images with shape (num_ims, im_x_dim, im_y_dim)

    interval : int
        every 'interval' number of images is placed into the training dataset
    
    dtype : np.dtype
        the data type to load the data into

    Returns
    -------
    nd.arrays
        np.arrays of order xtrain, ytrain, xtest, ytest
    """

    # Initiating the data arrays
    xtrain = []
    ytrain = []
    xtest = []
    ytest = []

    counter = 0
    # Segmenting out data
    for c_data, n_data in tqdm(zip(clean_data, noisy_data), desc='Formatting Data'):
        if counter % interval == 0:
            xtrain.append(n_data)
            ytrain.append(c_data)
        else:
            xtest.append(n_data)
            ytest.append(c_data)
        counter += 1
            
    # Forcing data into numpy arrays
    xtrain = np.asarray(xtrain, dtype=dtype)
    ytrain = np.asarray(ytrain, dtype=dtype)
    xtest = np.asarray(xtest, dtype=dtype)
    ytest = np.asarray(ytest, dtype=dtype)

    return xtrain, ytrain, xtest, ytest

def setup_data_tomogan(basedir, xtrain, ytrain, xtest, ytest, types='noise'):
    """Saves data into the proper location to be used by TomoGAN network.

    Parameters
    ----------
    basedir : str
        the path to the current project

    xtrain : np.array
        the first output from the tomosuite.tomogan.data_prep.format_data() function

    ytrain : np.array
        the second output from the tomosuite.tomogan.data_prep.format_data() function

    xtest : np.array
        the third output from the tomosuite.tomogan.data_prep.format_data() function

    ytest : np.array
        the fourth output from the tomosuite.tomogan.data_prep.format_data() function
        
    types : str
        the type of data being passed to TomoGAN. Example 'noise' or 'artifact'
    
    Returns
    -------
    None
        Saves data to the correct h5py format for TomoGAN.
    """
    
    data_shape = xtrain.shape
    
    if data_shape[1] < 384 or data_shape[2] < 384:
        raise ValueError(f'Image dimensions must be greater than 384 x 384. Current shape for xtrain is: {data_shape}. Please correct all train and test data.')

    location = f'{basedir}tomogan/'
    ident = f'tomogan_{types}_AI'

    # Creating hdf5 files to store data
    h5create_file(location, f'xtrain_{ident}')
    h5create_file(location, f'xtest_{ident}')

    h5create_file(location, f'ytrain_{ident}')
    h5create_file(location, f'ytest_{ident}')

    # Saving data to hdf5 files
    h5create_dataset(f'{location}xtrain_{ident}.h5', 'images', xtrain)
    h5create_dataset(f'{location}ytrain_{ident}.h5', 'images', ytrain)

    h5create_dataset(f'{location}xtest_{ident}.h5', 'images', xtest)
    h5create_dataset(f'{location}ytest_{ident}.h5', 'images', ytest)
