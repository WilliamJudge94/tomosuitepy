import os
import sys
import warnings
import numpy as np
from tqdm import tqdm
import sys, os, time, argparse, shutil, scipy, h5py, glob

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf 
    from util import save2img, save2img_tb
    from models import unet as make_generator_model
    from models import tomogan_disc as make_discriminator_model
    from data_processor import bkgdGen, gen_train_batch_bg, get1batch4test
    tf.enable_eager_execution()

sys.path.append(os.path.dirname(__file__))
path1 = os.path.dirname(__file__)
path2 = '/'.join(path1.split('/')[:-2])
vgg19_path = f'{path2}/hard_networks/TomoGAN/vgg19_weights_notop.h5'


def chunker(start, end, chunk_size):
    r = np.arange(start, end)
    store = []
    mini_store = []

    for idx, i in enumerate(r):
        if idx == 0:
            mini_store.append(i)
            
        elif idx % chunk_size != 0:
            mini_store.append(i)
            
        elif idx % chunk_size == 0:
            store.append(mini_store)
            mini_store = []
            mini_store.append(i)
            
        if idx == len(r)-1:
            store.append(mini_store)
            
    return store

def predict_tomogan(basedir,
                    data,
                    weights_iter,
                    chunk_size=5,
                    gpu='0', 
                    lunet=3,
                    in_depth=1,
                    data_type=np.float32,
                    verbose=False,
                    types='noise'):

    """Predict new images based on trained TomoGAN network
    
    Parameters
    ----------
    basedir : str
        the path to the current project
        
    data : np.array
        the data to predict
        
    weights_iter : int
        the epoch number of the trained TomoGAN network to do found in the basedir
        
    chunk_size : int
        chunk the data as to not OOM the GPU
        
    gpu : str
        to be passed to os.environ['CUDA_VISIBLE_DEVICES'] = 
        
    lunet : int
        keep the same as when training TomoGAN
        
    in_depth : int
        amount of images to take from the data and concatonate them. Should stay at a value of 1
        
    data_type : np.datatype
        the data dtype to keep the data
        
    verbose : bool
        outputs more useful text
        
    types : str
        incase TomoGAN is being used for 'noise' or 'artifact' removal
        
    Returns
    -------
    clean_data array, input_data array
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    if verbose:
        print('Starting Make Generator')

    generator = make_generator_model(input_shape=(None, None, in_depth), nlayers=int(lunet))
    generator.load_weights(f'{basedir}tomogan/{types}_experiment-itrOut/{types}_experiment-it{weights_iter}.h5')

    if verbose:
        print('Loading Data')

    def get1batch4test(X, in_depth, idx):
        batch_X = np.array([np.transpose(X[s_idx : (s_idx+in_depth)], (1, 2, 0)) for s_idx in idx])
        return batch_X.astype(data_type)

    if verbose:
        print('Predicting Data')
        
    chunks = chunker(0, len(data), chunk_size)
    output_store = []

    noise_data = data

    for c in tqdm(chunks):
        X_data = get1batch4test(noise_data, 1, idx = c)
        output = generator.predict(X_data)
        for ar in output:
            output_store.append(ar)
    output_store = np.asarray(output_store)
    output_store = np.asarray(output_store[:,:,:,0])
    
    return output_store, data


def save_predict_tomogan(basedir, good_data, bad_data, second_basedir=None, types='noise'):
    """Save the predicted Denoised Images from TomoGAN network.
    
    Parameters
    ----------
    basedir : str
        the path to the project associated to the trained TomoGAN network
        
    good_data : np.array
        the numpy array which is associated with the denoised data
        
    bad_data : np.array
        the numpy array which is associated with the noisy data
        
    second_basedir : str
        the path to the project associated to the noisy data project. If None this defaults to the basedir path
        
    types : str
        the type of data being passed to TomoGAN. Example 'noise' or 'artifact'
    
    Returns
    -------
    Nothing. Saves data to either the basedir or the second_basedir under tomogan/denoise_ _data.npy or tomogan/noise_exp_data.npy.
    """
    
    if second_basedir is None:
        np.save(f'{basedir}tomogan/de{types}_fake_data.npy', good_data)
        np.save(f'{basedir}tomogan/{types}_exp_data.npy', bad_data)
        
    else:
        np.save(f'{second_basedir}tomogan/de{types}_exp_data.npy', good_data)
        np.save(f'{second_basedir}tomogan/{types}_exp_data.npy', bad_data)