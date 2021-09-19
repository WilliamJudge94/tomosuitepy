import os, shutil
import imageio
import numpy as np
from tqdm import tqdm
import tifffile as tif


def format_data_noise2noise(datasets=[], dtype=np.float32):
    datasets = np.asarray(datasets, dtype=dtype)

    output_datasets = []
    for dataset in datasets:
        if np.min(dataset) < 0:
            dataset += np.abs(np.min(dataset))
            
        # Scale the slices to 255.0
        dataset /= np.max(dataset)
        dataset *= 255.0
        output_datasets.append(dataset)
        
    return output_datasets

def easy_save_noise2noise(output_path, dataset, im_type):
    
    total_length = len(dataset)
    zfill_length = len(str(total_length))
    

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)
        os.mkdir(output_path)

    for idx, data in enumerate(dataset):

        if im_type == 'png':
            imageio.imsave(f'{output_path}{str(idx).zfill(zfill_length)}.png', data)

        elif im_type == 'tif':
            im2save =  np.dstack((data, data, data))
            tif.imsave(f'{output_path}{str(idx).zfill(zfill_length)}.tif', im2save)

        else:
            raise Warning("im_type must be 'tif' or 'png'")

def setup_data_noise2noise(basedir, val_name, val_crop=10, datasets=[], names=[], im_type='tif'):
    val_store = {}
    
    for dataset, name in tqdm(zip(datasets, names), total=len(names), desc='Saving Datasets'):
        
        output_path = f'{basedir}noise2noise/{name}_recon/'
        
        easy_save_noise2noise(output_path, dataset, im_type)
        
        if name == val_name:
            output_path_val = f'{basedir}noise2noise/{name}_val_recon/'
            easy_save_noise2noise(output_path_val, dataset[::val_crop], im_type)
            slc_nums = np.arange(0, len(dataset))[::val_crop]
            
            
               
    try:
        test = output_path_val
        print(f'Validation Data Created For {val_name}')
        return slc_nums
    except:
        print(f'Validation Data NOT Created For {val_name}')

