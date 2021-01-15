import numpy as np
from tqdm import tqdm
from ...base.common import load_extracted_prj, loading_tiff_prj
from skimage.metrics import structural_similarity as ssim
from ...easy_networks.tomogan.data_prep import format_data_tomogan, setup_data_tomogan

def determine_ave_ssim(loaded_prj, split_amount):

    new_array = np.array_split(loaded_prj, split_amount)

    store = []
    for array in tqdm(new_array):
        master_im = array[0]
        mini_store = []
        for ar in array:
            mini_store.append(ssim(master_im, ar, data_range=master_im.max() - master_im.min()))
        store.append(np.mean(mini_store))
    return np.asarray(store)


def return_ssim_bound_images(loaded_prj, split_amount, threshold=None):
    
    ave_ssim = determine_ave_ssim(loaded_prj, split_amount)
    if threshold == None:
        print(ave_ssim)
        threshold = float(input('What is your threshold?'))
    new_array = np.array_split(loaded_prj, split_amount)
    finder = ave_ssim > threshold
    return_array = []
    
    for idx, value in enumerate(finder):
        if value:
            return_array.append(new_array[idx])
            
    return np.asarray(return_array, dtype=object)


def create_training_set(limited_prj):
    
    noisy_data = []
    clean_data = []
    
    for array in limited_prj:
        high_dose = np.mean(array, axis=0)
        
        for ar in array:
            noisy_data.append(ar)
            clean_data.append(high_dose)
             
    return np.array(clean_data), np.array(noisy_data)

def setup_experimental_noise_train(basedir, split_amount_exp, ssim_threshold=None, interval=2, dtype=np.float32):
    """Allows the User to set up experimental ML training with TomoGAN
    
    Parameters
    ----------
    basedir : str
        the path of the directory you would like to work in
        
    split_amount_exp : int
        the amount of duplicate angles one took during the experiment
        
    ssim_threshold : float
        a demical value that throws out data which doesnt meet the required image similarity test
        if None an input prompt will show all SSIM values and allow the User to pick the threshold
        
    interval : int
        every interval amount of images will be used for test data
        
    
    Returns
    -------
    Nothing. Saves training data in appropriate format for TomoGAN
    """
    loaded_prj = load_extracted_prj(basedir)
    new_array = return_ssim_bound_images(loaded_prj, split_amount_exp, ssim_threshold)
    clean_data, noisy_data = create_training_set(new_array)
    

    xtrain, ytrain, xtest, ytest = format_data_tomogan(clean_data=clean_data,
                                                       noisy_data=noisy_data,
                                                       interval=interval,
                                                       dtype=dtype)

    setup_data_tomogan(basedir=basedir,
                       xtrain=xtrain,
                       ytrain=ytrain,
                       xtest=xtest,
                       ytest=ytest,
                       types='noise')
    


    #save_tomogan_training(basedir, output, split_amount_ml)
    
def setup_experimental_noise_predict(basedir):
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