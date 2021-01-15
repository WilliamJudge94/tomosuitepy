import os
import cv2
import tomopy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.ndimage import median_filter
from ..base.common import loading_tiff_prj
from mpl_toolkits.axes_grid1 import make_axes_locatable


def colorbar(mappable, font_size=12):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=font_size)
    return fig.colorbar(mappable, cax=cax, )


def tomo_recon(prj, theta, rot_center):
    types='gridrec'
    
    #prj = tomopy.remove_stripe_ti(prj, 2)
    
    if types == 'gridrec':
        #recon = tomopy.recon(prj, theta, center=rot_center, algorithm='gridrec', ncore=16, filter_name='parzen')
        recon = tomopy.recon(prj, theta, center=rot_center, algorithm='gridrec', ncore=16)
        recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
        
    elif types == 'sirt':
        extra_options ={'MinConstraint':0}
        options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options': extra_options}
        recon = tomopy.recon(prj, theta, center=rot_center, algorithm=tomopy.astra, ncore=1, options=options)
    
    #Remove ring artifacts, this comes with a slight resolution cost
    #recon = tomopy.remove_ring(recon, center_x=None, center_y=None, thresh=300.0)
    
    return recon


def reconstruct_single_slice(prj_data, theta, rows=(604, 606), rot_center=True, med_filter=False, all_data_med_filter=False, kernel=(1, 3, 3), reconstruct_func=tomo_recon, recon_type='standard'):
    
    if med_filter and all_data_med_filter:
        print('Med Filter Applied Before')
        prj_data = median_filter(prj_data, size = kernel)

    
    if recon_type == 'standard':
        prj = prj_data[:, rows]
        
    elif recon_type == 'deepfillv2':
        prj = prj_data
    
    
    if med_filter and not all_data_med_filter:
        print('Med Filter Applied After')
        prj = median_filter(prj, size = kernel)


    recon = reconstruct_func(prj, theta, rot_center=rot_center)
    
    return recon


def prepare_deepfillv2(basedir, checkpoint_num, start_row, end_row):
    
    prj_data = np.load(f'{basedir}deepfillv2/predictions/{checkpoint_num}/ml_inpaint_data.npy')
    prj_data_shape = prj_data.shape
    
    prj_data = np.moveaxis(prj_data, 0, 1)
    theta = tomopy.angles(prj_data_shape[1], 0, 180)
    
    if start_row == None:
        start = 0
    else:
        start = start_row
        
    if end_row == None:
        end = prj_data_shape[0]
    else:
        end = end_row
        
    prj_data = prj_data[:, start:end, :]
    
    return start, end, prj_data, theta


def prepare_tomogan(basedir, types, second_basedir, wedge_removal,
                    sparse_angle_removal, start_row, end_row):
    
    prj_data = np.load(f'{basedir}tomogan/{types}_data.npy')
    
    if second_basedir is None:
        theta = np.load(f'{basedir}extracted/theta/theta.npy')
    else:
        theta = np.load(f'{second_basedir}extracted/theta/theta.npy')

    shape = prj_data.shape[0]

    prj_data = prj_data[wedge_removal:shape - wedge_removal, :, :]
    theta = theta[wedge_removal:shape - wedge_removal]

    prj_data = prj_data[::sparse_angle_removal]
    theta = theta[::sparse_angle_removal]
    
    if start_row == None:
        start = 0
    else:
        start = start_row
        
    if end_row == None:
        end = shape
    else:
        end = end_row
        
    return start, end, prj_data, theta


def deal_with_sparse_angle(prj_data, theta,
                           sparse_angle_removal,
                           double_sparse=None):
    
    if sparse_angle_removal != 1:
        new_prj =[]
        new_theta = []
        
        for idx, image in enumerate(prj_data):
            if idx % sparse_angle_removal == 0:
                new_prj.append(prj_data[idx])
                new_theta.append(theta[idx])

        prj_data = np.asarray(new_prj)
        theta = np.asarray(new_theta)
        
        if double_sparse is not None:
            new_prj2 =[]
            new_theta2 = []
        
            for idx, image in enumerate(prj_data):
                if idx % double_sparse == 0:
                    new_prj2.append(prj_data[idx])
                    new_theta2.append(theta[idx])
                
            prj_data = np.asarray(new_prj2)
            theta = np.asarray(new_theta2)
            
    
        print(f'Prj Data Shape {np.shape(prj_data)} --- Theta Shape {np.shape(theta)}')
        
    return prj_data, theta

def prepare_base(basedir, wedge_removal, sparse_angle_removal, start_row, end_row, double_sparse=None):
    
    prj_data = loading_tiff_prj(f'{basedir}extracted/projections/')
    theta = np.load(f'{basedir}extracted/theta/theta.npy')

    shape = prj_data.shape[0]

    prj_data = prj_data[wedge_removal:shape - wedge_removal, :, :]
    theta = theta[wedge_removal:shape - wedge_removal]
    
    prj_data, theta = deal_with_sparse_angle(prj_data, theta,
                                             sparse_angle_removal,
                                             double_sparse)
    
    if start_row == None:
        start = 0
    else:
        start = start_row
        
    if end_row == None:
        end = shape
    else:
        end = end_row

    return start, end, prj_data, theta



def prepare_dain(basedir, start_row, end_row):
    frames_loc = f'{basedir}output_frames/'
    
    files = os.listdir(frames_loc)
    files = sorted(files)
    new_files = [f'{frames_loc}{x}' for x in files]
    
    prj_data = []
    
    for file in tqdm(new_files):
        original = cv2.imread(file)
        grayscale = rgb2gray(original)
        prj_data.append(grayscale)
        
    prj_data = np.asarray(prj_data, dtype=np.float32)
        
    theta = tomopy.angles(prj_data.shape[0], 0, 180)
    
    if start_row == None:
        start = 0
    else:
        start = start_row
        
    if end_row == None:
        end = shape
    else:
        end = end_row
    
    
    return start, end, prj_data, theta


def reconstruct_data(basedir,
                     rot_center,
                     start_row=None,
                     end_row=None,
                     med_filter=False,
                     all_data_med_filter=False,
                     med_filter_kernel=(1, 3, 3),
                     reconstruct_func=tomo_recon,
                     network=None,
                     wedge_removal=0,
                     sparse_angle_removal=1,
                     types='denoise',
                     second_basedir=None,
                     checkpoint_num=None,
                     double_sparse=None):

    if network == None:
        recon_type = 'standard'
        start, end, prj_data, theta = prepare_base(basedir, wedge_removal,
                                                   sparse_angle_removal,
                                                   start_row, end_row, double_sparse)
        
    elif network == 'tomogan':
        recon_type = 'standard'
        start, end, prj_data, theta = prepare_tomogan(basedir, types,
                                                      second_basedir,
                                                      wedge_removal,
                                                      sparse_angle_removal,
                                                      start_row, end_row)
        
    elif network == 'deepfillv2':
        recon_type = 'deepfillv2'
        assert checkpoint_num != None
        start, end, prj_data, theta = prepare_deepfillv2(basedir, checkpoint_num, start_row, end_row)
        
        
    elif network == 'dain':
        recon_type = 'standard'
        start, end, prj_data, theta = prepare_dain(basedir, start_row, end_row)

    slc_proj = reconstruct_single_slice(prj_data.copy(), 
                                           theta,
                                           rot_center = rot_center, 
                                           rows=slice(start, end), 
                                           med_filter=med_filter, 
                                           all_data_med_filter=all_data_med_filter, 
                                           kernel=med_filter_kernel, 
                                           reconstruct_func=reconstruct_func,
                                           recon_type=recon_type)
    
    
    return slc_proj



def plot_reconstruction(slc_proj, figsize=(15, 15), clim=(0, 0.003)):
    """Allow the User to plot the data that was output from the reconstruction
    
    Parameters
    ----------
    slc_proj : nd.array
        the output from the tomosuite.base.reconstruc.reconstruct_data() function
    
    figsize : list
        the figsize to be passed to plt.figure()
        
    clim : list
        clim lower and upper limits to be passed into plt.clim()
        
    Returns
    -------
    Shows the plots for the given input data
    """
    for row, prj in enumerate(slc_proj):
        fig = plt.figure(figsize=figsize)
        plt.title(f'Row Num: {row}, Mean: {np.mean(prj)}')
        image = plt.imshow(prj)
        ax1 = plt.gca()
        ax1.tick_params(labelsize=15)
        colorbar(image)
        plt.clim(clim[0], clim[1])
        plt.show()
        
    return fig