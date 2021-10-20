import os
import cv2
import math
import time
import tomopy
import pathlib
import numpy as np
from tqdm import tqdm
import tifffile as tiff
import matplotlib.pyplot as plt
from pympler import muppy, summary
from skimage.color import rgb2gray
from scipy.ndimage import median_filter
from ..base.common import loading_tiff_prj, save_metadata
from ..base.email4recon import send_email
from ipywidgets import interact, interactive, fixed, widgets
from mpl_toolkits.axes_grid1 import make_axes_locatable 


def colorbar(mappable, font_size=12):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=font_size)
    return fig.colorbar(mappable, cax=cax, )


def save_load_delete_image_email(prj, basedir):

    fig = plt.figure(figsize=(12, 12))
    image = plt.imshow(prj, cmap='Greys_r')
    ax1 = plt.gca()
    ax1.tick_params(labelsize=15)
    colorbar(image)
    plt.clim(None, None)
    fig.savefig(f'{basedir}email_image.png')

    image = open(f'{basedir}email_image.png', 'rb').read()

    os.remove(f'{basedir}email_image.png')

    plt.close('all')

    return image


def tomo_recon(prj, theta, rot_center, user_extra=None):
    
    # Allow User to select what type of recon they want
    types='gridrec'
    
    #prj = tomopy.remove_stripe_ti(prj, 2)
    if types == 'gridrec':
        #recon = tomopy.recon(prj, theta, center=rot_center, algorithm='gridrec', ncore=16, filter_name='parzen')
        recon = tomopy.recon(prj, theta, center=rot_center, algorithm='gridrec', ncore=2)
        recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
        
    elif types == 'sirt':
        extra_options ={'MinConstraint':0}
        options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options': extra_options}
        recon = tomopy.recon(prj, theta, center=rot_center, algorithm=tomopy.astra, ncore=1, options=options)
    
    #Remove ring artifacts, this comes with a slight resolution cost
    #recon = tomopy.remove_ring(recon, center_x=None, center_y=None, thresh=300.0)
    
    return recon, user_extra


def obtain_rot_center_sinos(prj, shift_range=20):
    shifted_prj = []
    
    og_shape = prj.shape
    
    og_center = prj.shape[-1]/2
    
    range_array = np.arange(-shift_range, shift_range+1)
    
    for shift in range_array:
        shifted = np.roll(prj.copy(), shift, axis=2)
        shifted_prj.append(shifted)

    shifted_prj = np.asarray(shifted_prj)
    
    return_shifted = shifted_prj[:, :, :, shift_range:-shift_range]
    return_shifted_shape = return_shifted.shape
    
    print(f"Applying Rotation Finder Protocol - After Adjusting For Shift Changes Rotation Center Set To True Center Of {return_shifted_shape[-1]/2}")
    print(f"absolute_middle_rotation={(return_shifted_shape[-1]/2) + shift_range} - with a search range of {return_shifted_shape[-1]/2} - {(return_shifted_shape[-1]/2) + 2*shift_range}")
    
    old_shape = return_shifted.shape
    new_shape = (old_shape[1], 1, old_shape[-1])

    for idx, array in enumerate(return_shifted):
        if idx<1:
            storing = np.reshape(array[:, 0, :], new_shape)
        else:
            storing = np.append(storing, np.reshape(array[:, 0, :], new_shape), axis=1)
    
    return storing, return_shifted_shape[-1]/2, og_center, range_array

def reconstruct_single_slice(prj_data, theta, rows=(604, 606),
                             rot_center=True, med_filter=False,
                             all_data_med_filter=False, kernel=(1, 3, 3),
                             reconstruct_func=tomo_recon, recon_type='standard',
                             power2pad=False, edge_transition=None,
                             chunk_recon_size=1, dtypes=np.float32,
                             rot_center_shift_check=None,
                             muppy_amount=1000,
                             zero_pad_amount=None,
                             view_one=False,
                             minus_val=0,
                             chunker_save=False,
                             basedir=None,
                             emailer=None):
    
    prj_data -= minus_val
    
    # Apply a median filter on all data
    if med_filter and all_data_med_filter:
        print('Med Filter Applied Before')
        prj_data = median_filter(prj_data, size = kernel)

    # Setup projections based on the standard implementation
    if recon_type == 'standard':
        prj = prj_data[:, rows]
        if rot_center_shift_check is not None:
            prj, rot_center, og_center, range_array = obtain_rot_center_sinos(prj, rot_center_shift_check)
        
    # Setup projections based on deepfillv2 neural network predictions
    elif recon_type == 'deepfillv2':
        prj = prj_data
    
    # Apply median filter on only the rows
    if med_filter and not all_data_med_filter:
        print('Med Filter Applied After')
        prj = median_filter(prj, size = kernel)
        
    # Remove the harsh edge transition
    if edge_transition is not None:
        prj = prj[:, :, edge_transition:-edge_transition]
        rot_center = rot_center - edge_transition
        print(f"Applying Sinogram Edge Crop. New Rotation Center is - {rot_center}")
    
    # Pad projections based on the power of 2 shape
    if power2pad:

        shape_finder = prj.shape[2]
        power_of_2 = int(np.ceil(math.log(shape_finder, 2)))
        pad_value = 2 ** power_of_2
        mode='edge'
        
        if zero_pad_amount is not None:
            pad_value = zero_pad_amount + shape_finder
            mode='edge'
        
        pad = (pad_value - prj.shape[-1]) // 2
        pad1 = pad_value - prj.shape[-1] - pad
        prj = np.pad(
            prj,
            pad_width=((0, 0), (0, 0), (pad, pad1)),
            mode=mode,  # Pad with constant zero instead to get ring back
        )
        
        rot_center = rot_center + pad
        
    chunk_recon_store = []
    user_extra_store = []
    
    prj_shape = np.shape(prj)
    
    if view_one:
        plt.imshow(prj[:, 0, :])
        plt.title((prj[:, 0, :].min(), prj[:, 0, :].max()))
        plt.show()
        xxx = input('Press Any Key To Continue...')
    
    # Prefill Array For Better Speed + RAM Usage
    recon_store = np.zeros((prj_shape[1], prj_shape[2], prj_shape[2]))
    recon_store = recon_store.astype(dtypes)
    
    
    # Error out when chunker is not the right shape
    if prj_shape[1] % chunk_recon_size != 0:
        raise ValueError(f'Projection dimension {prj_shape[1]} cannot be divided evenly by chunk_recon_size={chunk_recon_size}. Remainder={prj_shape[1] % chunk_recon_size}')
    
    prj = prj.astype(dtypes)
    prj_chunked_main = np.array_split(prj, chunk_recon_size, axis=1)
    prj_chunked_main_shape = np.shape(prj_chunked_main)
    
    # Iterate through the recon chunks
    for idx, prj_chunked in enumerate(tqdm(prj_chunked_main, desc='Tomo Recon Progress', total=len(prj_chunked_main))):
        # Feed into reconstruction function
        recon, user_extra = reconstruct_func(prj_chunked.copy(), theta, rot_center=rot_center)
        recon_store[idx * prj_chunked_main_shape[2]: (idx + 1) * prj_chunked_main_shape[2]] = recon.copy()

        if chunker_save:
            if os.path.exists(f"{basedir}/tomsuitepy_recon_save_it_{str(idx).zfill(4)}.tiff"):
                
                if emailer is not None:
                    recipient, gmail_user, gmail_pass, divider = emailer
                    email_im = save_load_delete_image_email(recon[0, pad:-pad1, pad:-pad1], basedir)
                    send_email(recipient, email_im, f"ABOUT TO OVERWRITE FILES", f"Move chunked recon files in {basedir}.", gmail_user, gmail_pass )

                _ = input(f"You are about to overwrite chunked files. They are in {basedir}. Move them or hit ENTER to continue.")


            tiff.imsave(f"{basedir}/tomsuitepy_recon_save_it_{str(idx).zfill(4)}.tiff", recon[:, pad:-pad1, pad:-pad1])

        if emailer is not None:
            recipient, gmail_user, gmail_pass, divider = emailer
            if idx % divider == 0:
                email_im = save_load_delete_image_email(recon[0, pad:-pad1, pad:-pad1], basedir)
                send_email(recipient, email_im, f"Iteration {idx} out of {chunk_recon_size}", f"Recon {basedir}", gmail_user, gmail_pass )
                

        all_objects = muppy.get_objects()[:muppy_amount]
        sum1 = summary.summarize(all_objects)
        del recon
        #chunk_recon_store.append(recon)
        user_extra_store.append(user_extra)


    # Saving chunked data if requested and deleting the chunks
    if chunker_save:
        tiff.imsave(f"{basedir}/tomsuitepy_recon_FINAL.tiff", recon_store[:, pad:-pad1, pad:-pad1])
        for it in range(0, idx+1):
            os.remove(f"{basedir}/tomsuitepy_recon_save_it_{str(it).zfill(4)}.tiff")

    # Renaming data
    #np.concatenate(chunk_recon_store)
    recon = recon_store
    if user_extra != None:
        user_extra = np.concatenate(user_extra_store)
    
    # Crop to the original data if User pads
    if power2pad:
        recon = recon[:, pad:-pad1, pad:-pad1]
    
    if rot_center_shift_check is None:
        return recon, user_extra
    else:
        corrected_center = range_array + og_center
        recon = np.flip(recon, axis=0)

        return recon, corrected_center

def prepare_deepfillv2(basedir, checkpoint_num, start_row, end_row, verbose):
    
    import_file = f'{basedir}deepfillv2/predictions/{checkpoint_num}/ml_inpaint_data.npy'
    if verbose:
        print(f"Loading data from: {import_file}")
        time.sleep(0.5)
        
    prj_data = np.load(import_file)
    prj_data_shape = prj_data.shape
    
    if verbose:
            print(f"The shape of this data is: {prj_data_shape}")
    
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
                    sparse_angle_removal, start_row, end_row, verbose):
    
    import_file = f'{basedir}tomogan/{types}_data.npy'
    
    if verbose:
        print(f"Loading data from: {import_file}")
        time.sleep(0.5)
        
    prj_data = np.load(import_file)
    
    if second_basedir is None:
        theta = np.load(f'{basedir}extracted/theta/theta.npy')
    else:
        theta = np.load(f'{second_basedir}extracted/theta/theta.npy')

    shape = prj_data.shape[0]

    prj_data = prj_data[wedge_removal:shape - wedge_removal, :, :]
    theta = theta[wedge_removal:shape - wedge_removal]

    prj_data = prj_data[::sparse_angle_removal]
    theta = theta[::sparse_angle_removal]
    
    if verbose:
            print(f"The shape of this data is: {prj_data.shape}")
    
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
    "Also found in ...easy_networks.rife.data_prep"
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

def prepare_base(basedir, wedge_removal, sparse_angle_removal, start_row, end_row, double_sparse=None, verbose=False, ):
    
    import_file = f'{basedir}extracted/projections/'

    if verbose:
        print(f"Loading data from: {import_file}")
        time.sleep(0.5)
    
    prj_data = loading_tiff_prj(import_file)
    
    theta = np.load(f'{basedir}extracted/theta/theta.npy')
    
    shape = prj_data.shape[0]

    prj_data = prj_data[wedge_removal:shape - wedge_removal, :, :]
    theta = theta[wedge_removal:shape - wedge_removal]
    
    prj_data, theta = deal_with_sparse_angle(prj_data, theta,
                                             sparse_angle_removal,
                                             double_sparse)
    if verbose:
        print(f"The shape of this data is: {prj_data.shape}")
    
    if start_row == None:
        start = 0
    else:
        start = start_row
        
    if end_row == None:
        end = shape
    else:
        end = end_row

    return start, end, prj_data, theta



def prepare_rife(basedir, start_row, end_row, rife_types, verbose):


    frames_loc = f'{basedir}rife/{rife_types[0]}/'
    
    if verbose:
        print(f"Loading data from: {frames_loc}")
        time.sleep(0.5)
    
    files = os.listdir(frames_loc)
    files = sorted(files)
    new_files = [f'{frames_loc}{x}' for x in files if rife_types[1] in x]
    
    prj_data = []
    
    for file in tqdm(new_files, desc='Loading Data'):
        original = cv2.imread(file, -1)
        fixed_original = rgb2gray(original)
        if rife_types[2] == True:
            fixed_original = np.log(fixed_original)
        #fixed_original *= 255.0

        prj_data.append(fixed_original)

        
    prj_data = np.asarray(prj_data, dtype=np.float32)
    shape = prj_data.shape[0]
    theta = tomopy.angles(prj_data.shape[0], 0, 180)
    
    if verbose:
        print(f"The shape of this data is: {prj_data.shape}")
        if prj_data.shape == (0, ):
            raise ValueError(f'No data found in {frames_loc}') 
    
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
                     rife_types=['output_frames', '.png', False],
                     second_basedir=None,
                     checkpoint_num=None,
                     double_sparse=None,
                     power2pad=False,
                     edge_transition=None,
                     verbose=False,
                     chunk_recon_size=1,
                     dtypes=np.float32,
                     rot_center_shift_check=None,
                     muppy_amount=1000,
                     zero_pad_amount=None,
                     view_one=False,
                     minus_val=0,
                     chunker_save=False,
                     emailer=None):
    
    """Determine the tomographic reconstruction of data loaded into the TomoSuite data structure.
    
    Parameters
    ----------
    basedir : str
        Path to the current project where the User would like to load the projection data from.
    
    start_row : int or None
        If int the reconstruction will start at the designated row. Autosets to first row if None.
    
    end_row : int or None
        If int the reconstruction will end at the designated row. Autosets to last row if None.
        
    med_filter : bool
        If True it will apply a median filter to the data.
        
    all_data_med_filter : bool
        If True this will apple filter to entire dataset. If False then it will be applied to the the data selected from the start_row:end_row
        
    med_filter_kernel : array
        The median filter kernel
        
    reconstruct_func : function
        the implemetation of tomopy.recon the User would like a apply to the dataset.
        Inputs must be prj, theta, rot_center, user_extra=None while outputs must be recon, user_extra.
        
    network : None or str
        Allows the program to determine which data the User is importing. And puts the data into the
        right shape for the reconstruct_func. str examples are 'tomogan', 'deepfillv2', 'rife'.
        
    wedge_removal : int
        Zeroes out this many projections from the beginning and end of the projection list.
        
    sparse_angle_removal : int
        Take every sparse_angle_removal projection for the reconstruction.
        
    types : str
        Used to determine which datasets to load in for the tomogan network. 'denoise_fake',
        'denoise_exp', 'noise_exp', 
        
    rife_types : array
        An array with layout to rife_types=['folder', 'filetype', apply_log]
        
        folder : str
            the location inside basedir/rife/ to obtain the frames from. Standard is 'frames'
        
        filetype : str
            the string of the image file type. .png, .tif, or .tiff
            
        apply_log : bool
            if True this will apply np.log() to each of the frames.
            
    second_basedir : str
        A second directory used to load a different files theta data. Passed through to TomoGAN.
        
    checkpoint_num : str
        Used for 'deepfillv2' network. Allows the User to load in a certain network's checkpoint
        predictions for the reconstruction. 
        
    double_sparse : int
        If the User would like to take more data out from a previously sparsed dataset.
        
    power2pad : bool
        Pads the sinograms to a power of 2 size. Faster/better reconstruction.
        
    edge_transition : int
        power2pad may create a Ring Glitch in the reconstruction. This is due to a hash transition from
        the dataset to the padded area. This will remove sinogram edge columns to limit how harsh this
        boundary is.

    muppy_amount : int
        the amount of data to load when viewing the current RAM usage. This allows tomopy to refresh the RAM usage allowing for resource efficent tasks.

    zero_pad_amount : int
        the amount of zeros to pad to the sides of the sinogram.

    view_one : bool
        if True the User will be able to view a single sinogram as well as the min and max values. This allows for accurate recons for SIRT. If the min and max values are too high then the reconstruciton will be poor.

    minus_val : int
        the value to subtract from the sinogram to correct for large values in the sinogram. Allows for better SIRT reconstructions.
    
    chunker_save : bool
        If the User would like to chunk save their recons as well as save the final output then set this to True.

    emailer : array
        if not None the set to the following to send emails [recipient_email_str, sender_email_str, sender_pass_str, send_every_X_chunks_int]
        
    
    Returns
    -------
    The reconstructed data and a user_extra data that is output from the reconstruct_func()
    """

    # Saving MetaData Recon
    metadata_dic = {}
    metadata_keys = [var for var in locals().keys() if '__' not in var]
    for metadata_key in metadata_keys:
        metadata_dic[metadata_key] = locals()[metadata_key]
    save_metadata(basedir, metadata_dic, meta_data_type='recon')

    if network == None:
        recon_type = 'standard'
        start, end, prj_data, theta = prepare_base(basedir, wedge_removal,
                                                   sparse_angle_removal,
                                                   start_row, end_row, double_sparse, verbose=verbose)
        
    elif network == 'tomogan':
        recon_type = 'standard'
        start, end, prj_data, theta = prepare_tomogan(basedir, types,
                                                      second_basedir,
                                                      wedge_removal,
                                                      sparse_angle_removal,
                                                      start_row, end_row, verbose=verbose)
        
    elif network == 'deepfillv2':
        recon_type = 'deepfillv2'
        assert checkpoint_num != None
        start, end, prj_data, theta = prepare_deepfillv2(basedir, checkpoint_num, start_row, end_row, verbose=verbose)
        
        
    elif network == 'rife':
        recon_type = 'standard'
        start, end, prj_data, theta = prepare_rife(basedir, start_row, end_row, rife_types=rife_types, verbose=verbose)
        


    if chunk_recon_size > 1:
        p_cwd = pathlib.Path('.').absolute()
        print(f'Temporary files to be saved to {p_cwd} - Directory Real - {os.path.isdir(p_cwd)}')

    try:
        all_objects = muppy.get_objects()[:muppy_amount]
        sum1 = summary.summarize(all_objects)

    except Exception as ex:
        raise ValueError(f"Failed to initiate muppy RAM collection - Error: {ex}")

    slc_proj, user_extra = reconstruct_single_slice(prj_data.copy(), 
                                           theta,
                                           rot_center = rot_center, 
                                           rows=slice(start, end), 
                                           med_filter=med_filter, 
                                           all_data_med_filter=all_data_med_filter, 
                                           kernel=med_filter_kernel, 
                                           reconstruct_func=reconstruct_func,
                                           recon_type=recon_type,
                                           power2pad=power2pad,
                                           edge_transition=edge_transition,
                                           chunk_recon_size=chunk_recon_size,
                                           dtypes=dtypes,
                                           rot_center_shift_check=rot_center_shift_check,
                                           muppy_amount=muppy_amount,
                                           zero_pad_amount=zero_pad_amount,
                                           view_one=view_one, minus_val=minus_val,
                                           chunker_save=chunker_save, basedir=basedir, emailer=emailer)
        
        
    return slc_proj, user_extra



def plot_reconstruction(slc_proj, figsize=(15, 15), clim=(None, None), cmap='Greys_r', interactive=True):
    """Allow the User to plot the data that was output from the reconstruction
    
    Parameters
    ----------
    slc_proj : nd.array
        the output from the tomosuite.base.reconstruc.reconstruct_data() function
    
    figsize : list
        the figsize to be passed to plt.figure()
        
    clim : list
        clim lower and upper limits to be passed into plt.clim()
        
    cmap : str
        the cmap passed to plt.imshow()
        
    Returns
    -------
    Shows the plots for the given input data
    """
    if interactive:
        sliders = widgets.IntSlider(value=0, min=0, max=len(slc_proj)-1)
        interact(plotting_recons,
                 slcs_proj=fixed(slc_proj), idx=sliders, figsize=fixed(figsize), cmap=fixed(cmap), clim=fixed(clim))
        
    else:
        for row, prj in enumerate(slc_proj):
            fig = plt.figure(figsize=figsize)
            plt.title(f'Row Num: {row}, Mean: {np.mean(prj)}')
            image = plt.imshow(prj, cmap=cmap)
            ax1 = plt.gca()
            ax1.tick_params(labelsize=15)
            colorbar(image)
            plt.clim(clim[0], clim[1])
            plt.show()

        return fig

def plot_reconstruction_centers(slc_proj, figsize=(15, 15), clim=(None, None), cmap='Greys_r', absolute_middle_rotation=None, interactive=True):
    """Allow the User to plot the data that was output from the reconstruction
    
    Parameters
    ----------
    slc_proj : nd.array
        the output from the tomosuite.base.reconstruc.reconstruct_data() function
    
    figsize : list
        the figsize to be passed to plt.figure()
        
    clim : list
        clim lower and upper limits to be passed into plt.clim()
        
    cmap : str
        the cmap passed to plt.imshow()
        
    Returns
    -------
    Shows the plots for the given input data
    """

    starting_rotation_center = absolute_middle_rotation

    if starting_rotation_center is None:
        print('Please set starting_ration_center value')

    else:

        total = (len(slc_proj)+1)/2
        starting = starting_rotation_center - total + 1
        center_range_values = np.arange(starting, starting_rotation_center + total)
    
        if interactive:
        
            sliders = widgets.IntSlider(value=starting, min=starting, max=starting_rotation_center + total - 1)

            interact(plotting_center, slcs_proj=fixed(slc_proj),
            center_range_values=fixed(center_range_values),
            idx=sliders, starting_rotation_center=fixed(starting_rotation_center),
            figsize=fixed(figsize), cmap=fixed(cmap), clim=fixed(clim))
        
        else:

            for idx, item in enumerate(slc_proj):
                plt.figure(figsize=(figsize))
                plt.imshow(item, cmap=cmap)
                plt.clim(clim[0], clim[1])
                plt.title(f"Rotation Center of - {center_range_values[idx]}")
                plt.show()


def plotting_center(slcs_proj, center_range_values, idx, starting_rotation_center, figsize, cmap, clim):
    row = idx
    prj = slcs_proj[np.where(center_range_values == idx)[0]]
    starting = center_range_values[np.where(center_range_values == idx)[0]]
    
    fig = plt.figure(figsize=figsize)
    if starting_rotation_center is None:
        plt.title(f'Row Num: {row}, Mean: {np.mean(prj)}')
    else:
        plt.title(f'Row Num: {row}, Mean: {np.mean(prj)}, Rotation Center: {starting}')
        starting += 1
    image = plt.imshow(prj[0], cmap=cmap)
    ax1 = plt.gca()
    ax1.tick_params(labelsize=15)
    colorbar(image)
    plt.clim(clim[0], clim[1])
    plt.show()
    
    
def plotting_recons(slcs_proj, idx, figsize, cmap, clim):
    fig = plt.figure(figsize=figsize)
    prj = slcs_proj[idx]
    plt.title(f'Row Num: {idx}, Mean: {np.mean(prj)}')
    image = plt.imshow(prj, cmap=cmap)
    ax1 = plt.gca()
    ax1.tick_params(labelsize=15)
    colorbar(image)
    plt.clim(clim[0], clim[1])
    plt.show()
