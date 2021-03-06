==============================
 Dewedge with One Dataset - V1
==============================

In this section the User will learn how to dewedge tomographic projections by only training/using a neural network with the same experimental dataset. 

The first step is Sinogram Inpainting, while the second step is De-InpaintArtifacting. If your dataset does not perform well on Sinogram Inpainting please view the :ref:`Dewedge with One Dataset - V2` section of the documentation.

.. note::

    TomoSuite compiles several Neural Network Architectures out carry our various
    tasks related to Computed Tomography. The DeepFillV2 network was developed by
    https://github.com/JiahuiYu/generative_inpainting
    
    
    
DeepFillV2 Inpainting 
=====================

This secion of the DeWedging shows the User how to inpaint the sinograms based upon the avalable sinogram data. 


Importing TomoSuite
-------------------

.. code:: python

    import sys
    sys.path.append('/path/to/tomosuite/github/clone/tomosuite/')
    import tomosuite
    
    basedir = '/local/data/project_wedge01/'
    
    # Start project and extract files
    

Setting up Data
---------------
This allows a User to determine how their deepfillv2 machine learning will occur. If the User picks edge_or_central='edge',
then missing wedge rows in the sinogram will be kept at the beginning and end of the sinogram. If the User picks
edge_or_central='central', then half the sinogram is flipped and invertes to make a continuous image. 'central'
creates better predictions.

.. code:: python

    from tomosuite.easy_networks.deepfillv2.train import train_deepfillv2, tensorboard_command_deepfillv2
    from tomosuite.easy_networks.deepfillv2.data_prep import format_and_save_data4deepfillv2, 
    from tomosuite.easy_networks.deepfillv2.deepfillv2_prep import setup_inpainting, make_file_list4deepfillv2, determine_deepfillv2_mask_height
    
    
    # how many projections in the beginning and end of the scan are blank
    number2zero=44
    
    # This will take out X number of columns from the sinogram to keep image proportions square
    shrink_sinogram_width=False

    output = format_and_save_data4deepfillv2(basedir,
                                            number2zero=number2zero,
                                            types='base',
                                            downscale_shape=(512, 512),
                                            edge_or_central='central', # Where do you want to place the missing wedges
                                            shrink_sinogram_width=shrink_sinogram_width,
                                            lr_flip=True, # Flip training images left and right
                                            ud_flip=True, # Flip training images up and down
                                            val_ud_flip=True) # Flip validation images up and down
    
    # Giving outputs variable names
    downscaled_base_training_images, base_training_images, number2zero, max_intensity_values = output
    
    # Obtain images list for deepfillv2 to use
    make_file_list4deepfillv2(basedir)
    
    # Determine the rescaled number of missing wedge pixles
    height = determine_deepfillv2_mask_height(downscaled_base_training_images,
                                        base_training_images,
                                        number2zero=number2zero)
                                        

Set Up Inpainting
-----------------
This function allows the User to set up training parameters however they would like. To the batch size,
to the memory per job, to the static view size viewable through tensorboard. There are more parameters to 
tweek than what is listed blow, but these are the main ones Users should worry about. 

.. code:: python
    
    # Project path
    basedir = basedir
    
    # trippled shape of the images set bu format_and_save_data4deepfillv2()
    img_shapes = [512, 512, 3]
    
    # The value given by determine_deepfillv2_mask_height()
    height = height
    
    # If the User would like to randomly change the height of the missing wedge box (not advised)
    max_delta_height = 0
    
    # Amount of static images in tensorboard (lower if GPU runs out of memory)
    static_view_size = 5
    
    # Training batch size (lower if GPU runs out of memory)
    batch_size = 1
    
    # Number of epochs
    max_iters = 120000
    
    # Location of model to restore from
    model_restore = ''
    
    # How many GPU's to use during training
    num_gpus_per_job = 1
    
    # How many CPU's to use during training
    num_cpus_per_job = 4
    
    # How much memory to use during training
    memory_per_job = 11
    
    # Identifies what mask type to use. Should be same as edge_or_central
    mask_type = 'central'
    
    # If mask_type='edge' this allows the User to place the mask always in the upper location ("u"),
    #always in the down location ("d") or randomize it ("r")
    
    # If mask_type='central' this allows the user to randomize the central location ("r")
    #or always keep it in the center ("u" or "d")
    udr = 'r'


    setup_inpainting(basedir=basedir,
                    img_shapes=img_shapes,
                    height=height,
                    max_delta_height=max_delta_height, 
                    static_view_size=static_view_size,
                    batch_size=batch_size,
                    max_iters=max_iters,
                    model_restore=model_restore,
                    num_gpus_per_job=num_gpus_per_job,
                    num_cpus_per_job=num_cpus_per_job,
                    num_hosts_per_job=num_hosts_per_job,
                    memory_per_job=memory_per_job,
                    mask_type=mask_type, udr=udr)
                  

    
Train Inpainting
-----------------   

.. code:: python
    
    tensorboard_command_deepfillv2(basedir)
    
    train_deepfillv2(basedir, gpu='0,1,2,3,4,5,6,7')
    
    
    
    
Use Network For Predictions
---------------------------

.. code:: python

    from tomosuite.easy_networks.deepfillv2.predict import predict_deepfillv2
    from tomosuite.easy_networks.deepfillv2.data_prep import  convert2gray_rescale_save
    
    pred_images = predict_deepfillv2(basedir=basedir,
                                        checkpoint_num='120000', # epoch number of DeepfillV2 training
                                        image_height=512, # rescaled height
                                        image_width=512, # rescaled width
                                        gpu='0' # gpu ID)
    
    convert2gray_rescale_save(predicted_images=pred_images, # output of predict_deepfillv2
                                basedir=basedir,
                                number2zero=number2zero, # number of images associated with the missing wedge
                                checkpoint_num=checkpoint_num, # epoch number of DeepfillV2
                                downscale_shape=(512, 512), # rescaled image dimensions
                                edge_or_central='central',
                                shrink_sinogram_width=False,
                                val_ud_flip=True, # let the program know if you flipped the validation images)


View Inpainted Sinogram Data
----------------------------
Please visit :ref:`reconstructions`.


Noise2Noise Inpainting Artifact Removal
=======================================

While correcting the Wedge Artifacts of the dataset provided, the inpainter is not perfect. This means on reconstruction new streak artifacts take their place. Fortunatley, they streak artifacts differ for each epoch of the inpainting network. This means if we take different inpainter network epochs, predict the sinograms, the streak artifacts do not overlap. This allows the Users to use the Noise2Noise network to DeStreak Artifact the dataset.

In this section the User will learn how to de-streak artifact tomographic reconstructions by only training/using a neural network with the same experimental dataset. 

This is the second step in DeWedge Artifacting with One Dataset V1. If your dataset does not perform well on Sinogram Inpainting please view the :ref:`Dewedge with One Dataset - V2` section of the documentation.

.. note::

    TomoSuite compiles several Neural Network Architectures out carry our various tasks related to 
    Computed Tomography. The Noise2Noise network was developed by
    https://github.com/NVlabs/noise2noise, while https://github.com/yu4u/noise2noise made a
    Keras version of the architecture. The second repository was used for development of TomoSuite
    
    
Importing TomoSuite
-------------------

.. code:: python

    import sys
    sys.path.append('/path/to/tomosuite/github/clone/tomosuite/')
    import tomosuite
    import tomopy
    
    
Saving Multiple Recons
----------------------

.. code:: python


    # Import TomoSuite helper functions
    from tomosuite.base.reconstruct import reconstruct_data, plot_reconstruction

    # Define your own tomography reconstruction function. This is the TomoSuite's default
    def tomo_recon(prj, theta, rot_center, user_extra=None):
        recon = tomopy.recon(prj, theta,
                            center=rot_center,
                            algorithm='gridrec',
                            ncore=30)
        recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
        return recon, user_extra

    # Reconstruct the deepfillv2 projection data
    basedir = '/local/data/project_01/' 

    slcs, user_extra = reconstruct_data(basedir, rot_center=181, 
                               reconstruct_func=tomo_recon, network='deepfillv2',
                               checkpoint_num='120000', power2pad=True)


    np.save(f'{basedir}deepfillv2/predictions/120000_recon.npy', slcs)
    
    
    
    
    slcs, user_extra = reconstruct_data(basedir, rot_center=181, 
                               reconstruct_func=tomo_recon, network='deepfillv2',
                               checkpoint_num='90000', power2pad=True)


    np.save(f'{basedir}deepfillv2/predictions/90000_recon.npy', slcs)
    
    
    
Setting Up The Data For Noise2Noise - From Inpainting
-----------------------------------------------------

Make sure you are using the inpainting conda env

.. code:: python

    from tomosuite.easy_networks.noise2noise.data_prep import format_data_noise2noise, setup_data_noise2noise
    import numpy as np
    
    dataset1 = np.load(f'{basedir}deepfillv2/predictions/120000_recon.npy')
    dataset2 = np.load(f'{basedir}deepfillv2/predictions/90000_recon.npy')
    
    # You can add more than 2 datasets/recons - format the data to be stored
    dataset = format_data_noise2noise([dataset1, dataset2])
    
    # save the data in the correct location
    setup_data_noise2noise(basedir, 
                            val_name='120000', # name of the dataset used as validation data in tensorboard
                            val_crop=10, # every 10th images is used for validation
                            datasets=dataset,
                            names=['120000', '90000'], # the names of the dataests in the same order as placed in the format_data_noise2noise() function
                            )
                                         
                                         
Train Noise2Noise - For Inpainting Artifact
-------------------------------------------

.. code:: python

    from tomosuite.easy_networks.noise2noise.train import train_noise2noise, tensorboard_command_noise2noise
    
    tensorboard_command_noise2noise(basedir)
    
    train_noise2noise(basedir,
                        main_train_dir=['120000',], # The name of the main dataset
                        corresponding_train_dir=['90000', ], # all other corresponding recon datasets to use for image pair training
                        concat_train=False, # if True this will concatonate main_train_dir and corresponding_train_dir
                        crop_im_val=None, # crop the image left and right by this many pixles
                        single_image_train=None, # only train on a single image. use index number of image (sometimes gets better results)
                        single_image_val=None, 
                        im_type='tif',
                        image_size=64, # image random crop size
                        batch_size=16,
                        nb_epochs=60,
                        lr=0.01,
                        steps=100, # number of steps per epoch. Lower usually does better
                        loss_type='mae', # either 'mae' or 'l0'
                        weight=None,
                        model='srresnet',
                        save_best_only=True,
                        gpu='0', # id for which GPU to use
                        )


              
              
Predict Noise2Noise - For Inpainting Artifact
---------------------------------------------

.. code:: python             
              
    from tomosuite.easy_networks.noise2noise.predict import predict_noise2noise, save_predict_noise2noise
    
    
    main_train = '120000'
    test_dir = f'{basedir}noise2noise/{main_train}_recon/'
    weights = f'{basedir}noise2noise/output_model/weights.001-43.829-31.71315.hdf5'

    denoised_images, image_paths, out_images = predict_noise2noise(image_dir=test_dir, 
                                                                weight_file=weights,
                                                                amount2skip=100,
                                                                im_type='tif',
                                                                crop_im_val=None, 
                                                                gpu='0')
                                                                
                                                                
    save_predict_noise2noise(basedir,
                            denoised_images,
                            image_paths,
                            output_dir=None # set output path otherwise it will default to f'{basedir}noise2noise/output_validation'
                            )