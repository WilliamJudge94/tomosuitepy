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
    
    
Setting Up The Data - From Inpainting
-------------------------------------

Make sure you are using the inpainting conda env

.. code:: python

    from tomosuite.artifact.data_prep import setup_noise2noise_tomosuite_missingwedge
    
    setup_noise2noise_tomosuite_missingwedge(basedir, 
                                         ml_iterations=['42000', '52000'], 
                                         rot_center=256, img_shape=[512, 512, 3], 
                                         number2zero_shape_updated=145, add_flips=True, im_type='tif')
                                         
                                         
Train Noise2Noise - For Inpainting Artifact
-------------------------------------------

.. code:: python

    from tomosuite.artifact.noise2noise import train_n2n

    basedir = '/local/data/wjudge/TomoSuite/will/inpaint036_dual/'
    image_dir = f'{basedir}artifact/52000_recon/'
    test_dir = f'{basedir}artifact/52000_recon/'
    output_path = f'{basedir}artifact/output_model/'

    # Use mae or l0 - make sure you are using consistent images for input and output - 
    #chose the lowest validation PSNR - steps=150 - image_size=64
    
    # Make sure you are training on .png images, use a single image pair, train for a bit,
    #take the worst PSNR value, apply this network to the rest of the images

    source_noise_model = 'clean'
    target_noise_model = 'clean'
    val_noise_model = 'clean'
    loss_type='l0'


    image_size = 64
    batch_size = 1
    lr = 0.001
    steps=150 #150


    basedir = '/local/data/wjudge/TomoSuite/will/inpaint036_dual/'
    num_of_slcs = 1024
    main_train_dir = ['42000',]
    corresponding_train_dir = ['52000', ]

    concat_train = False
    crop_im_val = 60
    single_image_train = 100

    im_type = 'tif'


    train_n2n(image_dir,
              test_dir, 
              image_size=image_size, 
              batch_size=batch_size, 
              lr=lr, 
              output_path=output_path, 
              val_noise_model=val_noise_model,
              target_noise_model=target_noise_model, 
              source_noise_model=source_noise_model, 
              loss_type=loss_type, 
              save_best_only=False, 
              steps=steps, basedir=basedir, num_of_slcs=num_of_slcs, 
              main_train_dir=main_train_dir, corresponding_train_dir=corresponding_train_dir,
              concat_train=concat_train, crop_im_val=crop_im_val,
              single_image_train=single_image_train, im_type=im_type)
              
              
Predict Noise2Noise - For Inpainting Artifact
---------------------------------------------

.. code:: python             
              
    from tomosuite.artifact.predictions import predict_n2n

    test_dir = f'{basedir}artifact/52000_recon/'
    weights = '/local/data/wjudge/TomoSuite/will/inpaint036_dual/artifact/output_model/weights.001-43.829-31.71315.hdf5'

    predict_n2n(test_dir, weights,
                    output_dir=f'{basedir}artifact/output_validation',
                    test_noise_model='clean',
                    amount2skip=100, im_type='tif')