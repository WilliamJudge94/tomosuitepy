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
    

Setting up Data
---------------

.. code:: python

    from tomosuite.inpainting.data_prep import fake_missing_wedge, add_prj4missing_wedge

    # Number of projections missing from EACH SIDE OF THE TOMOGRAM
    num_of_proj_missing = 167

    # The data provided from the low_dose module - either 'noise' or 'denoise'
    data2use = 'noise_exp'

    # Reshape the images - works best on anything 768 or smaller
    new_shape_single = (768, 768)
    new_shape_triple = [768, 768, 3]

    # Apply fake missing wedge to see if the ML code works - Compare to Ground Truth
    number2zero_val = fake_missing_wedge(basedir,
                                            num_of_proj_missing,
                                            data2use)
    
    # Add in missing (blank) projections to experimental missing-wedge data
    number2zero =  add_prj4missing_wedge(basedir, types=data2use)
    
Creating the Training and Test Datasets
---------------------------------------
.. code:: python

    from tomosuite.inpainting.data_prep import create_testing_data, create_training_data
    
    output_testing = create_testing_data(basedir,
                                            data2use,
                                            number2zero,
                                            reshape=True,
                                            new_shape=new_shape_single)
                                            
    output_training = create_training_data(basedir,
                                            data2use,
                                            number2zero,
                                            reshape=True,
                                            new_shape=new_shape_single)
    

Prepare DeepFillV2 to Accept Your Data
--------------------------------------

.. code:: python

    from tomosuite.inpainting.data_prep import make_file_list, determine_train_height
    from tomosuite.inpainting.deepfillv2 import setup_inpainting
    import numpy as np

    make_file_list(basedir)

    fraction_correction = np.load(f'{basedir}inpainting/output_scaler.npy')

    corrected_missing_prj_amount = int(determine_train_height(new_shape_triple,
                                                                fraction_correction,
                                                                num_of_proj_missing))

    setup_inpainting(basedir, img_shapes=new_shape_triple,
                     height=corrected_missing_prj_amount,
                     static_view_size=10, batch_size=2)

Train Your Network
------------------

.. code:: python

    from tomosuite.inpainting.deepfillv2 import train_deepfillv2
    train_deepfillv2(basedir)
    

To View Training Process Use Tensorboard
    
.. code:: bash

    tensorboard --logdir=f'{basedir}inpainting/logs/'
    
    
Use Network For Predictions
---------------------------

.. code:: python

    from tomosuite.inpainting.predictions import predict_deepfillv2
    load_epoch = '40000'
    
    output = predict_deepfillv2(basedir,
                                load_epoch,
                                new_shape_single[0],
                                new_shape_single[1],
                                save=True)
    
    
    from tomosuite.inpainting.data_prep import retrieve_inpainting
    outs = retrieve_deepfillv2(basedir,
                                load_epoch,
                                corrected_missing_prj_amount)

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