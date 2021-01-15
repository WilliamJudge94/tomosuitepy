===================================
General Importer To Use Noise2Noise
===================================

If any User would like to use 2+ "poor-quality" image sets to clean the dataset, use the following steps.


Create A Project
================
Create two projects by following :ref:`starting_project`.

Project 1 basedir = '/local/data/artifact-removal/'



Preparing Data
==============

Prepare np.float16 or np.float32 RGB images (length, width, 3). Each dataset created must have the same image pair for every index. 

Example:

The artifact image found in 'dataset1[100]' must be the corresponding aftifact image that can be found in dataset2[100]


.. code:: python

    basedir = '/local/data/artifact-removal/'

    dataset1 = data1
    dataset2 = data2
    dataset3 = data3
    
    # Each dataset must be in the range of 0.0 - 255.0
    # Each dataset must be the same length

    from tomosuite.artifact.data_prep import setup_noise2noise_Users_data
    setup_noise2noise_Users_data(basedir,
                                    reconstructions=[dataset1, dataset2, dataset3],
                                    names=['1', '2', '3'])
    

Train Noise2Noise - For Inpainting Artifact
===========================================

Training progress can be viewed in Tensorboard by running this in the terminal

.. code:: python

    tensorboard --logdir=f'{basedir}artifact/logs/' --samples_per_plugin=images=300
    

.. code:: python

    from tomosuite.artifact.noise2noise import train_n2n
    
    basedir = '/local/data/artifact-removal/'
    image_dir = f'{basedir}artifact/1_recon/'
    test_dir = f'{basedir}artifact/1_recon/'
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


    num_of_slcs = len(dataset1)
    main_train_dir = ['1',]
    corresponding_train_dir = ['2', '3' ]

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
=============================================
The predicted images will be saved to f'{basedir}artifact/output_validation'

.. code:: python             
              
    from tomosuite.artifact.predictions import predict_n2n

    test_dir = f'{basedir}artifact/1_recon/'
    weights = f'{basedir}artifact/output_model/weights.001-43.829-31.71315.hdf5'

    predict_n2n(test_dir, weights,
                    output_dir=f'{basedir}artifact/output_validation',
                    test_noise_model='clean',
                    amount2skip=100, im_type='tif')