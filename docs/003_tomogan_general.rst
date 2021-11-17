.. _tomogan_general:

================================
General Importer To Use TomoGAN
================================

If any User would like to use "poor-quality" and "good-quality"
image pairs to clean "poor-quality" datasets use the following steps.

.. warning::

If you are using the PyPi version of TomoSuitePY,
Please view :ref:`installation` and :ref:`starting_project`
before attempting to load in the module. 

Create A Project
================
Create two projects by following :ref:`starting_project`.

.. code:: python

    #Project 1 - project to store the training data
    basedir_train = '/local/data/training_tomogan/'
    
    #Project 2 - project to store data to be predicted
    basedir_predict = '/local/data/prediction_tomogan/'


Preparing Data
==============

.. code:: python

    #Prepare 2 Grey-Scale image dataset with shape of: (number_of_images, x_im_dim, y_im_dim)
    
    # Clean grey-scale images - substitute your own data for the numpy command
    number_of_images, x_im_dim, y_im_dim = 1024, 1224, 1224
    clean_data = np.ones((number_of_images, x_im_dim, y_im_dim))
    
    # Noisy grey-scale images - subsititue your own data for the numpy command
    noisy_data = np.ones((number_of_images, x_im_dim, y_im_dim))
    
    # clean_data[100] should be the clean image for noisy_data[100]
    
    
.. code:: python

    # The image dimensions have to be => 384. If needed, increase your images size with the script below.

    new_noisy = []
    new_clean = []
    
    padding_value = 11

    for cl, no in zip(clean_data, noisy_data):
        new_noisy.append(np.pad(no, padding_value))
        new_clean.append(np.pad(cl, padding_value))
        
    new_noisy = np.asarray(new_noisy)
    new_clean = np.asarray(new_clean)
    
    # Make sure data is the correct shape
    print((new_noisy.shape, new_clean.shape))
    
    noisy_data = np.asarray(new_noisy)
    clean_data = np.asarray(new_clean)
    
    

.. code:: python


    from tomosuite.easy_networks.tomogan.data_prep import format_data_tomogan, setup_data_tomogan
    
    # Insert shuffling of data
    
    # Saves every 5th image for training data
    xtrain, ytrain, xtest, ytest = format_data_tomogan(clean_data,
                                                        noisy_data,
                                                        interval=5,
                                                        dtype=np.float32)

                                                        
    setup_data_tomogan(basedir_train, xtrain, ytrain, xtest, ytest, types='noise')
    
    

Training TomoGAN
================
Training progress can be viewed in Tensorboard by running this in the terminal

.. code:: python

    from tomosuite.easy_networks.tomogan.train import tensorboard_command_tomogan
    tensorboard_command_tomogan(basedir_train)

.. code:: python

    from tomosuite.easy_networks.tomogan.train import train_tomogan
    
    train_tomogan(basedir=basedir_train, epochs=120001, gpus='0',
                    lmse=0.5, lperc=2.0, 
                    ladv=20, lunet=3, depth=1,
                    itg=1, itd=2, mb_size=2,
                    img_size=512)


Setup Prediction Data
======================

Create a numpy array filled with grey-scale images that the User would like to apply the trained TomoGAN network to.
The shape should be (number_of_images, x_dimension, y_dimension)


Remember that the image dimensions have to be greater than 384 x 384. If needed, please use the script below to update the shape of your images.

.. code:: python

    # The image dimensions have to be => 384. If needed, increase your images size with the script below.

    new_pred_data = []
    
    padding_value = 11

    for pr in pred_data:
        new_pred_data.append(np.pad(pr, padding_value))
        
    new_pred_data = np.asarray(new_pred_data)
    
    # Make sure data is the correct shape
    print((new_pred_data.shape))
    
    pred_data = np.asarray(new_pred_data)


Predicting TomoGAN
==================
Once an appropriate epoch has been chosen through Tensorboard one can use this epoch to predict the denoised projections.
    
.. code:: python

    from tomosuite.easy_networks.tomogan.predict import predict_tomogan, save_predict_tomogan
    from tomosuite.base.common import load_extracted_prj

    # Loading in the Projection Data - substitute numpy command with your own data
    number_of_images, x_dim, y_dim = 1024, 1224, 1224
    
    # The dirty data the User wants to predict
    dirty_data = np.ones((number_of_images, x_dim, y_dim))

    clean_data, dirty_data = predict_tomogan(basedir_train,
                                    dirty_data,
                                    weights_iter='01000', # The epoch number to load weights of
                                    chunk_size=5, # Chunk the data so it doesnt overload GPU VRAM
                                    gpu='0', # Select which gpu to use
                                    lunet=3,
                                    in_depth=1,
                                    data_type=np.float32,
                                    verbose=False,
                                    types='noise')

    save_predict_tomogan(basedir=None,
                            good_data=clean_data,
                            bad_data=dirty_data,
                            second_basedir=basedir_predict,
                            types='noise')
                            
.. note::

    The predictions (clean_pred_data) are saved to: f'{basedir_predict}tomogan/denoise_exp_data.npy' or f'{basedir_predict}tomogan/deartifact_exp_data.npy'.
    This depends on what value the user sets 'types' to. Options are types='noise' or types='artifact'


View Denoised Data
==================
Please visit :ref:`reconstructions`.
                                        
                                        