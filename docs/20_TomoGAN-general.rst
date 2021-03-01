================================
General Importer To Use TomoGAN
================================

If any User would like to use "poor-quality" and "good-quality" image pairs to clean "poor-quality" datasets use the following steps.


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


    from tomosuite.easy_networks.tomogan.data_prep import format_data_tomogan, save_data_tomogan
    
    # Insert shuffling of data
    
    # Saves every 5th image for test data
    xtrain, ytrain, xtest, ytest = format_data_tomogan(clean_data,
                                                        noisy_data,
                                                        interval=5,
                                                        dtype=np.float32)
                                                        
    setup_data_tomogan(basedir_train, xtrain, ytrain, xtest, ytest, types='noise')
    
    

Training TomoGAN
================
Training progress can be viewed in Tensorboard by running this in the terminal

.. code:: python

    from tomosuite.easy_networks.tomogan.train tensorboard_command_tomogan
    tensorboard_command_tomogan(basedir_train)

.. code:: python

    from tomosuite.easy_networks.tomogan.train import train_tomogan,
    train_tomogan(basedir=basedir_train, epochs=120001, gpus='0',
                    lmse=0.5, lperc=2.0, 
                    ladv=20, lunet=3, depth=1,
                    itg=1, itd=2, mb_size=2,
                    img_size=512)


Setup Prediction Data
======================

Create a numpy array filled with grey-scale images that the User would like to apply the trained TomoGAN network to.
The shape should be (number_of_images, x_dimension, y_dimension)


Predicting TomoGAN
==================
Once an appropriate epoch has been chosen through Tensorboard one can use this epoch to predict the denoised projections.
    
.. code:: python

    from tomosuite.easy_networks.tomogan.predict import predict_tomogan, save_predict_tomogan
    from tomosuite.base.common import load_extracted_prj

    # Loading in the Projection Data - substitute numpy command with your own data
    number_of_images, x_dim, y_dim = 1024, 1224, 1224
    data = np.ones((number_of_images, x_dim, y_dim))

    clean_pred_data, dirty_data = predict_tomogan(basedir_train,
                                    data,
                                    weights_iter='01000', # The epoch number to load weights of
                                    chunk_size=5, # Chunk the data so it doesnt overload GPU VRAM
                                    gpu='0', # Select which gpu to use
                                    lunet=3,
                                    in_depth=1,
                                    data_type=np.float32,
                                    verbose=False,
                                    types='noise')

    save_predict_tomogan(basedir=None,
                            good_data=clean__pred_data,
                            bad_data=dirty_data,
                            second_basedir=basedir_predict,
                            types='noise')
                            
.. note::

    The predictions (clean_pred_data) are saved to: f'{basedir_predict}tomogan/denoise_exp_data.npy' 


View Denoised Data
==================
Please visit :ref:`reconstructions`.
                                        
                                        