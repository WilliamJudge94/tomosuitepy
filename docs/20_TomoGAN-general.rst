================================
General Importer To Use TomoGAN
================================

If any User would like to use "poor-quality" and "good-quality" image pairs to clean "poor-quality" datasets use the following steps.


Create A Project
================
Create two projects by following :ref:`starting_project`.

.. code:: python

    #Project 1 basedir = '/local/data/training_tomogan/' = basedir_train
    #Project 2 basedir = '/local/data/prediction_tomogan/' = basedir_predict


Preparing Data
==============

.. code:: python

    #Prepare a Grey-Scale image dataset with shape of: (im_pairs, 2, x_im_dim, y_im_dim)

    #dataset[0][0] is the first image pair and is the "poor-quality" image.
    #dataset[0][1] is the first image pair and is the "good-quality" image

.. code:: python

    from tomosuite.low_dose.data_prep import save_tomogan_training
    
    # Saves every 10th image inside your dataset as a validation image pair
    save_tomogan_training(basedir=basedir_train, data, interval=10)
    

Training TomoGAN
================
Training progress can be viewed in Tensorboard by running this in the terminal

.. code:: python

    tensorboard --logdir='/local/data/project_01/low_dose/logs/' --samples_per_plugin=images=300

.. code:: python

    from tomosuite.low_dose.tomogan import train_tomogan
    train_tomogan(basedir=basedir_train, epochs=120001, gpus='0',
                    lmse=0.5, lperc=2.0, 
                    ladv=20, lunet=3, depth=1,
                    itg=1, itd=2, mb_size=2,
                    img_size=896)


Saving Images To Predict
========================

.. note::

    Save images individually as '*.tiff' inside the f'{basedir_predict}extracted/projections/' folder
    
.. warning::
    
    MAKE SURE TO USE tiff AND NOT tif


Using TomoGAN to Predict Images
===============================

This function allows the User to apply the trained TomoGAN network on unseen projection data. 

.. note::

    The main difference between this function call and the one earlier in the Docs is that we have added the basedir=basedir_train and second_basedir=basedir_predict variable. What this tells tomogan is to use the model created by basedir_noise and apply it to the projections found in basedir_exp. Then save those denoised projections to basedir_exp.
    
.. code:: python

    denoised_epoch = '22000'

    from tomosuite.low_dose.tomogan import predict_tomogan
    output = predict_tomogan(basedir=basedir_train, 
                                        weights_iter=denoise_epoch,
                                        second_basedir=basedir_predict,
                                        chunk_size=5,
                                        noise=None,
                                        gpu='0',
                                        lunet=3,
                                        in_depth=1,
                                        data_type=np.float32,
                                        verbose=False)
                                        
                                        
                                        
.. note::

    The predictions are saved to: f'{basedir_predict}low_dose/denoise_exp_data.npy'