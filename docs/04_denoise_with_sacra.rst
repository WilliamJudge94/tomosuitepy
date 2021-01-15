============================
 Denoise with Second Dataset
============================

In this section the User will learn how to denoise tomographic projections by training a neural network on a sacraficial sample's data, then applying the trained network to another sample. 

Some Users would like the ability to perform a limited angle scan of a sacraficial sample and apply these noisy + clean image pairs to other datasets. This can be performed through this method.

.. note::

    TomoSuite compiles several Neural Network Architectures out carry our various tasks related
    to Computed Tomography. The TomoGAN network was developed by
    https://github.com/ramsesproject/TomoGAN
    

Importing TomoSuite
====================

.. code:: python

    import sys
    sys.path.append('/path/to/tomosuite/github/clone/tomosuite/')
    import tomosuite
    

Defining Experimental Noise Project Parameters
==============================================

One must first define locations of the experimental files in addition to
the output path for the tomosuite project.

.. warning:: This method assumes the user has taken a sacraficial sample scan with X number of angles and Y number of projections at that same angle.

.. code:: python

    # The folder location where the Users experimental .h5 file is located
    datadir_noise = '/home/will/data/beamtime_data/'
    
    # The name of the experimental file to load into tomosuite
    fname_noise = 'Experiment_noise.h5'
    
    # The name of the tomosuite project the User would like to create
    basedir_noise = '/home/will/projects/experiment_noise/'
    
    
    
    # The directory path where the raw experimental file lives
    datadir_exp = '/local/data/experimental/'
    
    # The file name of the data the User would like to import
    fname_exp = 'Experiment_01.h5'
    
    # The folder path where the User would like to store project data to
    basedir_exp = '/local/data/project_01/'
    
    
    
Starting A Project
==================
Now the User must start two different projects. basedir_noise is for the sacraficaial noise sample while basedir_exp is the experimental data you wan tto apply the trained denoising network to.


.. code:: python

    tomosuite.start_project(basedir=basedir_noise)
    tomosuite.start_project(basedir=basedir_exp)
    

    
Extract Projection Images and Theta Values For Both Datasets
============================================================
Allows the User to extract the dataset projections to the project folders.

.. code:: python
      
    # More extraction parameters can be found in the Doc-String or in the "Starting A Project" tab
    tomosuite.extract(datadir_noise, fname_noise, basedir_noise)
    tomosuite.extract(datadir_exp, fname_exp, basedir_exp)
    
    
Create TomoGAN Files
====================
In order to determine which data can be used for training (due to sample deformations), an SSIM is used to show the User the average Similarity Index of the first projection at each angle when compared to the rest of the projections for that same angle. A threshold can be set to eleiminate certain angles from trianing.
    
.. code:: python

    from tomosuite.low_dose.data_prep import setup_tomogan_exp_noise
    setup_tomogan_exp_noise(basedir_noise, split_amount_exp, ssim_threshold=None, split_amount_ml=2):
    
    
Training TomoGAN
================
This function allows the User to train the TomoGAN denoising network. Training progress can be viewed in Tensorboard by running tensorboard --logdir='/local/data/project_01/low_dose/logs/' --samples_per_plugin=images=300

.. code:: python

    from tomosuite.low_dose.tomogan import train_tomogan
    train_tomogan(basedir=basedir_noise, epochs=120001, gpus='0',
                    lmse=0.5, lperc=2.0, 
                    ladv=20, lunet=3, depth=1,
                    itg=1, itd=2, mb_size=2,
                    img_size=896)

    
Predicting TomoGAN
==================
This function allows the User to apply the trained TomoGAN network on unseen projection data. 

.. note::

    The main difference between this function call and the one earlier in the Docs is that we have added the basedir=basedir_noise and second_basedir=basedir_exp variable. What this tells tomogan is to use the model created by basedir_noise and apply it to the projections found in basedir_exp. Then save those denoised projections to basedir_exp.
    
.. code:: python

    denoised_epoch = '22000'

    from tomosuite.low_dose.tomogan import predict_tomogan
    output = predict_tomogan(basedir=basedir_noise, 
                                        weights_iter=denoise_epoch,
                                        second_basedir=basedir_exp,
                                        chunk_size=5,
                                        noise=None,
                                        gpu='0',
                                        lunet=3,
                                        in_depth=1,
                                        data_type=np.float32,
                                        verbose=False)

View Denoised Data
==================
Please visit :ref:`reconstructions`.