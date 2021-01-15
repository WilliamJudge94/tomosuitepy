=========================
 Denoise with One Dataset
=========================

In this section the User will learn how to denoise tomographic projections by only training/using a neural network with the same experimental dataset. 

.. note::

    TomoSuite compiles several Neural Network Architectures out carry our various tasks related
    to Computed Tomography. The TomoGAN network was developed by
    https://github.com/ramsesproject/TomoGAN


Importing TomoSuite
===================

.. code:: python

    import sys
    sys.path.append('/path/to/tomosuite/github/clone/tomosuite/')
    import tomosuite
    
    
Define Project Parameters
=========================

One must first define locations of the experimental files in addition to
the output path for the tomosuite project.

.. code:: python

    # The directory path where the raw experimental file lives
    datadir = '/local/data/experimental/'
    
    # The file name of the data the User would like to import
    fname = 'Experiment_01.h5'
    
    # The folder path where the User would like to store project data to
    basedir = '/local/data/project_01/'
    

Skip Denoising
==============

If the User would like to skip denoising and move onto sinogram inpainting or artifact removal please initiate the skip_lowdose() function

.. code:: python 

    tomosuite.skip_lowdose(basedir=basedir)
    
    
Desnoising Examples
===================

.. figure:: img/low_dose-raw.png
    :scale: 100%
    :align: center

    This is reconstruction (GridRec) of Raw Experimental data.
  
.. figure:: img/low_dose-fake-noise.png
    :scale: 100%
    :align: center

    This is the reconstruction (GridRec) of Denoised Experimental data by Fake Noise machine learning.
  
.. figure:: img/low_dose-sacra.png
    :scale: 100%
    :align: center

    This is the reconstruction (GridRec) of Denoised Experimental data by Sacraficial Sample machine learning.
    

Fake Noise - Denoising
======================

This is the first of two methods of experimental denoising. This method applies np.poisson() to the raw projection data, uses this new noise data as the "noisy training data", while the origianl data is used as the "clean data" during the network training process.
        

Determining Appropriate Fake Noise Level
-----------------------------------------
First we must determine the appropriate noise level to add to the experimental data. image_step allows us to apply noise to a select projection rather than waiting, while noise allows the User to set the np.poisson() noise level. This function will display two images, one of the origianl projections and one of the newly created noisy projection.
    
.. code:: python

    from tomosuite.low_dose.data_prep import noise_test_tomogan


    noise = 20
    image_step = 1000
    
    noise_test_tomogan(basedir,
                        image_step=image_step,
                        noise=noise,
                        figsize=(15, 15))

    
    
Create TomoGAN Files
--------------------
This function allows the User to apply the noise level to each projection in the project. These are seperated from the original projection files.
    
.. code:: python

    from tomosuite.low_dose.data_prep import setup_tomogan_fake_noise
    setup_tomogan_fake_noise(basedir,
                                noise=noise)
    
    
Training TomoGAN
================
Allows the User to train TomoGAN on these newly created noisy and clean image pairs. Training progress can be viewed in Tensorboard by running tensorboard --logdir='/local/data/project_01/low_dose/logs/' --samples_per_plugin=images=300

    
.. code:: python

    from tomosuite.low_dose.tomogan import train_tomogan
    train_tomogan(basedir, epochs=120001, gpus='0',
                    lmse=0.5, lperc=2.0, 
                    ladv=20, lunet=3, depth=1,
                    itg=1, itd=2, mb_size=2,
                    img_size=896)
    
    
Predicting TomoGAN
==================
Once an appropriate epoch has been chosen through Tensorboard one can use this epoch to predict the denoised projections.
    
.. code:: python

    denoised_epoch = '22000'

    from tomosuite.low_dose.tomogan import predict_tomogan
    output = tomosuite.predict_tomogan(basedir, 
                                        weights_iter=denoise_epoch,
                                        second_basedir=None,
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
