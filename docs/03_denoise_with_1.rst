======================
 Denoise with Type 1
======================

In this section the User will learn how to denoise tomographic projections by only training/using a neural network
with the same experimental dataset. 

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

    from tomosuite.base.start_project import start_project
    from tomosuite.base.extract_projections import extract

    from tomosuite.methods.denoise_type1 import denoise_t1_dataprep
    
    
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
    

Fake Noise - Denoising
======================

This is the first of two methods of experimental denoising.
This method applies np.poisson() to the raw projection data, uses this new noise data as the "noisy training data",
while the origianl data is used as the "clean data" during the network training process.
        

Determining Appropriate Fake Noise Level
-----------------------------------------
First we must determine the appropriate noise level to add to the experimental data. image_step allows us to
apply noise to a select projection rather than waiting, while noise allows the User to set the np.poisson() noise level.
This function will display two images, one of the origianl projections and one of the newly created noisy projection.
    
.. code:: python

    denoise_t1_dataprep.fake_noise_test(basedir,
                                    noise=125, # The noise level to apply to projections
                                    image_step=20, # Amount of images to skip (used to speed up code)
                                    plot=True,
                                    idx=0,
                                    figsize=(10, 10) )

    
    
Create TomoGAN Files
--------------------
This function allows the User to apply the noise level to each projection in the project. These are seperated from the original projection files.
    
.. code:: python

    denoise_t1_dataprep.setup_fake_noise_train(basedir,
                                            noise=125,
                                            interval=5, # Every 5th datapoint will be used for training
                                            dtype=np.float32)
    
    
Training TomoGAN
================
Allows the User to train TomoGAN on these newly created noisy and clean image pairs. Training progress can be viewed in Tensorboard by running tensorboard --logdir='/local/data/project_01/low_dose/logs/' --samples_per_plugin=images=300

    
.. code:: python

    from tomosuite.easy_networks.tomogan.train import train_tomogan, tensorboard_command_tomogan

    # Prints out a command line script which will initiate a tensorboard instance to view TomoGAN training
    tensorboard_command_tomogan(basedir)

    train_tomogan(basedir,
                    epochs=120001,
                    gpus='0', # Set the GPU to use
                    lmse=0.5,
                    lperc=2.0, 
                    ladv=20,
                    lunet=3,
                    depth=1,
                    itg=1,
                    itd=2,
                    mb_size=2, # Batch size
                    img_size=896, # Size of images to randomly crop to
                    types='noise')
    
    
Predicting TomoGAN
==================
Once an appropriate epoch has been chosen through Tensorboard one can use this epoch to predict the denoised projections.
    
.. code:: python

    from tomosuite.easy_networks.tomogan.predict import predict_tomogan, save_predict_tomogan

    # Loading in the Projection Data
    data = denoise_t1_dataprep.setup_fake_noise_predict(basedir)

    clean_data, dirty_data = predict_tomogan(basedir,
                                    data,
                                    weights_iter='01000' # The epoch number to load weights of
                                    chunk_size=5, # Chunk the data so it doesnt overload GPU VRAM
                                    gpu='0', # Select which gpu to use
                                    lunet=3,
                                    in_depth=1,
                                    data_type=np.float32,
                                    verbose=False,
                                    types='noise')

    save_predict_tomogan(basedir,
                            good_data=clean_data,
                            bad_data=dirty_data,
                            second_basedir=None,
                            types='noise')


View Denoised Data
==================
Please visit :ref:`reconstructions`.
