.. _tomogan_fake_full:

====================================================
Using TomoGAN Network to DeNoise - FAKE NOISE - Full
====================================================

.. warning::

If you are using the PyPi version of TomoSuitePY,
Please view :ref:`installation` and :ref:`starting_project`
before attempting to load in the module. 


TomoGAN Conda Environment Installation - Command Line
=====================================================

Install the required packages to be used for TomoGAN and Noise2Noise networks.
Only contains options to use GridRec for reconstructions. You do not need base
conda environment unless you plan to use SIRT for reconstructions. 

.. code:: python

    # Installing TomoGAN and Noise2Noise packages
    conda create -n test_tomogan python=3.6
    conda env update -n test_tomogan --file /location/to/tomosuitpy/github/clone/envs/tomogan.yml
    
    source activate test_tomogan
    
    pip install pandas
    pip install pympler
    pip install ipykernel
    
    ipython kernel install --user --name=test_tomogan


Basic Conda Environment Installation - Command Line
====================================================

Basics of tomosuitepy (excluding RIFE, Noise2Noise, TomoGAN, and Deepfill networks)
- Used for GridRec/SIRT Reconstructions as well as projection extraction.

.. code:: python

    # Installing basic packages
    conda create -n test_basic python=3.6
    conda env update -n test_basic --file /location/to/tomosuitpy/github/clone/envs/basic.yml
    
    source activate test_basic
    
    pip install pandas
    pip install pympler
    
    pip install ipykernel
    ipython kernel install --user --name=test_basic




Starting A Project - Jupyter
============================

In order to begin using TomoSuitePY, one must create a project for their .h5 data. Sometimes it is necessary
to create multiple projects for a single task, but this is only when one is to use a sacraficial sample for
network training. All use cases of a second project are detailed in the documentation of TomoSuite


It is also imperative that the User has the test_basic conda enviroment installed for this part of the tutorial. 


Importing TomoSuitePY
---------------------

.. code:: python

    import sys
    sys.path.append('/path/to/tomosuitepy/github/clone/tomosuitepy/')

    from tomosuitepy.base.start_project import start_project
    from tomosuitepy.base.extract_projections import extract

Starting A Project
------------------

.. code:: python

    # The directory path where the raw experimental file lives
    datadir = '/local/data/experimental/'
    
    # The file name of the data the User would like to import
    fname = 'Experiment_01.h5'
    
    # The folder path where the User would like to store project data to
    basedir = '/local/data/project_01/'
    

    start_project(basedir=basedir)


    extract(datadir=datadir,
                fname=fname,
                basedir=basedir,
                chunking_size=10)


    

Fake Noise - Denoising
======================

This is the first of two methods of experimental denoising. This method applies np.poisson()
to the raw projection data, uses this new noise data as the "noisy training data",
while the origianl data is used as the "clean data" during the network training process.
        

Determining Appropriate Fake Noise Level
-----------------------------------------
First we must determine the appropriate noise level to add to the experimental data. image_step allows us to
apply noise to a select projection rather than waiting, while noise allows the User to set the np.poisson() noise level.
This function will display two images, one of the origianl projections and one of the newly created noisy projection.

It is advised to aim for -0.6 - 0.6 differential in the clean and noisy images. These metrics are shows as the title of the last plot output of this function call.
    
.. code:: python

    from tomosuitepy.methods.denoise_type1 import denoise_t1_dataprep

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
Allows the User to train TomoGAN on these newly created noisy and clean image pairs.
Training progress can be viewed in Tensorboard by running tensorboard --logdir='/local/data/project_01/low_dose/logs/' --samples_per_plugin=images=300

    
.. code:: python

    from tomosuitepy.easy_networks.tomogan.train import train_tomogan, tensorboard_command_tomogan

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
                    img_size=512, # Size of images to randomly crop to
                    types='noise')
    
    
Predicting TomoGAN
==================
Once an appropriate epoch has been chosen through Tensorboard one can use this epoch to predict the denoised projections.
    
.. code:: python

    from tomosuitepy.easy_networks.tomogan.predict import predict_tomogan, save_predict_tomogan
    from tomosuitepy.base.common import load_extracted_prj

    # Loading in the Projection Data
    data = load_extracted_prj(basedir)

    clean_data, dirty_data = predict_tomogan(basedir,
                                    data,
                                    weights_iter='01000', # The epoch number to load weights of
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



Reconstructions - TomoGAN - FAKE NOISE
======================================

Once the User predicts through tomogan they now have the ability to reconstruct that predicted data.
In this case we are looking at DeNoise Type 1 or Type 2. Type 1 is where the User has imput fake noise into their projections,
and used tomogan to denoise the original projections. While Type 2 is where the User has a sacraficial sample, which contains multiple projections of the save FOV. 

The main concept is similar to that of the basic reconstruction. The main difference is now the User has to define the network='tomogan'
and the types='denoise_fake' for Type 1 or types='denoise_exp' for Type 2. This tells the reconstruct_data function to import the data
related to tomogan and make sure you import the denoised data based on the fake noise training or the sacraficial sample training. 

.. code:: python

    import tomosuitepy
    import tomopy

    # Import TomoSuite helper functions
    from tomosuitepy.base.reconstruct import reconstruct_data, plot_reconstruction

    # Define your own tomography reconstruction function. This is the TomoSuite's default
    def tomo_recon(prj, theta, rot_center, user_extra=None):
        recon = tomopy.recon(prj, theta,
                            center=rot_center,
                            algorithm='gridrec',
                            ncore=30)
        recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
        return recon, user_extra

    # Reconstruct the raw projection data
    basedir = '/local/data/project_01/' 

    slcs, user_extra = reconstruct_data(basedir,
                        rot_center=600,
                        reconstruct_func=tomo_recon, 
                        network='tomogan',
                        types='denoise_fake', # or denoise_exp if you are reconstructing Type 2
                        power2pad=True, # forces the sinogram to be in a power of 2 shape
                        edge_transition=5 # removes harsh edge on sinogram
                        )

    # Plot the reconstruction
    plot_reconstruction(slcs[0:10])
    
    
Command Line Interface (CLI)
============================

TomoSuitePY also comes with a command line interface for TomoGAN with fake poisson noise.
The following can be run in a bash terminal, however it does have limited features compared to 
it's Jupyter function counterpart.

.. bash::

    source activate basic_env
    
    cd /path/to/tomosuitepy/github/clone/
    cd /tomosuitepy/cli/
    
    python tomogan.py test-noise --help
    
    python tomogan.py setup-noise --help
    
    python tomogan.py train --help
     
    python tomogan.py predict --help
    
    python base.py find-centers --help
    
    python base.py recon --help
    
    python base.py recon --network tomogan --types denoise_fake


