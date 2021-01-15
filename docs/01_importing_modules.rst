===================
TomoSuite Functions 
===================

Unfortunatley, the compiled TomoSuite package use network architectures which are not compatable with one another with their python dependancies.
To mitigate the issue, it is necessary to import each function independently from one another depending on which section of the code the User would
like to use. Sorry, for this inconvenience. 


Importing Low Dose Functions
============================

.. code:: python

    # Requires tomogan_n2n env
    from tomosuite.low_dose.data_prep import noise_test_tomogan
    from tomosuite.low_dose.data_prep import setup_tomogan_fake_noise
    from tomosuite.low_dose.data_prep import setup_tomogan_exp_noise
    
    from tomosuite.low_dose.tomogan import train_tomogan
    from tomosuite.low_dose.tomogan import predict_tomogan
    

Importing Sinogram Inpainting Functions
=======================================

.. code:: python

    # Requires deepfill env
    import numpy as np

    from tomosuite.inpainting.data_prep import fake_missing_wedge
    from tomosuite.inpainting.data_prep import add_prj4missing_wedge
    from tomosuite.inpainting.data_prep import create_testing_data
    from tomosuite.inpainting.data_prep import create_training_data
    from tomosuite.inpainting.data_prep import make_file_list
    from tomosuite.inpainting.data_prep import determine_train_height
    from tomosuite.inpainting.data_prep import retrieve_inpainting

    from tomosuite.inpainting.deepfillv2 import setup_inpainting
    from tomosuite.inpainting.deepfillv2 import train_deepfillv2
    
    from tomosuite.inpainting.predictions import predict_deepfillv2
    
    
Importing Artifact Remover Functions
====================================

.. code:: python
    
    # Requires deepfill env
    from tomosuite.artifact.data_prep import setup_noise2noise_tomosuite_missingwedge
    
    # Required tomogan_n2n env
    from tomosuite.artifact.noise2noise import train_n2n
    from tomosuite.artifact.predictions import predict_n2n
    
Importing Reconstruction Functions
==================================


TomoGAN - Standard + Low Dose
-----------------------------

.. code:: python

    from tomosuite.base.reconstruct import reconstruct_data_tomogan
    from tomosuite.base.reconstruct import plot_reconstruction

DeepFillV2 - Sinogram Inpainting
--------------------------------


.. code:: python

    from tomosuite.base.reconstruct import reconstruct_data_deepfillv2
    from tomosuite.base.reconstruct import plot_reconstruction


