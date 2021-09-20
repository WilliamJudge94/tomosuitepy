========================================
DeSparse Angle With Rife Neural Network
========================================

If any User would like to improve sparse angle tomographic scans, please use the following protocols. 


Loading In TomoSuitePY
======================

.. code:: python

    import sys
    sys.path.append('/location/of/tomosuitepy/github_repo/')


Setting Up Data
===============

Please see the Start Project documentation


Placing Projections Into .MP4 - Jupyter
=======================================


.. code:: python

    from tomosuite.easy_networks.dain.data_prep import create_prj_mp4, rife_predict, obtain_frames


    output = create_prj_mp4(basedir, # Project file - definition in Start Project docs
                            sparse_angle_removal=1, # Use ever x frame
                            fps=10, # fps of output movie - 10fps is standard
                            apply_exp=False # If the User would like to apply a log to the frames
                            )
    

Obtain Network Prediction Command - Jupyter + Command Line
==========================================================
    
.. code:: python   

    # Take the output of this command and run it through your terminal with the rife conda environment activated
    rife_predict(basedir, exp=2)
    
 
Read The Network Prediction And Save New Projections - Jupyter
==============================================================

.. code:: python   

    frames = obtain_frames(basedir, video_type='predicted', output_folder='frames')
    


Obtain Rotation Center
======================

Please see Find Rotation Center documentation.
    
 
Use TomoSuite To Reconstruct New Frames - Jupyter
=================================================

.. code:: python   
     
    import tomopy
    from tomosuite.base.reconstruct import reconstruct_data, plot_reconstruction
    
    def tomo_recon(prj, theta, rot_center, user_extra=None):

        recon = tomopy.recon(prj, theta,
                            center=rot_center,
                            algorithm='gridrec',
                            ncore=8)
        return recon, user_extra


    frames_folder = 'frames'
    output_image_type = '.tif'
    apply_log = False

    slcs_v1, user_extra = reconstruct_data(basedir,
                                            rot_center=598, 
                                            reconstruct_func=tomo_recon,
                                            network='dain',
                                            start_row=None,
                                            end_row=None,
                                            power2pad=True,
                                            dain_types=[frames_folder, output_image_type, apply_log])


    fig = plot_reconstruction(slcs_v1[0:1], clim=(0, 0.28), cmap='rainbow')