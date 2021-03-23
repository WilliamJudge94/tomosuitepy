========================================
DeSparse Angle With Rife Neural Network
========================================

If any User would like to improve sparse angle tomographic scans, please use the following protocols. 


Placing Projections Into .MP4
==============================


.. code:: python

    from tomosuite.easy_networks.dain.data_prep import create_prj_mp4, rife_predict, obtain_frames


    output = create_prj_mp4(basedir, sparse_angle_removal=1, fps=10, apply_exp=False)
    

Obtain Network Prediction Command
=================================
    
.. code:: python   

    rife_predict(basedir, exp=2)
    
 
Read The Network Prediction And Save New Projections
====================================================

.. code:: python   

    frames = obtain_frames(basedir, video_type='predicted', output_folder='frames')
    
    
 
Use TomoSuite To Reconstruct New Frames
=======================================

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
                                            start_row=0,
                                            end_row=1,
                                            power2pad=True,
                                            dain_types=[frames_folder, output_image_type, apply_log])


    fig = plot_reconstruction(slcs_v1[0:1], clim=(0, 0.28), cmap='rainbow')