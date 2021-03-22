===================================
Finding The Rotation Center
===================================



General Trends of Rotation Center
=================================

.. code:: python

    from tomosuite.base.rotation_center import obtain_rotation_center

    out = obtain_rotation_center(basedir_6530,
                                pixel_shift_range=60, # The range of rotation centers to try based upon the absolute center of the image
                                number2zero=None, # Number of projections to zero out from the beginning and end
                                crop_sinogram=75, # Number of columns to remove of the sinogram - helps remove errors from experimental substrate
                                med_filter=False, # Apply a median filter
                                min_val=0.18 # Zero out values less than this value - All sinograms are normalized to 1
                                )
    
    
    
.. figure:: img/general_trends-TomoSuite.png
    :scale: 50%
    :align: center
    
    
Interactive Fine Tune Rotation Center
=====================================

.. code:: python


    from tomosuite.base.reconstruct import reconstruct_data, plot_reconstruction_centers
    basedir = '/local/data/path/'

    slcs, user_extra = reconstruct_data(basedir,
                        rot_center=616, # This has no relevence when rot_center_shift_check is enabled
                        start_row=500, # Keep this to a single image for rotation_center_check
                        end_row=501, # Keep this to a single image for rotation_center_check
                        reconstruct_func=tomo_recon, # Allows the user to define their own recon function
                        network=None, #  Keep this to None for rotation_center_check
                        power2pad=False, #  Keep this to False for rotation_center_check
                        edge_transition=None, # Keep this to None for rotation_center_check
                        chunk_recon_size=1, 
                        rot_center_shift_check=40 # Number of rotation centers to try before and after absolute image center
                                       )
     # absolute_middle_rotation is printed out when rot_center_shift_check is initalized                                 
                                       
    plot_reconstruction_centers(slcs[0:], clim=(0, 0.01), absolute_middle_rotation=612, figsize=(20, 20))
    

.. figure:: img/human_tuned_v2.png
    :scale: 50%
    :align: center