.. _reconstructions:

===============================
Reconstructions Through TomoPy
===============================



Raw .H5 Data
============

In order to reconstruct the raw .H5 data one must call the tomosuite.skip_lowdose() function. This saves the projection files to a numpy array inside the loction f'{basedir}low_dose/noise_exp_data.npy'. If the Low Dose portion of TomoSuite is used this file is created automatically. Along with a file falled f'{basedir}low_dose/denoise_exp_data.npy' OR f'{basedir}low_dose/denoise_fake_data.npy' depending on which method of denoising is performed.

View Denoised Data - Raw H5
---------------------------


.. code:: python
    
    
    from tomosuite.base.reconstruct import reconstruct_data_tomogan, plot_reconstruction

    tomosuite.skip_lowdose(basedir)
    
    slcs = reconstruct_data_tomogan(basedir,
                                    rot_center=500,
                                    start_row=None,
                                    end_row=None,
                                    wedge_removal=0,
                                    sparce_angle_removal=1,
                                    med_filter=False,
                                    all_data_med_filter=False,
                                    types='noise_exp',
                                    med_filter_kernel=(1, 3, 3),
                                    second_basedir=None,
                                    reconstruct_func=tomo_recon)
    plot_reconstruction(slcs[0:2])
    
    
Denoised Data
=============

View Denoised Data - Low Dose
-----------------------------

Once the Low Dose protocol is complete use the following methods to view the resulting data.

.. code:: python
    
    
    from tomosuite.base.reconstruct import reconstruct_data_tomogan, plot_reconstruction

    
    # Original
    slc = reconstruct_data_tomogan(basedir,
                                    rot_center=624,
                                    types='noise_exp')
    plot_reconstruction(slc[0:2])
    
    # Denoised
    slc = reconstruct_data_tomogan(basedir,
                                    types='denoise_exp')
    plot_reconstruction(slc[0:2])
    
    
Missing Wedge Data
==================
    
    
View Simulated Wedge Artifact
-----------------------------

After calling the 'fake_missing_wedge()' function for the DeWedge artifacting protocol, one might like to view the reconstruction of these projections.


View Inpainted Sinogram Missing Wedge Data
------------------------------------------


.. code:: python
    
    
    from tomosuite.base.reconstruct import reconstruct_data_deepfillv2, plot_reconstruction
    
    
    slc = reconstruct_data_deepfillv2(basedir, load_epoch, rot_center=383)
    plot_reconstruction(slc[0:10], clim=(0, 1))


Defining Your Own Recon Function
================================


.. code:: python 
    
    # The inputs have to be prj, theta, and rot_center
    # You can define this function however you like and pass your
    #new function into the tomosuite recon parameters
    
    # This is the standard defined tomo_recon function through TomoSuite

    def tomo_recon(prj, theta, rot_center):

        types='gridrec'

        #prj = tomopy.remove_stripe_ti(prj, 2)
        if types == 'gridrec':
            recon = tomopy.recon(prj, theta,
                                center=rot_center,
                                algorithm='gridrec',
                                ncore=16)             
            recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
            
        elif types == 'gridrec_parzen':
            recon = tomopy.recon(prj, theta,
                                center=rot_center,
                                algorithm='gridrec',
                                ncore=16,
                                filter_name='parzen')              
            recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)

        elif types == 'sirt':
            extra_options ={'MinConstraint':0}
            options = {'proj_type':'cuda', 'method':'SIRT_CUDA',
                        'num_iter':200, 'extra_options': extra_options}
            recon = tomopy.recon(prj, theta,
                                    center=rot_center,
                                    algorithm=tomopy.astra,
                                    ncore=1, options=options)

        #Remove ring artifacts, this comes with a slight resolution cost
        #recon = tomopy.remove_ring(recon, center_x=None, center_y=None, thresh=300.0)

        return recon
        
        
    from tomosuite.base.reconstruct import reconstruct_data_deepfillv2, plot_reconstruction
    
    
    slc = reconstruct_data_deepfillv2(basedir,
                                    load_epoch,
                                    rot_center=383,
                                    reconstruct_func=tomo_recon)

    plot_reconstruction(slc[0:10], clim=(0, 1))