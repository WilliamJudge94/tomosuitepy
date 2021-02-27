.. _reconstructions:

===============================
Reconstructions Through TomoPy
===============================



Basic Reconstructions
=====================

This section will show a User how to use the raw hdf5 projections to reconstruct a CT scan. The basics are to import the correct modules, define a reconstruction function, and pass that function to the reconstruct_data function which handels everything else. The reconstruct_data function is a wrapper which has the ability to pull data from many different directories. Most use cases will be explained somewhere on this page.

.. code:: python

    import sys
    sys.path.append('/path/to/tomosuite/github/clone/tomosuite/')
    import tomosuite
    import tomopy

    # Import TomoSuite helper functions
    from tomosuite.base.reconstruct import reconstruct_data, plot_reconstruction

    # Define your own tomography reconstruction function. This is the TomoSuite's default
    def tomo_recon(prj, theta, rot_center, user_extra=None):
        recon = tomopy.recon(prj, theta,
                            center=rot_center,
                            algorithm='gridrec',
                            ncore=30)
        return recon, user_extra

    # Reconstruct the raw projection data
    basedir = '/local/data/project_01/' 

    slcs, user_extra = reconstruct_data(basedir,
                        rot_center,
                        start_row=None, # If the User doesnt want to reconstruct all rows
                        end_row=None, # If the User doesnt want to reconstruct all rows
                        med_filter=False,
                        all_data_med_filter=False,
                        med_filter_kernel=(1, 3, 3),
                        reconstruct_func=tomo_recon, # Allows the user to define their own recon function
                        network=None, # None loads in the raw projection data
                        wedge_removal=0, # zero out the first and last wedge_removal number of projections
                        sparse_angle_removal=1, # only use every sparse_anlge_removal image for the recon
                        types='denoise', # used when network='TomoGAN'
                        second_basedir=None, # used to pull a second datasets data for the reconstruction
                        checkpoint_num=None, # used when network='Deepfillv2'
                        double_sparse=None, 
                        power2pad=False, # Force the sinogram to be of shape power2
                        edge_transition=None # eliminate columns of sinogram which cause a harsh ring effect in the recon)

    # Plot the reconstruction
    plot_reconstruction(slcs[0:1])


DeNoise Type 1 or Type 2
========================

Once the User predicts through tomogan they now have the ability to reconstruct that predicted data. In this case we are looking at DeNoise Type 1 or Type 2. Type 1 is where the User has imput fake noise into their projections, and used tomogan to denoise the original projections. While Type 2 is where the User has a sacraficial sample, which contains multiple projections of the save FOV. 

The main concept is similar to that of the basic reconstruction. The main difference is now the User has to define the network='tomogan' and the types='denoise_fake' for Type 1 or types='denoise_exp' for Type 2. This tells the reconstruct_data function to import the data related to tomogan and make sure you import the denoised data based on the fake noise training or the sacraficial sample training. 

.. code:: python

    import sys
    sys.path.append('/path/to/tomosuite/github/clone/tomosuite/')
    import tomosuite
    import tomopy

    # Import TomoSuite helper functions
    from tomosuite.base.reconstruct import reconstruct_data, plot_reconstruction

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

    


Defining Your Own Recon Function
================================

In this seciton the User will learn how to define their own reconstruction function. to do this one must follow the template below of a function defined as tomo_recon(prj, theta, rot_center, user_extra=None). The user_extra parameter allows the user to pass data out of the recon function. This is mainly for debugging purposes. Next the User has to make sure that this tomo_recon function returns recon, user_extra. Everything in between can be set by the User. make sure you are using tomopy.recon() to reconstruct the slices.


.. code:: python 
    
    # The inputs have to be prj, theta, and rot_center
    # You can define this function however you like and pass your
    #new function into the tomosuite recon parameters
    
    # This is the standard defined tomo_recon function through TomoSuite

    def tomo_recon(prj, theta, rot_center, user_extra=None):

        recon_type='gridrec'
        
        # Add preprocessing steps here
        #prj = tomopy.remove_stripe_ti(prj, 2)
        
        
        if recon_type == 'gridrec':
            recon = tomopy.recon(prj, theta,
                                center=rot_center,
                                algorithm='gridrec',
                                ncore=16)             
            recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
            
        elif recon_type == 'gridrec_parzen':
            recon = tomopy.recon(prj, theta,
                                center=rot_center,
                                algorithm='gridrec',
                                ncore=16,
                                filter_name='parzen')              
            recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)

        elif recon_type == 'sirt':
            extra_options ={'MinConstraint':0}
            options = {'proj_type':'cuda', 'method':'SIRT_CUDA',
                        'num_iter':200, 'extra_options': extra_options}
            recon = tomopy.recon(prj, theta,
                                    center=rot_center,
                                    algorithm=tomopy.astra,
                                    ncore=1, options=options)

        #Remove ring artifacts, this comes with a slight resolution cost
        #recon = tomopy.remove_ring(recon, center_x=None, center_y=None, thresh=300.0)

        return recon, user_extra
        
        
    from tomosuite.base.reconstruct import reconstruct_data, plot_reconstruction
    
    
    slcs, user_extra = reconstruct_data(basedir,
                        rot_center=600,
                        reconstruct_func=tomo_recon, 
                        network='tomogan',
                        types='denoise_fake', # or denoise_exp
                        )
                        
    plot_reconstruction(slc[0:10], clim=(0, 1))