.. _starting_project:

==================
Starting A Project 
==================

In order to begin using TomoSuite, one must create a project for their .h5 data. Sometimes it is necessary to create multiple projects for a single task, but this is only when one is to use a sacraficial sample for network training. All use cases of a second project are detailed in the documentation of TomoSuite


Importing TomoSuite
===================

.. code:: python

    import sys
    sys.path.append('/path/to/tomosuite/github/clone/tomosuite/')
    import tomosuite

    from tomosuite.base.start_project import start_project
    from tomosuite.base.extract_projections import extract

The Project
===========

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
                        extraction_func=dxchange.read_aps_32id,
                        binning=1, # shink size of projections
                        starting=0, # starting number for saved tiffs
                        dtype='float32', 
                        flat_roll=None, # roll the flat field image left or right
                        overwrite=True,
                        verbose=True,
                        save=True,
                        outlier_diff=None, # outlier_diff for remove_outlier
                        outlier_size=None, # outlier_size for remove_outlier
                        bkg_norm=True, # apply a background normalization
                        custom_dataprep=False) # skip all dataprep if True