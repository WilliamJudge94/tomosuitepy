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

The Project
===========

.. code:: python

    # The directory path where the raw experimental file lives
    datadir = '/local/data/experimental/'
    
    # The file name of the data the User would like to import
    fname = 'Experiment_01.h5'
    
    # The folder path where the User would like to store project data to
    basedir = '/local/data/project_01/'
    
    
    tomosuite.start_project(basedir=basedir)
    
    tomosuite.extract(datadir=datadir,
                        fname=fname,
                        basedir=basedir,
                        extraction_func=dxchange.read_aps_32id,
                        binning=1,
                        starting=0,
                        dtype='float32',
                        flat_roll=None,
                        overwrite=True,
                        verbose=True,
                        save=True)
                        
                        
    # Optional Step - skipping low-dose machine learning
    tomosuite.skip_lowdose(basedir=basedir)