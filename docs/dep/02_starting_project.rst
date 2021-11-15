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
                chunk_size4downsample=10) # Set chunk_size4downsample to 1 if you have a lot of RAM
