.. _starting_project:

==================
Starting A Project 
==================

In order to begin using TomoSuitePY, one must create a project for their .h5 data.
Sometimes it is necessary to create multiple projects for a single task,
but this is only when one is to use a sacraficial sample for network training.
All use cases of a second project are detailed in the documentation of TomoSuitePY


Importing TomoSuitePY
=====================

.. code:: python

    import sys
    sys.path.append('/path/to/tomosuitepy/github/clone/tomosuitepy/')
    import tomosuitepy

    from tomosuitepy.base.start_project import start_project
    from tomosuitepy.base.extract_projections import extract

The Project
===========

The project consists of the main folder structure and the extracted projections.
Extraction documentation can be found in the tomosuitepy.base.extract_projections.extract() function.
It is also possible to use your own extracted data instead of relying on dxchange.

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
                chunking_size=10,# Set chunk_size4downsample to 1 if you have a lot of RAM
                
                ) 
