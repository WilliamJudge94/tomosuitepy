.. _starting_project:

==================
Starting A Project 
==================

In order to begin using TomoSuitePY, one must create a project for their .h5 data.
Sometimes it is necessary to create multiple projects for a single task,
but this is only when one is to use a sacraficial sample for network training.
All use cases of a second project are detailed in the documentation of TomoSuitePY.


Importing TomoSuitePY (dev)
===========================

.. code:: python

    import sys
    sys.path.append('/path/to/tomosuitepy/github/clone/tomosuitepy/')
    import tomosuitepy

    from tomosuitepy.base.start_project import start_project
    from tomosuitepy.base.extract_projections import extract

    
Importing TomoSuitePY (PyPi)
===========================

.. warning::

The PyPi package has no dependencies listed.
Users must complete the conda install instructions (:ref:`installation`.)
before installing/using the PyPi version of TomoSuitePY.

.. code:: python

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
                chunking_size=10, # Set chunking_size to 1 for no chunking
                nan_inf_selective=True,
                remove_neg_vals=True,
                removal_val=0.001,
                )
                
.. note::
    
For most extractions the above is acceptable. However, when your material has high absorbance
the detector counts may equal the background counts. Creating significant zero or nan values in
the dataset. The nan_inf_selective variable is not well suited for this task. To mitigate the 
effects of highly absorbing materials please set the following:
nan_inf_selective=False, remove_nan_vals=True, remove_inf_vals=True, removal_val=0.001.
While the nan_inf_selective applies an intelligent median blur to the non-finite values,
these new settings will replace non-finite values with the value set as removal_val.
    
    
Command Line Interface (CLI)
============================

TomoSuitePY also comes with a command line interface for project setup and projection extraction.
The following can be run in a bash terminal, however it does have limited features compared to 
it's Jupyter function counterpart.

.. bash::

    source activate basic_env
    
    cd /path/to/tomosuitepy/github/clone/
    cd /tomosuitepy/cli/
    
    python base.py extract --help
    
    python base.py extract --file /path/2/h5/file/ --basedir /path/2/project/dir/2/create/
    

.. note::

Right now only APS Sector 32 is available for CLI integration. If Users would like to add more
please edit the base.py file. Go to the def extract() function and add in the desired dxchange
function to the dxchange_reader = {} dictionary.

