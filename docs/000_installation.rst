.. _installation:

===============================
Installation of TomoSuitePY
===============================

.. warning::

TomoSuitePY DOES NOT work with RedHat.


Installing Conda Environment To Jupyter
=======================================

For each environment below please use the following to add each environment to jupyter.
While replacing thre environment name with the appropriate name on your computer.

.. code:: python

    source activate env_tomosuite

    # Installing ipykernel so the environment can be used in jupyter
    ipython kernel install --user --name=env_tomosuite

    # install where you are running your jupyter lab - allows for interactive viewing
    jupyter labextension install @jupyter-widgets/jupyterlab-manager


Basic Conda Environment
=======================

Basics of tomosuite (excluding RIFE, Noise2Noise, TomoGAN, and Deepfill networks).

.. code:: python

    # Installing basic packages
    conda create -n test_basic python=3.6
    conda env update -n basic --file /location/to/tomosuitpy/github/clone/envs/basic.yml
    
    source activate basic
    
    # Initilizing env for Jupyter
    ipython kernel install --user --name=basic


TomoGAN and Noise2Noise Conda Environment
==========================================

Install the required packages to be used for TomoGAN, Noise2Noise, and RIFE networks.

.. code:: python

    # Installing TomoGAN and Noise2Noise packages
    conda create -n test_tomogan python=3.6
    conda env update -n test_tomogan --file /location/to/tomosuitpy/github/clone/envs/tomogan_n2n.yml
    
    source activate test_tomogan
    
    pip install pandas
    pip install pympler
    pip install ipykernel
    
    ipython kernel install --user --name=test_tomogan


RIFE Conda Environment
======================

Install the required packages to be used for RIFE networks.

.. code:: python

    # Installing RIFE packages
    conda create -n rife_tomosuitepy python=3.6
    conda env update -n rife_tomosuitepy --file /location/to/tomosuitpy/github/clone/envs/rife.yml
    
    source activate rife_tomosuitepy
    
    ipython kernel install --user --name=rife_tomosuitepy


Installing Through (PyPi)
==========================

.. warning::

The PyPi package has no dependencies listed.
Users must complete the instructions listed above
before installing the PyPi version of TomoSuitePY.

.. code:: bash
    
    source activate conda_env
    pip install tomosuitepy