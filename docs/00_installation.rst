.. _reconstructions:

===============================
Installation of TomoSuitePY
===============================



Creating Virtual Environment
============================

This will show users how to install tomosuitepy packages to be used for reconstructions through a jupyter lab environment.

.. code:: python

    # Creating a virtual environment through conda
    conda create -n env_tomosuite python=3.6

    source activate env_tomosuite

    # Installing ipykernel so the environment can be used in jupyter
    conda install ipykernel

    ipython kernel install --user --name=env_tomosuite



Basic Conda Environment
=======================

Basics of tomosuite (excluding RIFE, Noise2Noise, TomoGAN, and Deepfill networks).

.. code:: python

    # Installing basic packages
    conda env update -n basic_tomosuite --file /location/to/tomosuitpy/github/clone/envs/basic.yml
    pip install pandas
    pip install pympler


TomoGAN and Noise2Noise Conda Environment
==========================================

Install the required packages to be used for TomoGAN and Noise2Noise networks.

.. code:: python

    # Installing TomoGAN and Noise2Noise packages
    conda env update -n basic_tomosuite --file /location/to/tomosuitpy/github/clone/envs/tomogan_n2n.yml
    pip install pandas
    pip install pympler

DeepFillV2 Conda Environment
============================

Install the required packages to be used for DeepfillV2 networks.

.. code:: python

    # Installing TomoGAN and Noise2Noise packages
    conda env update -n basic_tomosuite --file /location/to/tomosuitpy/github/clone/envs/deepfillv2.yml
    pip install git+https://github.com/WilliamJudge94/neuralgym
    pip install pandas
    pip install pympler

RIFE Conda Environment
======================

Install the required packages to be used for RIFE networks.

.. code:: python

    # Installing RIFE packages
    conda env update -n basic_tomosuite --file /location/to/tomosuitpy/github/clone/envs/basic.yml
    pip install pandas
    pip install pympler

    cd /location/of/tomosuitepy_github/repo/hard_networks/RIFE/arXiv2020-RIFE/
    pip3 install -r requirements.txt
    pip install torchvision  

