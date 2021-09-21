.. _reconstructions:

===============================
Installation of TomoSuitePY
===============================



Installing Conda Environment To Jupyter
=======================================

For each environment below please use the following to add each environment to jupyter. Replace env_ with the appropriate name

.. code:: python

    source activate env_tomosuite

    # Installing ipykernel so the environment can be used in jupyter
    conda install ipykernel

    ipython kernel install --user --name=env_tomosuite



Basic Conda Environment
=======================

Basics of tomosuite (excluding RIFE, Noise2Noise, TomoGAN, and Deepfill networks).

.. code:: python

    # Installing basic packages
    conda create -n basic_tomosuite --file /location/to/tomosuitpy/github/clone/envs/basic.yml
    pip install pandas
    pip install pympler


TomoGAN and Noise2Noise Conda Environment
==========================================

Install the required packages to be used for TomoGAN, Noise2Noise, and RIFE networks.

.. code:: python

    # Installing TomoGAN and Noise2Noise packages
    conda create -n tomogan_n2n_tomosuite --file /location/to/tomosuitpy/github/clone/envs/tomogan_n2n.yml
    pip install pandas
    pip install pympler
    pip install itk
    pip install itkwidgets


DeepFillV2 Conda Environment
============================

Install the required packages to be used for DeepfillV2 networks.

.. code:: python

    # Installing TomoGAN and Noise2Noise packages
    conda create -n deepfill_tomosuite --file /location/to/tomosuitpy/github/clone/envs/deepfillv2.yml
    pip install git+https://github.com/WilliamJudge94/neuralgym
    pip install pandas
    pip install pympler
    pip install itk
    pip install itkwidgets

RIFE Conda Environment
======================

Install the required packages to be used for RIFE networks.

.. code:: python

    # Installing RIFE packages
    conda create -n rife_tomosuite --file /location/to/tomosuitpy/github/clone/envs/rife.yml
    pip install pandas
    pip install pympler
    pip install itk
    pip install itkwidgets


DO NOT USE
======================

.. code:: python

    # DO NOT INSTALL THESE - FOR TROUBLESHOOTING ONLY
    #cd /location/of/tomosuitepy_github/repo/hard_networks/RIFE/arXiv2020-RIFE/
    #pip3 install -r requirements.txt
    #pip install torchvision==0.9.0

    #conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

