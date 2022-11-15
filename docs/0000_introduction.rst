.. _installation:

======================
TomoSuitePY Overview
======================

.. warning::

TomoSuitePY DOES NOT work with RedHat.


Where To Use
=============

TomoSuitePY was built to be run on collected tomography datasets from various sychrotron light sources. The typical workflow is as follows:

- Collect sparse angle or low dosage tomography projections from a light source
- import data into tomosuitepy module format
- follow instructions to use RIFE or TomoGAN network for user defined needs

Use Cases
=========

TomoSuitePY's RIFE network works best on spherical data or roughly round objects. TomoGAN network has been validated for a variety of shapes.