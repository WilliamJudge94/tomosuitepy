TomoSuitePY
===========

[![Build Status](https://app.travis-ci.com/WilliamJudge94/tomosuitepy.svg?token=37Jbsp5zphrcthqkyvzn&branch=main)](https://app.travis-ci.com/WilliamJudge94/tomosuitepy)


TomoSuitePY is a culmination of machine learning networks and data preparation methods for those networks to enhance poor-quality tomographic datasets. Novel implementations of existing architectures are available for Users to de-noise, de-wedge artifact, de-sparse angle artifact, and de-ring artifact their own datasets without the need for obtaining any true ground truth datasets. This project aims not only to bring these networks into the tomographic world, but to make them as user friendly as possible, expanding the usability to novices in the machine learning domain.

Motivation & Features
---------------------

- Wrapper for the tomopy python module.
- Makes resource limited extraction and reconstruction of tomography datasets easy.
- Allows for easy use of TomoGAN network (Low Dose Noise Correction).
- Allows for easy use of RIFE network (Sparse Angle Interpolation)

BASIC Installation (git)
-------------------------

- conda create -n test_basic python=3.6
- conda env update -n test_basic --file /location/to/tomosuitpy/github/clone/envs/basic.yml

- source activate test_basic

- ipython kernel install --user --name=test_basic


Usage
-----

Please see the https://tomosuitepy.readthedocs.io/en/latest/ for more details.


License
-------

This project is released under the `GNU General Public License version 2`.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

GNU General Public License version 2: https://www.gnu.org/licenses/gpl-2.0.en.html

