TomoSuitePY
===========

[![Build Status](https://app.travis-ci.com/WilliamJudge94/tomosuitepy.svg?token=37Jbsp5zphrcthqkyvzn&branch=main)](https://app.travis-ci.com/WilliamJudge94/tomosuitepy)


TomoSuitePY is a culmination of machine learning networks and data preparation methods for those networks to enhance poor-quality tomographic datasets. Novel implementations of existing architectures are available for Users to de-noise, de-wedge artifact, de-sparse angle artifact, and de-ring artifact their own datasets without the need for obtaining any true ground truth datasets. This project aims not only to bring these networks into the tomographic world, but to make them as user friendly as possible, expanding the usability to novices in the machine learning domain.

Motivation & Features
---------------------



BASIC Installation (git)
-------------------------

# Installing basic packages
conda create -n test_basic python=3.6
conda env update -n test_basic --file /location/to/tomosuitpy/github/clone/envs/basic.yml

source activate test_basic

pip install pandas
pip install pympler

pip install ipykernel
ipython kernel install --user --name=test_basic


Usage
-----

Please see the https://tomosuitepy.readthedocs.io/en/latest/ for more details


License
-------

This project is released under the `GNU General Public License version 3`_.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

.. _GNU General Public License version 3: https://www.gnu.org/licenses/gpl-3.0.en.html

