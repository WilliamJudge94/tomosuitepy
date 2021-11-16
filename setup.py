import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name="tomosuitepy",
      version=read("VERSION.txt"),
      description="Machine learning pipelines for tomographic datasets",
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      author="William Judge",
      author_email="williamjudge94@gmail.com",
      url="https://github.com/WilliamJudge94/tomosuitepy",
      keywords="Tomography CT X-ray microscopy Argonne Advanced Photon Source",
      install_requires=[
      ],
      python_requires='==3.6',
      packages=['tomosuitepy',],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
          'Natural Language :: English',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Visualization',
      ]
)