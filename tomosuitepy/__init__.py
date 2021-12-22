import sys
import os
import pathlib

cwd1 = pathlib.Path.cwd()
cdw2 = cdw1.joinpath('hard_networks', 'TomoGAN')
cdw3 = cdw1.joinpath('hard_networks', 'generative_inpainting')
cdw4 = cdw1.joinpath('hard_networks', 'noise2noise')

sys.path.append(cdw1)
sys.path.append(cdw2)
sys.path.append(cdw3)
sys.path.append(cdw4)

#sys.path.append(os.path.dirname(__file__))
#sys.path.append(f'{os.path.dirname(__file__)}/hard_networks/TomoGAN/')
#sys.path.append(f'{os.path.dirname(__file__)}/hard_networks/generative_inpainting/')
#sys.path.append(f'{os.path.dirname(__file__)}/hard_networks/noise2noise/')

import imageio
import numpy as np
from tqdm import tqdm
import tifffile as tif
import matplotlib.pyplot as plt

#from tomosuite.base.common import load_extracted_prj, load_extracted_theta, skip_lowdose
#from tomosuite.base.start_project import start_project
#from tomosuite.base.extract_projections import extract
#from tomosuite.base.reconstruct import reconstruct_data_tomogan, reconstruct_data_deepfillv2, plot_reconstruction


#from tomosuite.low_dose.data_prep import setup_tomogan_fake_noise, noise_test_tomogan
#from tomosuite.low_dose.tomogan import train_tomogan, predict_tomogan, save_noise_tomogan


#from tomosuite.inpainting.data_prep import *
#from tomosuite.inpainting.inpainting import *
