import unittest
import os, sys
import numpy as np
from pympler import muppy, summary
import pathlib
import dxchange
import shutil

cwd1 = pathlib.Path('.').absolute().parents[0]
sys.path.append(f'{cwd1}')

cwd2 = pathlib.Path('..').absolute().parents[0]
sys.path.append(f'{cwd2}')

test_path = os.path.dirname(__file__)
main_path = '{}/'.format(test_path)

from tomosuitepy.easy_networks.tomogan.data_prep import *
from tomosuitepy.easy_networks.rife.data_prep import *
from tomosuitepy.base.start_project import start_project
from tomosuitepy.base.extract_projections import extract

muppy_ammount = 1000

class DataPrep(unittest.TestCase):

    def test_format_data_tomogan(self):
        clean_data = np.concatenate((np.ones((10, 100, 100)), np.zeros((10, 100, 100))))
        noisy_data = np.concatenate((np.zeros((10, 100, 100)), np.ones((10, 100, 100))))   
        xtrain, ytrain, xtest, ytest = format_data_tomogan(clean_data, noisy_data,
                                                           interval=5, dtype=np.float32)
                                    
        tf_xtrain = np.array_equal(xtrain, np.concatenate((np.zeros((2, 100, 100)), 
                                                           np.ones((2, 100, 100)))))
                                    
        tf_ytrain = np.array_equal(ytrain, np.concatenate((np.ones((2, 100, 100)),
                                                           np.zeros((2, 100, 100)))))
                                    
        tf_xtest = np.array_equal(xtest, np.concatenate((np.zeros((8, 100, 100)),
                                                         np.ones((8, 100, 100)))))
                                    
        tf_ytest = np.array_equal(ytest, np.concatenate((np.ones((8, 100, 100)),
                                                         np.zeros((8, 100, 100)))))
                                    
        self.assertTrue(tf_xtrain)
        self.assertTrue(tf_ytrain)
        self.assertTrue(tf_xtest)
        self.assertTrue(tf_ytest)
                                    
    
    def test_setup_data_tomogan(self):
        basedir = main_path + '/testing_setup/'
        start_project(basedir)
        
        clean_data = np.concatenate((np.ones((10, 400, 400)), np.zeros((10, 400, 400))))
        noisy_data = np.concatenate((np.zeros((10, 400, 400)), np.ones((10, 400, 400))))   
        xtrain, ytrain, xtest, ytest = format_data_tomogan(clean_data, noisy_data,
                                                           interval=5, dtype=np.float32)
        setup_data_tomogan(basedir, xtrain, ytrain, xtest, ytest, types='noise')
        
        location = basedir + 'tomogan/'
        ident = 'tomogan_noise_AI'
        for f in ['xtrain', 'ytrain', 'xtest', 'ytest']:
            f'{location}xtrain_{ident}.h5'
            self.assertTrue(os.path.exists(f'{location}xtrain_{ident}.h5'))
        
        
    def test_create_prj_mp4(self):
        basedir = main_path + '/testing_setup/'
        start_project(basedir)
        
        prj = np.concatenate((np.ones((10, 100, 100)), np.ones((10, 100, 100)))) * 2
        flat = np.concatenate((np.ones((3, 100, 100)), np.ones((3, 100, 100))))
        dark = prjs = np.concatenate((np.zeros((3, 100, 100)), np.zeros((3, 100, 100))))
        theta = np.linspace(0, np.pi, len(prjs))
        start_project(basedir)
        extract('', '', basedir, data=[prj, flat, dark, theta], chunking_size=1)

        output = create_prj_mp4(basedir, video_type='input', types='base', force_positive=False,
                   sparse_angle_removal=1, fps=2, torf=False, apply_exp=False)
        
        out_cmd = rife_predict(basedir, location_of_rife=rife_path, exp=2, scale=1.0,
                    gpu='0', video_input_type='input',
                    video_output_type='predicted',
                    python_location='')
        
        output2 = obtain_frames(basedir, video_type='input',
                                return_frames=False, output_folder='frames')

        self.assertTrue(os.path.exists(f'{basedir}/rife/video/input.mp4'))
        self.assertTrue(os.path.exists(f'{basedir}/rife/frames/prj_20.tif'))
        
        
    @classmethod
    def tearDownClass(cls):
        basedir = main_path + '/testing_setup/'
        shutil.rmtree(basedir)
        
if __name__ == '__main__':
    unittest.main()
