import unittest
import os, sys
import numpy as np
from pympler import muppy, summary
import pathlib
import dxchange

cwd1 = pathlib.Path('.').absolute().parents[0]
sys.path.append(f'{cwd1}')

cwd2 = pathlib.Path('..').absolute().parents[0]
sys.path.append(f'{cwd2}')

# h5 testing
testh5_path = os.path.dirname(__file__)
mainh5_path = '{}/'.format(testh5_path)
testh5_file_path = mainh5_path+'test.h5'

from tomosuitepy.base.common import *
from tomosuitepy.base.extract_projections import save_prj_ds_chunk, load_prj_ds_chunk, remove_saved_prj_ds_chunk, extract
from tomosuitepy.base.reconstruct import tomo_recon, reconstruct_single_slice
from tomosuitepy.base.email4recon import send_email
from tomosuitepy.base.rotation_center import *
from tomosuitepy.base.start_project import start_project

muppy_ammount = 1000

class TestEnv(unittest.TestCase):
    
    def test_h5create_file(self):

        h5create_file(mainh5_path, 'test')
        file_check = os.path.isfile('{}/test.h5'.format(testh5_path))
        self.assertTrue(file_check)
        
    def test_h5grab_data(self):
        file = testh5_file_path
        h5create_dataset(file, 'group1/group2/test_data2', [100])
        data = h5grab_data(file, 'group1/group2/test_data2')
        self.assertEqual(data, [100])
        
    def test_h5group_list(self):
        file = testh5_file_path
        result1 = h5group_list(file)[0][0]
        result2 = h5group_list(file, 'group1')[0][0]
        self.assertEqual(result1, 'group1')
        self.assertEqual(result2, 'group2')

    def test_save_prjs_ds_chunk(self):

        data = np.arange(0, 10)
        iteration = 0
        iterations = 1

        path_chunker = pathlib.Path('.').absolute()

        save_prj_ds_chunk(data, iteration, path_chunker)

        data_path = f'{path_chunker}/tomsuitepy_downsample_save_it_{str(iteration).zfill(5)}.npy'

        self.assertTrue(os.path.exists(data_path))
        self.assertTrue(np.array_equal(np.load(data_path), np.arange(0, 10)))

        remove_saved_prj_ds_chunk(iterations, path_chunker)

    def test_remove_saved_prj_ds_chunk(self):

        data = np.arange(0, 10)
        iteration = 0
        iterations = 1

        path_chunker = pathlib.Path('.').absolute()

        save_prj_ds_chunk(data, iteration, path_chunker)

        data_path = f'{path_chunker}/tomsuitepy_downsample_save_it_{str(iteration).zfill(5)}.npy'

        remove_saved_prj_ds_chunk(iterations, path_chunker)

        self.assertFalse(os.path.exists(data_path))


    def test_muppy(self):

        try:
            all_objects = muppy.get_objects()[:muppy_ammount]
            self.assertTrue(muppy_ammount == len(all_objects))
            sum1 = summary.summarize(all_objects)
            self.assertTrue(True)

        except Exception as ex:
            self.fail(f"Muppy raised ExceptionType unexpectedly! - {ex}")


    def test_pre_process_prjs(self):

        np.random.seed(1)
        prj = np.random.randint(100, size=(50, 100, 100))  
        flat = np.zeros(shape=(3, 100, 100))  
        dark = np.zeros(shape=(3, 100, 100)) 
        theta = np.linspace(0, 180, 100)

        data1 = extract(datadir='', fname='', basedir='/tmp/',
                    extraction_func=dxchange.read_aps_32id,
                    binning=1,
                    outlier_diff=None,
                    air=10,
                    outlier_size=None,
                    starting=0,
                    bkg_norm=False,
                    chunking_size=1,
                    force_positive=True, 
                    removal_val=0.001, 
                    custom_dataprep=False,
                    dtype='float32',
                    flat_roll=None,
                    overwrite=True,
                    verbose=False,
                    save=False,
                    minus_log=True,
                    remove_neg_vals=False,
                    remove_nan_vals=False,
                    remove_inf_vals=False,
                    correct_norma_extremes=False,
                    normalize_ncore=None,
                    data=[prj, flat, dark, theta])

        data2 = extract(datadir='', fname='', basedir='/tmp/',
                    extraction_func=dxchange.read_aps_32id,
                    binning=1,
                    outlier_diff=None,
                    air=10,
                    outlier_size=None,
                    starting=0,
                    bkg_norm=False,
                    chunking_size=10,
                    force_positive=True, 
                    removal_val=0.001, 
                    custom_dataprep=False,
                    dtype='float32',
                    flat_roll=None,
                    overwrite=True,
                    verbose=False,
                    save=False,
                    minus_log=True,
                    remove_neg_vals=False,
                    remove_nan_vals=False,
                    remove_inf_vals=False,
                    correct_norma_extremes=False,
                    normalize_ncore=None,
                    data=[prj, flat, dark, theta])


    def test_recon(self):
        np.random.seed(1)
        prj = np.random.randint(100, size=(50, 1000, 1000))  
        theta = np.linspace(0, 180, 50)
        rot_center = 500
        rows = slice(500, 600)
        power2pad = True
        
        inputs = np.load(f"{mainh5_path}recon_test.npy")
        m1_old, m2_old = inputs

        data1 = reconstruct_single_slice(prj, theta,
                                        rows=rows, rot_center=rot_center,
                                        power2pad=power2pad, chunk_recon_size=10)

        data2 = reconstruct_single_slice(prj, theta,
                                        rows=rows, rot_center=rot_center,
                                        power2pad=power2pad, chunk_recon_size=1)
        
        

        m1 = data1[0]
        m2 = data2[0]
        
        
        d1 = np.round(m1[::10, ::10, ::10], 4)
        d1_old = np.round(m1_old, 4)
        d2 = np.round(m2[::10, ::10, ::10], 4)
        d2_old = np.round(m2_old, 4)
        self.assertTrue(np.array_equal(d1.min(), d1_old.min()))
        self.assertTrue(np.array_equal(d2.min(), d2_old.min()))
        self.assertTrue(np.array_equal(d1.max(), d1_old.max()))
        self.assertTrue(np.array_equal(d2.max(), d2_old.max()))
        self.assertTrue(np.array_equal(d1.mean(), d1_old.mean()))
        self.assertTrue(np.array_equal(d2.mean(), d2_old.mean()))
        
        #np.save('/local/data/cabana-hpc1/github_repos/tomosuitepy/tests/recon_test.npy', [m1, m2])
        
        subs = np.subtract(m1, m2)
        value = np.sum(subs)
        tf = np.array_equal(value, 0.004251825623214245)
        self.assertTrue(value <= 0.005)

    def test_send_email(self):
        testing = ''
        try:
            send_email(recipient=['dummy_email@gmail.com', ],
                image='',
                subject='',
                body='',
                gmail_user='',
                gmail_pass='',
                send=False)
        except:
            testing = 'fail'
        
        self.assertTrue(testing=='')

    @classmethod
    def tearDownClass(cls):
        file = testh5_file_path
        h5delete_file(file)

if __name__ == '__main__':
    unittest.main()
