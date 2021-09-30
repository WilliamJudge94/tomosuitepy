import unittest
import os, sys
import numpy as np
from pympler import muppy, summary
import pathlib

cwd = pathlib.Path('.').absolute().parents[0]
sys.path.append(f'{cwd}')

from tomosuitepy.base.extract_projections import save_prj_ds_chunk, load_prj_ds_chunk, remove_saved_prj_ds_chunk, pre_process_prj
from tomosuitepy.base.reconstruct import tomo_recon, reconstruct_single_slice


class TestEnv(unittest.TestCase):

    def test_save_prjs_ds_chunk(self):

        data = np.arange(0, 10)
        iteration = 0
        iterations = 1

        path_chunker = pathlib.Path('.').absolute()

        save_prj_ds_chunk(data, iteration, path_chunker)

        data_path = f'{path_chunker}/tomsuitepy_downsample_save_it_{str(iteration).zfill(4)}.npy'

        self.assertTrue(os.path.exists(data_path))
        self.assertTrue(np.array_equal(np.load(data_path), np.arange(0, 10)))

        remove_saved_prj_ds_chunk(iterations, path_chunker)

    def test_remove_saved_prj_ds_chunk(self):

        data = np.arange(0, 10)
        iteration = 0
        iterations = 1

        path_chunker = pathlib.Path('.').absolute()

        save_prj_ds_chunk(data, iteration, path_chunker)

        data_path = f'{path_chunker}/tomsuitepy_downsample_save_it_{str(iteration).zfill(4)}.npy'

        remove_saved_prj_ds_chunk(iterations, path_chunker)

        self.assertFalse(os.path.exists(data_path))


    def test_muppy(self):

        try:
            all_objects = muppy.get_objects()[:100000]
            sum1 = summary.summarize(all_objects)
            self.assertTrue(True)

        except Exception as ex:
            self.fail(f"Muppy raised ExceptionType unexpectedly! - {ex}")


    def test_pre_process_prjs(self):

        np.random.seed(1)
        prj = np.random.randint(100, size=(50, 100, 100))  
        flat = np.zeros(shape=(3, 100, 100))  
        dark = np.zeros(shape=(3, 100, 100)) 

        data1 = pre_process_prj(prj, flat, dark,
                        flat_roll=None,
                        outlier_diff=None,
                        outlier_size=None,
                        air=10,
                        custom_dataprep=False,
                        binning=1,
                        bkg_norm=False,
                        chunk_size4bkg=10,
                        verbose=False,
                        force_positive=True,
                        removal_val=0.001,
                        minus_log=True,
                        remove_neg_vals=False,
                        remove_nan_vals=False,
                        remove_inf_vals=False,
                        correct_norma_extremes=True,
                        chunk_size4downsample=1)

        data2 = pre_process_prj(prj, flat, dark,
                        flat_roll=None,
                        outlier_diff=None,
                        outlier_size=None,
                        air=10,
                        custom_dataprep=False,
                        binning=1,
                        bkg_norm=False,
                        chunk_size4bkg=10,
                        verbose=False,
                        force_positive=True,
                        removal_val=0.001,
                        minus_log=True,
                        remove_neg_vals=False,
                        remove_nan_vals=False,
                        remove_inf_vals=False,
                        correct_norma_extremes=True,
                        chunk_size4downsample=10)

        self.assertTrue(np.array_equal(data1, data2))


    def test_recon(self):
        np.random.seed(1)
        prj = np.random.randint(100, size=(50, 1000, 1000))  
        theta = np.linspace(0, 180, 50)
        rot_center = 500
        rows = slice(500, 600)
        power2pad = True

        data1 = reconstruct_single_slice(prj, theta,
                                        rows=rows, rot_center=rot_center,
                                        power2pad=power2pad, chunk_recon_size=10)

        data2 = reconstruct_single_slice(prj, theta,
                                        rows=rows, rot_center=rot_center,
                                        power2pad=power2pad, chunk_recon_size=1)

        m1 = data1[0]
        m2 = data2[0]
        subs = np.subtract(m1, m2)
        value = np.sum(subs)
        tf = np.array_equal(value, 0.004251825623214245)
        self.assertTrue(tf)




if __name__ == '__main__':
    unittest.main()
