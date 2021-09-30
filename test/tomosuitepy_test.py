import unittest
import os, sys
import numpy as np

cwd = os.getcwd()[:-4]
sys.path.append(f'{cwd}')

from tomosuitepy.base.extract_projections import save_prj_ds_chunk, load_prj_ds_chunk, remove_saved_prj_ds_chunk


class TestEnv(unittest.TestCase):

    def test_save_prjs_ds_chunk(self):

        data = np.arange(0, 10)
        iteration = 0

        save_prj_ds_chunk(data, iteration)

        data_path = f'{os.getcwd()}/tomsuitepy_downsample_save_it_{str(iteration).zfill(4)}.npy'

        self.assertTrue(os.path.exists(data_path))
        self.assertTrue(np.array_equal(np.load(data_path), np.arange(0, 10)))

        remove_saved_prj_ds_chunk(iteration)

    def test_remove_saved_prj_ds_chunk(self):

        data = np.arange(0, 10)
        iteration = 0

        save_prj_ds_chunk(data, iteration)

        data_path = f'{os.getcwd()}/tomsuitepy_downsample_save_it_{str(iteration).zfill(4)}.npy'

        remove_saved_prj_ds_chunk(iteration)

        self.assertFalse(os.path.exists(data_path))
        



if __name__ == '__main__':
    unittest.main()