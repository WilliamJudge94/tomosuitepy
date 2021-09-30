import unittest
import os, sys

cwd = os.getcwd()[:-4]
sys.path.append(f'{cwd}')

print(cwd)

from tomosuitepy.base.extract_projections import save_prj_ds_chunk, load_prj_ds_chunk, remove_saved_prj_ds_chunk


class TestEnv(unittest.TestCase):

    def test_save_prjs_ds_chunk(data, iteration):

        data = np.arange(0, 10)
        iteration = 0

        save_prj_ds_chunk(data, iteration)

        data_path = f'{os.getcwd()}/tomsuitepy_downsample_save_it_{str(iteration).zfill(4)}.npy'

        assert os.path.exists(data_path)
        assert np.array_equal(np.load(data_path), np.arange(0, 1))