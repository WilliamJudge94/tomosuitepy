import unittest
import os, sys
import numpy as np
from pympler import muppy, summary

cwd = os.getcwd()[:-4]
sys.path.append(f'{cwd}')

from tomosuitepy.base.extract_projections import save_prj_ds_chunk, load_prj_ds_chunk, remove_saved_prj_ds_chunk


class TestEnv(unittest.TestCase):

    def test_save_prjs_ds_chunk(self):

        data = np.arange(0, 10)
        iteration = 0
        iterations = 1

        path_chunker = os.getcwd()

        save_prj_ds_chunk(data, iteration, path_chunker)

        data_path = f'{path_chunker}/tomsuitepy_downsample_save_it_{str(iteration).zfill(4)}.npy'

        self.assertTrue(os.path.exists(data_path))
        self.assertTrue(np.array_equal(np.load(data_path), np.arange(0, 10)))

        remove_saved_prj_ds_chunk(iterations, path_chunker)

    def test_remove_saved_prj_ds_chunk(self):

        data = np.arange(0, 10)
        iteration = 0
        iterations = 1

        path_chunker = os.getcwd()

        save_prj_ds_chunk(data, iteration, path_chunker)

        data_path = f'{path_chunker}/tomsuitepy_downsample_save_it_{str(iteration).zfill(4)}.npy'

        remove_saved_prj_ds_chunk(iterations, path_chunker)

        self.assertFalse(os.path.exists(data_path))


    def test_muppy(self):

        try:
            all_objects = muppy.get_objects()
            sum1 = summary.summarize(all_objects)
            self.assertTrue(True)
        except Exception as ex:
            self.fail(f"Muppy raised ExceptionType unexpectedly! - {ex}")


if __name__ == '__main__':
    unittest.main()
