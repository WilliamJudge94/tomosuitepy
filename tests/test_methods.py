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

from tomosuitepy.methods.denoise_type1.denoise_t1_dataprep import *

from tomosuitepy.methods.denoise_type2.denoise_t2_dataprep import *

muppy_ammount = 1001

class TestEnv(unittest.TestCase):

    def test_save_prjs_ds_chunk(self):

        pass

if __name__ == '__main__':
    unittest.main()
