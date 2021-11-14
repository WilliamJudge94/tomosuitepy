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

from tomosuitepy.easy_networks.tomogan.data_prep import *
from tomosuitepy.easy_networks.rife.data_prep import *

muppy_ammount = 1000

class TestEnv(unittest.TestCase):

    def test_save_prjs_ds_chunk(self):

        pass


if __name__ == '__main__':
    unittest.main()
