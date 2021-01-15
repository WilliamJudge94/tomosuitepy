from ...base.common import load_extracted_prj, load_extracted_theta
from ...base.reconstruct import deal_with_sparse_angle
from ..deepfillv2.data_prep import obtain_prj_data_deepfillv2

import cv2
import numpy as np
from tqdm import tqdm

def create_prj_mp4(basedir, output_file, types='base', sparse_angle_removal=0, fps=30, torf=False):
    prj_data, theta = obtain_prj_data_deepfillv2(basedir, types)
    
    prj_data, theta = deal_with_sparse_angle(prj_data, theta,
                                             sparse_angle_removal,
                                             double_sparse=None)
    
    prj_data = prj_data/np.max(prj_data)
    prj_data *= 255.0
    
    shapes = prj_data.shape
    size = shapes[1], shapes[2]
    out = cv2.VideoWriter(output_file,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (size[1], size[0]), False)
    for im in tqdm(prj_data, desc='Video Processing'):
        out.write((im).astype(np.uint8))
        
    out.release()