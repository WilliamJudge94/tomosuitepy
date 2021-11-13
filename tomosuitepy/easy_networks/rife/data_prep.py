from ...base.common import load_extracted_prj, load_extracted_theta
from ..deepfillv2.data_prep import obtain_prj_data_deepfillv2
from skimage.color import rgb2gray

import cv2
import time
import os
import shutil
import numpy as np
from tqdm import tqdm
import tifffile as tif
import sys

sys.path.append(os.path.dirname(__file__))
path1 = os.path.dirname(__file__)
path2 = '/'.join(path1.split('/')[:-2])
rife_path = f'{path2}/hard_networks/RIFE/arXiv2020-RIFE/'


def convert2gray(images):
    "Also found in ...easy_networks.deepfillv2.data_prep"
    greyscale = []
    for im in tqdm(images, desc='Converting to Grayscale'):
        greyscale.append(rgb2gray(im))
    return np.asarray(greyscale, dtype=np.float32)


def deal_with_sparse_angle(prj_data, theta,
                           sparse_angle_removal,
                           double_sparse=None):
    "Also found in ...base.reconstruct"

    if sparse_angle_removal != 1:
        new_prj = []
        new_theta = []

        for idx, image in enumerate(prj_data):
            if idx % sparse_angle_removal == 0:
                new_prj.append(prj_data[idx])
                new_theta.append(theta[idx])

        prj_data = np.asarray(new_prj)
        theta = np.asarray(new_theta)

        if double_sparse is not None:
            new_prj2 = []
            new_theta2 = []

            for idx, image in enumerate(prj_data):
                if idx % double_sparse == 0:
                    new_prj2.append(prj_data[idx])
                    new_theta2.append(theta[idx])

            prj_data = np.asarray(new_prj2)
            theta = np.asarray(new_theta2)

        print(
            f'Prj Data Shape {np.shape(prj_data)} --- Theta Shape {np.shape(theta)}')

    return prj_data, theta


def create_prj_mp4(basedir, video_type='input', types='base', force_positive=False,
                   sparse_angle_removal=0, fps=30, torf=False, apply_exp=False):
    """
    Prepare a mp4 video file of the projection files for RIFE.

    Parameters
    ----------
    basedir : str
        The path to the project

    video_type : str
        The name of the .mp4 file to be created and save in {basedir}rife/video/{video_type}.mp4

    types : str
        Where to retrieve the data from. types='base' pulls from the extracted files. types='noise' or types='artifact'
        uses the cleaned tomogan projection files. 

    force_positive : bool
        All projections to be positive numbers.

    sparse_angle_removal : int
        Remove every x amount of images from the projections

    fps : int
        Frame rate of the video

    torf : bool
        Sometimes the cv2.VideoWriter would like the User to use True instead of False. Not sure why.

    apply_exp : bool
        Apple np.exp() to data before saving projections to movie file.

    Returns
    -------
    None
        A mp4 file saved to the User designated loation. This is to be used by RIFE. 
    """
    prj_data, theta = obtain_prj_data_deepfillv2(basedir, types)

    print(f'The inital projection size is: {prj_data.shape}')

    prj_data, theta = deal_with_sparse_angle(prj_data, theta,
                                             sparse_angle_removal,
                                             double_sparse=None)

    print(f'The sparse angle projection size is: {prj_data.shape}')

    if apply_exp:
        prj_data = np.exp(prj_data)

    if force_positive:
        if np.nanmin(prj_data) < 0:
            prj_min = np.nanmin(prj_data)
            prj_max = np.nanmax(prj_data)
            prj_data += np.abs(prj_min)
            prj_new_min = np.nanmin(prj_data)
            prj_new_max = np.nanmax(prj_data)
            print(
                f'Forcing values to be positive. Before: Min/Max:{prj_min}/{prj_max} --- After: Min/Max:{prj_new_min}/{prj_new_max}')
    prj_data = prj_data/np.nanmax(prj_data)
    prj_data = prj_data * 255.0

    print(f"Projection Min: {prj_data.min()} --- Max: {prj_data.max()}")

    # prj_data = prj_data.astype(np.uint8)
    out_data = []

    output_file = f"{basedir}rife/video/{video_type}.mp4"

    print(f"Video saved to: {output_file}")

    # Wait 1 second before displaying the tqdm loading bar
    time.sleep(1)

    shapes = prj_data.shape
    size = shapes[1], shapes[2]
    out = cv2.VideoWriter(output_file,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (size[1], size[0]), torf)

    for im in tqdm(prj_data, desc='Video Processing'):

        # out.write(im)
        # out_data.append(im)

        out.write((im).astype(np.uint8))
        out_data.append((im).astype(np.uint8))

    out.release()

    return prj_data, np.asarray(out_data)


def rife_predict(basedir, location_of_rife=rife_path, exp=2, scale=1.0, gpu='0',
                 video_input_type='input', video_output_type='predicted'):
    """
    Use the neural network called RIFE to upscale the amount of projections.

    Parameters
    ----------
    basedir : str
        Path to the project.

    location_of_rife : str
        Path to the github repo of RIFE with / at the end.

    exp : int
        2 to the power of exp that the frames will be upscaled by

    scale : float
        If your frames are too large and using too much VRAM, you can scale the images down by a scaling factor. Fraction=smaller image.

    gpu : str
        The string index of the gpu to use.

    video_input_type : str
        The name of the mp4 file to use for the upscaling found at {basedir}rife/video/{video_input_type}.mp4

    video_output_type : str
        The name of the mp4 file to be created at {basedir}rife/video/{video_output_type}.mp4

    Returns
    -------
    command
        A command to be used in a terminal with the RIFE conda env variables installed.
    """

    pre = f'cd {location_of_rife} &&'
    first = f'python3 inference_video.py'
    second = f'--exp={exp}'
    third = f'--video={basedir}rife/video/{video_input_type}.mp4'
    fourth = f'--scale={scale}'
    fourth_inter = f'--gpu={gpu}'
    fifth = f"--output={basedir}rife/video/{video_output_type}.mp4"

    print(f"{pre} {first} {second} {third} {fourth} {fourth_inter} {fifth}")


def obtain_frames(basedir, video_type='predicted', return_frames=False, output_folder='frames'):
    """
    Based on the designated .mp4 file found in {basedir}rife/video/
    store the frames into {basedir}rife/{output_folder}

    Parameters
    ----------
    basedir : str
        The project directory

    video_type: str
        The name of the video found in {basedir}rife/video/{video_type}.mp4. It is either 'input' or 'predicted'

    return_frames : bool
        Allows the User to output the calculated frames while applying a RGB2Greyscale converter. 

    output_folder : str
        The name of the folder inside {basedir}rife/{output_folder} to store the projections to

    Returns
    -------
    None, nd.array
        Nothing unless User specifies to return the Greyscale projections with return_frames=True.    
    """

    vidcap = cv2.VideoCapture(f'{basedir}rife/video/{video_type}.mp4')
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    frame_number = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    save_dir = f"{basedir}rife/{output_folder}/"
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    print(f"Saving frames to: {save_dir}")

    projections = []

    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            #image = convert2gray(image)
            projections.append(image)
            tif.imsave(
                f"{basedir}rife/{output_folder}/prj_{str(count).zfill(len(str(int(frame_number))))}.tif", image.astype(np.float32))
            # cv2.imwrite("'/local/data/wjudge/image"+str(count)+".jpg", image)     # save frame as JPG file
        return hasFrames

    sec = 0
    frameRate = 1/fps  # //it will capture image in each 0.5 second
    count = 1
    success = getFrame(sec)
    for i in tqdm(range(int(frame_number)), desc=f'Saving Video Frames To TIF - Frame Number Is: {frame_number}'):
        count = count + 1
        sec = sec + frameRate
        #sec = round(sec, 8)
        success = getFrame(sec)

    if return_frames:
        return convert2gray(projections)


def create_prj_mp4_old(basedir, output_file, types='base', sparse_angle_removal=1, fps=30, torf=False):
    prj_data, theta = obtain_prj_data_deepfillv2(basedir, types)

    prj_data, theta = deal_with_sparse_angle(prj_data, theta,
                                             sparse_angle_removal,
                                             double_sparse=None)

    print(f"Min: {prj_data.min()} --- Max: {prj_data.max()}")
    prj_data = prj_data/np.max(prj_data)
    prj_data *= 255.0

    shapes = prj_data.shape
    size = shapes[1], shapes[2]
    out = cv2.VideoWriter(output_file,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (size[1], size[0]), False)
    for im in tqdm(prj_data, desc='Video Processing'):
        im /= np.max(im)
        im *= 255.0
        out.write((im).astype(np.uint8))

    out.release()
