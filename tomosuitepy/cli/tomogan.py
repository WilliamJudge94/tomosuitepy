import os
import sys
import typer
import shutil
import pathlib
import imageio
import numpy as np
import tifffile as tif
from time import sleep
from functools import partial

current_file = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(current_file))

from tomosuitepy.methods.denoise_type1 import denoise_t1_dataprep
from tomosuitepy.easy_networks.tomogan.train import train_tomogan, tensorboard_command_tomogan
from tomosuitepy.easy_networks.tomogan.predict import predict_tomogan, save_predict_tomogan
from tomosuitepy.base.common import load_extracted_prj

app = typer.Typer()


@app.command()
def test_noise(basedir: str = typer.Argument(..., help="path to the project"),
                    noise: float = typer.Option(125, help="amount of poisson noise to add"),
                    image_step: int = typer.Option(20, help="amount of images to skip"),
                    idx: int = typer.Option(0, help="index of noisy image to return"),
                    clim: str = typer.Option('None', help="clim of matplotlib plot"),
                 
                 ):
    
    if clim == 'None':
        clim = None
    else:
        clim = list(clim)
        
    org, noisy = denoise_t1_dataprep.fake_noise_test(basedir=basedir,
                                  noise=noise,
                                  image_step=image_step,
                                  plot=False,
                                  idx=idx,
                                  figsize=(10, 10),
                                  clim=clim)
    sleep(2)
    print("\n")
    difference = np.subtract(org, noisy)
    print(f"Min difference is: {difference.min()}, Max difference is: {difference.max()}\n")


@app.command()
def setup_noise(basedir: str = typer.Argument(..., help="path to the project"),
                    noise: float = typer.Option(125, help="amount of poisson noise to add"),
                    intervel: int = typer.Option(5, help="'interval' number of images is put in training set"),
               ):
    
    denoise_t1_dataprep.setup_fake_noise_train(basedir,
                                        noise=noise,
                                        interval=interval,
                                        dtype=np.float32)
    
    
@app.command()
def train(basedir: str = typer.Argument(..., help="path to the project"),
                    epochs: int = typer.Option(120001, help="amount of epochs"),
                    gpu: str = typer.Option('0', help="gpu to use"),
                    batch_size: int = typer.Option(2, help="the batch size"),
                    patch_size: int = typer.Option(512, help="the image patch size"),
                    types: str = typer.Option('noise', help="either 'noise' or 'artifact'"),
               ):


    print('\n')
    # Prints out a command line script which will initiate a tensorboard instance to view TomoGAN training
    tensorboard_command_tomogan(basedir=basedir)
    print('\n')

    train_tomogan(basedir=basedir,
                    epochs=epochs,
                    gpus=gpu,
                    lmse=0.5,
                    lperc=2.0,
                    ladv=20,
                    lunet=3,
                    depth=1,
                    itg=1,
                    itd=2,
                    mb_size=batch_size,
                    img_size=patch_size,
                    types=types)
    
    
@app.command()
def predict(basedir: str = typer.Argument(..., help="path to the project"),
            load_epoch: str = typer.Option('01000', help="full string number of epoch to load"),
            chunk_size: int = typer.Option(5, help="chunk data into GPU predict"),
            gpu: str = typer.Option('0', help="gpu to use"),
            types: str = typer.Option('noise', help="either 'noise' or 'artifact'"),

               ):
    
    data = load_extracted_prj(basedir=basedir)

    
    clean_data, dirty_data = predict_tomogan(basedir=basedir,
                                    data=data,
                                    weights_iter=load_epoch,
                                    chunk_size=chunk_size,
                                    gpu=gpu,
                                    lunet=3,
                                    in_depth=1,
                                    data_type=np.float32,
                                    verbose=False,
                                    types=types)


    save_predict_tomogan(basedir=basedir,
                            good_data=clean_data,
                            bad_data=dirty_data,
                            second_basedir=None,
                            types=types)

        
if __name__ == "__main__":
    app()