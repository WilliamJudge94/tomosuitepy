import os
import sys
import typer
import shutil
import pathlib
import imageio
import numpy as np
import tifffile as tif
from functools import partial

current_file = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(current_file))
rife_path = str(current_file) + "/hard_networks/RIFE/arXiv2020-RIFE/"

from tomosuitepy.easy_networks.rife.data_prep import create_prj_mp4, rife_predict, obtain_frames
from tomosuitepy.base.common import get_projection_shape

app = typer.Typer()


@app.command()
def create_mp4(basedir: str = typer.Argument(..., help="path to the project"),
                   fps: int = typer.Option(10, help="frames per second of video"),
                   apply_exp: bool = typer.Option(False, help="apply exponential to the data"),
                   sparse_angle_removal: int = typer.Option(1, help="take every X projections and save them"),
                   save_name: str = typer.Option('input', help="name of .mp4 file to save"),
                  ):
    
    
    output = create_prj_mp4(basedir,
                            sparse_angle_removal=sparse_angle_removal,
                            fps=fps,
                            apply_exp=apply_exp,
                            video_type=save_name
                            )

    
@app.command()
def rife_command(basedir: str = typer.Argument(..., help="path to the project"),
                 exp: int = typer.Option(2, help="2^exp improvment in prj density"),
                 gpu: str = typer.Option('0', help="index of gpu to use"),
                 input_dir: str = typer.Option('input', help="input .mp4 file to load"),
                 output_dir: str = typer.Option('predicted', help="name of saved .mp4 file"),
                 rife_loc: str = typer.Option(rife_path, help="path to RIFE directory"),
                 python_loc: str = typer.Option('', help="path to python to use"),
                 scale: float = typer.Option(1.0, help="scale of all the images"),
                 ):

    cmds = rife_predict(basedir=basedir,
                        location_of_rife=rife_loc,
                        exp=exp,
                        scale=scale,
                        gpu=gpu,
                        video_input_type=input_dir,
                        video_output_type=output_dir,
                        python_location=python_loc)
    
    print('\n copy and paste command into terminal with conda env activated for RIFE.\n')
    print(cmds)
    print('\n')
    
    
@app.command()
def extract_mp4(basedir: str = typer.Argument(..., help="path to the project"), 
                   input_dir: str = typer.Option('predicted', help="input .mp4 file to load"),
                   output_dir: str = typer.Option('frames', help="relative output dir to save frames to"),
                 
                 ):
                     

    obtain_frames(basedir=basedir,
                  video_type=input_dir,
                  return_frames=False,
                  output_folder=output_dir)
    
if __name__ == "__main__":
    app()