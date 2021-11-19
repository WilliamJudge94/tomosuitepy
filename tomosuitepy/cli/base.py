import os
import sys
import typer
import shutil
import tomopy
import pathlib
import imageio
import dxchange
import numpy as np
import tifffile as tif
from functools import partial
from skimage import img_as_ubyte


current_file = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(current_file))

from tomosuitepy.base.start_project import start_project as start_project_import
from tomosuitepy.base.extract_projections import extract as extract_import
from tomosuitepy.base.reconstruct import reconstruct_data as reconstruct_data_import
from tomosuitepy.base.common import get_projection_shape


def tomo_recon(prj, theta, rot_center,
               ncore=1, user_extra=None):

    recon = tomopy.recon(prj, theta,
                        center=rot_center,
                        algorithm='gridrec',
                        ncore=ncore)
    return recon, user_extra

dxchange_reader = {
    'aps_32id': dxchange.read_aps_32id,
}


app = typer.Typer()


@app.command()
def start_project(basedir: str = typer.Argument(..., help="directory absolute/relative path where\
                                                            the User would like to start a project")):
    start_project_import(basedir)

    
@app.command()
def extract(file: str = typer.Argument(..., help="the file to extract data from"),
            basedir: str = typer.Argument(..., help="the project directory"),
            extraction_func: str = typer.Option('aps_32id', help="the dxchange extraction"),
            binning: int = typer.Option(1, help="2^binning to downsample"),
            chunking_size: int = typer.Option(1, help="the amount of chunks to chunk data"),
            normalize_ncore: int = typer.Option(1, help="amount of cores used for normalization"),
            minus_log: bool = typer.Option(True, help="apply a minus log to data"),
            nan_inf_selective: bool = typer.Option(True, help="remove nonfinite values with median kernel"),
            kernel_selective: int = typer.Option(5, help="kernel size of filter"), 
            remove_neg_vals: bool = typer.Option(True, help="remove negative values"),
            removal_val: float = typer.Option(0.001, help="set negative values to this value"),
            dtype: str = typer.Option('float32', help="data type to save data to"),
            muppy_amount: int = typer.Option(1000, help="resets tomopy RAM usage"),
            verbose: bool = typer.Option(True, help="print useful extraction step names"),
           ):
    
    split_file = file.split('/')
    fname = split_file[-1]
    datadir = '/'.join(split_file[:-1])

    extract_import(datadir=datadir, fname=fname, basedir=basedir,
                extraction_func=dxchange_reader[extraction_func],
                binning=binning,
                starting=0,
                chunking_size=chunking_size,
                normalize_ncore=normalize_ncore,
                minus_log=minus_log,
                nan_inf_selective=nan_inf_selective,
                kernel_selective=kernel_selective,
                remove_neg_vals=remove_neg_vals,
                removal_val=removal_val,
                dtype=dtype,
                muppy_amount=muppy_amount,
                overwrite=True,
                verbose=verbose,
                save=True,
                custom_dataprep=False,
                data=None,
                outlier_diff=None,
                outlier_size=None,
                bkg_norm=False,
                air=10,
                flat_roll=None,
                force_positive=False,
                remove_nan_vals=False,
                remove_inf_vals=False,
                correct_norma_extremes=False,
                )

    
@app.command()
def find_centers(basedir: str = typer.Argument(..., help="the project directory"),
                 centers2check: int = typer.Option(40, help="how many rotation centers to check"),
                 start_row: int = typer.Option(500, help="the projection slice to find center of"),
                 edge_transition: int = typer.Option(15, help="remove this many pixles from the sides"),
                 chunking_size: int = typer.Option(1, help="chunk data for the recon"),
                 power2pad: bool = typer.Option(True, help="nearest power of 2 padding"),
                 ext: str = typer.Option('tiff', help="save as a multiple tiff or multiple png"),
                 
           ): 


    # Obtaining the rotation center test slices to be plotted at a later time.
    slcs, user_extra = reconstruct_data_import(basedir=basedir,
                        rot_center=616, 
                        start_row=start_row, 
                        end_row=start_row+1, 
                        reconstruct_func=tomo_recon, 
                        network=None,
                        power2pad=power2pad, 
                        edge_transition=edge_transition, 
                        chunk_recon_size=chunking_size,
                        rot_center_shift_check=centers2check 
                                       )
        
    og_prj_shape = get_projection_shape(basedir)
    half_shape = int(og_prj_shape[1] / 2)
    starting_number = int(half_shape - centers2check)
    
    save_path = f"{basedir}centers/"
    if os.path.exists(save_path):
        print(f"Removing Centers Dir - To Save New Centers - {save_path}")
        _ = input(f"You will be deleting {basedir}centers/ - press y to continue.")
        if _ == 'y':
            shutil.rmtree(save_path)
            os.mkdir(save_path)
        else:
            raise ValueError('Stopped Program')
    else:
        print(f"Making Centers Dir and Saving Centers - {save_path}")
        os.mkdir(save_path)
    
    if ext == 'tiff':
        for idx, slc in enumerate(slcs):
            tif.imsave(f"{save_path}rot_center-{str(idx+starting_number).zfill(len(str(len(slcs))))}.tiff", slc)

    elif ext == 'png':
        for idx, slc in enumerate(slcs):
            imageio.imsave(f"{save_path}rot_center-{str(idx+starting_number).zfill(len(str(len(slcs))))}.png", slc)


@app.command()
def recon(basedir: str = typer.Argument(..., help="directory absolute/relative path where\
                                                            the User would like to start a project"),
          rot_center: int = typer.Argument(..., help="rotation center"),
          recon_cores: int = typer.Option(1, help="amount of cores to use for grid-rec recon"),
          power2pad: bool = typer.Option(True, help="pad data to nearest power of 2"),
          edge_transition: int = typer.Option(15, help="remove x amount of pixles from sides of sinogram"),
          chunking_size: int = typer.Option(1, help="amount of data chunks to create"),
          med_filter: bool = typer.Option(False, help="apply a (1, 3, 3) median filter to data"),
          network: str = typer.Option('None', help="Network to use. None/tomogan/rife"),
          rife_frames: str = typer.Option('frames', help="dir inside rife dir to pull frames from"),
          types: str = typer.Option('denoise', help="data to load for tomogan"),
          second_basedir: str = typer.Option("None", help="data to predict for tomogan"),
          checkpoint_num: str = typer.Option('None', help="checkpoint to use for tomogan"),
          muppy_amount: int = typer.Option(1000, help="reset tomopy RAM usage"),
          ext: str = typer.Option('tiff', help="save as a multiple tiff or multiple png"),
          output_dir: str = typer.Option('recons/', help="dir relative to basedir to save recons to"),
         ):
    
    if network == 'None':
        network = None

    if second_basedir == 'None':
        second_basedir = None
    
    if checkpoint_num == 'None':
        checkpoint_num == None
        
    tomo_rec = partial(tomo_recon, ncore=recon_cores)
    
    slcs, user_extra = reconstruct_data_import(basedir=basedir,
                     rot_center=rot_center,
                     start_row=None,
                     end_row=None,
                     med_filter=med_filter,
                     all_data_med_filter=False,
                     med_filter_kernel=(1, 3, 3),
                     reconstruct_func=tomo_rec,
                     network=network,
                     wedge_removal=0,
                     sparse_angle_removal=1,
                     types=types,
                     rife_types=[rife_frames, '.tif', False],
                     second_basedir=second_basedir,
                     checkpoint_num=checkpoint_num,
                     double_sparse=None,
                     power2pad=power2pad,
                     edge_transition=edge_transition,
                     verbose=False,
                     chunk_recon_size=chunking_size,
                     dtypes=np.float32,
                     rot_center_shift_check=None,
                     muppy_amount=muppy_amount,
                     zero_pad_amount=None,
                     view_one=False,
                     minus_val=0,
                     chunker_save=False,
                     emailer=None,
                     select_prjs2use=None)
    
    
    
    save_path = f"{basedir}{output_dir}"
    if os.path.exists(save_path):
        print(f"Removing Centers Dir - To Save New Centers - {save_path}")
        _ = input(f"You will be deleting {basedir}centers/ - press y to continue.")
        if _ == 'y':
            shutil.rmtree(save_path)
            os.mkdir(save_path)
        else:
            raise ValueError('Stopped Program')
    else:
        print(f"Making Centers Dir and Saving Centers - {save_path}")
        os.mkdir(save_path)
    
    if ext == 'tiff':
        for idx, slc in enumerate(slcs):
            tif.imsave(f"{save_path}slc-{str(idx).zfill(len(str(len(slcs))))}.tiff", slc)

    elif ext == 'png':
        for idx, slc in enumerate(slcs):
            imageio.imsave(f"{save_path}slc-{str(idx).zfill(len(str(len(slcs))))}.png", slc)
    

if __name__ == "__main__":
    app()