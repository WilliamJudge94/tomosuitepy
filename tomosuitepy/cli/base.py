import os
import sys
import typer
import shutil
import tomopy
import pathlib
import imageio
import dxchange
import tifffile as tif
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
            

if __name__ == "__main__":
    app()