import os
import pathlib

def start_project(basedir):
    """Create directories necessary for a standard analysis package.
    
    Parameters
    ----------
    basedir : str
        the path used to create your experimental tomosuite directory. Example: /home/user/Destkop/experimental/
        
    Returns
    -------
    Nothing. Creates project directories.
    """
    create_folders = ['extracted/projections',
    
                    'extracted/theta', 
    
                    'tomogan',
                    'tomogan/logs/',

                    'deepfillv2', 
                    'deepfillv2/training_data/v1/training',
                    'deepfillv2/training_data/v1/validation',
                    'deepfillv2/logs/',
                    'deepfillv2/data_flist',
                    'deepfillv2/predictions',

                    'noise2noise',
                    'noise2noise/logs',
                    'noise2noise/output_model',
                     
                     'dain']
    
    create_folders = [f'{basedir}{fx}' for fx in create_folders]
    
    for folder in create_folders:
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 
        