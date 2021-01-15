from random import shuffle
import yaml
import os
import numpy as np

def shortened_training_list(array, every_n_data):

    array = np.asarray(array)
    sectioned_out_data = np.split(array, int(np.shape(array)[0]/2))
    sectioned_out_data_trim = sectioned_out_data[::every_n_data]
    shape_trim = np.shape(sectioned_out_data_trim)
    cropped_data = np.reshape(sectioned_out_data_trim, (int(shape_trim[0] * shape_trim[1], )))
    
    return list(cropped_data)

def make_file_list4deepfillv2(basedir, every_n_data_train=1, every_n_data_test=1):
    """Create the flist needed for DeepFill to work
    
    Parameters
    ----------
    basedir : str
        path to the project
        
    every_n_data : int
        take every nth pair of training data
    
    every_n_data : int
        take every nth pair of test/validation data
        
    Returns
    -------
    Nothing. Creates the flist files needed for DeepFill
    """


    folder_path = f'{basedir}deepfillv2/training_data/'
    train_filename = f'{basedir}deepfillv2/data_flist/train_shuffled.flist'
    validation_filename = f'{basedir}deepfillv2/data_flist/validation_shuffled.flist'
    is_shuffled = 1

    
    # get the list of directories
    dirs = os.listdir(folder_path)
    dirs_name_list = []

    # make 2 lists to save file paths
    training_file_names = []
    validation_file_names = []

    # print all directory names
    for dir_item in dirs:

        if '.ipynb_checkpoints' not in dir_item:
            # modify to full path -> directory
            dir_item = folder_path + dir_item

            training_folder = os.listdir(dir_item + "/training")
            for training_item in training_folder:
                if '.tif' in training_item:

                    training_item = dir_item + "/training" + "/" + training_item
                    training_file_names.append(training_item)

            training_file_names = shortened_training_list(training_file_names, every_n_data_train)
            
            validation_folder = os.listdir(dir_item + "/validation")
            for validation_item in validation_folder:
                if '.tif' in validation_item:
                    validation_item = dir_item + "/validation" + "/" + validation_item
                    validation_file_names.append(validation_item)
                    
            validation_file_names = shortened_training_list(validation_file_names, every_n_data_test)

    # shuffle file names if set
    if is_shuffled == 1:
        shuffle(training_file_names)
        shuffle(validation_file_names)

    # make output file if not existed
    if not os.path.exists(train_filename):
        os.mknod(train_filename)

    if not os.path.exists(validation_filename):
        os.mknod(validation_filename)

    # write to file
    fo = open(train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

    # print process
    print("Written file is: ", train_filename, ", is_shuffle: ", is_shuffled)
    
    
def setup_inpainting(basedir, img_shapes=[512, 512, 3], height=144, max_delta_height=0, 
                  static_view_size=5, batch_size=1, max_iters=120000, model_restore='',
                  num_gpus_per_job=1, num_cpus_per_job=4, num_hosts_per_job=1, memory_per_job=11,
                  mask_type='central', udr='r', 
                  
                  gpu_type="nvidia-tesla-p100", name="places2_gated_conv_v100", 
                  
                  random_crop=False, val=True, random_seed=False,
                  
                  gan="sngan", gan_loss_alpha=1, gan_with_mask=True, discounted_mask=True, padding="SAME",
                  
                  train_spe=4000, viz_max_out=10, val_psteps=2000,
                  
                  width='auto', max_delta_width=0, vertical_margin=0, horizontal_margin=0, 
                  
                  ae_loss=True, l1_loss=True, l1_loss_alpha=1, guided=False, edge_threshold=0.6, dataset='auto', data_flist='auto' ):
    
    
    """Setup all nesessary parameters for DeepFill to train to inpaint sinograms
    
    Parameters
    ----------
    basedir : str
        path to the project
        
    img_shapes : array
        the shape of the rgb images. Must be in terms of [X, X, 3]
        
    height : int
        the height of the missing angles in pixles on the sinogram based on the updated shape of the sinogram
        
    max_delta_height : int
        allows the User to train on varrying missing angles
        
    static_view_size : int
        the amount of testing images displayed in Tensorboard
    
    batch_size : int
        the batch size for training
        
    max_iters : int
        the number of Epochs
        
    model_restore : str
        full path to the previous log the User wants to resotre from
        
    num_gpus_per_job : int
        how many GPU's to distribute to
        
    num_cpus_per_job : int
        how many CPU's to distribute to
        
    num_hosts_per_job : int
        how many hosts to distribute to
        
    memory_per_job : int
        how much Memory per job
        
    mask_type : str
        the type of mask the User is applying. either 'edge' or 'central'
        
    udr : str
        if mask_type='edge' this will specify where to put the mask. 'u'=Upper, 'd'=Down, 'r'=Random 
        
        
    KEEP REST OF PARAMETERS CONSTANT
    
    Returns
    -------
    Nothing. Creates inpaint.yml file needed for DeepFillV2
    """
    yam_path = f'{basedir}deepfillv2/inpaint.yml'
    
    with open(yam_path, 'w+') as f:
        pass
    
    edit_yml_file(basedir=basedir, 
                  img_shapes=img_shapes, 
                  height=height,
                  mask_type=mask_type,
                  udr=udr,
                  max_delta_height=max_delta_height, 
                  static_view_size=static_view_size, 
                  batch_size=batch_size, 
                  max_iters=max_iters, 
                  model_restore=model_restore,
                  num_gpus_per_job=num_gpus_per_job, 
                  num_cpus_per_job=num_cpus_per_job, 
                  num_hosts_per_job=num_hosts_per_job, 
                  memory_per_job=memory_per_job, 
                  gpu_type=gpu_type, 
                  name=name, 
                  random_crop=random_crop, 
                  val=val, 
                  random_seed=random_seed, 
                  gan=gan, 
                  gan_loss_alpha=gan_loss_alpha, 
                  gan_with_mask=gan_with_mask, 
                  discounted_mask=discounted_mask, 
                  padding=padding,
                  train_spe=train_spe,
                  viz_max_out=viz_max_out,
                  val_psteps=val_psteps,
                  width=width,
                  max_delta_width=max_delta_width,
                  vertical_margin=vertical_margin,
                  horizontal_margin=horizontal_margin, 
                  ae_loss=ae_loss, 
                  l1_loss=l1_loss,
                  l1_loss_alpha=l1_loss_alpha,
                  guided=guided,
                  edge_threshold=edge_threshold,
                  dataset=dataset,
                  data_flist=data_flist)


def edit_yml_file(basedir, img_shapes=[512, 512, 3], height=144, max_delta_height=0, 
                  static_view_size=5, batch_size=1, max_iters=120000, model_restore='',
                  num_gpus_per_job=1, num_cpus_per_job=1, num_hosts_per_job=1, memory_per_job=11, 
                  mask_type='central', udr='r',
                  
                  gpu_type="nvidia-tesla-p100", name="places2_gated_conv_v100", 
                  
                  random_crop=False, val=True, random_seed=False,
                  
                  gan="sngan", gan_loss_alpha=1, gan_with_mask=True, discounted_mask=True, padding="SAME",
                  
                  train_spe=4000, viz_max_out=10, val_psteps=2000,
                  
                  width='auto', max_delta_width=0, vertical_margin=0, horizontal_margin=0, 
                  
                  ae_loss=True, l1_loss=True, l1_loss_alpha=1, guided=False, edge_threshold=0.6, dataset='auto', data_flist='auto'):
    
    
    width = img_shapes[0] - 1
    
    log_dir=f"{basedir}deepfillv2/logs"
    dataset = 'inpaint_sinogram'
    
    data_flist = {}
    data_flist[dataset] = [f'{basedir}deepfillv2/data_flist/train_shuffled.flist', f'{basedir}deepfillv2/data_flist/validation_shuffled.flist']
    
    arguments = locals()
    
    yml_path = f'{basedir}deepfillv2/inpaint.yml'
    
    save_dict = {}
    
    for arg in arguments:
        save_dict[arg] = arguments[arg]

    with open(yml_path) as f:
        list_doc = yaml.load(f, Loader=yaml.FullLoader)

    try:
        for arg in arguments:
            list_doc[arg] = arguments[arg] 

        with open(yml_path, "w") as f:
            yaml.dump(list_doc, f, default_flow_style=False, sort_keys=False)
            
    except:
        with open(yml_path, "w") as f:
            yaml.dump(save_dict, f, default_flow_style=False, sort_keys=False)
            
            
def determine_deepfillv2_mask_height(downscaled_base_training_images, base_training_images, number2zero):
    downscaled_base_shape = downscaled_base_training_images.shape[1]
    base_shape = base_training_images.shape[1]
    fraction = (2*number2zero)/base_shape
    return int(np.ceil(downscaled_base_shape * fraction))