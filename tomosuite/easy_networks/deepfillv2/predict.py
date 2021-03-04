import os
import cv2
import time
import argparse
import numpy as np
from tqdm import tqdm
import warnings
from skimage.color import rgb2gray

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    
    import tensorflow as tf
    import neuralgym as ng
    
from inpaint_model import InpaintCAModel

def predict_deepfillv2(basedir, checkpoint_num,  image_height, image_width, gpu='0', save=False):
    """Allow the User to predict the inpainting results on unseen data.
    Parameters
    ----------
    basedir : str
        the path to the project
    checkpoint_num : str
        the epoch in which the User wants to reload the model weights from
    image_height : int
        the image height
    image_widght : int
        the image width
    gpu : str
        define which computer gpu the User would like to use
    save : bool
        if True this will save the predictions
    Returns
    -------
    The predicted results
    """
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    validation_images = os.listdir(f'{basedir}deepfillv2/training_data/v1/validation/')
    validation_images = [f'{basedir}deepfillv2/training_data/v1/validation/{f}' for f in validation_images if '.tif' in f]
    
    zfi = len(validation_images)
    zfil = str(zfi)
    zfills = len(zfil)
    
    checkpoint_dir = f'{basedir}deepfillv2/logs/snap-{checkpoint_num}'
    
    FLAGS = ng.Config(f'{basedir}deepfillv2/inpaint.yml', show_config=False)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, image_height, image_width*2, 3))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    
    grid = 4
    return_array = []
    
        
    for image_loc in tqdm(validation_images, desc='Predictions', position=0, leave=True):
        
        
        mask = create_mask(FLAGS.img_shapes, FLAGS.height)
        og_mask_shape = mask.shape
        mask = mask[:image_height//grid*grid, :image_width//grid*grid, :]
        mask = np.expand_dims(mask, 0)
        
        
        image = cv2.imread(image_loc, -1)

        assert image.shape == og_mask_shape

        h, w, _ = image.shape
        image = image[:h//grid*grid, :w//grid*grid, :]

        image = np.expand_dims(image, 0)
        input_image = np.concatenate([image, mask], axis=2)

        # load pretrained model
        result = sess.run(output, feed_dict={input_image_ph: input_image})
        return_array.append(result[0][:, :, ::-1])

    #return_array = convert2gray(return_array)
    
    if save:
        
        output_path = f'{basedir}deepfillv2/predictions/{checkpoint_num}/'
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(f'{output_path}n2n_data/')
            
        np.save(f'{output_path}prj_data.npy', return_array)
                
        #for im_counter, im in tqdm(enumerate(return_array), total=len(return_array), desc='Saving Images'):
        #    cv2.imwrite(f'{output_path}{str(im_counter).zfill(zfills)}.png', im)
        
    return return_array




def create_mask(img_shapes, height):
    
    
    master_shape = img_shapes[0]
    half_master_shape = int(master_shape/2)
    master_height = height
    half_master_height = int(master_height/2)
    start_position = half_master_shape - half_master_height
    
    
    image = np.zeros((master_shape, master_shape))
    ones = np.ones((1, master_shape))
    ones *= 255
    
    for idx in range(start_position-1, start_position + height):
        image[idx] = ones
        
    new_out = image.astype(np.float32)

    new_out2 = np.dstack((new_out, new_out, new_out))
    
    return new_out2
