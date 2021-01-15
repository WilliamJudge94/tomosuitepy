import numpy as np
from tqdm import tqdm
import tifffile as tif
from skimage.transform import resize
from ...base.common import load_extracted_prj, load_extracted_theta


def obtain_prj_data_deepfillv2(basedir, types):
    if types == 'base':
        prj_data = load_extracted_prj(basedir)
        theta = load_extracted_theta(basedir)
        
    elif types != 'base':
        prj_data = np.load(f'{basedir}tomogan/{types}_data.npy')
        theta = np.load(f'{basedir}extracted/theta/theta.npy')
        
    return prj_data, theta

def determine_max_number2zero_value(prj_data):
    shape = prj_data.shape
    sino_length = shape[0]
    sino_width = shape[2]
    number2zero_max_val = (sino_width - sino_length)/-2
    return number2zero_max_val

def determine_delta_sinogram_width(prj_data, number2zero):
    shape = prj_data.shape
    sino_length = shape[0]
    sino_width = shape[2]
    
    need_width = sino_length - 2*number2zero
    delta_width = sino_width - need_width
    
    
    return delta_width, delta_width/2
    

def zero_prj4missing_wedge(prj_data, number2zero):
    
    if number2zero > determine_max_number2zero_value(prj_data):
        delta_width_change, half_delta_width_change = determine_delta_sinogram_width(prj_data, number2zero)
        
        raise Warning(f"number2zero > max value of {determine_max_number2zero_value(prj_data)} for this dataset. "\
                      "Unable to make square aspect ratio training data. Sinogram Width needs to be "\
                      f"decreased by {half_delta_width_change} on each side "\
                      "Please decrease Sinogram Width using "\
                      "tomosuite.easy_networks.deepfillv2.data_prep.decrease_sinogram_width() or "\
                      f"set an integer value >= {half_delta_width_change} to shrink_sinogram_width "\
                      "to allow for a higher 'number2zero' value. ")
    
    shape = prj_data.shape
    zer = np.zeros((number2zero, shape[1], shape[2]))

    prj_data[0:number2zero] = zer
    prj_data[-number2zero:] = zer

    return prj_data


def decrease_sinogram_width(prj_data, decrease_number):
    return prj_data[:,:, decrease_number:-decrease_number]


def obtain_prj_sinograms(prj_data):
    rows = np.arange(0, np.shape(prj_data)[1])
    sino = prj_data[:, rows]
    
    sino = np.asarray(sino)
    
    data = []
    for index in tqdm(range(0, np.shape(sino)[1]), desc='Obtaining Sino'):
        data.append(sino[:, index, :])
        
    return np.asarray(data)

def obtain_max_intensity_values4sino(prj_sinograms):
    max_intensity_values = np.max(prj_sinograms, axis=(1, 2))
    return max_intensity_values


def normalize_sinograms(prj_sinograms):
    prj_sinograms = prj_sinograms / np.max(prj_sinograms, axis=(1,2), keepdims=True)
    prj_sinograms *= 255.0
    return prj_sinograms


def determine_how_many_sinogram_training_images(prj_sinograms, number2zero):
    shape = prj_sinograms.shape
    sino_length = shape[1]
    sino_width = shape[2]
    updated_sino_length = sino_length - number2zero
    number_of_training_images = np.ceil(updated_sino_length/sino_width)
    return int(number_of_training_images)

def obtain_base_training_images(prj_sinograms, number2zero, number_of_training_images):
    remove_blank_data_sino = prj_sinograms[:, number2zero:-number2zero, :]
    shape = remove_blank_data_sino.shape
    sino_length = shape[1]
    sino_width = shape[2]
    difference_in_shape = sino_length - sino_width
    changer = int(difference_in_shape/number_of_training_images)
    
    main_train = []
    
    for sino in tqdm(remove_blank_data_sino,  desc='Training Images'):
        for iteration in range(number_of_training_images):
            starting = iteration * changer
            ending = starting + sino_width
            main_train.append(sino[starting:ending])
            
    return np.asarray(main_train)


def obtain_base_images(prj_sinograms, number2zero):
    base_data = []
    
    for sino in tqdm(prj_sinograms, desc='Base Images'):
        base_data.append(sino[number2zero:-number2zero])   
        
    return np.asarray(base_data)

def obtain_base_testing_images(prj_sinograms, number2zero, number_of_training_images):
    shape = prj_sinograms.shape
    sino_length = shape[1]
    sino_width = shape[2]
    
    main_test = []
    
    for sino in tqdm(prj_sinograms, desc='Testing Images'):
        pre = []
        pre.append(sino[:sino_width])
        pre.append(sino[-sino_width:])
        main_test.append(pre)
        
    return np.asarray(main_test)
        

def obtain_central_testing_images(base_testing_images, shift=0):
    
    shape = base_testing_images.shape
    square_shape = shape[2]
    
    central_test = []
    
    for sinos in tqdm(base_testing_images, desc='Central Images'):
        
        first = sinos[0][0:int(square_shape/2)]
        second = sinos[1][-int(square_shape/2):]

        first = np.flipud(first)
        second = np.flipud(second)

        second_lr_flip = np.fliplr(second)
        second_lf_flip_shift = flips = np.roll(second_lr_flip, shift, axis=1)

        version1 = np.vstack((first, second_lf_flip_shift))
        
        central_test.append(version1)
          
    return np.asarray(central_test)


def invert_base_testing_images(base_images, base_testing_images, number2zero):
    
    main_output = []
    
    for base_image, base_testing_image in tqdm(zip(base_images, base_testing_images), total=len(base_images), desc='Inverting Base Testing'):

        top_wedge = base_testing_image[0][:number2zero]
        bottom_wedge = base_testing_image[1][-number2zero:]
        
        og_image = np.concatenate((top_wedge, base_image, bottom_wedge))

        main_output.append(og_image)
                
    return np.asarray(main_output)
        

def invert_central_testing_images(base_images, central_testing_images, number2zero):
    
    shape = central_testing_images.shape
    middle_index = int(shape[1]/2)
    starting = middle_index - number2zero
    ending = middle_index + number2zero
    
    main_output = []
    
    for base_image, central_testing_image in tqdm(zip(base_images, central_testing_images), total=len(base_images), desc='Inverting Central Testing'):
        first = central_testing_image[starting:middle_index]
        second = central_testing_image[middle_index:ending]
        second = np.fliplr(second)
        second = np.flipud(second)
        
        og_image = np.concatenate((first, base_image, second))
        
        main_output.append(og_image)
        
    return np.asarray(main_output)


def invert_central_prediction_images(base_images, central_testing_images, number2zero):
    
    shape = central_testing_images.shape
    middle_index = int(shape[1]/2)
    starting = middle_index - number2zero
    ending = middle_index + number2zero
    
    main_output = []
    
    for base_image, central_testing_image in tqdm(zip(base_images, central_testing_images), total=len(base_images), desc='Inverting Central Testing2'):
        first = central_testing_image[starting:middle_index]
        second = central_testing_image[middle_index:ending]
        
        first = np.flipud(first)
        second = np.flipud(second)
        second = np.fliplr(second)
        
        og_image = np.concatenate((first, base_image, second))
        
        main_output.append(og_image)
        
    return np.asarray(main_output)


def upscale_images(images, new_shape):
    upscaled_images = []
    for im in tqdm(images, desc='Upscaling Images'):
        upscaled_images.append(resize(im, new_shape))
        
    return np.asarray(upscaled_images)
    

def downscale_base_testing_images(base_testing_images, new_shape):
    shape = base_testing_images.shape
    first_shape = int(shape[0] * shape[1])
    reduce_dimensions_data = np.reshape(base_testing_images, (first_shape, shape[2], shape[3]))
    
    main_output = []
    
    for image in tqdm(reduce_dimensions_data, desc='Reducing Base Test Image Size'):
        main_output.append(resize(image, new_shape))
        
    return np.asarray(main_output)
    
    
def downscale_central_testing_images(central_testing_images, new_shape):
    main_output = []
    for image in tqdm(central_testing_images, desc='Reducing Central Test Image Size'):
        main_output.append(resize(image, new_shape))
        
    return np.asarray(main_output)


def downscale_base_training_images(base_training_images, new_shape):
    main_output = []
    for image in tqdm(base_training_images, desc='Reducing Base Training Image Size'):
        main_output.append(resize(image, new_shape))
        
    return np.asarray(main_output)

def up_flip_every_other_validation():
    pass


def save_training_images(basedir, training_images, validation_images, lr_flip=False, ud_flip=False, val_ud_flip=True):
    
    training_path = f'{basedir}deepfillv2/training_data/v1/training/'
    val_path = f'{basedir}deepfillv2/training_data/v1/validation/'
    
    training_zfill_pre1 = len(training_images)
    val_zfill = len(str(len(validation_images)))
    
    zfill_add = np.sum([lr_flip, ud_flip])
    training_zfill_pre2 = training_zfill_pre1 * (1+zfill_add)
    training_zfill = len(str(training_zfill_pre2))
    
    counter = 0
    
    
    for train_image in tqdm(training_images, desc='Saving Training Data'):
        
        training_loc1 = f'{training_path}{str(counter).zfill(training_zfill)}.tif'
        image1 = np.dstack((train_image, train_image, train_image))
        tif.imsave(training_loc1, image1)
        counter += 1
        
        if lr_flip:
            lr_image = np.fliplr(train_image)
            training_loc2 = f'{training_path}{str(counter).zfill(training_zfill)}.tif'
            image_lr = np.dstack((lr_image, lr_image, lr_image))
            tif.imsave(training_loc2, image_lr)
            counter += 1
            
        if ud_flip:
            ud_image = np.flipud(train_image)
            training_loc3 = f'{training_path}{str(counter).zfill(training_zfill)}.tif'
            image_ud = np.dstack((ud_image, ud_image, ud_image))
            tif.imsave(training_loc3, image_ud)
            counter += 1
            
    counter = 0
    for val_image in tqdm(validation_images, desc='Saving Validation Data'):
        validation_loc1 = f'{val_path}{str(counter).zfill(val_zfill)}.tif'
        if val_ud_flip:
            if counter % 2 != 0:
                val_image = np.flipud(val_image)
        image_val = np.dstack((val_image, val_image, val_image))
        tif.imsave(validation_loc1, image_val)
        counter += 1
        
def format_and_save_data4deepfillv2(basedir, number2zero, types='base', downscale_shape=(512, 512),
                                    edge_or_central='central', shrink_sinogram_width=False,
                                    lr_flip=True, ud_flip=True,  val_ud_flip=True):

    prj_data, theta = obtain_prj_data_deepfillv2(basedir, types)

    if shrink_sinogram_width is not False:
        prj_data = decrease_sinogram_width(prj_data, shrink_sinogram_width)

    prj_data_wedge = zero_prj4missing_wedge(prj_data, number2zero)
    prj_sinograms = obtain_prj_sinograms(prj_data_wedge)
    max_intensity_values = obtain_max_intensity_values4sino(prj_sinograms)
    prj_sinograms = normalize_sinograms(prj_sinograms)

    number_of_training_images = determine_how_many_sinogram_training_images(prj_sinograms, number2zero)
    base_images = obtain_base_images(prj_sinograms, number2zero)

    base_training_images = obtain_base_training_images(prj_sinograms, number2zero, number_of_training_images)
    base_testing_images = obtain_base_testing_images(prj_sinograms, number2zero, number_of_training_images)
    central_testing_images = obtain_central_testing_images(base_testing_images)

    assert base_training_images.shape[1] == base_testing_images.shape[2]
    assert base_training_images.shape[2] == base_testing_images.shape[3]
    assert base_training_images.shape[1] == central_testing_images.shape[1]
    assert base_training_images.shape[2] == central_testing_images.shape[2]

    downscaled_base_training_images = downscale_base_training_images(base_training_images, downscale_shape)
        

    if edge_or_central == 'central':
        downscaled_central_testing_images = downscale_central_testing_images(central_testing_images, downscale_shape)
        save_training_images(basedir, downscaled_base_training_images, downscaled_central_testing_images, lr_flip, ud_flip)
        
    elif edge_or_central == 'edge':
        downscaled_base_testing_images = downscale_base_testing_images(base_testing_images, downscale_shape)
        save_training_images(basedir, downscaled_base_training_images, downscaled_base_testing_images, lr_flip, ud_flip,  val_ud_flip=val_ud_flip)

    else:
        raise Warning(f"edge_or_central value must equal 'central' or 'edge'")
        
    return downscaled_base_training_images, base_training_images, number2zero