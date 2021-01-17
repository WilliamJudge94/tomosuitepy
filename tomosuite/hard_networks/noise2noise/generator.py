from pathlib import Path
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import warnings  

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    from keras.utils import Sequence


class NoisyImageGenerator(Sequence):
    def __init__(self, image_dir, source_noise_model, target_noise_model, basedir=None, batch_size=32, image_size=64, 
                 main_train_dir=[], corresponding_train_dir=[], concat_train=False, num_of_slcs=1024, crop_im_val=None, im_type='tif',
                 single_image_train=None):
        
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp", ".tif")

        self.image_paths = []
        for p in Path(image_dir).glob("**/*"):
            if p.suffix.lower() in image_suffixes:
                if '.ipynb_checkpoints' not in os.fspath(p):
                    self.image_paths.append(p)
        
        #self.image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes] 
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.image_num = len(self.image_paths)
        self.batch_size = batch_size
        self.image_size = image_size
        self.basedir = basedir
        
        self.num_of_slcs_og = num_of_slcs[1]
        self.num_of_slcs_real = num_of_slcs[0]
        
        self.main_train_dir = main_train_dir
        self.corresponding_train_dir = corresponding_train_dir
        self.concat_train = concat_train
        self.crop_im_val = crop_im_val
        self.single_image_train = single_image_train
        self.im_type = im_type
        
        if self.im_type == 'png':
            self.im_dtype = np.uint8
        elif self.im_type == 'tif':
            self.im_dtype = np.float16

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=self.im_dtype)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=self.im_dtype)
        sample_id = 0

        while True:
            
            
            
            if self.single_image_train is not None:
                random_image_value_pre = self.single_image_train
                #random_image_value_post = str(self.single_image_train).zfill(zfills)      
            else:
                random_image_value_pre = random.randint(0, self.num_of_slcs_real-1)
                
            zfills = len(str(self.num_of_slcs_og))
            random_image_value_post = str(random_image_value_pre).zfill(zfills)
            
            if self.concat_train:
                train_files = self.main_train_dir + self.corresponding_train_dir
                selected_files = random.sample(train_files, 2)
              
                image_path1 = f'{self.basedir}noise2noise/{selected_files[0]}_recon/{random_image_value_post}.{self.im_type}'
                image_path2 = f'{self.basedir}noise2noise/{selected_files[1]}_recon/{random_image_value_post}.{self.im_type}'

            else:
                selected_files = random.sample(self.corresponding_train_dir, 1)
                    
                image_path1 = f'{self.basedir}noise2noise/{self.main_train_dir[0]}_recon/{random_image_value_post}.{self.im_type}'
                image_path2 = f'{self.basedir}noise2noise/{selected_files[0]}_recon/{random_image_value_post}.{self.im_type}'

            
            if self.im_type == 'png':
                image1 = cv2.imread(str(image_path1))
                image2 = cv2.imread(str(image_path2))
            elif self.im_type == "tif":
                image1 = cv2.imread(str(image_path1), -1)
                image2 = cv2.imread(str(image_path2), -1)      
            
            if self.crop_im_val is not None:
                num = self.crop_im_val
                image1 = image1[num:-num, num:-num, :]
                image2 = image2[num:-num, num:-num, :]

            image = image1.copy()           
            
            h, w, _ = image.shape

            if h >= image_size and w >= image_size:
                h, w, _ = image.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                
                clean_patch1 = image1[i:i + image_size, j:j + image_size]
                clean_patch2 = image2[i:i + image_size, j:j + image_size]
                
                x[sample_id] = clean_patch1
                y[sample_id] = clean_patch2

                sample_id += 1

                if sample_id == batch_size:
                    return x, y


class ValGenerator(Sequence):
    def __init__(self, image_dir, val_noise_model, single_image_train, im_type, crop_im_val):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp", ".tif")

        image_paths = []
        for p in Path(image_dir).glob("**/*"):
            if p.suffix.lower() in image_suffixes:
                if '.ipynb_checkpoints' not in os.fspath(p):
                    image_paths.append(p)
        
        #image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.single_image_train = single_image_train
        self.im_type = im_type
        self.crop_im_val = crop_im_val
        self.image_num = len(image_paths)
        zfills = len(str(self.image_num))
        self.data = []

        if single_image_train is not None:
            checker = str(self.single_image_train).zfill(zfills)
        else:
            checker = '.'
        
        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

        for image_path in image_paths:
            if checker in str(image_path):
                
                if self.im_type == 'png':
                    y = cv2.imread(str(image_path))

                elif self.im_type == "tif":
                    y = cv2.imread(str(image_path), -1)

                if self.crop_im_val is not None:
                    num = self.crop_im_val
                    y = y[num:-num, num:-num, :]
                    
                h, w, _ = y.shape
                y = y[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
                x = val_noise_model(y)
                self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])
                
                
        if single_image_train is not None:
            if self.data == []:
                raise Warning(f'No images found in Validation set with number {single_image_train}. Please set single_image_val=None or save image {single_image_train} to the validation folder.')
                

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]
