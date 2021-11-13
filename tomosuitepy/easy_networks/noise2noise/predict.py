import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import tifffile as tif
from pathlib import Path
from model import get_model
from noise_model import get_noise_model


def get_image(image, im_type):
    if im_type == 'png':
        image = np.clip(image, 0, 255)
        return image.astype(dtype=np.uint8)

    elif im_type == 'tif':
        return image.astype(dtype=np.float32)


def predict_noise2noise(image_dir,
                        weight_file,
                        model="srresnet",
                        amount2skip=1,
                        im_type='tif',
                        crop_im_val=None,
                        gpu='0'):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    test_noise_model = "clean"
    val_noise_model = get_noise_model(test_noise_model)
    model = get_model(model)
    model.load_weights(weight_file)

    image_paths = []
    image_suffixes = (".jpeg", ".jpg", ".png", ".bmp", ".tif")
    for p in Path(image_dir).glob("**/*"):
        if p.suffix.lower() in image_suffixes:
            if '.ipynb_checkpoints' not in os.fspath(p):
                image_paths.append(p)
    image_paths = image_paths[::amount2skip]

    out_images = []
    denoised_images = []

    for image_path in tqdm(image_paths):

        if im_type == 'png':
            im_dtype = np.uint8
            image = cv2.imread(str(image_path))

        elif im_type == 'tif':
            im_dtype = np.float32
            image = cv2.imread(str(image_path), -1)

        if crop_im_val is not None:
            num = crop_im_val
            image = image[num:-num, num:-num, :]

        h, w, _ = image.shape
        # for stride (maximum 16)
        image = image[:(h // 16) * 16, :(w // 16) * 16]
        h, w, _ = image.shape

        out_image = np.zeros((h, w * 3, 3), dtype=im_dtype)
        noise_image = image  # val_noise_model(image)

        pred = model.predict(np.expand_dims(noise_image, 0))

        denoised_image = get_image(pred[0], im_type)
        out_image[:, :w] = image
        out_image[:, w:w * 2] = noise_image
        out_image[:, w * 2:] = denoised_image

        out_images.append(out_image)
        denoised_images.append(denoised_image)

    return np.asarray(denoised_images), np.asarray(image_paths), np.asarray(out_images)


def save_predict_noise2noise(basedir, denoised_images, image_paths, output_dir=None):

    if output_dir == None:
        output_dir = f'{basedir}noise2noise/output_validation'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for data, image_path in zip(denoised_images, image_paths):
        tif.imwrite(str(output_dir.joinpath(image_path.name))
                    [:-4] + ".tif", data)

    print(f'Saved data to {output_dir}')
