import sys
import warnings
import argparse
import os
import shutil
import numpy as np
from pathlib import Path

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback
    from keras.optimizers import Adam
    from model import get_model, PSNR, L0Loss, UpdateAnnealingParameter
    from generator import NoisyImageGenerator, ValGenerator
    from noise_model import get_noise_model
    from tensorflow.python.eager.context import context, EAGER_MODE, GRAPH_MODE


def switch_to(mode):
    ctx = context()._eager_context
    ctx.mode = mode
    ctx.is_eager = mode == EAGER_MODE


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125


class N2NCallback(Callback):

    def __init__(self, val_generator, batch_size, summary_writer):
        super().__init__()
        self.val_generator = val_generator
        self.batch_size = batch_size
        self.summary_writer = summary_writer

    def on_epoch_end(self, epoch, logs=None):

        data = []
        for im in self.val_generator:
            data.append(im[0][0])
        data = np.asarray(data)

        image_og = data

        prediction = []
        for im_pred in image_og:
            im_pred = np.expand_dims(im_pred, axis=0)
            prediction.append(self.model.predict(im_pred)[0])

        #image_og = np.asarray(self.val_generator)[0,0,0,:,:,:]
        #image = np.expand_dims(image_og, axis=0)
        #prediction = self.model.predict(image)
        #dual_image = np.vstack((image, prediction))
        switch_to(EAGER_MODE)

        with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.image('input', tf.constant(
                image_og), step=epoch, max_images=15)
            tf.contrib.summary.image('prediction', tf.constant(
                prediction), step=epoch, max_images=15)
            #tf.contrib.summary.image('pred_v_input', tf.constant(prediction), step=(epoch*2), max_images=15)
            #tf.contrib.summary.image('pred_v_input', tf.constant(image_og), step=(epoch*2)+1, max_images=15)

        self.summary_writer.flush()
        sys.stdout.flush()

        switch_to(GRAPH_MODE)


def train_noise2noise(basedir,
                      main_train_dir,
                      corresponding_train_dir,
                      concat_train=False,
                      crop_im_val=None,
                      single_image_train=None,
                      single_image_val=None,
                      im_type='tif',
                      image_size=64,
                      batch_size=16,
                      nb_epochs=60,
                      lr=0.01,
                      steps=1000,
                      loss_type="mae",
                      weight=None,
                      model="srresnet",
                      save_best_only=True,
                      gpu='0'):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    image_dir = f'{basedir}noise2noise/{main_train_dir[0]}_recon/'
    test_dir = f'{basedir}noise2noise/{main_train_dir[0]}_val_recon/'
    output_path = f'{basedir}noise2noise/output_model/'
    source_noise_model = "clean"
    target_noise_model = "clean"
    val_noise_model = "clean"

    # Sometimes this can be wrong if there are .ipynb checkpoint files.
    # generator.py addresses this issue
    if single_image_train == None:
        num_of_slcs = len(os.listdir(image_dir)), len(os.listdir(image_dir))
    else:
        num_of_slcs = 1, len(os.listdir(image_dir))

    if len(os.listdir(image_dir)) < batch_size:
        raise Warning(
            "Batch Size is Greater Than Total Number of Directory Images")

    output_path = Path(__file__).resolve().parent.joinpath(output_path)
    model = get_model(model)
    if weight is not None:
        model.load_weights(weight)

    opt = Adam(lr=lr)
    callbacks = []

    if loss_type == "l0":
        l0 = L0Loss()
        callbacks.append(UpdateAnnealingParameter(
            l0.gamma, nb_epochs, verbose=1))
        loss_type = l0()

    model.compile(optimizer=opt, loss=loss_type, metrics=[PSNR])
    source_noise_model = get_noise_model(source_noise_model)
    target_noise_model = get_noise_model(target_noise_model)
    val_noise_model = get_noise_model(val_noise_model)

    generator = NoisyImageGenerator(image_dir, source_noise_model, target_noise_model, batch_size=batch_size,
                                    image_size=image_size, basedir=basedir, num_of_slcs=num_of_slcs,
                                    main_train_dir=main_train_dir,
                                    corresponding_train_dir=corresponding_train_dir,
                                    concat_train=concat_train, crop_im_val=crop_im_val,
                                    single_image_train=single_image_train, im_type=im_type)

    val_generator = ValGenerator(test_dir, val_noise_model,
                                 single_image_train=single_image_val, im_type=im_type, crop_im_val=crop_im_val)

    switch_to(EAGER_MODE)
    logdir = f'{basedir}noise2noise/logs/'

    if os.path.isdir(logdir):
        shutil.rmtree(logdir)
    os.mkdir(logdir)

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    summary_writer = tf.contrib.summary.create_file_writer(logdir=logdir)
    switch_to(GRAPH_MODE)

    output_path.mkdir(parents=True, exist_ok=True)
    callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))
    callbacks.append(ModelCheckpoint(str(output_path) + "/weights.{epoch:03d}-{val_loss:.3f}-{val_PSNR:.5f}.hdf5",
                                     monitor="val_PSNR",
                                     verbose=1,
                                     mode="max",
                                     save_best_only=save_best_only))
    callbacks.append(N2NCallback(val_generator, batch_size, summary_writer))

    hist = model.fit_generator(generator=generator,
                               steps_per_epoch=steps,
                               epochs=nb_epochs,
                               validation_data=val_generator,
                               verbose=1, callbacks=callbacks)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)


def tensorboard_command_noise2noise(basedir):
    """Return the command line command used to launch a tensorboard instance for tracking Noise2Noise's progress

    Parameters
    ----------
    basedir : str
        the path to the project

    Returns
    -------
    The command used to launch tensorboard instance for TomoGAN
    """
    command = f"tensorboard --logdir='{basedir}noise2noise/logs/' --samples_per_plugin=images=300"
    print(command)
