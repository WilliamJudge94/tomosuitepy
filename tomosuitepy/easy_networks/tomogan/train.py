import os
import sys
import warnings
import numpy as np
from tqdm import tqdm
from util import save2img, save2img_tb
from models import unet as make_generator_model
import sys, os, time, argparse, shutil, scipy, h5py, glob
from models import tomogan_disc as make_discriminator_model
from data_processor import bkgdGen, gen_train_batch_bg, get1batch4test

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf 
    tf.enable_eager_execution()

sys.path.append(os.path.dirname(__file__))
path1 = os.path.dirname(__file__)
path2 = '/'.join(path1.split('/')[:-2])
vgg19_path = f'{path2}/hard_networks/TomoGAN/vgg19_weights_notop.h5'

def train_tomogan(basedir, epochs=120001, gpus='0', lmse=0.5, lperc=2.0, 
                  ladv=20, lunet=3, depth=1, itg=1, itd=2, mb_size=2, img_size=896, types='noise'):
    """Allows the User to use TomoGAN to denoise a dataset with no extra data. Please use tomosuite.noise_test_tomogan to 
    
    Parameters
    ----------
    basedir : str
        the path to the project
        
    epochs : int
        the number of epochs to train TomoGAN
        
    gpus : str
        the string value of which gpus one would like to be available to TomoGAN
        
    lmse : float
        unknown
        
    lperc : float
        unknown
        
    ladv : int
        unknown
        
    lunet : int
        the number of layers of UNet to use
        
    depth : int
        the depth of the input image
        
    itg : int
        the iterations of the generator
        
    itd : int
        the interations of the descriminator
        
    mb_size : int
        the batch size
        
    img_size : int
        the size of the image (must be square)
        
    types : str
        the type of data being passed to TomoGAN. Example 'noise' or 'artifact'
    
    Returns
    -------
    Allows User to train TomoGAN for denoising tomographic projections
    """

    # Setting up TomoGAN parameters
    expName = f'{basedir}tomogan/{types}_experiment'
    location = f'{basedir}tomogan/'

    xtrain = f'{location}xtrain_tomogan_{types}_AI.h5'
    ytrain = f'{location}ytrain_tomogan_{types}_AI.h5'
    xtest = f'{location}xtest_tomogan_{types}_AI.h5'
    ytest = f'{location}ytest_tomogan_{types}_AI.h5'

    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    
    # disable printing INFO, WARNING, and ERROR
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.99
    sess = tf.compat.v1.Session(config = config)
    tf.compat.v1.keras.backend.set_session(sess)

    in_depth = depth
    disc_iters, gene_iters = itd, itg
    lambda_mse, lambda_adv, lambda_perc = lmse, ladv, lperc

    # Logging parameters
    itr_out_dir = expName + '-itrOut'
    logdir = f"{basedir}tomogan/logs"

    if os.path.isdir(logdir): 
        shutil.rmtree(logdir)
    os.mkdir(logdir)
    
    if os.path.isdir(itr_out_dir): 
        shutil.rmtree(itr_out_dir)
    os.mkdir(itr_out_dir)

    print('X train: {}\nY train: {}\nX test: {}\nY test: {}'.format(xtrain, ytrain, xtest, ytest))

    # build minibatch data generator with prefetch
    mb_data_iter = bkgdGen(data_generator=gen_train_batch_bg(x_fn=xtrain, \
                                          y_fn=ytrain, mb_size=mb_size, \
                                          in_depth=in_depth, img_size=img_size), \
                           max_prefetch=16)   

    generator = make_generator_model(input_shape=(None, None, in_depth), nlayers=int(lunet) ) 
    discriminator = make_discriminator_model(input_shape=(img_size, img_size, 1))
    feature_extractor_vgg = tf.keras.applications.VGG19(\
                            weights=vgg19_path, \
                            include_top=False)

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def adversarial_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    gen_optimizer  = tf.compat.v1.train.AdamOptimizer(1e-4)
    disc_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
    
    summary_writer = tf.contrib.summary.create_file_writer(logdir=logdir)
    tf.contrib.summary.always_record_summaries()
    
    for epoch in range(epochs):
        time_git_st = time.time()
        print(epoch)
        for _ge in range(gene_iters):
            X_mb, y_mb = mb_data_iter.next() # with prefetch
            with tf.GradientTape() as gen_tape:
                gen_tape.watch(generator.trainable_variables)

                gen_imgs = generator(X_mb, training=True)
                disc_fake_o = discriminator(gen_imgs, training=False)

                loss_mse = tf.losses.mean_squared_error(gen_imgs, y_mb)
                loss_adv = adversarial_loss(disc_fake_o)

                vggf_gt  = feature_extractor_vgg.predict(tf.concat([y_mb, y_mb, y_mb], 3).numpy())
                vggf_gen = feature_extractor_vgg.predict(tf.concat([gen_imgs, gen_imgs, gen_imgs], 3).numpy())
                perc_loss= tf.losses.mean_squared_error(vggf_gt.reshape(-1), vggf_gen.reshape(-1))

                gen_loss = lambda_adv * loss_adv + lambda_mse * loss_mse + lambda_perc * perc_loss

                gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
                gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

        itr_prints_gen = '[Info] Epoch: %05d, gloss: %.2f (mse%.3f, adv%.3f, perc:%.3f), gen_elapse: %.2fs/itr' % (\
                         epoch, gen_loss, loss_mse*lambda_mse, loss_adv*lambda_adv, perc_loss*lambda_perc, \
                         (time.time() - time_git_st)/gene_iters, )
        
        time_dit_st = time.time()

        for _de in range(disc_iters):
            X_mb, y_mb = mb_data_iter.next() # with prefetch        
            with tf.GradientTape() as disc_tape:
                disc_tape.watch(discriminator.trainable_variables)

                gen_imgs = generator(X_mb, training=False)

                disc_real_o = discriminator(y_mb, training=True)
                disc_fake_o = discriminator(gen_imgs, training=True)

                disc_loss = discriminator_loss(disc_real_o, disc_fake_o)

                disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

        print('%s; dloss: %.2f (r%.3f, f%.3f), disc_elapse: %.2fs/itr, gan_elapse: %.2fs/itr' % (itr_prints_gen,\
              disc_loss, disc_real_o.numpy().mean(), disc_fake_o.numpy().mean(), \
              (time.time() - time_dit_st)/disc_iters, time.time()-time_git_st))

        if epoch % (200//gene_iters) == 0:
            X222, y222 = get1batch4test(x_fn=xtest, y_fn=ytest, in_depth=in_depth)
            pred_img = generator.predict(X222[:1])

            idxxx = 0
            generator.save("%s/%s-it%05d.h5" % (itr_out_dir, f'{types}_experiment', epoch), \
                           include_optimizer=True)
            
            x = tf.constant(1.8, dtype=tf.float32)
            
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('gen_loss', tf.constant(gen_loss), step=epoch)
                tf.contrib.summary.scalar('gen_loss-mse_lambda-mse', tf.constant(loss_mse*lambda_mse), step=epoch)
                tf.contrib.summary.scalar('gen_loss-adv_lambda-adv', tf.constant(loss_adv*lambda_adv), step=epoch)
                tf.contrib.summary.scalar('gen_perc-loss_lambda-perc', tf.constant(perc_loss*lambda_perc), step=epoch)

                tf.contrib.summary.scalar('disc_loss', tf.constant(disc_loss), step=epoch)
                tf.contrib.summary.scalar('disc_real_o', tf.constant(disc_real_o.numpy().mean()), step=epoch)
                tf.contrib.summary.scalar('disc_fake_o', tf.constant(disc_fake_o.numpy().mean()), step=epoch)

                if epoch == 0:
                    ground_truth_image = save2img_tb(y222[idxxx,:,:,idxxx])
                    ground_truth_image =  np.expand_dims(ground_truth_image, 0)
                    ground_truth_image =  np.expand_dims(ground_truth_image, 3)
                    
                    noisy_image = save2img_tb(X222[idxxx,:,:,in_depth//2])
                    noisy_image = np.expand_dims(noisy_image, 0)
                    noisy_image = np.expand_dims(noisy_image, 3)
                    
                    tf.contrib.summary.image('gtruth', tf.constant(ground_truth_image), step=epoch, max_images=1)
                    tf.contrib.summary.image('noisy', tf.constant(noisy_image), step=epoch, max_images=1)

                predicted_image = save2img_tb(pred_img[idxxx,:,:,idxxx])
                predicted_image = np.expand_dims(predicted_image, 0)
                predicted_image = np.expand_dims(predicted_image, 3)
                tf.contrib.summary.image('predictions', tf.constant(predicted_image), step=epoch, max_images=1)

        sys.stdout.flush()
        
def tensorboard_command_tomogan(basedir):
    """Return the command line command used to launch a tensorboard instance for tracking TomoGAN's progress
    
    Parameters
    ----------
    basedir : str
        the path to the project
        
    Returns
    -------
    The command used to launch tensorboard instance for TomoGAN
    """
    command = f"tensorboard --logdir='{basedir}tomogan/logs/' --samples_per_plugin=images=300"
    print(command)