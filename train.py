import os.path
import shutil
import time
from glob import glob

import h5py
import tensorflow as tf

import os
import numpy as np
import logging
import glob
import matplotlib.pyplot as plt
from packaging import version
from tensorflow.keras import Model, layers
from tfwrapper import metrics
import utils
import model as model
import configuration as config
import augmentation as aug
from background_generator import BackgroundGenerator

logging.basicConfig(
    level=logging.INFO # allow DEBUG level messages to pass through the logger
    )

log_dir = os.path.join(config.log_root, config.experiment_name)

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "this notebook requires Tensorflow 2.0 or above"

def run_training(continue_run):

    logging.info('EXPERIMENT NAME: %s' % config.experiment_name)

    init_step = 0

    # Load data train
    data = h5py.File(os.path.join(config.data_root, 'train.hdf5'), 'r')
    
    # the following are HDF5 datasets, not numpy arrays
    images_train = data['images_train'][()]
    labels_train = data['masks_train'][()]
    data.close()

    if not train_on_all_data:
        data = h5py.File(os.path.join(config.data_root, 'val.hdf5'), 'r')
        images_val = data['images_train'][()]
        labels_val = data['masks_train'][()]
        data.close()
        
    logging.info('Data summary:')
    logging.info(' - Training Images:')
    logging.info(images_train.shape)
    logging.info(images_train.dtype)
    logging.info(' - Training Labels:')
    logging.info(labels_train.shape)
    logging.info(labels_train.dtype)
    if not train_on_all_data:
        logging.info(' - Validation Images:')
        logging.info(images_val.shape)
        logging.info(images_val.dtype)
    
    nlabels = config.nlabels
    
    #restore previous session
    if continue_run:
        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!! Continuing previous run !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        try:
            
            init_epoch = last_epoch + 1
        except:
            logging.warning('!!! Didnt find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0   
    
    elif continue_run == False:
        
        step = init_step
        curr_lr = config.learning_rate
        init_epoch = 0
        
        no_improvement_counter = 0
        best_val = np.inf
        last_train = np.inf
        best_dice = 0
        train_loss_history = []
        val_loss_history = []
        train_dice_history = []
        val_dice_history = []
        lr_history = []
        
        # Build a model
        model = model_zoo.get_model(images_train, nlabels, config)
        model.summary()
        
        logging.info('compiling model...')
        model.compile(optimizer=Adam(lr=curr_lr),
                      loss=metrics.unified_focal_loss, 
                      metrics=metrics.dice_coefficient)
        
        for epoch in range(init_epoch, config.max_epochs):

            logging.info('EPOCH %d/%d' % (epoch, config.max_epochs)
            
            train_temp = []
            val_temp = []

            for batch in iterate_minibatches(images_train,
                                             labels_train,
                                             batch_size=config.batch_size,
                                             augment_batch=config.augment_batch):
                
                start_time = time.time()
                         
                x, y = batch

                # TEMPORARY HACK (to avoid incomplete batches)
                if y.shape[0] < config.batch_size:
                    step += 1
                    continue

                hist = model.train_on_batch(x,y)

                loss_value, _ = hist

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if step % 20 == 0:
                    # Print status to stdout.
                    logging.info('Step %d: loss = %.3f (%.3f sec)' % (step, loss_value, duration))
                
                if (step + 1) % config.train_eval_frequency == 0:

                    logging.info('Training Data Eval:')
                    [train_loss, train_dice] = do_eval(sess,
                                                       eval_loss,
                                                       images_pl,
                                                       labels_pl,
                                                       training_pl,
                                                       images_train,
                                                       labels_train,
                                                       config.batch_size)

                    train_summary_msg = sess.run(train_summary, feed_dict={train_error_: train_loss,
                                                                           train_dice_: train_dice}
                                                 )
                    summary_writer.add_summary(train_summary_msg, step)

                    train_temp.append([train_loss, train_dice])
                    
                    curr_lr = config.learning_rate * train_loss
                    logging.info('Learning rate change to: %f' % curr_lr)
                    
                    if train_loss < last_train:  # best_train found:
                        no_improvement_counter = 0
                        logging.info('Decrease in training error!')
                    else:
                        no_improvement_counter = no_improvement_counter+1
                        logging.info('No improvment in training error for %d steps' % no_improvement_counter)

                    last_train = train_loss

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % config.val_eval_frequency == 0:

                    checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                    filelist = glob.glob(os.path.join(log_dir, 'model.ckpt*'))
                    for file in filelist:
                        os.remove(file)
                    saver.save(sess, checkpoint_file, global_step=step)
                    # Evaluate against the training set.

                    if not train_on_all_data:

                        # Evaluate against the validation set.
                        logging.info('Validation Data Eval:')
                        [val_loss, val_dice] = do_eval(sess,
                                                       eval_loss,
                                                       images_pl,
                                                       labels_pl,
                                                       training_pl,
                                                       images_val,
                                                       labels_val,
                                                       config.batch_size)
                        
                        val_temp.append([val_loss, val_dice])
                        val_summary_msg = sess.run(val_summary, feed_dict={val_error_: val_loss, val_dice_: val_dice}
                        )
                        summary_writer.add_summary(val_summary_msg, step)

                        if val_dice > best_dice:
                            best_dice = val_dice
                            best_file = os.path.join(log_dir, 'model_best_dice.ckpt')
                            filelist = glob.glob(os.path.join(log_dir, 'model_best_dice*'))
                            for file in filelist:
                                os.remove(file)
                            saver_best_dice.save(sess, best_file, global_step=step)
                            logging.info('Found new best dice on validation set! - %f -  Saving model_best_dice.ckpt' % val_dice)

                        if val_loss < best_val:
                            best_val = val_loss
                            best_file = os.path.join(log_dir, 'model_best_loss.ckpt')
                            filelist = glob.glob(os.path.join(log_dir, 'model_best_loss*'))
                            for file in filelist:
                                os.remove(file)
                            saver_best_loss.save(sess, best_file, global_step=step)
                            logging.info('Found new best loss on validation set! - %f -  Saving model_best_loss.ckpt' % val_loss)

                step += 1
                
            # end epoch
            if len(train_temp)!=0:
                lr_history.append(curr_lr)
                sum_dice = 0
                sum_loss = 0
                for i in range(len(train_temp)):
                    sum_loss += train_temp[i][0]
                    sum_dice += train_temp[i][1]
                train_loss_history.append(sum_loss/len(train_temp))
                train_dice_history.append(sum_dice/len(train_temp))
                sum_dice = 0
                sum_loss = 0
                for i in range(len(val_temp)):
                    sum_loss += val_temp[i][0]
                    sum_dice += val_temp[i][1]
                val_loss_history.append(sum_loss/len(val_temp))
                val_dice_history.append(sum_dice/len(val_temp))
            
        sess.close()
        
        #plot history (loss, dice, lr)
        plt.figure()
        plt.plot(train_loss_history, label='train_loss')
        plt.plot(val_loss_history, label='val_loss')
        plt.title('model loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        plt.figure()
        plt.plot(train_dice_history, label='train_dice')
        plt.plot(val_dice_history, label='val_dice')
        plt.title('model dice')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('dice')
        plt.show()
        plt.figure()
        plt.plot(lr_history)
        plt.title('model learning rate')
        plt.xlabel('epoch')
        plt.ylabel('learning rate')
        plt.show() 


def do_eval(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            training_time_placeholder,
            images,
            labels,
            batch_size):

    '''
    Function for running the evaluations every X iterations on the training and validation sets. 
    :param sess: The current tf session 
    :param eval_loss: The placeholder containing the eval loss
    :param images_placeholder: Placeholder for the images
    :param labels_placeholder: Placeholder for the masks
    :param training_time_placeholder: Placeholder toggling the training/testing mode. 
    :param images: A numpy array or h5py dataset containing the images
    :param labels: A numpy array or h5py dataset containing the corresponding labels 
    :param batch_size: The batch_size to use. 
    :return: The average loss (as defined in the experiment), and the average dice over all `images`. 
    '''

    loss_ii = 0
    dice_ii = 0
    num_batches = 0

    for batch in BackgroundGenerator(iterate_minibatches(images, labels, batch_size=batch_size, augment_batch=False)):  # No aug in evaluation
    # you can wrap the iterate_minibatches function in the BackgroundGenerator class for speed improvements
    # but at the risk of not catching exceptions

        x, y = batch

        if y.shape[0] < batch_size:
            continue

        feed_dict = { images_placeholder: x,
                      labels_placeholder: y,
                      training_time_placeholder: False}

        closs, cdice = sess.run(eval_loss, feed_dict=feed_dict)
        loss_ii += closs
        dice_ii += cdice
        num_batches += 1

    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches

    logging.info('  Average loss: %0.04f, average dice: %0.04f' % (avg_loss, avg_dice))

    return avg_loss, avg_dice


def iterate_minibatches(images, labels, batch_size, augment_batch=False):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: hdf5 dataset
    :param labels: hdf5 dataset
    :param batch_size: batch size
    :return: mini batches
    '''

    random_indices = np.arange(images.shape[0])
    np.random.shuffle(random_indices)

    n_images = images.shape[0]

    for b_i in range(0,n_images,batch_size):

        if b_i + batch_size > n_images:
            continue

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]
        y = labels[batch_indices, ...]
        #Xid = id_img[batch_indices]

        image_tensor_shape = [X.shape[0]] + list(config.image_size) + [1]
        X = np.reshape(X, image_tensor_shape)
        
        if augment_batch:
            X, y = aug.augmentation_function(X, y)

            
        yield X, y


def main():

    continue_run = True
    if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)
        continue_run = False

    # Copy experiment config file
    shutil.copy(config.__file__, log_dir)

    run_training(continue_run)


if __name__ == '__main__':
 
    main()
