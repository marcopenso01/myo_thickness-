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
import math
import matplotlib.pyplot as plt
from packaging import version
from tensorflow.keras.models import load_model
from tfwrapper import metrics
import utils
import configuration as config
import augmentation as aug
from background_generator import BackgroundGenerator
import model_zoo as model_zoo

logging.basicConfig(
    level=logging.INFO # allow DEBUG level messages to pass through the logger
    )

log_dir = os.path.join(config.log_root, config.experiment_name)

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "this notebook requires Tensorflow 2.0 or above"

def run_training(continue_run):

    logging.info('EXPERIMENT NAME: %s' % config.experiment_name)
    print_txt(log_dir, ['\nEXPERIMENT NAME: %s' % config.experiment_name])

    # Load data train
    data = h5py.File(os.path.join(config.data_root, 'train.hdf5'), 'r')
    
    # the following are HDF5 datasets, not numpy arrays
    images_train = data['images_train'][()]
    labels_train = data['masks_train'][()]
    data.close()

    if not config.train_on_all_data:
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
    if not config.train_on_all_data:
        logging.info(' - Validation Images:')
        logging.info(images_val.shape)
        logging.info(images_val.dtype)
    
    print_txt(log_dir, ['\nData summary:'])
    print_txt(log_dir, ['\n - Training Images:\n'])
    print_txt(log_dir, str(images_train.shape))
    print_txt(log_dir, ['\n'])
    print_txt(log_dir, str(images_train.dtype))
    print_txt(log_dir, ['\n - Training Labels:\n'])
    print_txt(log_dir, str(labels_train.shape))
    print_txt(log_dir, ['\n'])
    print_txt(log_dir, str(labels_train.dtype))
    if not config.train_on_all_data:
        print_txt(log_dir, ['\n - Validation Images:\n'])
        print_txt(log_dir, str(images_val.shape))
        print_txt(log_dir, ['\n'])
        print_txt(log_dir, str(images_val.dtype))
    
    nlabels = config.nlabels
    
    #restore previous session
    if continue_run:
        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!! Continuing previous run !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        try:
            filename, last_epoch, last_step, best_loss, best_dice = utils.get_latest_model_checkpoint_path(log_dir)
            logging.info('loading model...')
            print_txt(log_dir, ['\nloading model...'])
            model = load_model(os.path.join(log_dir, filename))
            logging.info('Latest epoch was: %d' % last_epoch)
            print_txt(log_dir, ['\nLatest epoch was: %d' % last_epoch])
            logging.info('Latest step was: %d' % last_step)
            print_txt(log_dir, ['\nLatest step was: %d' % last_step])
            init_epoch = last_epoch + 1 #plus 1 otherwise repeats last epoch
            init_step = last_step + 1
            curr_lr = model.optimizer.learning_rate.numpy()
        except:
            logging.warning('!!! Didnt find init checkpoint. Maybe first run failed. Disabling continue mode...')
            print_txt(log_dir, ['\n!!! Didnt find init checkpoint. Maybe first run failed. Disabling continue mode...'])
            continue_run = False
            
    if not continue_run:
        
        init_epoch = 0
        init_step = 0 
        best_loss = np.inf
        best_dice = 0
        curr_lr = config.learning_rate
        
        # Build a model
        model = model_zoo.get_model(images_train, nlabels, config)
        model.summary()
        with open(log_dir + 'summary_report.txt','a') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
        
        logging.info('compiling model...')
        print_txt(log_dir, ['\ncompiling model...'])
        
        if config.loss_type == 'binary_crossentropy':
            loss = metrics.binary_cross_entropy_loss()
        elif config.loss_type == 'dice':
            loss = metrics.dice_loss(delta = 0.5)
        elif config.loss_type == 'tversky':
            loss = metrics.tversky_loss(delta = 0.7)
        elif config.loss_type == 'focal_tversky':
            loss = metrics.focal_tversky_loss(delta=0.7, gamma=0.75)
        elif config.loss_type == 'focal_loss':
            loss = metrics.focal_loss(alpha=None, beta=None, gamma_f=2.)
        elif config.loss_type == 'hybrid_focal':
            loss = metrics.hybrid_focal_loss(weight=None, alpha=None, beta=None, gamma=0.75, gamma_f=2.)
        elif config.loss_type == 'asymmetric_focal':
            loss = metrics.asymmetric_focal_loss(delta=0.25, gamma=2.)
        elif config.loss_type == 'asymmetric_focal_tversky':
            loss = metrics.asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75)
        elif config.loss_type == 'unified_focal':
            loss = metrics.unified_focal_loss(weight=0.5, delta=0.6, gamma=0.2)
            
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=curr_lr),
                      loss=loss, 
                      metrics=metrics.dice_coefficient())
    
    step = init_step
    train_loss_history = []
    val_loss_history = []
    train_dice_history = []
    val_dice_history = []
    lr_history = []
        
    for epoch in range(init_epoch, config.max_epochs):

        logging.info('EPOCH %d/%d' % (epoch, config.max_epochs))
        print_txt(log_dir, ['\nEPOCH %d/%d' % (epoch, config.max_epochs)])

        train_temp = []

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

            [loss_value, dice_value] = model.train_on_batch(x,y)

            train_temp.append([loss_value, dice_value])
            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 20 == 0:
                # Print status to stdout.
                logging.info('Epoch %d (Step %d): loss = %.3f - lr = %.6f (%.3f sec)' % (epoch, step, loss_value, curr_lr, duration))
                print_txt(log_dir, ['\nEpoch %d (Step %d): loss = %.3f - lr = %.6f (%.3f sec)' % (epoch, step, loss_value, curr_lr, duration)])
            '''
            if config.adaptive_decay:
                if (step + 1) % config.train_eval_frequency == 0:
                    logging.info('Training Data Eval:')
                    [train_loss, train_dice] = do_eval(model
                                                       images_train,
                                                       labels_train,
                                                       batch_size=config.batch_size)

                    curr_lr = config.learning_rate * train_loss
                    logging.info('Learning rate change to: %f' % curr_lr)
            '''

            # --- end batch ---
            step += 1

        # --- end epoch ---
        # Training Data Eval          
        if len(train_temp)!=0:
            sum_dice = 0
            sum_loss = 0
            for i in range(len(train_temp)):
                sum_loss += train_temp[i][0]
                sum_dice += train_temp[i][1]
            train_loss_history.append(sum_loss/len(train_temp))
            train_dice_history.append(sum_dice/len(train_temp))

        # Validation Data Eval
        if not config.train_on_all_data:
            logging.info('Validation Data Eval:')
            print_txt(log_dir, ['\nValidation Data Eval:'])
            [val_loss, val_dice] = do_eval(model,
                                           images_val,
                                           labels_val,
                                           batch_size=config.batch_size)
            val_loss_history.append(val_loss)
            val_dice_history.append(val_dice)

            if val_dice > best_dice:
                logging.info('Found new best dice on validation set! - %f - saving model' % val_dice)
                print_txt(log_dir, ['\nFound new best dice on validation set! - %f - saving model' % val_dice])
                best_dice = val_dice
                filelist = glob.glob(os.path.join(log_dir, 'model_best_dice*'))
                for file in filelist:
                    os.remove(file)
                model.save(os.path.join(log_dir, 'model_best_dice.h5'))

            if val_loss < best_loss:
                logging.info('Found new best loss on validation set! - %f - saving model' % val_loss)
                print_txt(log_dir, ['\nFound new best loss on validation set! - %f - saving model' % val_loss])
                best_loss = val_loss
                filelist = glob.glob(os.path.join(log_dir, 'model_best_loss*'))
                for file in filelist:
                    os.remove(file)
                model.save(os.path.join(log_dir, 'model_best_loss.h5'))

        # Save a checkpoint
        if (step + 1) % config.val_eval_frequency == 0:
            filelist = glob.glob(os.path.join(log_dir, 'checkpoint*'))
            for file in filelist:
                os.remove(file)
            model.save(os.path.join(log_dir, ('checkpoint' + '_epoch_' + str(epoch) + '_step_' + str(step) + 
                                              '_loss_' + str(best_loss) + '_dice_' + str(best_dice) + '.h5')))
        
        # Learning rate Eval       
        lr_history.append(curr_lr)
        # Decay learning rate
        if config.time_decay:
            #decay_rate = config.learning_rate / config.max_epochs
            decay_rate = 1E-4
            curr_lr *= (1. / (1. + decay_rate * epoch))
            model.optimizer.learning_rate.assign(curr_lr)
        elif config.step_decay:
            drop = 0.5
            epochs_drop = 40.0
            curr_lr = config.learning_rate * math.pow(drop,
                      math.floor((1+epoch)/epochs_drop))
            model.optimizer.learning_rate.assign(curr_lr)
        elif config.exp_decay:
            qq = 0.01
            curr_lr = config.learning_rate * math.exp(-qq*epoch)
            model.optimizer.learning_rate.assign(curr_lr)
        elif config.adaptive_decay:
            curr_lr = config.learning_rate * val_loss
            model.optimizer.learning_rate.assign(curr_lr)
        else:
            curr_lr = curr_lr * 0.985
            model.optimizer.learning_rate.assign(curr_lr)

    # --- end training ---
    #plot history (loss, metrics, lr)
    logging.info('Saving plots...')
    plt.figure()
    plt.plot(train_loss_history, label='train_loss')
    plt.plot(val_loss_history, label='val_loss')
    plt.title('model loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(log_dir,'loss.png'))
    plt.figure()
    plt.plot(train_dice_history, label='train_dice')
    plt.plot(val_dice_history, label='val_dice')
    plt.title('model dice')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.savefig(os.path.join(log_dir,'dice.png'))
    plt.figure()
    plt.plot(lr_history)
    plt.title('model learning rate')
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.savefig(os.path.join(log_dir,'learning_rate.png')) 
    logging.info('END')

def do_eval(model, images, labels, batch_size):
    '''
    Function for running the evaluations every X iterations on the training and validation sets. 
    :param images: A numpy array containing the images
    :param labels: A numpy array containing the corresponding labels 
    :param batch_size: batch size
    :return: The average loss and the average dice over all `images`.
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

        closs, cdice = model.test_on_batch(x,y)

        loss_ii += closs
        dice_ii += cdice
        num_batches += 1

    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches

    logging.info('Average loss: %0.04f, average dice: %0.04f' % (avg_loss, avg_dice))
    print_txt(log_dir, ['\nAverage loss: %0.04f, average dice: %0.04f' % (avg_loss, avg_dice)])

    return avg_loss, avg_dice


def iterate_minibatches(images, labels, batch_size, augment_batch=False):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: tensor
    :param labels: tensor
    :param batch_size: batch size
    :return: mini batches
    '''

    random_indices = np.arange(images.shape[0])
    np.random.shuffle(random_indices)

    n_images = images.shape[0]

    for b_i in range(0,n_images,batch_size):

        if b_i + batch_size > n_images:
            continue

        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]
        y = labels[batch_indices, ...]
        #Xid = id_img[batch_indices]

        image_tensor_shape = [X.shape[0]] + list(config.image_size) + [1]
        X = np.reshape(X, image_tensor_shape)
        
        if augment_batch:
            X, y = aug.augmentation_function(X, y)
            
        yield X, y

 
def print_txt(output_dir, stringa):
    out_file = os.path.join(output_dir, 'summary_report.txt')
    with open(out_file, "a") as text_file:
        text_file.writelines(stringa)


def main():

    continue_run = True
    if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)
        out_file = os.path.join(log_dir, 'summary_report.txt')
        with open(out_file, "w") as text_file:
            text_file.write('\n\n-------------------------------------------------------------------------------------\n')
            text_file.write('Model summary\n')
            text_file.write('-------------------------------------------------------------------------------------\n\n')
        continue_run = False

    # Copy experiment config file
    shutil.copy(config.__file__, log_dir)

    run_training(continue_run)


if __name__ == '__main__':
 
    main()
