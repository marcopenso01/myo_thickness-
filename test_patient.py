import logging
import os
import time

import h5py
import numpy as np
import tensorflow as tf

import configuration as config
import image_utils
import metrics
import model as model
import utils

logging.basicConfig(
    level=logging.INFO  # allow DEBUG level messages to pass through the logger
)


def score_data(input_folder, output_folder, model_path, config, do_postprocessing=False, dice=True):
    nx, ny = config.image_size[:2]
    batch_size = 1
    num_channels = config.nlabels

    image_tensor_shape = [batch_size] + list(config.image_size) + [1]
    images_pl = tf.compat.v1.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    mask_pl, softmax_pl = model.predict(images_pl, config)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        if dice:
            nn = 'model_best_dice.ckpt'
            data_file_name = 'pred_on_dice.hdf5'
        else:
            nn = 'model_best_loss.ckpt'
            data_file_name = 'pred_on_loss.hdf5'

        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, nn)
        saver.restore(sess, checkpoint_path)
        init_iteration = int(checkpoint_path.split('/')[-1].split('-')[-1])
        total_time = 0
        total_volumes = 0

        data_file_path = os.path.join(output_folder, data_file_name)
        out_file = h5py.File(data_file_path, "w")

        RAW = []
        PRED = []
        PAZ = []
        PHS = []
        MASK = []
        CIR_MASK = []

        for paz in os.listdir(input_path):

            start_time = time.time()
            logging.info('Reading %s' % paz)
            data = h5py.File(os.path.join(input_path, paz, 'pre_proc', 'artefacts.hdf5'), 'r')

            n_file = len(data['img_raw'][()])
            for ii in range(n_file):
                RAW.append(data['img_raw'][ii])
                PAZ.append(paz)
                PHS.append(data['phase'][ii])
                if config.gt_exists:
                    MASK.append(data['mask'][ii])
                    CIR_MASK.append(data['mask_cir'][ii])

                img = data['img_raw'][ii].copy()
                if config.standardize:
                    img = image_utils.standardize_image(img)
                if config.normalize:
                    img = image_utils.normalize_image(img)

                # GET PREDICTION
                feed_dict = {
                    images_pl: img,
                }

                mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict=feed_dict)
                prediction_cropped = np.squeeze(logits_out[0, ...])

                prediction = np.uint8(np.argmax(prediction_cropped, axis=-1))

                if do_postprocessing:
                    prediction = image_utils.keep_largest_connected_components(prediction)

                PRED.append(prediction)

            data.close()
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            total_volumes += 1
            logging.info('Evaluation of volume took %f secs.' % elapsed_time)

        n_file = len(PRED)
        dt = h5py.special_dtype(vlen=str)
        out_file.create_dataset('img_raw', [n_file] + [nx, ny], dtype=np.float32)
        out_file.create_dataset('pred', [n_file] + [nx, ny], dtype=np.uint8)
        out_file.create_dataset('paz', (n_file,), dtype=dt)
        out_file.create_dataset('phase', (n_file,), dtype=dt)
        if config.gt_exists:
            out_file.create_dataset('mask', [n_file] + [nx, ny], dtype=np.uint8)
            out_file.create_dataset('mask_cir', [n_file] + [nx, ny], dtype=np.uint8)

        for i in range(n_file):
            out_file['img_raw'][i, ...] = RAW[i]
            out_file['pred'][i, ...] = PRED[i]
            out_file['paz'][i, ...] = PAZ[i]
            out_file['phase'][i, ...] = PHS[i]
            if config.gt_exists:
                out_file['mask'][i, ...] = MASK[i]
                out_file['mask_cir'][i, ...] = CIR_MASK[i]

        # free memory
        RAW = []
        PRED = []
        PAZ = []
        PHS = []
        MASK = []
        CIR_MASK = []

        out_file.close()

        logging.info('Average time per volume: %f' % (total_time / total_volumes))


if __name__ == '__main__':
    log_root = config.log_root
    model_path = os.path.join(log_root, config.experiment_name)
    logging.info(model_path)

    logging.warning('EVALUATING ON TEST SET')
    input_path = config.test_data_root
    output_path = os.path.join(model_path, 'predictions')

    score_data(input_path,
               output_path,
               model_path,
               config=config,
               do_postprocessing=True,
               dice=True)

    if config.gt_exists:
        metrics.main(output_path)
