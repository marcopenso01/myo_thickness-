import nibabel as nib
import numpy as np
import os
import glob

def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


def get_latest_model_checkpoint_path(folder):
    '''
    :param folder: Folder where the checkpoints are saved
    :return: The model of the latest iteration
    '''
    for file in glob.glob(os.path.join(folder, 'checkpoint*.h5')):

        filename = file.split('/')[-1]
        epoch = filename.split('_')[2]
        step = filename.split('_')[4]
        loss = filename.split('_')[6]
        dice = filename.split('_')[8].split('.h5')[0]

    return filename, int(epoch), int(step), float(loss), float(dice)
