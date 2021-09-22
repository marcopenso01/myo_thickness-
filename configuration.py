import os
import tensorflow as tf
import model_structure

experiment_name = 'test1'

# Model settings Unet2D
weight_init = 'he_normal'    # xavier_uniform/ xavier_normal/ he_normal /he_uniform /caffe_uniform/ simple/ bilinear
model_handle = model_structure.unet2D_same
#model_handle = model_structure.ResUNet
#model_handle = model_structure.DenseUNet
#model_handle = model_structure.net1

# Data settings
data_mode = '2D' 
image_size = (192, 192)
nlabels = 4
train_on_all_data = False 
gt_exists = True    #True if it exists the ground_trth images, otherwise False.

# Training settings
batch_size = 4 
learning_rate = 0.001
optimizer_handle = tf.compat.v1.train.AdamOptimizer     #(beta1 = 0.9, beta2 = 0.999, epsilon=1e-08)
schedule_lr = False       #decrease 10 times the LR when loss gradient lower than threshold
weight_decay = 0 
momentum = None
loss_type = 'crossentropy_and_dice'     #'weighted_crossentropy'/'crossentropy'/'dice'/'dice_onlyfg'/'crossentropy_and_dice (alfa,1-alfa)'/'tversky'/'focal_tversky'/crossentropy_and_focal_tversky(alfa,1-alfa) --> https://arxiv.org/pdf/2006.14822.pdf
alfa = 0.6    #[0-1]    
augment_batch = True

# Augmentation settings
do_rotation_range = True   #random rotation in range "rg" (min,max)
rg = (-20,20)     
gamma = True               #transforms the input image pixelwise according to the equation O = I**gamma
prob = 1                   #Probability [0.0/1.0] (0 no augmentation, 1 always)

# Paths settings (need to mount MyDrive before)
data_root = 'F:\ARTEFACTS'      
test_data_root = 'F:\ARTEFACTS'
project_root = 'F:\ARTEFACTS'                       
log_root = os.path.join(project_root, 'artefact_logdir')
weights_root = os.path.join(log_root, experiment_name)

# Pre-process settings
standardize = False
normalize = True

# Rarely changed settings
max_epochs = 1000

train_eval_frequency = 200
val_eval_frequency = 150 
