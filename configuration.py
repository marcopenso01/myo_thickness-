import os
import model_zoo

experiment_name = 'test1'

# Model settings
model_handle = model_zoo.unet2D_same
#model_handle = model_zoo.net1

# Data settings
data_mode = '2D' 
image_size = (160, 160)
nlabels = 2
train_on_all_data = False 
gt_exists = True    #True if it exists the ground_trth images, otherwise False.

# Training settings
batch_size = 4 
learning_rate = 0.001
loss_type = 'unified_focal'  #'binary_crossentropy' 'dice' 'tversky' 'focal_tversky' 'focal_loss' 'hybrid_focal' 'asymmetric_focal' 'asymmetric_focal_tversky' 'unified_focal'
augment_batch = True

# Decay Learning rate
time_decay = False      #LearningRate = LearningRate * 1/(1 + decay * epoch)
step_decay = False      #LearningRate = InitialLearningRate * DropRate^floor(epoch / epochDrop)
exp_decay = False       #LearningRate = InitialLearningRate * exp^(-decay * epoch)
adaptive_decay = True  #LearningRate = InitialLearningRate * loss_function

# Augmentation settings
do_rotation_range = True   #random rotation in range "rg" (min,max)
rg = (0,359)
do_fliplr = True           #Flip array in the left/right direction
do_flipud = True           #Flip array in the up/down direction.
gamma = True               #transforms the input image pixelwise according to the equation O = I**gamma
prob = 1                   #Probability [0.0/1.0] (0 no augmentation, 1 always)

# Paths settings (need to mount MyDrive before)
data_root = 'F:\ARTEFACTS'      
test_data_root = 'F:\ARTEFACTS'
project_root = 'F:\ARTEFACTS'                       
log_root = os.path.join(project_root, 'MyoMR_logdir')
weights_root = os.path.join(log_root, experiment_name)

# Pre-process settings
standardize = False
normalize = True

# Rarely changed settings
max_epochs = 1000

train_eval_frequency = 200
