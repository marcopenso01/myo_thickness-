import tensorflow as tf
from tensorflow.keras import losses, metrics, backend
import numpy as np


def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


################################
#  Binary cross entropy loss   #
################################
def binary_cross_entropy_loss(y_true, y_pred):
    '''
    binary cross entropy loss
    '''
    loss = losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
    return loss
       
        
################################
#      Dice coefficient        #
################################
def dice_coefficient(delta = 0.5, smooth = 0.000001):
    """
    Parameters
    ----------
    delta : float, optional
    controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
    smoothing constant to prevent division by zero errors, by default 0.000001
    """
    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        tp = tf.math.reduce_sum(y_true * y_pred, axis=axis)
        fn = tf.math.reduce_sum(y_true * (1-y_pred), axis=axis)
        fp = tf.math.reduce_sum((1-y_true) * y_pred, axis=axis)
        # Calculate Dice score
        dice_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        dice = tf.math.reduce_mean(dice_class)

        return dice
    return loss_function


################################
#           Dice loss          #
################################
def dice_loss(delta=0.5, smooth=0.000001):
    """
    Dice loss originates from Sørensen–Dice coefficient, which is a statistic developed in 1940s to gauge the similarity between two samples.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.5
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = tf.math.reduce_sum(y_true * y_pred, axis=axis)
        fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=axis)
        fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=axis)
        # Calculate Dice score
        dice_class = (tp + smooth) / (tp + delta * fn + (1 - delta) * fp + smooth)
        # Average class scores
        dice_loss = tf.math.reduce_mean(1 - dice_class)

        return dice_loss

    return loss_function


################################
#         Tversky loss         #
################################
def tversky_loss(delta = 0.7, smooth = 0.000001):
    """Tversky loss function
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
        delta=0.5 : dice coefficient
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """    
    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = tf.math.reduce_sum(y_true * y_pred, axis=axis)
        fn = tf.math.reduce_sum(y_true * (1-y_pred), axis=axis)
        fp = tf.math.reduce_sum((1-y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        tversky_loss = tf.math.reduce_mean(1-tversky_class)

        return tversky_loss
    return loss_function

################################
#      Focal Tversky loss      #
################################
def focal_tversky_loss(delta=0.7, gamma=0.75, smooth=0.000001):
    """Focal Tversky loss
    Parameters
    ----------
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        epsilon = backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        axis = identify_axis(y_true.get_shape())
        tp = tf.math.reduce_sum(y_true * y_pred, axis=axis)
        fn = tf.math.reduce_sum(y_true * (1-y_pred), axis=axis)
        fp = tf.math.reduce_sum((1-y_true) * y_pred, axis=axis)
        tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
        # Average class scores
        focal_tversky_loss = tf.math.reduce_mean(tf.math.pow((1-tversky_class), gamma))

        return focal_tversky_loss
    return loss_function

################################
#          Focal loss          #
################################
def focal_loss(alpha=None, beta=None, gamma_f=2.):
    """Focal loss
    Parameters
    ----------
    alpha : float, optional
        controls weight given to each class, by default None
    beta : float, optional
        controls relative weight of false positives and false negatives. Beta > 0.5 penalises false negatives more than false positives, by default None
    gamma_f : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 2.
    """
    def loss_function(y_true, y_pred):
        epsilon = backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        if beta is not None:
            beta_weight = np.array([beta, 1-beta])
            cross_entropy = beta_weight * cross_entropy

        if alpha is not None:
            alpha_weight = np.array(alpha, dtype=np.float32)
            focal_loss = alpha_weight * tf.math.pow(1 - y_pred, gamma_f) * cross_entropy
        else:
            focal_loss = tf.math.pow(1 - y_pred, gamma_f) * cross_entropy

        focal_loss = tf.math.reduce_mean(tf.math.reduce_sum(focal_loss, axis=[-1]))

        return focal_loss
    return loss_function


################################
#       Hybrid Focal loss      #
################################
def hybrid_focal_loss(weight=None, alpha=None, beta=None, gamma=0.75, gamma_f=2.):
    """Default is the linear unweighted sum of the Focal loss and Focal Tversky loss
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to Focal Tversky loss and Focal loss, by default None
    alpha : float, optional
        controls weight given to each class, by default None
    beta : float, optional
        controls relative weight of false positives and false negatives. Beta > 0.5 penalises  false negatives more than false positives, by default None
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 0.75
    gamma_f : float, optional
        Focal loss' focal parameter controls degree of down-weighting of easy examples, by default 2.
    """
    def loss_function(y_true,y_pred):
        focal_tversky = focal_tversky_loss(gamma=gamma)(y_true,y_pred)
        focal = focal_loss(alpha=alpha, beta=beta, gamma_f=gamma_f)(y_true,y_pred)
        # return weighted sum of Focal loss and Focal Dice loss
        if weight is not None:
            return (weight * focal_tversky) + ((1-weight) * focal)
        else:
            return focal_tversky + focal

    return loss_function

################################
#     Asymmetric Focal loss    #
################################
def asymmetric_focal_loss(delta=0.25, gamma=2.):
    """For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.25
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """
    def loss_function(y_true, y_pred):
        epsilon = backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)

        #calculate losses separately for each class, only suppressing background class
        back_ce = tf.math.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - delta) * back_ce

        fore_ce = cross_entropy[:,:,:,1]
        fore_ce = delta * fore_ce

        loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.stack([back_ce, fore_ce],axis=-1),axis=-1))
        return loss
    return loss_function


#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    value > 0.5 penalising FN predictions more than FP
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):
        epsilon = backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = tf.math.reduce_sum(y_true * y_pred, axis=axis)
        fn = tf.math.reduce_sum(y_true * (1-y_pred), axis=axis)
        fp = tf.math.reduce_sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:,0])
        fore_dice = (1-dice_class[:,1]) * tf.math.pow(1-dice_class[:,1], -gamma)

        # Average class scores
        loss = tf.math.reduce_mean(tf.stack([back_dice,fore_dice],axis=-1))
        return loss
    return loss_function


################################
#      Unified Focal loss      #
################################
def unified_focal_loss(weight=0.5, delta=0.6, gamma=0.2):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to Asymmetric Focal Tversky loss and Asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.2
    """
    def loss_function(y_true, y_pred):
        asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true,y_pred)
        # Obtain Asymmetric Focal loss
        asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true,y_pred)
        # return weighted sum of Asymmetrical Focal loss and Asymmetric Focal Tversky loss
        if weight is not None:
            return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl

    return loss_function
