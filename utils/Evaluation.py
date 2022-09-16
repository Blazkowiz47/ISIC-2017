import numpy as np
import tensorflow as tf

def get_confusion_matrix(mask, prediction,threshold = None): 
    assert(threshold)
    prediction = convert_to_ones_and_zeros(prediction,threshold)
    mask = convert_to_ones_and_zeros(mask)
    prediction = tf.reshape(prediction,[-1])
    mask = tf.reshape(mask,[-1])
    conf_mat = tf.math.confusion_matrix(mask,prediction)
    return conf_mat
   
def convert_to_ones_and_zeros(image, threshold = None):
    if not tf.is_tensor(image): 
        image = tf.convert_to_tensor(image)
    if threshold:
        image = image > threshold
        return image
    image = image == tf.reduce_max(image)
    return image

def get_dice_coefficient(confusion_matrix: np.ndarray):
    TP,FP,TN,FN = get_tuple(confusion_matrix)
    return 2*TP / (2*TP + FN + FP)

def get_jaccard_index(confusion_matrix: np.ndarray):
    TP,FP,TN,FN = get_tuple(confusion_matrix)
    return TP / (TP + FN + FP)

def get_sensitivity(confusion_matrix: np.ndarray):
    TP,FP,TN,FN = get_tuple(confusion_matrix)
    return TP / (TP + FN)

def get_specificity(confusion_matrix: np.ndarray):
    TP,FP,TN,FN = get_tuple(confusion_matrix)
    return TN / (TN + FP)

def get_accuracy(confusion_matrix: np.ndarray):
    TP,FP,TN,FN = get_tuple(confusion_matrix)
    return (TP + TN) / tf.reduce_sum(confusion_matrix)

def get_tuple(confusion_matrix):
    '''
        return: TP,FP,TN,FN 
    '''
    return confusion_matrix[1][1] ,confusion_matrix[0][1], confusion_matrix[0][0], confusion_matrix[1][0]