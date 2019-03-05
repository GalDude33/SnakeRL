import tensorflow as tf


def huber_loss(y_true, y_pred):
    """huber loss - supposed to improve stability"""
    return tf.losses.huber_loss(y_true, y_pred)
