import tensorflow as tf
import cv2   
    

def transform(state, size = (84, 84)):
    output = tf.image.resize(state, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
    output = tf.squeeze(output)
    return output.numpy()