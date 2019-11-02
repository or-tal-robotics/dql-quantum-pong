import tensorflow as tf
import cv2   
    

def transform(state, size):
    output = tf.image.resize(test_img, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
    return output.numpy()

if __name__ == "__main__":
    test_img = cv2.imread("test_transformer.jpg")
    output = transform(test_img, size= [500,500])
    cv2.imshow("img_test", output)