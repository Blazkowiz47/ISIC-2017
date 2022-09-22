import numpy as np
import cv2

def kMeans(image,out_shape=None, K=2):
    # convert to np.float32
    image = np.float32(image)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2

    _,label,center = cv2.kmeans(image,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    result = center[label.flatten()]
    if out_shape:
        result = result.reshape(out_shape)
    return result

