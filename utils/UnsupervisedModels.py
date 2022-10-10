import numpy as np
import cv2
from sklearn.cluster import DBSCAN, Birch, AffinityPropagation, MeanShift, OPTICS, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

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

def dbscan(image, out_shape,min_samples=9,eps=0.25):
    # define the model
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(image)
    labels = reform(np.array(labels),out_shape)
    return labels

def gaussian_mixture(image, out_shape=None, n_components=2):
    labels = GaussianMixture(n_components=n_components).fit_predict(image)
    labels = reform(np.array(labels),out_shape)
    return labels

def birch(image, out_shape=None,n_clusters=2):
    labels = Birch(n_clusters=n_clusters).fit_predict(image)
    labels = reform(np.array(labels),out_shape)
    return labels

def affinity_propogation(image, out_shape=None,random_state=2):
    labels = AffinityPropagation(random_state=random_state).fit_predict(image)
    labels = reform(np.array(labels),out_shape)
    return labels

def mean_shift(image, out_shape=None,bandwidth=2):
    labels = MeanShift(bandwidth=bandwidth).fit_predict(image)
    labels = reform(np.array(labels),out_shape)
    return labels

def optics(image, out_shape=None,min_samples=2,max_eps=0.25):
    labels = OPTICS(min_samples=min_samples,max_eps=max_eps).fit_predict(image)
    labels = reform(np.array(labels),out_shape)
    return labels

def agglomerative(image, out_shape=None,n_clusters=2):
    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(image)
    labels = reform(np.array(labels),out_shape)
    return labels

def reform(labels, out_shape= None):
    if out_shape:
        labels = labels.reshape(out_shape)
    labels = (labels - labels.min()) / (labels.max() - labels.min())
    labels = labels * 255
    return labels

