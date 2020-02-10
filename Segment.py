from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture as GMM

def suma_xy(new,factor):
    for i in range(0,len(new)):
        a=[]
        for j in range(0,len(new[0])):
            if j==0:
                a=[*new[i][j],*[fator*i/len(new),fator*j/len(new[0])]]
            elif j==1:
                a=[a,[*new[i][j],*[fator*i/len(new),fator*j/len(new[0])]]]
            else:
                a=[*a,[*new[i][j],*[fator*i/len(new),fator*j/len(new[0])]]]
        if i==0:
            b=a
        elif i==1:
            b=[b,a]
        else:
            b=[*b,a]
    b=np.array(b)
    return b

def k_means(k,img):
    Z = img.reshape((-1,len(img[0][0])))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 14
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    img_labels=label.reshape((img.shape[0],img.shape[1]))
    return img_labels

def gmm(n,img):
    Z=img.reshape((-1,len(img[0][0])))
    gmm_model=GMM(n_components=n,covariance_type='tied').fit(Z)
    gmm_labels=gmm_model.predict(Z)
    img_labels=gmm_labels.reshape((img.shape[0],img.shape[1]))
    return img_labels


def segmentByClustering(rgbImage, colorSpace, clusteringMethod, numberOfClusters):
    a=b
    return segmentation
