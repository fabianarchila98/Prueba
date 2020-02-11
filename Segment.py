from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import AgglomerativeClustering as AC

def suma_xy(new,factor):

    for i in range(0,len(new)):
        a=[]
        for j in range(0,len(new[0])):
            if j==0:
                a=[*new[i][j],*[factor*i/len(new),factor*j/len(new[0])]]
            elif j==1:
                a=[a,[*new[i][j],*[factor*i/len(new),factor*j/len(new[0])]]]
            else:
                a=[*a,[*new[i][j],*[factor*i/len(new),factor*j/len(new[0])]]]
        if i==0:
            b=a
        elif i==1:
            b=[b,a]
        else:
            b=[*b,a]
    b=np.array(b)
    return b
def hierarchical(n,img):

    Z_2=img.reshape((-1,len(img[0][0])))
    # print(Z_2)
    # ac_model=AC(n_clusters=n,linkage='average',compute_full_tree='false',affinity='cosine')
    ac_model=AC(n_clusters=n)

    ac_labels=ac_model.fit_predict(Z_2)
    img_labels_3=ac_labels.reshape((img.shape[0],img.shape[1]))
    return img_labels_3

def k_means(k,img):
    Z = img.reshape((-1,len(img[0][0])))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    img_labels=label.reshape((img.shape[0],img.shape[1]))
    return img_labels

def gmm(n,img):
    Z=img.reshape((-1,len(img[0][0])))
    gmm_model=GMM(n_components=n,covariance_type='tied').fit(Z)
    gmm_labels=gmm_model.predict(Z)
    img_labels=gmm_labels.reshape((img.shape[0],img.shape[1]))
    return img_labels

def watershed(rgb):

    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((2,2),np.uint8)
    #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(closing,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)

    # Threshold
    ret, sure_fg = cv2.threshold(dist_transform,0.07*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(rgb,markers)
    rgb[markers == -1] = [255,0,0]
    return markers


def segmentByClustering(rgbImage, colorSpace, clusteringMethod, numberOfClusters):

    if clusteringMethod=='hierarchical':
        if colorSpace=='rgb':
            scale_percent = 40 # percent of original size
            width = int(rgbImage.shape[1] * scale_percent / 100)
            height = int(rgbImage.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(rgbImage, dim, interpolation = cv2.INTER_AREA)
            segmentation=hierarchical(numberOfClusters,resized)
        elif colorSpace=='lab':
            lab_image=color.rgb2lab(rgbImage)
            scale_percent = 40 # percent of original size
            width = int(lab_image.shape[1] * scale_percent / 100)
            height = int(lab_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(lab_image, dim, interpolation = cv2.INTER_AREA)
            segmentation=hierarchical(numberOfClusters,resized)
        elif colorSpace=='hsv':
            hsv_image=color.rgb2hsv(rgbImage)
            scale_percent = 40 # percent of original size
            width = int(hsv_image.shape[1] * scale_percent / 100)
            height = int(hsv_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(hsv_image, dim, interpolation = cv2.INTER_AREA)
            segmentation=hierarchical(numberOfClusters,resized)
        elif colorSpace=='rgb+xy':
            scale_percent = 40 # percent of original size
            width = int(rgbImage.shape[1] * scale_percent / 100)
            height = int(rgbImage.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(rgbImage, dim, interpolation = cv2.INTER_AREA)
            xy_sum=suma_xy(resized,50)
            segmentation=hierarchical(numberOfClusters,xy_sum)
        elif colorSpace=='lab+xy':
            lab_image=color.rgb2lab(rgbImage)
            scale_percent = 40 # percent of original size
            width = int(lab_image.shape[1] * scale_percent / 100)
            height = int(lab_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(lab_image, dim, interpolation = cv2.INTER_AREA)
            xy_sum=suma_xy(resized,10)
            segmentation=hierarchical(numberOfClusters,xy_sum)

        elif colorSpace=='hsv+xy':
            hsv_image=color.rgb2hsv(rgbImage)
            scale_percent = 40 # percent of original size
            width = int(hsv_image.shape[1] * scale_percent / 100)
            height = int(hsv_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(hsv_image, dim, interpolation = cv2.INTER_AREA)
            xy_sum=suma_xy(resized,0.2)
            segmentation=hierarchical(numberOfClusters,xy_sum)

    else:

        if colorSpace=='rgb':
            image=rgbImage
        elif colorSpace=='lab':
            image=color.rgb2lab(rgbImage)
        elif colorSpace=='hsv':
            image=color.rgb2hsv(rgbImage)
        elif colorSpace=='rgb+xy':
            image=suma_xy(rgbImage,50)
        elif colorSpace=='lab+xy':
            lab_image=color.rgb2lab(rgbImage)
            image=suma_xy(lab_image,10)
        elif colorSpace=='hsv+xy':
            hsv_image=color.rgb2hsv(rgbImage)
            image=suma_xy(hsv_image,0.2)

        if clusteringMethod=='kmeans':
            segmentation=k_means(numberOfClusters,image)
        elif clusteringMethod=='gmm':
            segmentation=gmm(numberOfClusters,image)
        elif clusteringMethod=='watershed':
            segmentation=watershed(image)



    return segmentation
