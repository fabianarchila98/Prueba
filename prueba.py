from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import cv2
from Segment import k_means ,gmm
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import AgglomerativeClustering as AC
img = 'BSDS_small/train/22090.jpg'
rgb = io.imread(img)
new = color.rgb2hsv(rgb)

# res=np.zeros([len(new),len(new[0])])
fator=0.05
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
# print(b[1])
# img = cv2.imread('home.jpg')
Z = b.reshape((-1,len(b[0][0])))
# print(Z)

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 14
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((b.shape))
img_labels=label.reshape((b.shape[0],b.shape[1]))
# print()
# print(label)
# plt.imshow(img_labels, cmap=plt.get_cmap('hot'))
# plt.colorbar()
# plt.show()
# plt.imshow(k_means(10,new), cmap=plt.get_cmap('hot'))
# plt.colorbar()
# plt.show()
# Z_2=b.reshape((-1,len(b[0][0])))
# print(Z_2)
# gmm_model=GMM(n_components=4,covariance_type='tied').fit(Z_2)
# gmm_labels=gmm_model.predict(Z_2)
# img_labels_2=label.reshape((b.shape[0],b.shape[1]))
scale_percent = 40 # percent of original size
width = int(rgb.shape[1] * scale_percent / 100)
height = int(rgb.shape[0] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(rgb, dim, interpolation = cv2.INTER_AREA)
Z_2=resized.reshape((-1,len(resized[0][0])))
print(Z_2)
ac_model=AC(n_clusters=14,linkage='average',compute_full_tree='false',affinity='cosine')
ac_labels=ac_model.fit_predict(Z_2)
img_labels_3=ac_labels.reshape((resized.shape[0],resized.shape[1]))
#
plt.imshow(img_labels_3, cmap=plt.get_cmap('hot'))
plt.colorbar()
plt.show()

# cv2.imshow('res2',res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
