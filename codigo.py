import scipy.io as sio
import matplotlib.pyplot as plt
import imageio
import Segment
from skimage import io, color
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import AgglomerativeClustering as AC
import numpy as np
from sklearn import metrics

img = 'BSDS_small/train/22090.jpg'
rgb = io.imread(img)
# plt.imshow(imageio.imread(img))
# plt.show()

# Load .mat
gt=sio.loadmat(img.replace('jpg', 'mat'))

#Load segmentation from sixth human
segm=gt['groundTruth'][0,5][0][0]['Segmentation']
print(segm.shape)
# plt.imshow(segm, cmap=plt.get_cmap('hot'))
# plt.colorbar()
# plt.show()

labels=Segment.k_means(2,rgb)
labels_r=labels.reshape(-1,1)
segm_r=segm.reshape(-1,1)
labels_u=np.unique(labels_r)
segm_u=np.unique(segm_r)


# for i in segm_u:
#     valor=0
#     count=np.zeros(len(labels_u))
#     count_i=0
#     for j in range(0,len(segm_r)):
#         if segm_r[j]==i:
#             count[labels_r[j]]+=1
#     print(count)
#     a=np.amax(count)/np.sum(count)
#     print(a)

segm_tr=segm_r=segm.reshape(1,-1)
labels_tr=labels.reshape(1,-1)
# print(metrics.adjusted_rand_score(segm_tr[0], labels_tr[0]))
print(metrics.homogeneity_score(segm_tr[0], labels_tr[0]))

print(len(segm_r))
print(len(labels_r))
# print(np.zeros(len(labels_u))[0])

# #Boundaries from third human
# segm=gt['groundTruth'][0,2][0][0]['Boundaries']
# plt.imshow(segm)
# plt.colorbar()
# plt.show()
