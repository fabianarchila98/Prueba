from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import cv2
import Segment
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import AgglomerativeClustering as AC

img = 'BSDS_small/train/22090.jpg'
rgb = io.imread(img)
new = color.rgb2lab(rgb)

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
# print(new[0])
# a=Segment.segmentByClustering(rgb,'hsv+xy','gmm',2)
#
# plt.imshow(a, cmap=plt.get_cmap('hot'))
# plt.colorbar()
# plt.show()
# b,g,r = cv2.split(rgb)
# b = cv2.merge([r,g,b])

l=Segment.watershed(rgb)


plt.imshow(l, cmap=plt.get_cmap('cool'))
plt.colorbar()
plt.show()
