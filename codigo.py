import scipy.io as sio
import matplotlib.pyplot as plt
import imageio

img = 'BSDS_small/train/22090.jpg'
# plt.imshow(imageio.imread(img))
# plt.show()

# Load .mat
gt=sio.loadmat(img.replace('jpg', 'mat'))

#Load segmentation from sixth human
segm=gt['groundTruth'][0,5][0][0]['Segmentation']
print(segm.shape)
plt.imshow(segm, cmap=plt.get_cmap('hot'))
plt.colorbar()
plt.show()
print(segm.reshape(-1,1))
import collections
print([item for item, count in collections.Counter(segm.reshape((-1,1))).items() if count > 1])


# #Boundaries from third human
# segm=gt['groundTruth'][0,2][0][0]['Boundaries']
# plt.imshow(segm)
# plt.colorbar()
# plt.show()
