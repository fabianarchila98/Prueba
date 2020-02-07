from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
img = 'BSDS_small/train/22090.jpg'
rgb = io.imread(img)
lab = color.rgb2xyz(rgb)
print(lab)
print(rgb)
plt.imshow(rgb) # ???
plt.show()
#try to visualize lab image
plt.imshow(lab) # ???
plt.show()

# def debugImg(rawData):
#   import cv2
#   toShow = np.zeros((rawData.shape), dtype=np.uint8)
#   cv2.normalize(rawData, toShow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#   # cv2.imwrite('name', toShow)
#   plt.imshow(toShow)
#   plt.show()
#
# debugImg(lab) # Different enough?
