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
# plt.imshow(segm, cmap=plt.get_cmap('hot'))
# plt.colorbar()
# plt.show()

def unique(list1):

    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

ll=segm.reshape(-1,1)

aa=unique(ll)
print(aa[1])


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i

    return num ,counter

List = [2, 1, 2, 2, 1, 3]
print(most_frequent(List))

# #Boundaries from third human
# segm=gt['groundTruth'][0,2][0][0]['Boundaries']
# plt.imshow(segm)
# plt.colorbar()
# plt.show()
