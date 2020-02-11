
def imshow(img, seg, title='Image'):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.imshow(seg, cmap=plt.get_cmap('rainbow'), alpha=0.5)
    cb = plt.colorbar()
    cb.set_ticks(range(seg.max()+1))
    plt.title(title)
    plt.axis('off')
    plt.show()

def groundtruth(img_file):
    import scipy.io as sio
    img = imageio.imread(img_file)
    gt=sio.loadmat(img_file.replace('jpg', 'mat'))
    segm=gt['groundTruth'][0,5][0][0]['Segmentation']
    imshow(img, segm, title='Groundtruth')
    return segm

def check_dataset(folder):
    import os
    if not os.path.isdir(folder):
        # Download it.
        # Put your code here. Then remove the 'pass' command.
        pass

if __name__ == '__main__':
    import argparse
    import imageio
    from Segment import segmentByClustering # Change this line if your function has a different name
    parser = argparse.ArgumentParser()

    parser.add_argument('--color', type=str, default='rgb', choices=['rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy', 'hsv+xy']) # If you use more please add them to this list.
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--method', type=str, default='watershed', choices=['kmeans', 'gmm', 'hierarchical', 'watershed'])
    parser.add_argument('--img_file', type=str, required=True)

    opts = parser.parse_args()

    check_dataset(opts.img_file.split('/')[0])

    img = imageio.imread(opts.img_file)
    labels = segmentByClustering(rgbImage=img, colorSpace=opts.color, clusteringMethod=opts.method, numberOfClusters=opts.k)
    imshow(img, labels, title='Prediction')
    segm=groundtruth(opts.img_file)
    import numpy as np
    if labels.shape!=segm.shape:
        aa=segm.shape
        cc=np.copy(segm)
        for i in range(0,int(aa[0]/2)+1):
            cc=np.delete(cc,i,0)

        for i in range(0,int(aa[1]/2)+1):
            cc=np.delete(cc,i,1)
        segm=cc


    from sklearn import metrics
    labels_r=labels.reshape(-1,1)
    segm_r=segm.reshape(-1,1)
    labels_u=np.unique(labels_r)
    segm_u=np.unique(segm_r)
    segm_tr=segm_r=segm.reshape(1,-1)
    labels_tr=labels.reshape(1,-1)
    print('Clustering metric:')
    print(metrics.homogeneity_score(segm_tr[0], labels_tr[0]))
