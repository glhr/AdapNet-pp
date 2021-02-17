import numpy as np
import cv2
import matplotlib.pyplot as plt

adaptnet_labels = {
0: (  0,  0,  0),
1: ( 70,130,180),
2: ( 70, 70, 70),
3: (128, 64,128),
4: (244, 35,232),
5: (190,153,153),
6: (107,142, 35),
7: (153,153,153),
8: (  0,  0,142),
9: (220,220,  0),
10: (220, 20, 60),
11: (  0,  0,230)
}

def label2rgb(labels, n_classes=12):
    """
    Convert a labels image to an rgb image using a matplotlib colormap
    """
    blank = np.zeros_like(labels)
    blank = np.stack((blank,)*3, axis=-1)
    for cls in range(n_classes):
        blank[labels==cls] = adaptnet_labels[cls]
    return cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)

def combine_result(img,rgb):
    return np.concatenate((img,rgb), axis=1)

if __name__ == '__main__':
  labels = np.arange(256).astype(np.uint8)[np.newaxis, :]
  lut = gen_lut()
  rgb = labels2rgb(labels, lut)
