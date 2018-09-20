import numpy as np
import pandas as pd
import time
import pickle
from PIL import Image
import skimage.transform as trans
import skimage.io as io
import os
import matplotlib.pyplot as plt

class Data():
    def __init__(self, name, img, mask, depth, mask_area, mask_clusters, mask_avgclust, img_avg, img_max, img_min):
        self.name = name
        self.img = img
        self.mask = mask
        self.depth = depth
        self.mask_area = mask_area
        self.mask_num_clust = mask_clusters
        self.mask_avg_clust = mask_avgclust
        self.img_avg = img_avg
        self.img_max = img_max
        self.img_min = img_min

data = pickle.load(open('prepdata.dat', 'rb'))

var1 = []
var2 = []

for dat in data:
    var1.append(dat.depth)
    var2.append(dat.img_avg)

plt.plot(var1, var2, 'ro', markersize=2)
plt.xlabel('Depth')
plt.ylabel('img')
plt.show()