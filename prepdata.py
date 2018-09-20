
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import pickle
from PIL import Image
import skimage.transform as trans
import skimage.io as io
import os
from pylab import *
from scipy.ndimage import measurements

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


masks = 'data/salt/pro/masks'
images = 'data/salt/pro/images'
df = pd.read_csv('depths.csv')
avoid = ['05b69f83bf.png', '0d8ed16206.png', '10833853b3.png', '135ae076e9.png', '1a7f8bd454.png', '1b0d74b359.png', '1c0b2ceb2f.png', '1efe1909ed.png', '1f0b16aa13.png', '1f73caa937.png', '20ed65cbf8.png', '287b0f197f.png', '2fb6791298.png', '37df75f3a2.png', '3ee4de57f8.png', '3ff3881428.png', '40ccdfe09d.png', '423ae1a09c.png', '4f30a97219.png', '51870e8500.png', '573f9f58c4.png', '58789490d6.png', '590f7ae6e7.png', '5aa0015d15.png', '5edb37f5a8.png', '5ff89814f5.png', '6b95bc6c5f.png', '6f79e6d54b.png', '755c1e849f.png', '762f01c185.png', '7769e240f0.png', '808cbefd71.png', '8c1d0929a2.png', '8ee20f502e.png', '9260b4f758.png', '96049af037.png', '96d1d6138a.png', '97515a958d.png', '99909324ed.png', '9aa65d393a.png', 'a2b7af2907.png', 'a31e485287.png', 'a3e0a0c779.png', 'a48b9989ac.png', 'a536f382ec.png', 'a56e87840f.png', 'a8be31a3c1.png', 'a9e940dccd.png', 'a9fd8e2a06.png', 'aa97ecda8e.png', 'acb95dd7c9.png', 'b11110b854.png', 'b552fb0d9d.png', 'b637a7621a.png', 'b8c3ca0fab.png', 'b9bf0422a6.png', 'bedb558d15.png', 'c1c6a1ebad.png', 'c20069b110.png', 'c3589905df.png', 'c8404c2d4f.png', 'cc15d94784.png', 'd0244d6c38.png', 'd0e720b57b.png', 'd1665744c3.png', 'd2e14828d5.png', 'd6437d0c25.png', 'd8bed49320.png', 'd93d713c55.png', 'dcca025cc6.png', 'e0da89ce88.png', 'e51599adb5.png', 'e7da2d7800.png', 'e82421363e.png', 'ec542d0719.png', 'f0190fc4b4.png', 'f26e6cffd6.png', 'f2c869e655.png', 'f9fc7746fb.png']

array = []

filenames = [i for i in os.listdir(images)]
for i in filenames:
    if i != 'Thumbs.db' and i not in avoid:
        #links
        mask = masks + '/' + i
        image = images + '/' + i

        #basic
        img = io.imread(os.path.join(image),as_gray = False)
        mask = io.imread(os.path.join(mask),as_gray = True)
        depth = df[df['id'] == i.split('.')[0]]['z'].tolist()[0]

        #mask ops
        mask = mask / 65535
        mask_area = np.count_nonzero(mask)
        lw, num = measurements.label(mask)
        clusters = measurements.sum(mask, lw, index=arange(lw.max() + 1))
        mask_clusters = len(clusters)
        mask_avgclust = np.mean(clusters)

        #img ops
        img_avg = np.average(img)
        img_max = np.max(img)
        img_min = np.min(img)
        
        array.append(Data(i, img, mask, depth, mask_area, mask_clusters, mask_avgclust, img_avg, img_max, img_min))
"""
for i in array:
    print(i.mask_avg_clust, i.mask_num_clust, i.img_avg)
"""

pickle.dump(array, open('prepdata.dat', 'wb'), -1)