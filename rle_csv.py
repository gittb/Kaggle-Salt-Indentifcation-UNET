# Run-Length Encode and Decode

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import pickle
from PIL import Image
import skimage.transform as trans
import skimage.io as io
import os


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    index_offset = 1
    mask = img.T
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated.
    If index_offset = 1 we assume arrays start at 1.
    '''

    inds = np.append(np.insert(mask.flatten(), 0, 0), 0) 
    runs = np.where(inds[1:] != inds[:-1])[0]
    runs[1::2] = runs[1::2] - runs[:-1:2] 

    if index_offset > 0:
        runs[0::2] += index_offset

    rle = ' '.join([str(r) for r in runs])

    return rle

 
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


base = 'data/salt/test/masks'
blackbox = []
indexes = []
filenames = pickle.load(open('filenames.dat', 'rb'))
print(len(filenames))

for i in range(len(filenames) ):
#for i in range(0, 15 ):
    flink = base + '/' + str(i + 1) + '.png'

    img = io.imread(os.path.join(flink),as_gray = True)
    #print(np.max(img))
    img = img / 65535
    #time.sleep(10)
    #print(np.max(img))
    rle = rle_encode(img)
    #print(len(rle))
    """
    if len(rle) > 0:
        rle_real = str(rle[0])
        for x in range(0, len(rle)):
            rle_real += ' '
            rle_real += str(rle[x])
    else:
        rle_real = ''
    """
    if i % 1000 == 0:
        print(i)
    blackbox.append(rle)
    indexes.append(filenames[i].split('.')[0])


col =['rle_mask']
df = pd.DataFrame(blackbox, columns=col, index=indexes)
df.index.name = 'id'

df.to_csv("Sub_new.csv")
