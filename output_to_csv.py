import numpy as np 
import pandas as pd 
import time
import pickle
from PIL import Image
import skimage.transform as trans
import skimage.io as io
import os
from model import *
from data import *


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

#config
blackbox = []
indexes = []
filenames = pickle.load(open('filenames.dat', 'rb'))
target_size = (101,101)
num_images = 18000
model = unet(pretrained_weights='unet_salt_new.hdf5')
testGene = testGenerator("data/salt/test/images/", num_images)


#tf
results = model.predict_generator(testGene,num_images,verbose=1)

print("results captured")

print('decoding...')
print(len(filenames))

for i in range(len(results) ):
    img = results[i]
    img = img[:,:,0]
    img = trans.resize(img,target_size)
    img[img > 0.5] = 1
    img[img <= 0.5] = 0
    rle = rle_encode(img)

    if i % 1000 == 0:
        print(i)
    blackbox.append(rle)
    indexes.append(filenames[i].split('.')[0])


col =['rle_mask']
df = pd.DataFrame(blackbox, columns=col, index=indexes)
df.index.name = 'id'

df.to_csv("Sub_new.csv")
