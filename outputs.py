from model import *
from data import *

model = unet(pretrained_weights='unet_salt_new.hdf5')

num_images = 18000

testGene = testGenerator("data/salt/test/images/", num_images)
results = model.predict_generator(testGene,num_images,verbose=1)
saveResult("data/salt/test/masks/", results, (101,101))