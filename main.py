from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=.2,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(2,'data/salt/train','images','masks',data_gen_args,save_to_dir = None)

model = unet(pretrained_weights='unet_salt_new.hdf5')
model_checkpoint = ModelCheckpoint('unet_salt_new.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=5000,epochs=10,callbacks=[model_checkpoint])