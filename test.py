from model import *
from data import *

model = unet(pretrained_weights='unet_salt_new.hdf5')

num_images = 3920

train_dict = "data/salt/train/image2/"
mask_dict = "data/salt/train/mask2/"



testGene = testGenerator(train_dict, num_images)
results = model.predict_generator(testGene,num_images,verbose=1)


compare = []
for i in range(1, num_images+1):
    mask = io.imread(os.path.join(mask_dict+ str(i) + '.png'),as_gray = False)
    mask = mask / 65535
    compare.append(mask)


values = []
for i in range(len(compare)):
    realarea = np.count_nonzero(compare[i])
    predarea = np.count_nonzero(results[i])
    values.append(abs(realarea - predarea))

print(np.max(values))
print(np.min(values))
print(np.mean(values))


high = []
for i in range(len(compare)):
    if values[i] > 16000:
        high.append(i)

print(high)
print(len(high))
