import os
import pickle
test_path = 'data/salt/train/images'
test_path2 = 'data/salt/train/masks'
filelinks = [i for i in os.listdir(test_path)]

#pickle.dump(filelinks, open('filenames.dat', 'wb'), -1)
c = 1
for i in range(len(filelinks)):
    os.rename(test_path + '/' + filelinks[i], test_path + '/' + str(c) + '.png')
    os.rename(test_path2 + '/' + filelinks[i], test_path2 + '/' + str(c) + '.png')
    c += 1
