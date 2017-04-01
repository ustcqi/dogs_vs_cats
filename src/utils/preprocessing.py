import os
import shutil

train_filenames = os.listdir('../data/train')
train_cat = filter(lambda x:x[:3] == 'cat', train_filenames)
train_dog = filter(lambda x:x[:3] == 'dog', train_filenames)


def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)

rmrf_mkdir('../data/train2')
os.mkdir('../data/train2/cat')
os.mkdir('../data/train2/dog')

rmrf_mkdir('../data/test2')
os.symlink('../data/test/', '../data/test2/test')


for filename in train_cat:
    os.symlink('../data/train/'+filename, '../data/train2/cat/'+filename)

for filename in train_dog:
    os.symlink('../data/train/'+filename, '../data/train2/dog/'+filename)

