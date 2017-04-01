from keras.models import *
from keras.layers import *
#from keras.applications import  Xception, InceptionV3, VGG16, VGG19
from keras.preprocessing.image import *

import h5py

#https://github.com/abnera/image-classifier/blob/master/code/fine_tune.py

train_data_dir = '../data/train'
test_data_dir = '../data/test/'
transformation_ratio = .05
print(os.getcwd())

def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func is not None:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator() #rescale=1. / 255
    train_generator = gen.flow_from_directory(train_data_dir, image_size, shuffle=False, batch_size=16)
    test_generator = gen.flow_from_directory(test_data_dir, image_size, shuffle=False, batch_size=16, class_mode=None)
    print(train_generator.samples)
    print(test_generator.samples)
    train = model.predict_generator(train_generator, train_generator.samples)
    test = model.predict_generator(test_generator, test_generator.samples)
    with h5py.File("../data/features_h5y/gap_%s.h5" % MODEL.func_name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)

#import models.resnet50 as resnet50
#write_gap(resnet50.ResNet50, (224, 224))
import models.xception as xception
write_gap(xception.Xception, (299, 299), xception.preprocess_input)

''''
import models.inception_v3 as inception_v3
write_gap(inception_v3.InceptionV3, (299, 299), inception_v3.preprocess_input)

import models.vgg16 as vgg16
write_gap(vgg16.VGG16, (224, 224))

import models.vgg19 as vgg19
write_gap(vgg19.VGG19, (224, 224))
'''
