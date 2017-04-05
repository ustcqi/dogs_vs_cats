# _*_ coding:utf-8 _*_
from keras.models import *
from keras.layers import *
#from keras.applications import  Xception, InceptionV3, VGG16, VGG19
from keras.preprocessing.image import *

import h5py

class FeatureExtractor:
    def __init__(self):
        self.train_data_dir = '../data/train2'
        self.test_data_dir = '../data/test2'
        self.features_h5y_root = '../data/h5y/'
        self.trasformation_ratio = 0.05
        self.image_size = (299, 299)
        self.channels = 3
        self.batch_size = 20

    def extract_feature(self, MODEL, image_size=None, func=None):
        if image_size == None:
            image_size = self.image_size
        width = image_size[0]
        height = image_size[1]
        input_tensor = Input((height, width, self.channels))
        x = input_tensor
        print(np.shape(input_tensor))
        if func is not None:
            x = Lambda(func)(x)
        print('base model input tensor shape: ', np.shape(x))
        base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
        print('base model name: ', base_model.name)
        print('base model output tensor shape ', np.shape(base_model.output))

        model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
        print('model name: ', model.name)
        print('model outout tensor shape ', np.shape(model.output))

        gen = ImageDataGenerator()
        train_generator = gen.flow_from_directory(self.train_data_dir,
                                                  image_size,
                                                  shuffle=False,
                                                  batch_size=self.batch_size)
        test_generator = gen.flow_from_directory(self.test_data_dir,
                                                 image_size,
                                                 shuffle=False,
                                                 batch_size=self.batch_size,
                                                 class_mode=None)

        print(train_generator.samples,  train_generator.classes)
        print(test_generator.samples, test_generator.classes)
        train = model.predict_generator(train_generator, train_generator.samples/self.batch_size, workers=4)
        test = model.predict_generator(test_generator, test_generator.samples/self.batch_size, workers=4)
        print(type(train), np.shape(train))
        print(type(test), np.shape(test))
        dumped_fname = self.features_h5y_root + 'feature_%s.h5' % MODEL.func_name
        with h5py.File(dumped_fname) as h:
            h.create_dataset("train", data=train)
            h.create_dataset("test", data=test)
            h.create_dataset("label", data=train_generator.classes)


if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    import models.resnet50 as resnet50
    feature_extractor.extract_feature(resnet50.ResNet50, image_size=(229, 229))

    import models.vgg16 as vgg16
    feature_extractor.extract_feature(vgg16.VGG16, image_size=(224, 224))

    import models.xception as xception
    feature_extractor.extract_feature(xception.Xception,
                                      image_size=(299, 299),
                                      func=xception.preprocess_input)
    '''
    import models.vgg19 as vgg19
    feature_extractor.extract_feature(vgg19.VGG19, image_size=(224, 224))
    import models.inception_v3 as inception_v3
    feature_extractor.extract_feature(inception_v3.InceptionV3,
                                      image_size=(229, 229),
                                      func=inception_v3.preprocess_input)
    '''
