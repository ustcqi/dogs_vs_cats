# _*_ coding:utf-8 _*_
from sklearn.utils import shuffle
import pandas as pd
import h5py
import numpy as np

from keras.models import *
from keras.layers import *
from keras.preprocessing.image import *

np.random.seed(2017)


class DogCatClassifier:

    def __init__(self):
        self.batch_size = 128
        self.nb_epoch = 8
        self.validation_split = 0.2
        self.dropout = 0.5

        self.feature_h5_list = ["feature_ResNet50.h5",
                                "feature_Xception.h5",
                                "feature_InceptionV3.h5",
                                "feature_vgg16.h5",
                                "feature_vgg19.h5"]
        self.features_h5y_path = '../features_h5y/'

    def merge_features(self, feature_list=None, features_path=None):
        if features_path is None:
            features_path = self.features_h5y_path
        if feature_list is None:
            feature_list = self.feature_h5_list
        X_train = []
        X_test = []
        for filename in self.feature_list:
            fname = features_path + filename
            with h5py.File(fname, 'r') as h:
                X_train.append(np.array(h['train_h5y']))
                X_test.append(np.array(h['test_h5y']))
                y_train = np.array(h['label_h5y'])
        X_train = np.concatenate(X_train, axis=1)
        X_test = np.concatenate(X_test, axis=1)
        X_train, y_train = shuffle(X_train, y_train)
        return (X_train, X_test, y_train)

    def train_model(self, X_train, y_train):
        input_tensor = Input(X_train.shape[1:])
        x = input_tensor
        x = Dropout(0.5)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(input_tensor, x)
        model.compile(optimizer='adadelta',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train,
                  batch_size=self.batch_size,
                  nb_epoch=self.nb_epoch,
                  validation_split=self.validation_split)
        return model


    def predict(self, model, X_test):
        y_pred = model.predict(X_test, verbose=1)
        y_pred = y_pred.clip(min=0.005, max=0.995)
        return y_pred


def generate_submission(y_pred, fname):
    df = pd.read_csv("sample_submission.csv")

    image_size = (224, 224)
    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory("../data/test/",
                                             image_size,
                                             shuffle=False,
                                             batch_size=16,
                                             class_mode=None)
    for i, fname in enumerate(test_generator.filenames):
        index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
        df.set_value(index-1, 'label', y_pred[i])

    df.to_csv(fname, index=None)
    df.head(10)

if __name__ == "__main__":
    dog_cat_cls = DogCatClassifier()
    (X_train, X_test, y_train) = dog_cat_cls.merge_features()
    model = dog_cat_cls.train_model(X_train, y_train)
    y_pred = dog_cat_cls.predict(model, X_test)
    submit_fname = 'pred.csv'
    generate_submission(y_pred, submit_fname)

