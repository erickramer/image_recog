from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
import os
import scipy.misc

def VGG_expanded(weights_path="./data/vgg_expanded.h5"):

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,4*224,4*224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, 7, 7))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, 1, 1))
    model.add(Dropout(0.5))
    model.add(Convolution2D(1000, 1, 1, activation='linear'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def VGG_16(weights_path="./data/vgg16_weights.h5"):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def normalize_image(f, size=(224, 224)):

    mean_pixel = [103.939, 116.779, 123.68]

    img = scipy.misc.imread(f)

    img = img[:, :, 0:3]
    img = scipy.misc.imresize(img, size)
    img = img.astype(np.float32, copy=False)

    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]

    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0)

    return img

class ImageScorer(object):

    def __init__(self, tag_path="./data/tags.txt"):
        self._vgg = VGG_16()
        #self._vgg_expanded = VGG_expanded()

        tags = open(tag_path).readlines()
        self._tags = [tag.strip('\n') for tag in tags]

    def __call__(self, f):
        return self.score(f)

    def _score_vgg(self, f):
        img = normalize_image(f, size=(224,224))

        p = self._vgg.predict(img)
        res = []
        for i, score in enumerate(p[0, :]):
            res.append({'tag': self._tags[i], 'score': float(score)})

        return res

    def _score_vgg_expanded(self, f):
        img = normalize_image(f, size=(4*224, 4*224))

        p = self._vgg_expanded.predict(img)[0, :, :, :]
        res = []

        for i in range(p.shape[0]):
            tag = self._tags[i]
            scores = []
            for j in range(p.shape[1]):
                for k in range(p.shape[2]):
                    score = {'x': i, 'y': j, 'score': float(p[i, j, k])}
                    scores.append(score)
            res.append({'tag': tag, 'scores': scores})

        return res

    def score(self, f):
        vgg = self._score_vgg(f)
        #vgg_expanded = self._score_vgg_expanded(f)
        vgg_expanded = None

        return {'vgg': vgg, 'vgg_expanded': vgg_expanded}
