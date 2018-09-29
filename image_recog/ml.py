import keras as ks
import numpy as np
import os
import re
import scipy.misc

class ImageScorer(object):

    def __init__(self, tag_path="./data/tags.txt", logger=None,
        init_models = ['vgg', 'mobile'], data_dir = "./data/images"):

        # load labels for image categories
        tags = open(tag_path).readlines()
        tags = [tag.strip('\n') for tag in tags]
        self._tags = [re.sub("^n[0-9]+ ", "", tag) for tag in tags]

        # logger
        self._logger = logger

        if 'mobile' in init_models:
            self._init_mobile()
        else:
            self._mobile = None

        if 'vgg' in init_models:
            self._init_vgg()
        else:
            self._vgg = None

    def _init_vgg(self, data_dir = None):
        if self._logger is not None:
            self._logger.info("Initializing VGG")

        self._vgg = ks.applications.vgg16.VGG16()

        if data_dir is not None:
            cat_path = os.path.join(data_dir, 'cat.jpeg')
            try:
                scores = self.score(cat_path, size = (224, 224), model = "vgg")
            except:
                if self._logger is not None:
                    self._logger.error("Failed to initialize VGG")
                pass

    def _init_mobile(self, data_dir = None):
        if self._logger is not None:
            self._logger.info("Initializing MobileNet")

        self._mobile = ks.applications.mobilenet.MobileNet(alpha=0.25)

        if data_dir is not None:
            cat_path = os.path.join(data_dir, 'cat.jpeg')
            try:
                scores = self.score(cat_path, size = (224, 224), model = "mobile")
            except:
                if self._logger is not None:
                    self._logger.error("Failed to initialize MobileNet")


    def _normalize(self, f, size, model):

        if self._logger is not None:
            self._logger.info("Normalizing image file")

        img = scipy.misc.imread(f)
        img = img[:, :, 0:3]
        img = scipy.misc.imresize(img, size)
        img = img.astype(np.float32, copy=False)

        if model == "mobile":
            img = 2*(img / 255) - 1 # normalizing for mobilenet

        arr = np.expand_dims(img, axis=0)

        if self._logger is not None:
            self._logger.info("Normalizing created: %s", arr.__str__())

        return arr


    def _predict(self, arr, model):
        if self._logger is not None:
            self._logger.info("Predicting with input: %s", arr.__str__())

        if model == "mobile":
            if self._mobile is None:
                self._init_mobile()

            p = self._mobile.predict(arr)
        else:
            if self._vgg is None:
                self._init_vgg()

            p = self._vgg.predict(arr)

        if self._logger is not None:
            self._logger.info("Created predictions: ", p.__str__())

        return p

    def score(self, f, size = (224, 224), model = "mobile"):

        # normalizing input
        try:
            arr = self._normalize(f, size, model)
        except: # this shouldn't throw an error
            raise

        # scoring model
        try:
            p = self._predict(arr, model)
            error = False
            scores = []
            for i, score in enumerate(p[0, :]):
                scores.append({'tag': self._tags[i], 'score': float(score)})

        except ValueError as err:
            if self._logger is not None:
                self._logger.error("Fail to predict: ", err.__str__())
            scores = []
            error = True

        return {'scores': scores, 'error': error}
