import os
import tensorflow as tf


class Predictor:

    __instance = None
    __models = dict({})
    __models_dir = ''

    def __init__(self, models_dir='models/'):
        Predictor.__models_dir = models_dir
        self.__load_models()
        """ Virtually private constructor. """
        if Predictor.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Predictor.__instance = self

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Predictor.__instance == None:
            Predictor()
        return Predictor.__instance

    @classmethod
    def __load_models(cls):
        list_model = [model for model in os.listdir(cls.__models_dir) if ('.tflite' in model)]
        for model_name in list_model:
            # get model path
            model_path = os.path.join(cls.__models_dir, model_name)
            # loading and active model
            model = tf.lite.Interpreter(model_path=model_path)
            model.allocate_tensors()
            # get name model
            name = str(model_name.split('.tflite')[0])
            # append to array models
            cls.__models[name] = model
        print('Load model successfully!')

    @classmethod
    def customize_size(original_size, target_size):
        # default ratio = 1
        ratio = 1
        # get size width, height of image
        width, height = original_size
        # get ratio from size of image
        ratio = (width / target_size) if (width < height) else (height / target_size)
        # return new width, height size
        return int(width / ratio), int(height / ratio)
