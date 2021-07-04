import os
import numpy as np
import tensorflow as tf

from skimage import io
from PIL import Image, ImageOps

print(tf.__version__)


class Predictor:

    __instance = None
    __models = dict({})
    __models_dir = ''
    __target_size = 224
    __shift_pixel = 50

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
    def __load_models(self):
        list_model = [model for model in os.listdir(
            self.__models_dir) if ('.tflite' in model)]
        for model_name in list_model:
            # get model path
            model_path = os.path.join(self.__models_dir, model_name)
            # loading and active model
            model = tf.lite.Interpreter(model_path=model_path)
            model.allocate_tensors()
            # get name model
            name = str(model_name.split('.tflite')[0])
            # append to array models
            self.__models[name] = model
        print('Load model successfully!')

    @classmethod
    def customize_size(self, original_size, target_size):
        # default ratio = 1
        ratio = 1
        # get size width, height of image
        width, height = original_size
        # get ratio from size of image
        ratio = (width / target_size) if (width <
                                          height) else (height / target_size)
        # return new width, height size
        return int(width / ratio), int(height / ratio)

    @classmethod
    def soft_voting(self, results):
        sum_prob = np.zeros(5)
        for result in results:
            sum_prob = sum_prob + result
        return (sum_prob)/len(results)

    @classmethod
    # jumps is the number of pixels
    def demo_crop(self, image_path, target_size=224, shift_pixel=50):
        image = io.imread(image_path)
        img = Image.fromarray(image.astype('uint8'), 'RGB')
        img = img.convert('RGB')
        img = img.resize(self.customize_size(img.size, target_size))
        x_max, y_max = np.array(img.size) - target_size
        images = []
        for random_x in range(0, x_max + 1, shift_pixel):
            for random_y in range(0, y_max + 1, shift_pixel):
                area = (random_x, random_y, random_x +
                        target_size, random_y + target_size)
                c_img = img.crop(area)
                fit_img_h = ImageOps.fit(
                    c_img, (target_size, target_size), Image.ANTIALIAS)
                image = np.array(fit_img_h) / 255
                image = image.reshape(
                    1, target_size, target_size, 3).astype(np.float32)
                images.append(image)
        return images

    @classmethod
    def predict(self, model, image):
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], image)
        model.invoke()
        output = model.get_tensor(output_details[0]['index'])
        # classes = np.argmax(output, axis = 1)
        return output[0]

    @classmethod
    def predict_with_image_croped(self, model, list_image):
        results = []
        for image in list_image:
            results.append(self.predict(model, image))
        return self.soft_voting(results)

    @staticmethod
    def predict_with_all_model(image_path):
        results = []
        for model in Predictor.__models.values():
            results.append(Predictor.predict_with_image_croped(
                model, Predictor.demo_crop(image_path=image_path)))
        return Predictor.soft_voting(results)
