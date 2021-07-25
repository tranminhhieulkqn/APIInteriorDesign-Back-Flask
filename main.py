# NOTE: run command line to get library needed before deploy
# pip3 freeze > requirements.txt

import os
import gc
import time
import numpy as np

from flask import Flask, request, jsonify
from flask.helpers import send_from_directory

from source.Predictor_TFLite import Predictor_TFLite
from source.Predictor_Keras import Predictor_Keras

app = Flask(__name__)

predictor = Predictor_Keras.getInstance()
# labels = ['Art Decor', 'Hi-Tech', 'IndoChinese', 'Industrial', 'Scandinavian']
labels = ['ArtDecor', 'HiTech', 'Indochina', 'Industrial', 'Scandinavian']

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')


@app.route("/", methods=['GET'])
def home():
    return 'Welcome to Interior Design Predictor API!!!'


@app.route("/predict", methods=['POST'])
def predict():
    url = request.json['url']
    if url is not None:
        try:
            start = time.time()
            # image = io.imread(url)
            output = predictor.ensemble_predict(image_url=url)
            end = time.time() - start
            print('time: ', end)
        except():
            return jsonify({
                "success": False,
                "message": "File not exist!"
            }), 404

        return jsonify({
            "success": True,
            "message": "Predicted Results",
            "result": (output*100).tolist(),
            "label": labels[int(np.argmax(output))],
            "score": np.max(output*100).tolist()
        }), 200
    else:
        return jsonify({
            "success": False,
            "message": "File not exist!"
        }), 404


if __name__ == '__main__':
    # run with environment production (deploy)
    app.run()
    # run with environment development (debug)
    # app.run(debug=True, port=5000)
