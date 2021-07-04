# NOTE: run command line to get library needed before deploy
# pip3 freeze > requirements.txt

import os
import time
import numpy as np
from skimage import io
from flask import Flask, request, jsonify
from flask.helpers import send_from_directory

from source.Predictor import Predictor

app = Flask(__name__)

predictor = Predictor.getInstance()
labels = ['Art Decor', 'Hi-Tech', 'IndoChinese', 'Industrial', 'Scandinavian']


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
            # start = time.time()
            # urls = demo_crop(url)
            # end = time.time()- start
            # print(end)
            
            start = time.time()
            # image = io.imread(url)
            output = predictor.predict_with_all_model(image_path=url)
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
            # "result": (output*100).tolist(),
            # "label": labels[int(np.argmax(output))],
            # "score": np.max(output*100).tolist()
        }), 200
    else:
        return jsonify({
            "success": False,
            "message": "File not exist!"
        }), 404
    return jsonify({
        "success": "True",
    }), 200


if __name__ == '__main__':
    # run with environment production (deploy)
    # app.run()
    # run with environment development (debug)

    app.run(debug=True, port=5000)
