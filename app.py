# NOTE: run command line to get library needed before deploy
# pip3 freeze > requirements.txt

import os
from flask import Flask
import flask
from flask.helpers import send_from_directory

from source.Predictor import Predictor

app = Flask(__name__)


@app.route('/favicon.ico', methods=['POST'])
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')


@app.route("/")
def home():
    return 'Welcome to Interior Design Predictor API!!!'


@app.route("/predict")
def predict():
    return flask.request.args


if __name__ == '__main__':
    predictor = Predictor.getInstance()
    # run with environment production (deploy)
    app.run()
    # run with environment development (debug)
    # app.run(debug=True, port=5000)
