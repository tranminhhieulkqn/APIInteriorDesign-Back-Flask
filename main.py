# NOTE: run command line to get library needed before deploy
# pip3 freeze > requirements.txt

import os
from flask import Flask
from flask.helpers import send_from_directory

from source.Predictor import Predictor

app = Flask(__name__)

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')


@app.route("/")
def home():
    return 'Hello World!!!'

if __name__ == '__main__':
    ## run with environment production (deploy)
    # app.run()
    ## run with environment development (debug)
    predictor = Predictor.getInstance()
    app.run(debug=True, port=5000)