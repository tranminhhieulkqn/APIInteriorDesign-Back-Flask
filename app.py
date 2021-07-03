# NOTE: run command line to get library needed before deploy
# pip3 freeze > requirements.txt

import os
import tensorflow as tf
from flask import Flask
from flask.helpers import send_from_directory

print("Tensorflow version : " + tf.__version__)

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

app = Flask(__name__)

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')


@app.route("/")
def home():
    return "Hello, Flask!"

if __name__ == '__main__':
    ## run with 
    # app.run()
    app.run(debug=True, port=5000)