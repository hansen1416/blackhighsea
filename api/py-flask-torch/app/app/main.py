import csv
import logging
import os
import sys

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

TMP_DIR = '/app/tmp/'

@app.route("/", method=["POST"])
def cartoon_image():

    return {1:1}

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=os.environ['HOME']=='development', port=80)