import csv
import logging
import os
import sys

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DATA_DIR = '/app/data/'

@app.route("/prediction/<stock_code>")
def prediction(stock_code):

    dates = []
    real_high = []
    real_low = []
    pred_high = []
    pred_low = []

    with open(os.path.join(DATA_DIR, '{0}.csv'.format(stock_code)), newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            dates.append(row['date'])
            real_high.append(row['real_high'])
            real_low.append(row['real_low'])
            pred_high.append("%.2f" % float(row['pred_high']))
            pred_low.append("%.2f" % float(row['pred_low']))

    return {
        "date": dates,
        "real_high": real_high,
        "real_low": real_low,
        "pred_high": pred_high,
        "pred_low": pred_low,
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=80)