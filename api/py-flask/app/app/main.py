# import csv
import logging
import os
import sys
import socket
import shutil

from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# log_format = "%(levelname)s %(asctime)s - %(message)s"

# logging.basicConfig(stream = sys.stdout, format = log_format, level = logging.INFO)
# logger = logging.getLogger()

# from stockname import StockNames

# app = Flask(__name__, static_url_path='/sharedvol')
app = Flask(__name__)
# CORS(app)

app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="http://localhost:4600")

# @socketio.on('message')
# def handle_message(data):
#     print('received message: ' + data)

@app.route('/')
def index():
    return {1: 2}

@socketio.event
def my_event(message):
    emit('my response', {'data': 'got it!'})



# @app.route("/stylize", methods = ['POST', 'OPTIONS'])
# def stylize():

#     model_path = os.path.join('/sharedvol', 'gan-generator.pt')
#     input_image = os.path.join('/sharedvol', 'test.jpg')
#     output_image = os.path.join('/sharedvol', 'test_out.jpg')

#     with open(input_image, 'wb') as f:
#         shutil.copyfileobj(request.files['origin_image'], f)

#     HOST, PORT = "py-cartoongan", 65432

#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#     s.connect((HOST, PORT))

#     send_msg = model_path + " " + input_image + " " + output_image

#     s.send(send_msg.encode('ascii'))

#     recv_msg = str(s.recv(1024))

#     return {
#         "output": 'dasdasd',
#     }


# LSTM_RESULT_DIR = '/app/data/'

# @app.route("/stocklist")
# def stocklist():

#     lstm_results = [f.strip('.csv') for f in os.listdir(LSTM_RESULT_DIR) \
#         if os.path.isfile(os.path.join(LSTM_RESULT_DIR, f))]

#     stocks = {code: StockNames[code] for code in lstm_results if StockNames.get(code)}

#     return stocks

# @app.route("/prediction/<stock_code>")
# def prediction(stock_code):

#     dates = []
#     real_close = []
#     pred_close = []

#     with open(os.path.join(LSTM_RESULT_DIR, \
#         '{0}.csv'.format(stock_code)), newline='') as csvfile:
#         reader = csv.DictReader(csvfile, delimiter=',')
#         for row in reader:
#             dates.append(row['date'])
#             real_close.append(row['real_close'])
#             pred_close.append("%.2f" % float(row['pred_close']))

#     return {
#         "date": dates,
#         "real_close": real_close,
#         "pred_close": pred_close,
#     }

if __name__ == "__main__":
    # app.run(host="0.0.0.0", debug=os.environ['FLASK_ENV']=='development', port=80)
    socketio.run(app, debug=os.environ['FLASK_ENV']=='development', port=80)