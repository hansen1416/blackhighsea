from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return {'hello': 'world'}

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)