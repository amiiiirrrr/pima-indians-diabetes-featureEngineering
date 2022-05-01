"""
testing the model output quality by flask
"""
from flask import Flask, jsonify, request
from test_model import TestModel

__author__ = "Seyed Amir Mousavi"
__license__ = "Public Domain"
__version__ = "1.0.0"
__status__ = "Production"

APP = Flask(__name__)


# routes
@APP.route('/', methods=['POST', 'GET'])
def predict():
    """
    to test the quality of target prediction
    :return: json
    """
    data = request.get_json(force=True)		# get json request
    ouput_dict = INS.run_flask(data)		# run_flask return results
    return jsonify(ouput_dict)


if __name__ == "__main__":
    INS = TestModel()
    APP.run(debug=True, host="0.0.0.0", port=8027, threaded=True)
