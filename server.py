import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
vect = pickle.load(open('vect.pkl', 'rb'))
tfidf_transformer = pickle.load(open('tfidf_transformer.pkl', 'rb'))

severity = {0: 'Minor', 1:  'Normal',  2: 'Major', 4: 'Critical', 5: 'Blocker'}


@app.route("/", methods=['GET'])
def hello_world():
    return "hello world"


@app.route("/predict", methods=["POST"])
def predict():

    # Make prediction using model loaded from disk as per the data.
    data = request.get_json(force=True)
    # preporcess the post data
    ip = [data['bug_description']]
    ip_dtm = vect.transform(ip)
    ip_tfidf = tfidf_transformer.transform(ip_dtm)
    # use the input to predict
    prediction = model.predict(ip_tfidf)
    # return the output
    output = severity[prediction[0]]
    return jsonify(output)
