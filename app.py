import numpy as np
import pandas as pd
import pickle
from flask import Flask, jsonify, render_template, request

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import h5py
from keras.models import load_model

# load the dataset but only keep the top n words, zero the rest
top_words = 10000
max_words = 500

#load the csv file saved

df = pd.read_csv('./data/movie_data.csv', encoding='utf-8')

tokenizer_obj = Tokenizer(num_words=top_words)
tokenizer_obj.fit_on_texts(df.loc[:50000, 'review'].values)

def pred(usermoviereview):
    test_samples = [usermoviereview]
    review_tokens = tokenizer_obj.texts_to_sequences(test_samples)
    review_tokens_pad = pad_sequences(review_tokens, maxlen=max_words)

    print("call predict")
    # Load in pretrained model
    loaded_model = load_model('./models/movie_sa_model.h5')
    print("Loaded model from disk")
    sentiment = loaded_model.predict(x=review_tokens_pad)
    print(sentiment)
    if sentiment[0] > 0.5:
        sentiment_str = "You like the movie:" + "{0:.2%}".format(float(sentiment[0]))
    else:
        sentiment_str = "You didn't like the movie:" + "{0:.2%}".format(float(sentiment[0]))
    return sentiment_str

# webapp
app = Flask(__name__, template_folder='./')


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == "POST":
        message = request.form['message']
        print(message)
        response =  pred(message)
        print(response)
        return jsonify(response)
    return jsonify("Input text")

@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)