
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the trained model
model = load_model('Flask Project/sentiment_analysis_model_main.h5')

# Load the tokenizer
with open('Flask Project/main_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        review = review.lower()
        sequence = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequence, maxlen=100)
        score = model.predict(padded_sequence)[0][0]
        sentiment = 'positive' if score >= 0.5 else 'negative'
        return jsonify({'prediction': f'Sentiment: {sentiment}, Score: {score:.2f}', 'sentiment': sentiment})
    
if __name__ == '__main__':
    app.run(debug=True)
