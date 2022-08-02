import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import model

app = Flask(__name__)
nb_model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))
# text_transform = pickle.load(open('text_transform.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    class_map = {0: "Ham", 1: "Spam"}

    test_text = [str(x) for x in request.form.values()][0]

    test_text = model.text_transform(test_text)

    test_text = tfidf.transform(pd.Series(test_text)).toarray()

    prediction = nb_model.predict(test_text)[0]

    output = class_map[prediction]
    # output = test_text
    return render_template('index.html', prediction_text='Detected message type is: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)