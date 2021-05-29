from flask import Flask,request,render_template,url_for
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import pandas as pd
import joblib


clf = pickle.load(open('model1.pkl','rb'))
cv = pickle.load(open('transform12.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST']) 
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_pred = clf.predict(vect)

        return render_template('index.html',pred = my_pred)
    return render_template('index.html')    




if __name__ == "__main__":
    app.run(debug=True)