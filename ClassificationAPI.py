from flask import Flask, make_response, request, jsonify
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import os
import pickle
import numpy as np


app = Flask(__name__)

#load the model
file = os.getcwd()+"\Models\\"
countVectorizer = pickle.load(open(file+"vectorizerCV.pkl", "rb"))
tfidfVectorizer = pickle.load(open(file+"vectorizerTFIDF.pkl", "rb"))
lda = pickle.load(open(file+"LDA.pkl", "rb"))
tsvd = pickle.load(open(file+"TSVD.pkl", "rb"))
rfc_CV_Lda = pickle.load(open(file+"RF_CV_LDA.pkl", "rb"))
rfc_TFIDF_Tsvd = pickle.load(open(file+"RF_TFIDF_TSVD.pkl", "rb"))

nltk.download('punkt')


#method to convert feature to pca vectors
def preProcessText(text):
    nonPuncuated = text.translate(str.maketrans('','',string.punctuation))
    STOPWORDS = set(stopwords.words("english"))
    nonStopword = " ".join(word for word in nonPuncuated.split() if word not in STOPWORDS)
    tokenizedText = word_tokenize(nonStopword)
    
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokenizedText)
    lemmatizedWords = []
    for word, tag in pos_tags:
        if tag.startswith('N'):
            pos = 'n'
        elif tag.startswith('V'):
            pos = 'v'
        elif tag.startswith('R'):
            pos = 'r'
        else:
            pos = 'n'
        
        lemma = lemmatizer.lemmatize(word, pos)
        lemmatizedWords.append(lemma)
    joinedText = ' '.join(lemmatizedWords)

    tfidfVectorizedText = tfidfVectorizer.transform([joinedText])
    cvVectorizedText = countVectorizer.transform([joinedText])

    ldaText = lda.transform(cvVectorizedText)
    tsvdText = tsvd.transform(tfidfVectorizedText)
    
    
    return ldaText, tsvdText

#method to predict models
def predictingModels(text):
    ldaText, tsvdText = preProcessText(text)
    model1_Score = rfc_CV_Lda.predict(ldaText)[0]
    model2_Score = rfc_TFIDF_Tsvd.predict(tsvdText)[0]
    print(f'model 1 : {model1_Score}, model 2: {model2_Score}')
    return (model1_Score+model2_Score)//2

@app.route('/sopPrediction', methods = ['POST'])
def getText():
    if request.method == "POST":
        data = request.get_json()
        text = data['text']
        if len(text) != 0:
            prediction = predictingModels(text)
            response = jsonify({'score':prediction})
            return response
        else:
            return f"Input doesn't meet prerequisite, Input length is {len(text)}"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 105)