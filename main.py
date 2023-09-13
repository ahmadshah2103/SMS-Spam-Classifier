import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk
import streamlit as st
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


def text_preprocess(text):
    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []
    for t in text:
        if t.isalnum():
            y.append(t)

    text = y[:]
    y.clear()

    for t in text:
        if t not in stopwords.words('english') and t not in string.punctuation:
            y.append(t)

    text = y[:]
    y.clear()

    for t in text:
        y.append(ps.stem(t))

    return ' '.join(y)


vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('spam_filter_model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter your message')

if st.button('Check Spam'):
    #STEPS:
    # Step.01: Transform text
    transformed_text = text_preprocess(input_sms)
    # Step .02: Vectorize Text
    vectorized_text = vectorizer.transform([transformed_text]).toarray()
    # Step .03: Predict
    output = model.predict(vectorized_text)
    if output:
        st.header(:red['Spam'])
    else:
        st.header(:green['Not Spam'])


