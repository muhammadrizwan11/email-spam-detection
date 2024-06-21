import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load necessary NLTK resources
nltk.download('stopwords')

# Preprocess function
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = text.split()
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in words if word not in stop_words])  # Remove stopwords
    return text

# Load the trained model and vectorizer
loaded_model = joblib.load('multinomial_naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Make sure to save and load the vectorizer as well

# Streamlit app
st.title('Email Spam Detection')
st.write('Enter the email content below to check if it is spam or not.')

# Text input for the email content
email_input = st.text_area("Email Content", height=200)

if st.button('Predict'):
    if email_input:
        # Preprocess and transform the input email
        preprocessed_email = preprocess_text(email_input)
        vectorized_email = vectorizer.transform([preprocessed_email])

        # Predict using the loaded model
        prediction = loaded_model.predict(vectorized_email)

        # Display the prediction
        if prediction[0] == 1:
            st.error("The email is classified as Spam.")
        else:
            st.success("The email is classified as Ham (Not Spam).")
    else:
        st.warning("Please enter the email content.")
