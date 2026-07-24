import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vect.pkl")

st.title(" 😍 Sentimental Analysis App")

st.write("Enter any sentence below.")

text = st.text_area("Enter Text")

if st.button("Predict"):
    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)


    if prediction[0] == "Positive":
        st.success("😉 Positive")

    elif prediction[0] == "Negative":
        st.success("😟 Negative")
    else:
        st.info("😐 Neutral")
        
