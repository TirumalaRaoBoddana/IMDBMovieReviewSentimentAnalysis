import pickle
import numpy as np
import warnings
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load the model and vectorizer
with open('model.pkl', 'rb') as file:
    loaded_model, loaded_cv = pickle.load(file)

# Streamlit app configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.sidebar.title("🎥 Movie Review Sentiment Analysis")
st.sidebar.markdown("""
This project uses a machine learning model to analyze the sentiment of a given movie review. 

### Key Features:
- **Sentiment Analysis**: Detect if a review is **Positive** or **Negative**.
- **Interactive Input**: Enter any movie review to get instant feedback.
- **User-Friendly Design**: Simple and intuitive interface.
💡 **Note**: This analysis is based on patterns in the training data and may not capture sarcasm or nuanced sentiments.

Enjoy exploring!
""")

# Main Title
st.header("🎥 Movie Review Sentiment Analysis")
st.markdown("""
        ### How It Works:
        1. Input a movie review.
        2. The review is processed using a **CountVectorizer** to transform text into a numerical format.
        3. The transformed input is passed to a **trained classification model** to predict the sentiment.
    """)
review_input = st.text_area(
    label="Write your movie review here:",
    placeholder="E.g., The movie was fantastic! The story was gripping, and the characters were well-developed.",
    height=150
)

# Analyze button
if st.button("Analyze Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    elif len(review_input.split()) <= 40:
        st.warning("Your review must be longer than 40 words. Please elaborate and try again.")
    else:
        # Preprocess the input
        processed_input = loaded_cv.transform([review_input])
        # Predict sentiment
        prediction = loaded_model.predict(processed_input)
        # Display result
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.success(f"The sentiment of the review is **{sentiment}**!")
