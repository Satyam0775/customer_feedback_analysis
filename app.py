"""
app.py — Minimal Gradio Interface for Sentiment Prediction
Author: Satyam
"""

import joblib
import gradio as gr
import re
import nltk
from nltk.corpus import stopwords
import os

# === Paths ===
MODEL_DIR = r"C:\Users\satya\Downloads\customer_feedback_analysis\task\models"
MODEL_PATH = os.path.join(MODEL_DIR, "feedback_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# === Load resources ===
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# === Load model and vectorizer ===
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# === Text cleaning (same as training) ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# === Prediction function ===
def predict_sentiment(text):
    if not text.strip():
        return "⚠️ Please enter some text!"
    text_clean = clean_text(text)
    text_vec = vectorizer.transform([text_clean])
    prediction = model.predict(text_vec)[0]
    return f"Predicted Sentiment: {prediction}"

# === Gradio Interface ===
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type your review here..."),
    outputs="text",
    title="🧠 Customer Feedback Sentiment Classifier",
    description="Enter a customer review to predict whether it's Positive, Negative, or Neutral.",
    examples=[
        ["I love this product! It works perfectly."],
        ["Worst experience ever, totally disappointed."],
        ["It was okay, not too great, not too bad."]
    ]
)

# === Launch Gradio app ===
if __name__ == "__main__":
    demo.launch(share=False, server_port=None)  # auto-select free port
