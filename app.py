import os
import zipfile
import subprocess
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
import altair as alt


# -------------------- Step 1: Download model if missing --------------------
MODEL_DIR = "goemotions_ekman_model"
ZIP_FILE = "goemotions_ekman_model.zip"
GOOGLE_DRIVE_FILE_ID = "1VLjrztZDnSjfovvTRv2hA47OdedKLcSv"


def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        st.info("📦 Downloading model...")
        if not os.path.exists(ZIP_FILE):
            subprocess.run(
                ["gdown", "--id", GOOGLE_DRIVE_FILE_ID, "-O", ZIP_FILE],
                check=True
            )
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(".")
        st.success("✅ Model downloaded and extracted!")


download_and_extract_model()


# -------------------- Step 2: Load model and tokenizer --------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()
model.eval()


# -------------------- Step 3: Emotion mappings --------------------
id_to_emotion = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "sadness",
    5: "surprise",
    6: "neutral"
}


emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "joy": "🤗",
    "neutral": "😐",
    "sadness": "😔",
    "surprise": "😮"
}


# -------------------- Step 4: Emotion prediction --------------------
def predict_emotions(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_id = int(np.argmax(probs))
    return id_to_emotion[predicted_id], probs


# -------------------- Step 5: Streamlit App --------------------
def main():
    st.title("🧠 Text Emotion Detection (BERT - GoEmotions)")
    st.subheader("Detect Emotions In Text Using a Fine-tuned BERT Model")

    with st.form(key='emotion_form'):
        raw_text = st.text_area("Enter your text here:")
        submit_button = st.form_submit_button(label="Analyze")

    if submit_button and raw_text.strip():
        col1, col2 = st.columns(2)
        emotion, probs = predict_emotions(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Predicted Emotion")
            emoji = emotions_emoji_dict.get(emotion, "❓")
            confidence = np.max(probs) * 100
            st.write(f"{emotion.capitalize()} {emoji}")
            st.write(f"Confidence: {confidence:.2f}%")

        with col2:
            st.success("Prediction Probability Distribution")
            proba_df = pd.DataFrame({
                "emotion": list(id_to_emotion.values()),
                "probability": probs
            })
            chart = (
                alt.Chart(proba_df)
                .mark_bar()
                .encode(
                    x='emotion',
                    y='probability',
                    color='emotion'
                )
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("👈 Enter some text and click Analyze to see results.")


if __name__ == '__main__':
    main()
