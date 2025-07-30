import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(
    page_title="Klasifikasi Sentimen",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("yeaylow/indobert-sentimen-berita")
    tokenizer = AutoTokenizer.from_pretrained("yeaylow/indobert-sentimen-berita")
    return model, tokenizer

model, tokenizer = load_model()
label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
color_map = {
    "Negatif": "#d9534f",  
    "Netral": "#5bc0de",   
    "Positif": "#5cb85c"   
}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=1).item()
    return label_map[pred], probs[0][pred].item()

st.set_page_config(
    page_title="Klasifikasi Sentimen",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    body {
        background-color: #0f1c2e;
        color: #f5f5f5;
    }
    .main {
        background-color: #0f1c2e;
    }
    .title {
        color: #f0f0f0;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    .description {
        color: #bbbbbb;
        text-align: center;
        font-size: 18px;
    }
    .box {
        background-color: #1e2a38;
        padding: 25px;
        border-radius: 10px;
        margin-top: 20px;
        color: #ffffff;
        font-size: 18px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #aaaaaa;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Klasifikasi Sentimen Berita Daerah Buleleng</div>', unsafe_allow_html=True)

text = st.text_area("Masukkan judul berita", height=150)

if st.button("Prediksi Sentimen"):
    if not text.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
    else:
        label, confidence = predict(text)
        color = color_map[label]

        st.markdown(f"""
        <div style='
            background-color: {color};
            color: white;
            padding: 20px;
            border-radius: 10px;
            font-size: 22px;
            margin-top: 20px;
            text-align: center;
        '>
            <b>{label}</b><br>
            Confidence: {confidence:.2%}
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="footer">@DewaMahattama</div>', unsafe_allow_html=True)
