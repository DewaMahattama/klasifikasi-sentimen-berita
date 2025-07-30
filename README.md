# Klasifikasi Sentimen Berita Daerah Buleleng

[![Streamlit App](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)](https://klasifikasi-sentimen-berita-5cq7ezwcidd88zzzaciypv.streamlit.app/)

Sistem berbasis Streamlit yang digunakan untuk mengklasifikasikan sentimen dari judul berita daerah Buleleng ke dalam tiga kategori: **Positif**, **Netral**, dan **Negatif**. Sistem ini memanfaatkan model fine-tuned IndoBERT yang diunggah ke [Hugging Face Hub](https://huggingface.co/yeaylow/indobert-sentimen-berita) untuk klasifikasi senteimen berita.

## 🔗 Demo 

[Link Sistem](https://klasifikasi-sentimen-berita-5cq7ezwcidd88zzzaciypv.streamlit.app/)

---

## 🧠 Model

Model yang digunakan adalah versi fine-tuned dari IndoBERT untuk tugas klasifikasi sentimen multi-kelas. Model diunduh langsung dari Hugging Face:

- 🔗 https://huggingface.co/yeaylow/indobert-sentimen-berita

---

## Library

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [Transformers (Hugging Face)](https://huggingface.co/transformers/)
- [Torch](https://pytorch.org/)
- [Pandas, NumPy, scikit-learn, matplotlib](https://scikit-learn.org/stable/)

---

## Cara Menjalankan (Lokal)

1. **Clone repo ini:**
   ```bash
   git clone https://github.com/username/klasifikasi-sentimen-berita.git
   cd klasifikasi-sentimen-berita
2. **Install Dependensi ini:**
   pip install -r requirements.txt
3. **Jalankan:**
   streamlit run app.py


