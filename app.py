import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Sentiment Analysis TikTok", layout="wide")

# ================= LOAD DATA =================
df = pd.read_csv("hasil_labeling_sentimen.csv")
df.columns = df.columns.str.lower()

st.write(df.columns)
# ================= SESSION =================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ================= HOME =================
if st.session_state.page == "home":

    st.markdown("<h1 style='text-align:center;'>Analisis SEntimen Masyarakat terhadap Program Makan Bergizi Gratis pada Tiktok</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;'>Dashboard hasil penelitian berbasis lexicon sentiment analysis</p>", unsafe_allow_html=True)

    if os.path.exists("images/alur.png"):
        st.image("images/alur.png")

    st.markdown("### Alur Penelitian")
    st.write("Data → Preprocessing → Pelabelan → Analisis → Visualisasi")

    col1, col2 = st.columns(2)

    if col1.button("📊 Lihat Hasil Penelitian"):
        st.session_state.page = "hasil"

    if col2.button("🔍 Analisis Baru"):
        st.session_state.page = "analisis"


# ================= HASIL =================
elif st.session_state.page == "hasil":

    st.title("DASHBOARD HASIL PENELITIAN")

    st.markdown("## Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Data", len(df))
    col2.metric("Positif", len(df[df["sentiment"].astype(str)=="positif"]
    col3.metric("Netral", len(df[df["sentiment"].astype(str)=="netral"]
    col3.metric("Negatif", len(df[df["sentiment"].astype(str)=="negatif"]
    
    st.write("---")

    st.markdown("## Tahapan Analisis")

    st.markdown("### 1. Pengumpulan Data")
    st.write("Data diambil dari komentar TikTok sebagai objek penelitian.")

    st.markdown("### 2. Preprocessing")
    st.write("Cleaning, case folding, tokenizing, stopword removal.")

    st.markdown("### 3. Pelabelan Sentimen")
    st.write("Menggunakan pendekatan lexicon-based untuk klasifikasi sentimen.")

    st.write("---")

    st.markdown("## Visualisasi Hasil")

    if os.path.exists("wordcloud_hasil_preprocessing.png"):
        st.image("wordcloud_hasil_preprocessing.png", caption="WordCloud")

    if os.path.exists("distribusi_hasil_pelabelan_sentimen.png"):
        st.image("distribusi_hasil_pelabelan_sentimen.png", caption="Distribusi Sentimen")

    if os.path.exists("confusion_matrix_5fold.png"):
        st.image("confusion_matrix_5fold.png", caption="Confusion Matrix")

    st.write("---")

    if st.button("⬅ Kembali"):
        st.session_state.page = "home"


# ================= ANALISIS BARU =================
elif st.session_state.page == "analisis":

    st.title("ANALISIS BARU")

    text = st.text_area("Masukkan komentar:")

    if st.button("Proses Analisis"):
        if text.strip() == "":
            st.warning("Masukkan teks dulu")
        else:
            st.success("Prototype aktif (belum terhubung model)")

    if st.button("⬅ Kembali"):
        st.session_state.page = "home"
