import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Analisis Sentimen TikTok", layout="wide")

# ================= LOAD DATA =================
file_path = "hasil_labeling_sentimen.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
else:
    st.error("File CSV tidak ditemukan di GitHub (root folder)")
    st.stop()

# ================= SESSION STATE =================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ================= HALAMAN HOME =================
if st.session_state.page == "home":
    st.title("ANALISIS SENTIMEN TIKTOK")

    st.markdown("### Alur Penelitian")
    st.write("Pengumpulan Data → Preprocessing → Pelabelan → Model → Evaluasi → Visualisasi")

    if os.path.exists("images/alur.png"):
        st.image("images/alur.png")

    st.write("---")

    col1, col2 = st.columns(2)

    if col1.button("📊 Lihat Hasil Penelitian"):
        st.session_state.page = "hasil"

    if col2.button("✍️ Analisis Baru"):
        st.session_state.page = "analisis"

# ================= HALAMAN HASIL =================
elif st.session_state.page == "hasil":
    st.title("HASIL PENELITIAN")

    st.subheader("Overview Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Data", len(df))
    col2.metric("Positif", len(df[df["sentiment"] == "positif"]))
    col3.metric("Netral", len(df[df["sentiment"] == "netral"]))

    st.write("---")

    st.subheader("Pengumpulan Data")
    st.write("Data dikumpulkan dari komentar TikTok sebagai dataset penelitian.")

    st.subheader("Preprocessing")
    st.write("Tahap cleaning, case folding, tokenizing, stopword removal.")

    st.subheader("Pelabelan")
    st.write("Menggunakan lexicon-based sentiment analysis.")

    st.write("---")

    st.subheader("Visualisasi Hasil")

    if os.path.exists("wordcloud_hasil_preprocessing.png"):
        st.image("wordcloud_hasil_preprocessing.png")

    if os
