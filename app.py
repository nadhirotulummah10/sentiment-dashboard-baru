import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Analisis Sentimen TikTok", layout="wide")

# ================= LOAD DATA =================
df = pd.read_csv("hasil_labeling_sentimen.csv")

# ================= STATE =================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ================= HOME =================
if st.session_state.page == "home":
    st.title("ANALISIS SENTIMEN TIKTOK")

    st.write("Alur Analisis:")
    st.write("Pengumpulan Data → Preprocessing → Pelabelan → Visualisasi")

    if os.path.exists("images/alur.png"):
        st.image("images/alur.png")

    col1, col2 = st.columns(2)

    if col1.button("📊 Lihat Hasil Penelitian"):
        st.session_state.page = "hasil"

    if col2.button("✍️ Analisis Baru"):
        st.session_state.page = "analisis"

# ================= HASIL =================
elif st.session_state.page == "hasil":
    st.title("HASIL PENELITIAN")

    st.subheader("Overview Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Data", len(df))
    col2.metric("Positif", len(df[df["sentiment"] == "positif"]))
    col3.metric("Netral", len(df[df["sentiment"] == "netral"]))

    st.write("---")

    st.subheader("Pengumpulan Data")
    st.write("Data dikumpulkan dari komentar TikTok.")

    st.subheader("Preprocessing")
    st.write("Cleaning, tokenizing, stopword removal.")

    st.subheader("Pelabelan")
    st.write("Lexicon-based sentiment analysis.")

    st.write("---")

    st.subheader("Visualisasi")

    if os.path.exists("wordcloud_hasil_preprocessing.png"):
        st.image("wordcloud_hasil_preprocessing.png")

    if os.path.exists("distribusi_hasil_pelabelan_sentimen.png"):
        st.image("distribusi_hasil_pelabelan_sentimen.png")

    if os.path.exists("confusion_matrix_5fold.png"):
        st.image("confusion_matrix_5fold.png")

    if st.button("⬅ Kembali"):
        st.session_state.page = "home"

# ================= ANALISIS =================
elif st.session_state.page == "analisis":
    st.title("ANALISIS BARU")

    text = st.text_area("Masukkan komentar:")

    if st.button("Proses"):
        st.info("Prototype belum terhubung model")

    if st.button("⬅ Kembali"):
        st.session_state.page = "home"
