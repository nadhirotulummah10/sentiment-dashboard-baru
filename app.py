import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Analisis Sentimen TikTok", layout="wide")

# ================= LOAD DATA (AMAN) =================
try:
    df = pd.read_csv("hasil_labeling_sentimen.csv")
    df.columns = df.columns.str.lower()
except:
    df = pd.DataFrame()

# ================= INIT PAGE =================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ================= HOME =================
def home():
    st.title("ANALISIS SENTIMEN TIKTOK")

    st.write("Alur Analisis:")
    st.write("Data → Preprocessing → Pelabelan → Visualisasi")

    if os.path.exists("images/alur.png"):
        st.image("images/alur.png")

    st.write("---")

    col1, col2 = st.columns(2)

    if col1.button("📊 Lihat Hasil Penelitian"):
        st.session_state.page = "hasil"
        st.rerun()

    if col2.button("✍️ Analisis Baru"):
        st.session_state.page = "analisis"
        st.rerun()

# ================= HASIL =================
def hasil():

    st.title("HASIL PENELITIAN")

    if df.empty:
        st.error("Data tidak ditemukan")
        return

    if "sentiment" not in df.columns:
        st.error("Kolom 'sentiment' tidak ada di CSV")
        st.write("Kolom yang ada:", df.columns)
        return

    total = len(df)
    positif = len(df[df["sentiment"] == "positif"])
    netral = len(df[df["sentiment"] == "netral"])
    negatif = len(df[df["sentiment"] == "negatif"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total", total)
    col2.metric("Positif", positif)
    col3.metric("Netral", netral)
    col4.metric("Negatif", negatif)

    st.write("---")

    st.subheader("Pengumpulan Data")
    st.write("Data komentar TikTok dikumpulkan sebagai dataset penelitian.")

    st.subheader("Preprocessing")
    st.write("Cleaning, tokenizing, stopword removal.")

    st.subheader("Pelabelan")
    st.write("Lexicon-based sentiment analysis.")

    st.write("---")

    if os.path.exists("wordcloud_hasil_preprocessing.png"):
        st.image("wordcloud_hasil_preprocessing.png")

    if os.path.exists("distribusi_hasil_pelabelan_sentimen.png"):
        st.image("distribusi_hasil_pelabelan_sentimen.png")

    if os.path.exists("confusion_matrix_5fold.png"):
        st.image("confusion_matrix_5fold.png")

    if st.button("⬅ Kembali"):
        st.session_state.page = "home"
        st.rerun()

# ================= ANALISIS =================
def analisis():

    st.title("ANALISIS BARU")

    text = st.text_area("Masukkan komentar:")

    if st.button("Proses"):
        if text.strip() == "":
            st.warning("Masukkan teks dulu")
        else:
            st.success("Prototype aktif (belum terhubung model)")

    if st.button("⬅ Kembali"):
        st.session_state.page = "home"
        st.rerun()

# ================= ROUTER =================
if st.session_state.page == "home":
    home()
elif st.session_state.page == "hasil":
    hasil()
elif st.session_state.page == "analisis":
    analisis()
