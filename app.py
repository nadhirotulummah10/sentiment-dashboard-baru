import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Analisis Sentimen TikTok", layout="wide")

# ================= LOAD DATA (AMAN) =================
file_path = "hasil_labeling_sentimen.csv"

if not os.path.exists(file_path):
    st.error("File CSV tidak ditemukan di GitHub")
    st.stop()

df = pd.read_csv(file_path)
df.columns = df.columns.str.lower()

# kalau kolom sentiment tidak ada, stop biar jelas
if "sentiment" not in df.columns:
    st.error("Kolom 'sentiment' tidak ada di CSV")
    st.stop()

# ================= STATE =================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ================= HOME =================
if st.session_state.page == "home":

    st.title("ANALISIS SENTIMEN TIKTOK")

    st.write("Alur penelitian:")
    st.write("Data → Preprocessing → Pelabelan → Visualisasi")

    if os.path.exists("images/alur.png"):
        st.image("images/alur.png")

    col1, col2 = st.columns(2)

    if col1.button("📊 Lihat Hasil Penelitian"):
        st.session_state.page = "hasil"

    if col2.button("✍️ Analisis Baru"):
        st.session_state.page = "analisis"

# ================= HASIL =================
elif st.session_state.page == "hasil":

    st.title("DASHBOARD HASIL PENELITIAN")

    st.write("---")

    total = len(df)
    positif = len(df[df["sentiment"].astype(str) == "positif"])
    netral = len(df[df["sentiment"].astype(str) == "netral"])
    negatif = len(df[df["sentiment"].astype(str) == "negatif"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total", total)
    col2.metric("Positif", positif)
    col3.metric("Netral", netral)
    col4.metric("Negatif", negatif)

    st.write("---")

    st.subheader("Pengumpulan Data")
    st.write("Data komentar TikTok dikumpulkan sebagai dataset penelitian.")

    st.subheader("Preprocessing")
    st.write("Cleaning, case folding, tokenizing, stopword removal.")

    st.subheader("Pelabelan")
    st.write("Lexicon-based sentiment analysis digunakan untuk memberi label.")

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
        if text.strip() == "":
            st.warning("Masukkan teks dulu")
        else:
            st.success("Prototype berjalan (belum terhubung model ML)")

    if st.button("⬅ Kembali"):
        st.session_state.page = "home"
