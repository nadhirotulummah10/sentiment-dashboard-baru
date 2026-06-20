
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Analisis Sentimen", layout="wide")

df = pd.read_csv("hasil_labeling_sentimen.csv")

# ================= STATE PAGE =================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ================= HALAMAN 1 =================
if st.session_state.page == "home":
    st.title("Analisis Sentimen Tiktok Terhadap Program Makan Bergizi Gratis pada Tiktok Menggunakan Logistic Regression")

    st.write("### Alur Analisis:")
    st.write("Pengumpulan Data → Preprocessing → TF-IDF → Model → Evaluasi → Visualisasi")

    col1, col2 = st.columns(2)

    if col1.button("📊 Lihat Hasil Penelitian"):
        st.session_state.page = "hasil"

    if col2.button("✍️ Analisis Baru"):
        st.session_state.page = "analisis"

# ================= HALAMAN 2 =================
elif st.session_state.page == "hasil":
    st.title("HASIL PENELITIAN")

    st.write("## Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Data", len(df))
    col2.metric("Positif", len(df[df["sentiment"]=="positif"]))
    col3.metric("Netral", len(df[df["sentiment"]=="netral"]))

    st.write("---")

    st.write("## Pengumpulan Data")
    st.write("Data komentar TikTok dikumpulkan dan diproses.")

    st.write("## Preprocessing")
    st.write("Cleaning, tokenizing, stopword removal.")

    st.write("## Pelabelan")
    st.write("Menggunakan lexicon-based sentiment labeling.")

    st.write("## Hasil Visualisasi")

    st.image("images/wordcloud.png")
    st.image("images/distribusi.png")
    st.image("images/confusion_matrix.png")

    if st.button("⬅ Kembali"):
        st.session_state.page = "home"

# ================= HALAMAN 3 =================
elif st.session_state.page == "analisis":
    st.title("ANALISIS BARU")

    text = st.text_area("Masukkan komentar:")

    if st.button("Proses"):
        st.info("Fitur analisis belum diaktifkan (prototype)")

    if st.button("⬅ Kembali"):
        st.session_state.page = "home"
