
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

    if col2.button("✍️ Analisis Baruimport streamlit as st
import pandas as pd

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

df = pd.read_csv("hasil_labeling_sentimen.csv")
df.columns = df.columns.str.lower()

# ================= SIDEBAR =================
menu = st.sidebar.selectbox(
    "📊 Menu Navigasi",
    ["Overview", "Pengumpulan Data", "Preprocessing", "Pelabelan", "Visualisasi", "Analisis Baru"]
)

# ================= OVERVIEW =================
if menu == "Overview":
    st.title("OVERVIEW DATASET")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Data", len(df))
    col2.metric("Positif", len(df[df["sentiment"]=="positif"]))
    col3.metric("Netral", len(df[df["sentiment"]=="netral"]))

# ================= PENGUMPULAN DATA =================
elif menu == "Pengumpulan Data":
    st.title("PENGUMPULAN DATA")
    st.write("Data dikumpulkan dari komentar TikTok...")

# ================= PREPROCESSING =================
elif menu == "Preprocessing":
    st.title("PREPROCESSING")
    st.write("Cleaning, case folding, tokenizing...")

# ================= PELABELAN =================
elif menu == "Pelabelan":
    st.title("PELABELAN SENTIMEN")
    st.write("Menggunakan lexicon-based approach...")

# ================= VISUALISASI =================
elif menu == "Visualisasi":
    st.title("VISUALISASI HASIL")

    st.image("wordcloud_hasil_preprocessing.png")
    st.image("distribusi_hasil_pelabelan_sentimen.png")
    st.image("confusion_matrix_5fold.png")

# ================= ANALISIS BARU =================
elif menu == "Analisis Baru":
    st.title("ANALISIS BARU")

    text = st.text_area("Masukkan komentar")

    if st.button("Proses"):
        st.info("Prototype belum aktif model")"):
        st.session_state.page = "analisis"

# ================= HALAMAN 2 =================
elif st.session_state.page == "hasil":
    st.title("HASIL PENELITIAN")

    st.write("## Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Data", len(df))
    col2.metric("Positif", len(df[df["sentimen"]=="positif"]))
    col3.metric("Netral", len(df[df["sentimen"]=="netral"]))

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
