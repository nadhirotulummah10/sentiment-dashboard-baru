import streamlit as st
import pandas as pd
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Sentimen TikTok", layout="wide")

# Navigasi Sidebar
with st.sidebar:
    st.title("Navigasi")
    page = st.radio("Pilih Halaman:", ["🏠 Beranda & Alur", "📊 Hasil Penelitian"])

# ==========================================
# HALAMAN 1: BERANDA & ALUR (Penjelasan Saja)
# ==========================================
if page == "🏠 Beranda & Alur":
    st.title("🏠 Beranda & Alur Penelitian")
    st.write("Penelitian ini menganalisis sentimen masyarakat di TikTok terkait program Makan Bergizi Gratis.")
    
    st.subheader("Alur Kerja Sistem")
    st.markdown("""
    1. **Pengumpulan Data:** Melakukan *scraping* komentar dari platform TikTok.
    2. **Preprocessing:** Membersihkan data dari *noise* (emoji, angka, simbol) dan normalisasi kata.
    3. **Pelabelan:** Mengklasifikasikan komentar menjadi Positif, Netral, atau Negatif menggunakan *Lexicon-Based*.
    4. **Balancing (SMOTE):** Menyeimbangkan data agar model tidak bias.
    5. **TF-IDF:** Mengekstraksi fitur kata menjadi bobot numerik.
    6. **Evaluasi:** Menguji model menggunakan *5-Fold Cross Validation*.
    """)

# ==========================================
# HALAMAN 2: HASIL PENELITIAN (Tabel & Gambar)
# ==========================================
elif page == "📊 Hasil Penelitian":
    st.title("📊 Hasil Penelitian & Data Eksperimen")
    
    # 1. Scraping
    st.subheader("1. Data Scraping")
    df_scrap = pd.DataFrame({'User': ['user_1', 'user_2'], 'Komentar': ['Gaji 💀', 'Semenjak ada MBG...']})
    st.dataframe(df_scrap)
    st.caption("Penjelasan: Data mentah hasil scraping sebelum diproses.")

    # 2. Preprocessing
    st.subheader("2. Hasil Preprocessing")
    df_pre = pd.DataFrame({'Awal': ['Gaji 💀'], 'Hasil': ['gaji']})
    st.table(df_pre)
    st.caption("Penjelasan: Teks sudah bersih dari noise dan menjadi kata dasar.")

    # 3. Pelabelan
    st.subheader("3. Pelabelan Sentimen")
    df_label = pd.DataFrame({'Kelas': ['Positif', 'Negatif', 'Netral'], 'Jumlah': [975, 1428, 562]})
    st.table(df_label)
    st.caption("Penjelasan: Distribusi kelas sebelum balancing.")

    # 4. Balancing (SMOTE)
    st.subheader("4. Balancing Data (SMOTE)")
    df_smote = pd.DataFrame({'Kelas': ['Positif', 'Negatif', 'Netral'], 'Sebelum': [975, 1428, 562], 'Sesudah': [1428, 1428, 1428]})
    st.table(df_smote)
    st.caption("Penjelasan: Data disetarakan menjadi 1.428 per kelas.")

    # 5. TF-IDF
    st.subheader("5. Matriks TF-IDF")
    df_tfidf = pd.DataFrame({'adik': [0.0], 'ahli': [0.0], 'Label': [1]})
    st.dataframe(df_tfidf)
    st.caption("Penjelasan: Representasi numerik fitur kata.")

    # 6. Evaluasi
    st.subheader("6. Hasil Evaluasi Model (5-Fold)")
    df_eval = pd.DataFrame({'Fold': [1, 2, 3, 4, 5, 'Rata-rata'], 'Accuracy': [0.805, 0.785, 0.793, 0.777, 0.806, 0.793]})
    st.table(df_eval)
    st.caption("Penjelasan: Akurasi rata-rata model adalah 79.34%.")
