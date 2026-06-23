import streamlit as st
import pandas as pd

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Sentimen", layout="wide")

# Navigasi Sidebar (Tetap ada dan tidak akan hilang)
with st.sidebar:
    st.title("Menu Utama")
    page = st.radio("Pilih Halaman:", ["Beranda & Alur", "Hasil Penelitian"])

# Halaman 1: Beranda & Alur
if page == "Beranda & Alur":
    st.title("Beranda & Alur Penelitian")
    st.write("Sistem Analisis Sentimen Komentar TikTok: Makan Bergizi Gratis")
    st.markdown("""
    **Alur Penelitian:**
    1. Pengumpulan Data (Scraping)
    2. Preprocessing Teks
    3. Pelabelan Sentimen (Lexicon)
    4. Balancing Data (SMOTE)
    5. Ekstraksi Fitur (TF-IDF)
    6. Klasifikasi & Evaluasi (Logistic Regression)
    """)

# Halaman 2: Hasil Penelitian
elif page == "Hasil Penelitian":
    st.title("Hasil Penelitian")
    
    # Contoh menampilkan tabel hasil penelitianmu
    st.subheader("Data Hasil Eksperimen")
    
    # Kamu bisa ganti dataframe ini dengan tabel hasil asli dari Colab-mu
    data = {
        'Tahapan': ['Scraping', 'Preprocessing', 'Balancing', 'Evaluasi'],
        'Hasil/Metrik': ['3.320 baris', '2.965 baris', '4.284 baris', 'Akurasi 79.34%']
    }
    df = pd.DataFrame(data)
    st.table(df)
    
    st.write("Penjelasan: Tabel di atas menunjukkan ringkasan data dari awal hingga tahap evaluasi akhir.")
