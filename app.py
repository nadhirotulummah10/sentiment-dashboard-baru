import streamlit as st
import pandas as pd

st.set_page_config(page_title="Analisis Sentimen TikTok", layout="wide")

st.title("Sistem Analisis Sentimen TikTok: Makan Bergizi Gratis")

# 1. PENGUMPULAN DATA
st.header("1. Pengumpulan Data (Scraping)")
st.write("Proses penarikan 3.320 data mentah dari komentar TikTok.")
df_scrap = pd.DataFrame({'User': ['user_1', 'user_2'], 'Komentar': ['Gaji 💀', 'Semenjak ada MBG...']})
st.dataframe(df_scrap)
st.caption("Penjelasan: Data mentah diambil menggunakan scraping untuk menangkap opini masyarakat.")

# 2. PREPROCESSING
st.header("2. Preprocessing Data")
st.write("Pembersihan teks (Case Folding, Normalisasi, Tokenizing, Stopword, Stemming).")
df_pre = pd.DataFrame({'Awal': ['Gaji 💀'], 'Hasil': ['gaji']})
st.table(df_pre)
st.caption("Penjelasan: Menghilangkan noise (emoji/simbol) dan mengubah kata ke bentuk dasar.")

# 3. PELABELAN
st.header("3. Pelabelan (Lexicon-Based)")
st.write("Penentuan kelas berdasarkan skor kamus kata.")
df_label = pd.DataFrame({'Kelas': ['Positif', 'Negatif', 'Netral'], 'Jumlah': [975, 1428, 562]})
st.table(df_label)
st.caption("Penjelasan: Data dilabeli secara otomatis menggunakan pendekatan lexicon.")

# 4. BALANCING (SMOTE)
st.header("4. Balancing Data (SMOTE)")
st.write("Mengatasi imbalance data agar tiap kelas memiliki jumlah 1.428 baris.")
df_smote = pd.DataFrame({'Kelas': ['Positif', 'Negatif', 'Netral'], 'Sebelum': [975, 1428, 562], 'Sesudah': [1428, 1428, 1428]})
st.table(df_smote)
st.caption("Penjelasan: SMOTE menyeimbangkan data agar model tidak bias ke kelas mayoritas.")

# 5. TF-IDF
st.header("5. Ekstraksi Fitur (TF-IDF)")
st.write("Transformasi teks ke matriks numerik (612 fitur).")
df_tfidf = pd.DataFrame({'adik': [0.0], 'ahli': [0.0], 'Label': [1]})
st.dataframe(df_tfidf)
st.caption("Penjelasan: Mengubah kata menjadi bobot numerik untuk model pembelajaran mesin.")

# 6. EVALUASI (5-FOLD)
st.header("6. Evaluasi (5-Fold Cross Validation)")
st.write("Pengujian performa model dengan akurasi rata-rata 79.34%.")
df_eval = pd.DataFrame({'Fold': [1, 2, 3, 4, 5, 'Rata-rata'], 'Accuracy': [0.80, 0.78, 0.79, 0.77, 0.80, 0.79]})
st.table(df_eval)
st.caption("Penjelasan: Nilai rata-rata akurasi 79.34% menunjukkan model stabil dalam memprediksi sentimen.")
