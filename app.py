import streamlit as st
import pandas as pd

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Sentimen TikTok", layout="wide")

# Navigasi Sidebar
with st.sidebar:
    st.title("Menu Utama")
    page = st.radio("Pilih Halaman:", ["Beranda & Alur", "Hasil Penelitian"])

# Halaman 1: Beranda & Alur
if page == "Beranda & Alur":
    st.title("🏠 Beranda & Alur Penelitian")
    st.write("Sistem Analisis Sentimen Komentar TikTok terkait program **Makan Bergizi Gratis**.")
    
    st.markdown("### Alur Penelitian:")
    st.markdown("""
    1. **Pengumpulan Data:** *Scraping* komentar dari platform TikTok.
    2. **Preprocessing:** *Cleaning*, *Case Folding*, Normalisasi, *Stopword Removal*, dan *Stemming*.
    3. **Pelabelan:** Penentuan sentimen (*Lexicon-Based*).
    4. **Balancing:** Penyeimbangan data menggunakan metode **SMOTE**.
    5. **Ekstraksi Fitur:** Transformasi teks ke numerik menggunakan **TF-IDF**.
    6. **Klasifikasi & Evaluasi:** Pemodelan menggunakan *Multinomial Logistic Regression* dengan *5-Fold Cross Validation*.
    """)

# Halaman 2: Hasil Penelitian
elif page == "Hasil Penelitian":
    st.title("📊 Hasil Penelitian & Eksperimen")
    
    # 1. Scraping
    st.subheader("1. Data Hasil Scraping (Total: 3.320 Baris)")
    df_scrap = pd.DataFrame({
        'Komentar': ['gaji💀', 'semenjak ada MBG aku bisa nabung🤍🙋🏼‍♀️', 'maaf...', 'boleh jujur??...', 'gra² mbg aku'],
        'Tanggal_Komentar': ['2026-02-02 23:36:29', '2026-02-02 08:24:10', '2026-02-02 10:22:26', '2026-02-02 10:37:20', '2026-02-02 08:16:53']
    })
    st.dataframe(df_scrap, use_container_width=True)

    # 2. Preprocessing
    st.subheader("2. Hasil Preprocessing (Total: 2.965 Baris)")
    df_pre = pd.DataFrame({
        'Komentar': ['gaji💀', 'semenjak ada MBG...'],
        'cleaning': ['gaji', 'semenjak ada MBG aku bisa nabung'],
        'Final_Text': ['gaji', 'semenjak makan gizi gratis nabung']
    })
    st.dataframe(df_pre, use_container_width=True)

    # 3. Ringkasan Dataset
    st.subheader("3. Ringkasan Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Dataset", "2.965 Baris")
    col2.metric("Positif (Label 2)", "975 (32.9%)")
    col3.metric("Netral (Label 1)", "562 (19.0%)")
    col4.metric("Negatif (Label 0)", "1.428 (48.2%)")
    
    df_label = pd.DataFrame({
        'Sentimen': ['Positif (Label 2)', 'Negatif (Label 0)', 'Netral (Label 1)'],
        'Jumlah Ulasan': [975, 1428, 562],
        'Persentase (%)': ['32.9%', '48.2%', '19.0%']
    })
    st.table(df_label)

    # 4. Balancing SMOTE
    st.subheader("4. Balancing Data dengan SMOTE")
    st.write("Setelah proses SMOTE, distribusi data menjadi seimbang:")
    st.info("Total Data Uji Evaluasi: 4.284 Baris (Masing-masing kelas 1.428 baris)")

    # 5. TF-IDF
    st.subheader("5. Matriks TF-IDF")
    df_tfidf = pd.DataFrame({'adik': [0.0]*5, 'ahli': [0.0]*5, 'Label': [1, 2, 0, 0, 0]})
    st.dataframe(df_tfidf, use_container_width=True)

    # 6. Evaluasi Model
    st.subheader("6. Hasil Logistic Regression (5-Fold)")
    df_eval = pd.DataFrame({
        'Fold': ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Average'],
        'Accuracy': [0.8051, 0.7853, 0.7935, 0.7771, 0.8061, 0.7934],
        'Precision': [0.8125, 0.7897, 0.8023, 0.7826, 0.8115, 0.7997],
        'Recall': [0.8051, 0.7853, 0.7935, 0.7771, 0.8061, 0.7934],
        'F1-Score': [0.8045, 0.7852, 0.7928, 0.7770, 0.8061, 0.7931]
    })
    st.table(df_eval)
