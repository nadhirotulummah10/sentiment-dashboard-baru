import streamlit as st
import pandas as pd
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="Sistem Analisis Sentimen TikTok",
    page_icon="📊",
    layout="wide"
)

# CSS Custom
st.markdown("""
<style>
.main-title { font-size: 32px; font-weight: bold; color: #1E3A8A; }
.subtitle { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
.section-header { font-size: 22px; font-weight: bold; color: #1E3A8A; border-bottom: 2px solid #E5E7EB; margin-top: 20px; }
.card { background-color: #F9FAFB; padding: 20px; border-radius: 10px; border-left: 5px solid #3B82F6; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigasi
with st.sidebar:
    st.markdown("### Navigasi Sistem")
    page = st.radio("Pilih Halaman:", ["🏠 Beranda & Alur", "📊 Hasil Penelitian & Evaluasi Model"])
    st.markdown("---")
    st.markdown("**Metodologi:** Multinomial Logistic Regression & TF-IDF.")

# Halaman 1: Beranda & Alur (Penjelasan Konsep)
if page == "🏠 Beranda & Alur":
    st.markdown('<div class="main-title">ANALISIS SENTIMEN KOMENTAR TIKTOK</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Sistem Prototype Klasifikasi Sentimen Makan Bergizi Gratis</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Pipeline Sistem</div>', unsafe_allow_html=True)
    
    steps = [
        ("1. Pengumpulan Data (Scraping)", "Proses penarikan komentar dari TikTok terkait program Makan Bergizi Gratis (MBG)."),
        ("2. Preprocessing Data", "Pembersihan teks: case folding, normalisasi kata gaul, tokenizing, stopword removal, dan stemming."),
        ("3. Pelabelan (Lexicon-Based)", "Menentukan kelas Positif, Netral, atau Negatif menggunakan kamus sentimen."),
        ("4. Balancing Data (SMOTE)", "Penyeimbangan jumlah data kelas untuk mengatasi ketimpangan distribusi (imbalance)."),
        ("5. Ekstraksi Fitur (TF-IDF)", "Mengubah teks menjadi matriks bobot angka agar dapat dipahami model."),
        ("6. Klasifikasi & Validasi", "Pelatihan model Multinomial Logistic Regression dengan validasi 5-Fold Cross Validation.")
    ]
    
    for title, desc in steps:
        st.markdown(f'<div class="card"><strong>{title}</strong><br>{desc}</div>', unsafe_allow_html=True)

# Halaman 2: Hasil Penelitian (Data Tabel & Gambar)
elif page == "📊 Hasil Penelitian & Evaluasi Model":
    st.markdown('<div class="main-title">HASIL PENELITIAN & EVALUASI MODEL</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Eksplorasi Data Eksperimen</div>', unsafe_allow_html=True)
    tabs = st.tabs(["📋 Scraping", "⚙️ Preprocessing", "⚖️ SMOTE", "🔢 TF-IDF"])
    
    with tabs[0]:
        st.dataframe(pd.DataFrame({'Komentar': ["gaji💀", "semenjak ada MBG...", "maaf..."], 'Tanggal': ["2026-02-02", "2026-02-02", "2026-02-02"]}))
    with tabs[1]:
        st.dataframe(pd.DataFrame({'Komentar': ["gaji💀"], 'Final_Text': ["gaji"]}))
    with tabs[2]:
        st.table(pd.DataFrame({'Kondisi': ['Awal', 'Pasca-SMOTE'], 'Jumlah': ['2965', '4284']}))
    with tabs[3]:
        st.dataframe(pd.DataFrame({'adik': [0.0], 'ahli': [0.0], 'Label': [1]}))

    st.markdown('<div class="section-header">Evaluasi Model</div>', unsafe_allow_html=True)
    eval_data = pd.DataFrame({
        'Fold': ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Average'],
        'Accuracy': ['0.8051', '0.7853', '0.7935', '0.7771', '0.8061', '0.7934']
    })
    st.table(eval_data)

    st.markdown('<div class="section-header">Visualisasi Grafik</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    # Pastikan file gambar ada di folder yang sama atau sesuaikan path-nya
    if os.path.exists("wordcloud_hasil_preprocessing.png"): c1.image("wordcloud_hasil_preprocessing.png")
    if os.path.exists("distribusi_hasil_pelabelan_sentimen.png"): c2.image("distribusi_hasil_pelabelan_sentimen.png")
    if os.path.exists("confusion_matrix_5fold.png"): c3.image("confusion_matrix_5fold.png")
