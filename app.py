import streamlit as st
import pandas as pd
import os

# 1. Set page config paling atas
st.set_page_config(
    page_title="Sistem Analisis Sentimen TikTok",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Modern Layout
st.markdown("""
<style>
.main-title {
    font-size: 32px;
    font-weight: bold;
    color: #1E3A8A;
    margin-bottom: 5px;
}
.subtitle {
    font-size: 16px;
    color: #4B5563;
    margin-bottom: 25px;
}
.section-header {
    font-size: 22px;
    font-weight: bold;
    color: #1E3A8A;
    border-bottom: 2px solid #E5E7EB;
    padding-bottom: 8px;
    margin-top: 20px;
    margin-bottom: 15px;
}
.card {
    background-color: #F9FAFB;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #3B82F6;
    margin-bottom: 15px;
}
.step-title {
    font-weight: bold;
    color: #2563EB;
    font-size: 16px;
}
.caption-text {
    font-size: 13px;
    color: #6B7280;
    margin-top: -5px;
    margin-bottom: 15px;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR NAVIGATION =================
with st.sidebar:
    st.markdown("### Navigasi Sistem")
    page = st.radio(
        "Pilih Halaman:",
        ["🏠 Beranda & Alur", "📊 Hasil Penelitian & Evaluasi Model"]
    )
    st.markdown("---")
    st.markdown("**Metodologi:**\nMultinomial Logistic Regression dengan Ekstraksi Fitur TF-IDF.")

# ================= PAGE 1: HOME & ALUR (HANYA PENJELASAN) =================
if page == "🏠 Beranda & Alur":
    st.markdown('<div class="main-title">ANALISIS SENTIMEN KOMENTAR TIKTOK</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Sistem Prototype Klasifikasi Sentimen Masyarakat Menggunakan Multinomial Logistic Regression & TF-IDF</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Pipeline & Alur Sistem Analisis</div>', unsafe_allow_html=True)
    st.markdown("Berikut adalah tahapan pemrosesan data dalam penelitian analisis sentimen program *Makan Bergizi Gratis*:")

    st.markdown("""
    <div class="card">
        <span class="step-title">1. Pengumpulan Data (Scraping TikTok)</span><br>
        <p style="margin-top: 5px; margin-bottom: 0px; color: #374151; font-size: 14px;">
        Proses penarikan data komentar dari platform TikTok menggunakan kata kunci terkait program <b>Makan Bergizi Gratis (MBG)</b>. 
        Data mentah ini menangkap opini, tanggal, dan ekspresi asli masyarakat.
        </p>
    </div>
    
    <div class="card">
        <span class="step-title">2. Preprocessing Data (Pembersihan Teks)</span><br>
        <p style="margin-top: 5px; margin-bottom: 0px; color: #374151; font-size: 14px;">
        Mengubah teks tidak terstruktur menjadi token bersih siap olah. Tahapan meliputi:
        <b>Case Folding</b> (menyeragamkan huruf kecil), <b>Cleaning</b> (hapus emoji/simbol), 
        <b>Normalization</b> (mengubah kata gaul ke baku), <b>Tokenizing</b> (pemotongan kata), 
        <b>Stopword Removal</b> (buang kata tidak penting), dan <b>Stemming</b> (mencari kata dasar).
        </p>
    </div>
    
    <div class="card">
        <span class="step-title">3. Pelabelan Sentimen (Lexicon-Based Labeling)</span><br>
        <p style="margin-top: 5px; margin-bottom: 0px; color: #374151; font-size: 14px;">
        Proses penentuan kelas awal sentimen menjadi kelompok <b>Positif</b>, <b>Netral</b>, atau <b>Negatif</b> secara otomatis berdasarkan kecocokan skor bobot pada kamus kata (*Lexicon Dictionary*).
        </p>
    </div>
    
    <div class="card">
        <span class="step-title">4. Penyeimbangan Data (SMOTE Resampling)</span><br>
        <p style="margin-top: 5px; margin-bottom: 0px; color: #374151; font-size: 14px;">
        Mengatasi ketimpangan sebaran data awal (*data imbalance*) di mana salah satu kelas terlalu mendominasi. 
        Metode <b>SMOTE</b> mensintesis data baru pada kelas minoritas agar distribusi jumlah data menjadi seimbang sempurna sebelum masuk ke tahap pelatihan model.
        </p>
    </div>
    
    <div class="card">
        <span class="step-title">5. Ekstraksi Fitur (TF-IDF Vectorizer)</span><br>
        <p style="margin-top: 5px; margin-bottom: 0px; color: #374151; font-size: 14px;">
        Mentransformasikan kata-kata tekstual menjadi bentuk matriks pembobotan numerik berdasarkan frekuensi kemunculan kata di dalam sebuah komentar (*Term Frequency*) dan kelangkaannya di seluruh dokumen (*Inverse Document Frequency*).
        </p>
    </div>
    
    <div class="card">
        <span class="step-title">6. Klasifikasi & Evaluasi (Multinomial Logistic Regression 5-Fold)</span><br>
        <p style="margin-top: 5px; margin-bottom: 0px; color: #374151; font-size: 14px;">
        Melatih model statistik regresi logistik multinomial untuk mengenali pola bobot fitur TF-IDF terhadap 3 kelas sentimen. Kinerja model divalidasi menggunakan pengujian silang <b>5-Fold Cross Validation</b> guna menjamin keakuratan yang objektif dan stabil.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ================= PAGE 2: HASIL PENELITIAN & DATA ASLI COLAB =================
elif page == "📊 Hasil Penelitian & Evaluasi Model":
    st.markdown('<div class="main-title">HASIL PENELITIAN & EVALUASI MODEL</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Kompilasi Data Eksperimen Asli dan Hasil Pengujian Validasi Model</div>', unsafe_allow_html=True)
    
    # --- RINGKASAN METRIK UTAMA ---
    st.markdown('<div class="section-header">Metrik Ringkasan Dataset Pasca-SMOTE (Overview)</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data Uji (Pasca-SMOTE)", "4.284 Baris")
    col2.metric("Sentimen Negatif (Label 0)", "1.428 (33.3%)")
    col3.metric("Sentimen Netral (Label 1)", "1.428 (33.3%)")
    col4.metric("Sentimen Positif (Label 2)", "1.428 (33.3%)")
    
    # --- TABEL DATASET EKSPERIMEN ---
    st.markdown('<div class="section-header">Eksplorasi Data Berdasarkan Tahapan Eksperimen</div>', unsafe_allow_html=True)
    
    exp_tab1, exp_tab2, exp_tab3, exp_tab4 = st.tabs([
        "📋 Data Scraped Awal", 
        "⚙️ Hasil Preprocessing", 
        "⚖️ Hasil Balancing (SMOTE)", 
        "🔢 Matriks Fitur TF-IDF"
    ])
    
    with exp_tab1:
        st.write("**Sampel Dataset Hasil
