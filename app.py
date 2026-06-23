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
}
</style>
""", unsafe_allow_html=True)

# 3. Fungsi Load Data Distribusi Hasil Riset (Total 2965 baris bersih atau dummy default 4284 sesuai support model)
@st.cache_data
def load_data():
    file_path = "hasil_labeling_sentimen.csv"
    if os.path.exists(file_path):
        try:
            data = pd.read_csv(file_path)
            data.columns = data.columns.str.strip()
            
            target_col = None
            for c in data.columns:
                if c.lower() in ['y_original_mapped', 'sentiment', 'label', 'prediksi']:
                    target_col = c
                    break
            
            if target_col:
                def map_label(val):
                    v_str = str(val).strip().lower()
                    if v_str in ['0', '0.0', 'negatif', 'negative']: return 'negatif'
                    elif v_str in ['1', '1.0', 'netral', 'neutral']: return 'netral'
                    elif v_str in ['2', '2.0', 'positif', 'positive']: return 'positif'
                    return 'netral'
                data['sentiment'] = data[target_col].apply(map_label)
                return data
        except:
            pass
    # Menyesuaikan total support 4284 data seimbang setelah SMOTE (1428 x 3) sesuai classification report terbarumu
    return pd.DataFrame({'sentiment': ['negatif'] * 1428 + ['positif'] * 1428 + ['netral'] * 1428})

df = load_data()

# ================= SIDEBAR NAVIGATION =================
with st.sidebar:
    st.markdown("### Navigasi Sistem")
    # DI SINI SUDAH DISINKRONKAN DENGAN STRIP EMOJI BIAR LOGIKA IF JALAN
    page = st.radio(
        "Pilih Halaman:",
        ["🏠 Beranda & Alur", "📊 Hasil Penelitian & Evaluasi Model"]
    )
    st.markdown("---")
    st.markdown("**Metodologi:**\nMultinomial Logistic Regression dengan Ekstraksi Fitur TF-IDF.")

# ================= PAGE 1: HOME & ALUR =================
if page == "🏠 Beranda & Alur":
    st.markdown('<div class="main-title">ANALISIS SENTIMEN KOMENTAR TIKTOK</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Sistem Prototype Klasifikasi Sentimen Masyarakat Menggunakan Multinomial Logistic Regression & TF-IDF</div>', unsafe_allow_html=True)
    
    # --- TABEL PERBANDINGAN PREPROCESSING AGAR BERANDA TIDAK POLOS ---
    st.markdown('<div class="section-header">📋 Sampel Transformasi Data Preprocessing</div>', unsafe_allow_html=True)
    st.markdown("Berikut adalah contoh perbandingan data komentar mentah dari TikTok sebelum dan sesudah melalui tahapan *text preprocessing* di Google Colab:")
    
    sample_data = {
        'Komentar Asli (Raw Data)': [
            "Waaah mantap bgt programnya!! moga berkah & tepat sasaran yaa 🥰👍 @jokowi #makanmalam",
            "program makan gratis ini cuma habisin anggaran aja, mending buat fasilitas sekolah keles",
            "Klo menurut gw sih biasa aja ya.. kita liat aja nanti realisasinya gmn pas udh jln."
        ],
        'Hasil Preprocessing (Clean Teks)': [
            "mantap program moga berkah tepat sasaran",
            "program makan gratis habis anggaran mending fasilitas sekolah",
            "biasa lihat realisasi jalan"
        ]
    }
    st.table(pd.DataFrame(sample_data))
    
    st.markdown('<div class="section-header">Pipeline & Alur Sistem Analisis</div>', unsafe_allow_html=True)
    
    steps = [
        ("1. Pengumpulan Data", "Proses scraping data komentar dari platform TikTok menggunakan kata kunci program terkait."),
        ("2. Preprocessing Data", "Tahapan pembersihan data teks (text cleaning) meliputi Case Folding, Removal (URL, angka, username, emoji, tanda baca), Tokenizing, Normalization, dan Filtering/Stopword Removal."),
        ("3. Pelabelan (Labeling)", "Proses penentuan kelas awal sentimen menjadi kelas Positif, Netral, atau Negatif menggunakan pendekatan berbasis kamus kata (Lexicon-Based)."),
        ("4. Balance Data (Penyeimbangan Data)", "Mengatasi ketimpangan sebaran data (data imbalance) antar-kelas sentimen menggunakan SMOTE agar seimbang masing-masing 1.428 data per kelas."),
        ("5. Ekstraksi Fitur TF-IDF", "Metode Term Frequency - Inverse Document Frequency (TF-IDF) diterapkan untuk mentransformasikan token teks menjadi representasi vektor numerik."),
        ("6. Pemodelan (Multinomial Logistic Regression)", "Tahap pelatihan (training) algoritma Logistic Regression multi-kelas untuk mempelajari pola relasi data menggunakan Stratified K-Fold Cross Validation.")
    ]
    
    for title, desc in steps:
        st.markdown(f"""
        <div class="card">
            <span class="step-title">{title}</span><br>
            <p style="margin-top: 5px; margin-bottom: 0px; color: #374151; font-size: 14px;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
    if os.path.exists("images/alur.png"):
        st.image("images/alur.png", caption="Diagram Alir Arsitektur Sistem", use_column_width=True)

# ================= PAGE 2: HASIL PENELITIAN & EVALUASI MODEL =================
elif page == "📊 Hasil Penelitian & Evaluasi Model":
    st.markdown('<div class="main-title">HASIL PENELITIAN & EVALUASI MODEL</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Eksplorasi Data, Distribusi Kelas, dan Metrik Kinerja Akurasi</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Metrik Ringkasan Dataset Pasca-SMOTE (Overview)</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    # Menyesuaikan dengan total seimbang dari Classification Report terbaru (Total: 4284)
    col1.metric("Total Data Uji Evaluasi", "4.284 Baris")
    col2.metric("Sentimen Positif (Label 2)", "1.428 (33.3%)")
    col3.metric("Sentimen Netral (Label 1)", "1.428 (33.3%)")
    col4.metric("Sentimen Negatif (Label 0)", "1.428 (33.3%)")
    
    st.markdown('<div class="section-header">Visualisasi Hasil Penelitian & Evaluasi Model</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["☁️ WordCloud", "📊 Distribusi Sentimen", "🎯 Confusion Matrix", "📈 Metrik Evaluasi (5-Fold)"])
    
    with tab1:
        st.write("**WordCloud Hasil Preprocessing**")
        if os.path.exists("wordcloud_hasil_preprocessing.png"):
            st.image("wordcloud_hasil_preprocessing.png", use_column_width=True)
        else:
            st.info("💡 Grafik [wordcloud_hasil_preprocessing.png] belum terbaca di folder utama.")
            
    with tab2:
        st.write("**Distribusi Hasil Pelabelan Sentimen**")
        if os.path.exists("distribusi_hasil_pelabelan_sentimen.png"):
            st.image("distribusi_hasil_pelabelan_sentimen.png", use_column_width=True)
        else:
            st.info("💡 Grafik [distribusi_hasil_pelabelan_sentimen.png] belum terbaca di folder utama.")
            
    with tab3:
        st.write("**Evaluasi 5-Fold Cross Validation (Confusion Matrix)**")
        if os.path.exists("confusion_matrix_5fold.png"):
            st.image("confusion_matrix_5fold.png", use_column_width=True)
        else:
            st.info("💡 Grafik [confusion_matrix_5fold.png] belum terbaca di folder utama.")

    with tab4:
        st.write("**Tabel Metrik Performa Model - Logistic Regression (5-Fold Cross Validation)**")
        
        # BARU: 100% SINKRON SESUAI SCREENSHOT TERBARU KAMU (image_b95a6f.png / image_b9b0ca.png)
        eval_data = {
            'Fold': ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Average'],
            'Accuracy': ['0.8051', '0.7853', '0.7935', '0.7771', '0.8061', '0.7934'],
            'Precision': ['0.8125', '0.7897', '0.8023', '0.7826', '0.8115', '0.7997'],
            'Recall': ['0.8051', '0.7853', '0.7935', '0.7771', '0.8061', '0.7934'],
            'F1-Score': ['0.8045', '0.7852', '0.7928', '0.7770', '0.8061', '0.7931']
        }
        st.table(pd.DataFrame(eval_data))
        st.success("🎯 Model Multinomial Logistic Regression menunjukkan performa yang optimal dengan nilai rata-rata akurasi sebesar 0.7934.")
