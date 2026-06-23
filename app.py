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
    margin-bottom: 10px;
}
.step-title {
    font-weight: bold;
    color: #2563EB;
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

# 3. Fungsi Load Data Distribusi Hasil Riset 
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
    # Mengembalikan total data seimbang pasca-SMOTE (1428 x 3 = 4284) sesuai log Colab
    return pd.DataFrame({'sentiment': ['negatif'] * 1428 + ['positif'] * 1428 + ['netral'] * 1428})

df = load_data()

# ================= SIDEBAR NAVIGATION =================
with st.sidebar:
    st.markdown("### Navigasi Sistem")
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
    
    st.markdown('<div class="section-header">Pipeline & Alur Sistem Analisis</div>', unsafe_allow_html=True)
    st.markdown("Berikut adalah visualisasi rincian data asli dari tahapan eksperimen Google Colab:")

    # ----------------------------------------------------
    # TAHAP 1: PENGUMPULAN DATA
    # ----------------------------------------------------
    st.markdown("""
    <div class="card">
        <span class="step-title">1. Pengumpulan Data (Scraping TikTok)</span><br>
        <p style="margin-top: 5px; margin-bottom: 0px; color: #374151; font-size: 14px;">
        Proses penarikan data komentar masyarakat dari platform TikTok mengenai program <b>Makan Bergizi Gratis (MBG)</b>. 
        Total data awal terkumpul sebanyak <b>3.320 baris data</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sesuai Foto 1 (image_ba34c0.png)
    df_foto1 = pd.DataFrame({
        'Komentar': [
            "gaji💀",
            "semenjak ada MBG aku bisa nabung🤍🙋🏼‍♀️",
            "maaf...",
            "boleh jujur??...",
            "gra² mbg aku"
        ],
        'Tanggal_Komentar': [
            "2026-02-02 23:36:29",
            "2026-02-02 08:24:10",
            "2026-02-02 10:22:26",
            "2026-02-02 10:37:20",
            "2026-02-02 08:16:53"
        ]
    })
    st.dataframe(df_foto1, use_container_width=True)
    st.markdown('<div class="caption-text">Tabel 1.1: Cuplikan dataset mentah hasil scraping awal (Shape: 3320 baris)</div>', unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAHAP 2: PREPROCESSING DATA
    # ----------------------------------------------------
    st.markdown("""
    <div class="card">
        <span class="step-title">2. Preprocessing Data (Pembersihan Teks multi-tahap)</span><br>
        <p style="margin-top: 5px; margin-bottom: 0px; color: #374151; font-size: 14px;">
        Tahapan normalisasi teks mentah menjadi kata baku terstruktur. Proses ini meliputi pembersihan karakter, penyeragaman huruf, perbaikan kata gaul (*normalization*), pemotongan kata (*tokenization*), penghapusan kata tidak penting (*stopword*), dan pencarian kata dasar (*stemming*). Total data bersih setelah reduksi duplikat adalah <b>2.965 baris</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sesuai Foto 2 (image_ba38e0.png)
    df_foto2 = pd.DataFrame({
        'Komentar': ["gaji💀", "semenjak ada MBG aku bisa nabung🤍🙋🏼‍♀️", "maaf...", "boleh jujur??...", "gra² mbg aku"],
        'cleaning': ["gaji", "semenjak ada MBG aku bisa nabung", "maaf", "boleh jujur", "gra mbg aku"],
        'case_folding': ["gaji", "semenjak ada mbg aku bisa nabung", "maaf", "boleh jujur", "gra mbg aku"],
        'normalisasi': ["gaji", "semenjak ada makan bergizi gratis aku bisa nabung", "maaf", "boleh jujur", "gara makan bergizi gratis aku"],
        'tokenisasi': ["[gaji]", "[semenjak, ada, makan, bergizi, gratis, aku, b...]", "[maaf]", "[boleh, jujur]", "[gara, makan, bergizi, gratis, aku]"],
        'stopword': ["[gaji]", "[semenjak, makan, bergizi, gratis, nabung]", "[maaf]", "[jujur]", "[gara, makan, bergizi, gratis]"],
        'stemming': ["gaji", "semenjak makan gizi gratis nabung", "maaf", "jujur", "gara makan gizi gratis"],
        'Final_Text': ["gaji", "semenjak makan gizi gratis nabung", "maaf", "jujur", "gara makan gizi gratis"]
    })
    st.dataframe(df_foto2, use_container_width=True)
    st.markdown('<div class="caption-text">Tabel 1.2: Jalur transformasi pipa data preprocessing teks hingga menjadi text bersih (Final_Text)</div>', unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAHAP 3: PELABELAN SENTIMEN
    # ----------------------------------------------------
    st.markdown("""
    <div class="card">
        <span class="step-title">3. Pelabelan Sentimen (Lexicon-Based Output)</span><br>
        <p style="margin-top: 5px; margin-bottom: 0px; color: #374151; font-size: 14px;">
        Hasil distribusi kelas sentimen yang diperoleh menggunakan pendekatan kamus kata (Lexicon-Based). Data awal memperlihatkan ketimpangan jumlah (imbalance data) yang cukup mencolok antar-kelas.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sesuai Foto 3 (image_ba3fea.png)
    df_foto3 = pd.DataFrame({
        'Sentimen': ['positif', 'negatif', 'netral'],
        'Jumlah Ulasan': [975, 1428, 562],
        'Persentase (%)': [32.88, 48.16, 18.95]
    })
    st.table(df_foto3)
    st.markdown('<div class="caption-text">Tabel 1.3: Ringkasan sebaran sekuensial label hasil kalkulasi pendekatan Lexicon</div>', unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAHAP 4: BALANCE DATA (SMOTE)
    # ----------------------------------------------------
    st.markdown("""
    <div class="card">
        <span class="step-title">4. Balancing Dataset (SMOTE Resampling)</span><br>
        <p style="margin-top: 5px; margin-bottom: 0px; color: #374151; font-size: 14px;">
        Untuk mencegah model bias kognitif ke kelompok mayoritas (Negatif), diterapkan metode <b>SMOTE</b> pada data latih. Hasil penyeimbangan menghasilkan jumlah baris data yang seimbang sempurna, yakni masing-masing <b>1.428 data per kelas</b> (Total: 4.284 baris).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sesuai Foto 4 (image_ba43a5.png)
    df_foto4 = pd.DataFrame({
        'Kondisi Dataset': ['X_original_tfidf (Awal)', 'y_original_mapped (Awal)', 'X_resampled (Pasca-SMOTE)', 'y_resampled (Pasca-SMOTE)'],
        'Dimensi / Shape': ['(2965, 612)', '(2965,)', '(4284, 612)', '(4284,)'],
        'Detail Distribusi Sentimen': [
            "Negatif: 1428 | Positif: 975 | Netral: 562",
            "Negatif: 1428 | Positif: 975 | Netral: 562",
            "Netral: 1428 | Positif: 1428 | Negatif: 1428",
            "Netral: 1428 | Positif: 1428 | Negatif: 1428"
        ]
    })
    st.table(df_foto4)
    st.markdown('<div class="caption-text">Tabel 1.4: Perbandingan dimensi matriks fitur sebelum dan sesudah rekonstruksi SMOTE</div>', unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAHAP 5: EKSTRAKSI FITUR (TF-IDF)
    # ----------------------------------------------------
    st.markdown("""
    <div class="card">
        <span class="step-title">5. Ekstraksi Fitur Teks (Matriks Vektor TF-IDF)</span><br>
        <p style="margin-top: 5px; margin-bottom: 0px; color: #374151; font-size: 14px;">
        Transformasi kata bersih menjadi nilai bobot numerik representatif. Berdasarkan analisis, terbentuk <b>612 fitur kata unik</b> yang menjadi kolom parameter latih untuk Logistic Regression model.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sesuai Foto 5 (image_ba4441.png)
    df_foto5 = pd.DataFrame({
        'adik': [0.0, 0.0, 0.0, 0.0, 0.0],
        'ah': [0.0, 0.0, 0.0, 0.0, 0.0],
        'ahli': [0.0, 0.0, 0.0, 0.0, 0.0],
        'ahli gizi': [0.0, 0.0, 0.0, 0.0, 0.0],
        'ajar': [0.0, 0.0, 0.0, 0.0, 0.0],
        '...': ['...', '...', '...', '...', '...'],
        'warung': [0.0, 0.0, 0.0, 0.0, 0.0],
        'ya allah': [0.0, 0.0, 0.0, 0.0, 0.0],
        'ya makan': [0.0, 0.0, 0.0, 0.0, 0.0],
        'Label': [1, 2, 0, 0, 0]
    }, index=[0, 1, 2, 3, 4])
    st.dataframe(df_foto5, use_container_width=True)
    st.markdown('<div class="caption-text">Tabel 1.5: Struktur matriks bobot TF-IDF (klasifikasi_tfidf_features.csv) berdimensi 5 baris &times; 613 kolom</div>', unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAHAP 6: VALIDASI EVALUASI (5-FOLD CROSS VALIDATION)
    # ----------------------------------------------------
    st.markdown("""
    <div class="card">
        <span class="step-title">6. Hasil Akurasi Validasi Kinerja (Multinomial Logistic Regression 5-Fold)</span><br>
        <p style="margin-top: 5px; margin-bottom: 0px; color: #374151; font-size: 14px;">
        Metrik performa yang diperoleh melalui skema iterasi 5-Fold Cross Validation secara stabil.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sesuai Foto 5-Fold Utama (image_b95a6f.png / image_b9b0ca.png)
    eval_data = {
        'Fold': ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Average'],
        'Accuracy': ['0.8051', '0.7853', '0.7935', '0.7771', '0.8061', '0.7934'],
        'Precision': ['0.8125', '0.7897', '0.8023', '0.7826', '0.8115', '0.7997'],
        'Recall': ['0.8051', '0.7853', '0.7935', '0.7771', '0.8061', '0.7934'],
        'F1-Score': ['0.8045', '0.7852', '0.7928', '0.7770', '0.8061', '0.7931']
    }
    st.table(pd.DataFrame(eval_data))
    st.markdown('<div class="caption-text">Tabel 1.6: Log performa akurasi final pengujian model</div>', unsafe_allow_html=True)

# ================= PAGE 2: HASIL PENELITIAN & EVALUASI MODEL =================
elif page == "📊 Hasil Penelitian & Evaluasi Model":
    st.markdown('<div class="main-title">HASIL PENELITIAN & EVALUASI MODEL</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Eksplorasi Data, Distribusi Kelas, dan Metrik Kinerja Akurasi</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Metrik Ringkasan Dataset Pasca-SMOTE (Overview)</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data Uji Evaluasi", "4.284 Baris")
    col2.metric("Sentimen Positif (Label 2)", "1.428 (33.3%)")
    col3.metric("Sentimen Netral (Label 1)", "1.428 (33.3%)")
    col4.metric("Sentimen Negatif (Label 0)", "1.428 (33.3%)")
    
    st.markdown('<div class="section-header">Visualisasi Hasil Penelitian & Evaluasi Model</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["☁️ WordCloud", "📊 Distribusi Sentimen", "🎯 Confusion Matrix"])
    
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
