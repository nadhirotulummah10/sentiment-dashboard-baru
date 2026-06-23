# ================= PAGE 2: HASIL PENELITIAN & EVALUASI =================
elif page == "📊 Hasil Penelitian & Eksperimen":
    st.markdown('<div class="main-title">HASIL PENELITIAN & EVALUASI MODEL</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Eksplorasi Data, Distribusi Kelas, dan Metrik Kinerja Akurasi</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Metrik Ringkasan Dataset (Overview)</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    total_data = len(df)
    pos_data = len(df[df["sentiment"] == "positif"])
    net_data = len(df[df["sentiment"] == "netral"])
    neg_data = len(df[df["sentiment"] == "negatif"])
    
    # Persentase dibulatkan ke bilangan bulat
    p_pos = (pos_data/max(1, total_data)*100)
    p_net = (net_data/max(1, total_data)*100)
    p_neg = (neg_data/max(1, total_data)*100)
    
    col1.metric("Total Dataset", f"{total_data} Baris")
    col2.metric("Sentimen Positif (Label 2)", f"{pos_data} ({p_pos:.0f}%)")
    col3.metric("Sentimen Netral (Label 1)", f"{net_data} ({p_net:.0f}%)")
    col4.metric("Sentimen Negatif (Label 0)", f"{neg_data} ({p_neg:.0f}%)")
    
    st.markdown('<div class="section-header">Visualisasi Hasil Eksperimen & Evaluasi Model</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["☁️ WordCloud", "📊 Distribusi Sentimen", "🎯 Confusion Matrix", "📈 Metrik Evaluasi (5-Fold)"])
    
    with tab1:
        st.write("**WordCloud Hasil Preprocessing**")
        if os.path.exists("wordcloud_hasil_preprocessing.png"):
            st.image("wordcloud_hasil_preprocessing.png", use_column_width=True)
            
    with tab2:
        st.write("**Distribusi Hasil Pelabelan Sentimen**")
        if os.path.exists("distribusi_hasil_pelabelan_sentimen.png"):
            st.image("distribusi_hasil_pelabelan_sentimen.png", use_column_width=True)
            
    with tab3:
        st.write("**Evaluasi 5-Fold Cross Validation (Confusion Matrix)**")
        if os.path.exists("confusion_matrix_5fold.png"):
            st.image("confusion_matrix_5fold.png", use_column_width=True)

    with tab4:
        st.write("**Tabel Metrik Performa Model - Logistic Regression (5-Fold Cross Validation)**")
        
        # Data disesuaikan persis dengan screenshot yang Anda berikan
        eval_data = {
            'Fold': ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Average'],
            'Accuracy': [0.8051, 0.7853, 0.7935, 0.7771, 0.8061, 0.7934],
            'Precision': [0.8125, 0.7897, 0.8023, 0.7826, 0.8115, 0.7997],
            'Recall': [0.8051, 0.7853, 0.7935, 0.7771, 0.8061, 0.7934],
            'F1-Score': [0.8045, 0.7852, 0.7928, 0.7770, 0.8061, 0.7931]
        }
        st.table(pd.DataFrame(eval_data))
        st.success("🎯 Model Multinomial Logistic Regression menunjukkan performa yang stabil dengan akurasi rata-rata 79.34%.")
