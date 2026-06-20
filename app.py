elif st.session_state.page == "hasil":

    st.title("HASIL PENELITIAN")

    # ================= CEK DATA =================
    try:
        st.write("Preview Data")
        st.dataframe(df.head())

        st.write("Kolom dataset:", df.columns)

    except Exception as e:
        st.error("Data tidak bisa dibaca")
        st.stop()

    st.write("---")

    # ================= CEK KOLOM SENTIMEN =================
    if "sentiment" not in df.columns:
        st.error("Kolom 'sentiment' tidak ditemukan di CSV")
        st.stop()

    # ================= HITUNG AMAN =================
    total = len(df)

    positif = len(df[df["sentiment"].astype(str) == "positif"])
    netral = len(df[df["sentiment"].astype(str) == "netral"])
    negatif = len(df[df["sentiment"].astype(str) == "negatif"])

    # ================= OVERVIEW =================
    st.subheader("Overview Dataset")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total", total)
    col2.metric("Positif", positif)
    col3.metric("Netral", netral)
    col4.metric("Negatif", negatif)

    st.write("---")

    # ================= PENJELASAN =================
    st.subheader("Pengumpulan Data")
    st.write("Data komentar TikTok dikumpulkan sebagai dataset penelitian.")

    st.subheader("Preprocessing")
    st.write("Cleaning, tokenizing, stopword removal.")

    st.subheader("Pelabelan")
    st.write("Lexicon-based sentiment analysis digunakan untuk memberi label.")

    st.write("---")

    # ================= VISUALISASI =================
    st.subheader("Visualisasi Hasil")

    if os.path.exists("wordcloud_hasil_preprocessing.png"):
        st.image("wordcloud_hasil_preprocessing.png")

    if os.path.exists("distribusi_hasil_pelabelan_sentimen.png"):
        st.image("distribusi_hasil_pelabelan_sentimen.png")

    if os.path.exists("confusion_matrix_5fold.png"):
        st.image("confusion_matrix_5fold.png")

    st.write("---")

    if st.button("⬅ Kembali"):
        st.session_state.page = "home"
