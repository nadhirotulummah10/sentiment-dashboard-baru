import streamlit as st
import pandas as pd
import os

# ... (bagian set_page_config dan CSS Anda tetap di sini) ...

# ================= SIDEBAR NAVIGATION =================
with st.sidebar:
    st.markdown("### Navigasi Sistem")
    page = st.radio(
        "Pilih Halaman:",
        ["🏠 Beranda & Alur", "📊 Hasil Penelitian & Eksperimen"]
    )

# ================= KONDISI UTAMA =================
# Pastikan Halaman 1 dimulai dengan IF
if page == "🏠 Beranda & Alur":
    st.markdown('<div class="main-title">ANALISIS SENTIMEN KOMENTAR TIKTOK</div>', unsafe_allow_html=True)
    # ... (isi konten beranda Anda) ...

# Halaman 2 HARUS dimulai dengan ELIF
elif page == "📊 Hasil Penelitian & Eksperimen":
    st.markdown('<div class="main-title">HASIL PENELITIAN & EVALUASI MODEL</div>', unsafe_allow_html=True)
    # ... (isi konten hasil penelitian Anda) ...
