import streamlit as st
import pickle
import pandas as pd

# ===== Styling Sidebar =====
st.markdown("""
    <style>
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f0f4f8;
            padding-top: 20px;
        }
        /* Judul Sidebar */
        .sidebar-title {
            font-size: 22px;
            font-weight: bold;
            color: #2c3e50;
            padding-bottom: 10px;
            text-align: center;
        }
        /* Radio button */
        .stRadio > label {
            font-weight: bold;
            color: #34495e;
        }
    </style>
""", unsafe_allow_html=True)

# ===== Sidebar Navigasi =====
st.sidebar.markdown('<div class="sidebar-title">📌 Menu Navigasi</div>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["🏠 Beranda", "📊 Prediksi"])

# ===== Halaman Beranda =====
if page == "🏠 Beranda":
    st.title("📊 Beranda")
    st.markdown("## 🍼 Tentang Stunting")
    st.write("""
    **Stunting** adalah kondisi ketika tinggi badan anak lebih rendah dari standar usianya akibat kekurangan gizi kronis.
    Kondisi ini dapat mempengaruhi perkembangan fisik maupun kognitif anak.

    **Pencegahan stunting** meliputi:
    - 🍼 Pemberian ASI eksklusif hingga 6 bulan
    - 🍚 MPASI bergizi seimbang
    - 💉 Imunisasi lengkap
    - 🧼 Kebersihan lingkungan
    """)

    st.markdown("## 📏 Penjelasan Status Gizi")
    st.markdown("""
    <div style="background-color:#e8f5e9;padding:10px;border-radius:10px;margin-bottom:10px">
    <b>1. 🟢 Normal</b> — Anak anda sehat.<br>
    <i>Tips:</i> Tetap berikan gizi seimbang, rutin cek kesehatan, dan jaga kebersihan.
    </div>

    <div style="background-color:#fff3e0;padding:10px;border-radius:10px;margin-bottom:10px">
    <b>2. 🟡 Stunted</b> — Anak mengalami hambatan pertumbuhan.<br>
    <i>Tips:</i> Perlu peningkatan asupan gizi dan perbaikan pola makan.
    </div>

    <div style="background-color:#ffebee;padding:10px;border-radius:10px;margin-bottom:10px">
    <b>3. 🔴 Severely Stunted</b> — Anak sangat terhambat pertumbuhannya.<br>
    <i>Tips:</i> Segera konsultasikan ke tenaga kesehatan.
    </div>

    <div style="background-color:#e3f2fd;padding:10px;border-radius:10px">
    <b>4. 🔵 Tinggi</b> — Tinggi badan anak di atas rata-rata usianya.<br>
    <i>Tips:</i> Tetap jaga pola makan bergizi seimbang dan aktivitas fisik yang sesuai.
    </div>
    """, unsafe_allow_html=True)

# ===== Halaman Prediksi =====
elif page == "📊 Prediksi":
    MODEL_FILE = "svm_model.pkl"  

    with open(MODEL_FILE, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        model = obj["model"]
        le_gender = obj.get("encoder_gender", None)
        le_status = obj.get("encoder_status", None)
        scaler = obj.get("scaler", None)
    elif isinstance(obj, (tuple, list)) and len(obj) == 3:
        model, le_gender, scaler = obj
        le_status = None
    else:
        st.error("Format file model tidak dikenali. Harap periksa file pickle.")
        st.stop()

    st.title("🩺 Prediksi Status Gizi Balita")

    gender_display = st.selectbox("Jenis Kelamin", ["laki-laki", "perempuan"])
    umur = st.number_input("Umur (bulan)", 0, 60, 24)
    tinggi = st.number_input("Tinggi Badan (cm)", 40.0, 150.0, 85.0)

    if st.button("Prediksi"):
        df_input = pd.DataFrame({
            "Umur (bulan)": [umur],
            "Jenis Kelamin": [gender_display],
            "Tinggi Badan (cm)": [tinggi]
        })

        if le_gender is not None:
            df_input["Jenis Kelamin"] = le_gender.transform(df_input["Jenis Kelamin"])

        if scaler is not None:
            num_cols = ["Tinggi Badan (cm)", "Umur (bulan)"]
            df_input[num_cols] = scaler.transform(df_input[num_cols])

        y_pred_enc = model.predict(df_input)
        if le_status is not None:
            y_pred = le_status.inverse_transform(y_pred_enc)
        else:
            y_pred = y_pred_enc

        st.success(f"Hasil Prediksi Status Gizi: {y_pred[0]}")
