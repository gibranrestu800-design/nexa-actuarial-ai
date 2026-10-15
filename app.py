import streamlit as st
import pandas as pd
import joblib

# ==============================
# NEXA ACTUARIAL AI
# Dashboard Analisis Risiko
# ==============================

st.set_page_config(
    page_title="NEXA Actuarial AI",
    page_icon="📊",
    layout="wide"
)

# JUDUL
st.title("NEXA Actuarial AI")
st.caption("Dashboard Analisis Risiko Berbasis Aturan")

st.info(
    "NEXA merupakan prototipe pembelajaran untuk "
    "analisis skor risiko. Skor belum divalidasi "
    "dengan data historis."
)

st.divider()

# INPUT NASABAH
st.subheader("Data Pengemudi dan Kendaraan")

col1, col2, col3 = st.columns(3)

with col1:
    umur_pengemudi = st.number_input(
        "Usia pengemudi (tahun)",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

    umur_kendaraan = st.number_input(
        "Usia kendaraan (tahun)",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )

    tenaga_kendaraan = st.number_input(
        "Tenaga kendaraan (VehPower)",
        min_value=1,
        max_value=15,
        value=6,
        step=1
    )

    bonus_malus = st.number_input(
        "Bonus-malus",
        min_value=50,
        max_value=350,
        value=100,
        step=1
    )

with col2:
    area = st.selectbox(
        "Area",
        ["A", "B", "C", "D", "E", "F"]
    )

    merek_kendaraan = st.selectbox(
        "Merek kendaraan",
        ["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"]
    )

    jenis_bahan_bakar = st.selectbox(
        "Jenis bahan bakar",
        ["Regular", "Diesel"]
    )

with col3:
    kepadatan_wilayah = st.number_input(
        "Kepadatan wilayah",
        min_value=1,
        max_value=50000,
        value=100,
        step=10
    )

    wilayah = st.selectbox(
    "Wilayah",
    [
        "R11", "R21", "R22", "R23", "R24", "R25",
        "R26", "R31", "R41", "R42", "R43", "R52",
        "R53", "R54", "R72", "R73", "R74", "R82",
        "R83", "R91", "R93", "R94"
    ]
)
    pendapatan = st.number_input(
    "Pendapatan per bulan (Rp)",
    min_value=0,
    max_value=1000000000,
    value=5000000,
    step=500000
)
    jumlah_klaim = st.number_input(
    "Jumlah klaim sebelumnya",
    min_value=0,
    max_value=100,
    value=0,
    step=1
)

    exposure = st.number_input(
        "Durasi perlindungan (tahun)",
        min_value=0.01,
        max_value=1.0,
        value=1.0,
        step=0.05
    )

# ==============================
# PERHITUNGAN SKOR
# ==============================

risk_score = 0
faktor = []
kontribusi = {}

# FAKTOR UMUR
if umur_pengemudi >= 60:
    skor_umur = 30
    faktor.append("Umur 60 tahun atau lebih (+30)")
elif umur_pengemudi >= 40:
    skor_umur = 20
    faktor.append("Umur 40–59 tahun (+20)")
else:
    skor_umur = 10
    faktor.append("Umur di bawah 40 tahun (+10)")

risk_score += skor_umur
kontribusi["Umur"] = skor_umur

# FAKTOR PENDAPATAN
if pendapatan < 3000000:
    skor_pendapatan = 30
    faktor.append("Pendapatan di bawah Rp3 juta (+30)")
elif pendapatan < 5000000:
    skor_pendapatan = 20
    faktor.append(
        "Pendapatan Rp3 juta hingga di bawah Rp5 juta (+20)"
    )
else:
    skor_pendapatan = 10
    faktor.append("Pendapatan Rp5 juta atau lebih (+10)")

risk_score += skor_pendapatan
kontribusi["Pendapatan"] = skor_pendapatan

# FAKTOR RIWAYAT KLAIM
if jumlah_klaim > 0:
    skor_klaim = 15
    faktor.append("Memiliki riwayat klaim (+15)")
else:
    skor_klaim = 0
    faktor.append("Belum ada klaim sebelumnya (+0)")

risk_score += skor_klaim
kontribusi["Riwayat klaim"] = skor_klaim

# ==============================
# MENENTUKAN KATEGORI RISIKO
# ==============================

if risk_score >= 60:
    risk_level = "Tinggi"
elif risk_score >= 30:
    risk_level = "Sedang"
else:
    risk_level = "Rendah"

# ==============================
# HASIL ANALISIS
# ==============================

st.divider()
st.header("Hasil Analisis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Skor Risiko", risk_score)

with col2:
    st.metric("Jumlah Klaim", jumlah_klaim)

with col3:
    st.metric("Faktor Penilaian", len(faktor))
 # PREDIKSI PROBABILITAS KLAIM DENGAN MODEL ML
st.divider()
st.subheader("Prediksi Probabilitas Klaim (Machine Learning)")

try:
    model = joblib.load("nexa_model_nyata.joblib")

    data_nasabah = pd.DataFrame([{
        "DrivAge": umur_pengemudi,
        "VehAge": umur_kendaraan,
        "VehPower": tenaga_kendaraan,
        "BonusMalus": bonus_malus,
        "Density": kepadatan_wilayah,
        "Exposure": exposure,
        "Area": area,
        "VehBrand": merek_kendaraan,
        "VehGas": jenis_bahan_bakar,
        "Region": wilayah
    }])

    probabilitas = model.predict_proba(data_nasabah)[0][1]

    st.metric(
        "Estimasi Probabilitas Klaim",
        f"{probabilitas:.1%}"
    )

    st.caption(
        "Estimasi eksperimental berdasarkan dataset klaim "
        "asuransi kendaraan. Belum divalidasi untuk keputusan "
        "asuransi nyata atau probabilitas klaim 12 bulan."
    )

except Exception as e:
    st.error(f"Model belum dapat digunakan: {e}")
    # Penjelasan faktor risiko
    st.subheader("Faktor yang Memengaruhi Prediksi")

    kontribusi = []

    for calibrated in model.calibrated_classifiers_:
            pipeline = calibrated.estimator
            scaler = pipeline.named_steps["scaler"]
            logistic = pipeline.named_steps["logistic"]

            nilai_standar = scaler.transform(data_nasabah)[0]
            koefisien = logistic.coef_[0]
            kontribusi.append(nilai_standar * koefisien)

    rata_rata = sum(kontribusi) / len(kontribusi)

    nama_fitur = [
            "Umur",
            "Pendapatan bulanan",
            "Jumlah klaim sebelumnya"
        ]

    for nama, nilai in sorted(
            zip(nama_fitur, rata_rata),
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            arah = (
                "Mendorong skor naik"
                if nilai > 0
                else "Mendorong skor turun"
                if nilai < 0
                else "Kontribusi mendekati nol"
            )
            st.write(f"**{nama}:** {arah}")

except Exception as e:
    st.warning(f"Prediksi ML belum dapat ditampilkan: {e}")   

st.subheader("Kategori Risiko")

if risk_level == "Tinggi":
    st.error("🔴 Risiko Tinggi")
elif risk_level == "Sedang":
    st.warning("🟡 Risiko Sedang")
else:
    st.success("🟢 Risiko Rendah")

# GRAFIK KONTRIBUSI FAKTOR
st.subheader("Kontribusi Faktor Risiko")

# GRAFIK KONTRIBUSI FAKTOR RISIKO
st.subheader("Kontribusi Faktor Risiko")

kontribusi_grafik = pd.DataFrame({
    "Faktor": [
        "Umur",
        "Pendapatan bulanan",
        "Jumlah klaim sebelumnya"
    ],
    "Poin skor": [
        skor_umur,
        skor_pendapatan,
        skor_klaim
    ]
})

st.bar_chart(
    kontribusi_grafik.set_index("Faktor"),
    y_label="Poin skor",
    x_label="Faktor"
)

# PENJELASAN
st.subheader("Faktor yang Memengaruhi")

for item in faktor:
    st.write(f"- {item}")

st.subheader("Interpretasi Hasil")

if risk_level == "Tinggi":
    st.write(
        "Skor masuk kategori tinggi berdasarkan "
        "ambang batas simulasi NEXA."
    )
elif risk_level == "Sedang":
    st.write(
        "Skor masuk kategori sedang berdasarkan "
        "ambang batas simulasi NEXA."
    )
else:
    st.write(
        "Skor masuk kategori rendah berdasarkan "
        "ambang batas simulasi NEXA."
    )

st.info(
    "Catatan: skor ini merupakan hasil aturan "
    "sederhana, bukan probabilitas klaim. "
    "Bobot faktor dan ambang kategori belum "
    "divalidasi menggunakan data historis. "
    "Jangan gunakan hasil ini untuk keputusan "
    "asuransi nyata."
)

st.caption("NEXA Actuarial AI | Prototype v0.2")