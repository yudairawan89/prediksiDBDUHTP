import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# ================================
# LOAD MODEL & PREPROCESSOR
# ================================
model = joblib.load("model_stacking_dbd.pkl")
scaler = joblib.load("scaler_dbd.pkl")
le = joblib.load("label_encoder_dbd.pkl")

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="Prediksi Risiko DBD", layout="wide")
st.title("Dashboard Prediksi Risiko DBD Berbasis Machine Learning")

st.markdown("""
Alat bantu ini dirancang untuk mendeteksi tingkat risiko DBD berdasarkan data lingkungan, cuaca, dan sosial per wilayah.
Silakan unggah data dalam format **.csv**.
""")

# ================================
# UPLOAD DATA
# ================================
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data yang Diupload")
    st.dataframe(df.head())

    # Pastikan urutan fitur sesuai model
    fitur = [
        'jumlah_kasus_dbd', 'curah_hujan', 'jumlah_tps_liar',
        'suhu_rata_rata', 'jumlah_fogging', 'jumlah_genangan_air',
        'kelembaban', 'pengangguran', 'tingkat_pendidikan'
    ]

    try:
        X = df[fitur]
        X_scaled = scaler.transform(X)
        prediksi = model.predict(X_scaled)
        prediksi_label = le.inverse_transform(prediksi)
        df['Prediksi Risiko DBD'] = prediksi_label

        def rekomendasi(label):
            if label == 'Tinggi':
                return 'Prioritaskan fogging & edukasi warga'
            elif label == 'Sedang':
                return 'Pemantauan & penyuluhan ringan'
            else:
                return 'Pertahankan kondisi dan monitoring berkala'

        df['Rekomendasi'] = df['Prediksi Risiko DBD'].apply(rekomendasi)

        output = df[['kecamatan', 'latitude', 'longitude', 'Prediksi Risiko DBD', 'Rekomendasi']].copy()
        output.insert(0, 'No', range(1, len(output) + 1))

        st.subheader("Hasil Prediksi dan Rekomendasi")
        st.dataframe(output.drop(columns=['latitude', 'longitude']))

        # ================================
        # TAMPILKAN PETA WILAYAH
        # ================================
        st.subheader("Visualisasi Peta Risiko DBD")
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11)

        color_map = {
            'Rendah': 'green',
            'Sedang': 'orange',
            'Tinggi': 'red'
        }

        for _, row in output.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=f"{row['kecamatan']}\nRisiko: {row['Prediksi Risiko DBD']}\n{row['Rekomendasi']}",
                color=color_map.get(row['Prediksi Risiko DBD'], 'blue'),
                fill=True,
                fill_opacity=0.7
            ).add_to(m)

        st_data = st_folium(m, width=800, height=500)

        # Unduh hasil
        csv = output.drop(columns=['latitude', 'longitude']).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Prediksi (CSV)",
            data=csv,
            file_name="hasil_prediksi_risiko_dbd.csv",
            mime="text/csv",
        )

    except KeyError:
        st.error("Kolom pada file CSV tidak sesuai. Pastikan menyertakan kolom: kecamatan, latitude, longitude, dan fitur model.")
        st.markdown("### Kolom yang diperlukan:")
        st.code(", ".join(fitur + ['kecamatan', 'latitude', 'longitude']))

else:
    st.info("Silakan unggah data untuk memulai prediksi.")
