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
# KOORDINAT DEFAULT PER KECAMATAN
# ================================
kecamatan_coords = {
    "Sukajadi": [-0.5176592, 101.4367539],
    "Senapelan": [-0.5361938, 101.4367539],
    "Pekanbaru Kota": [-0.5070677, 101.4477793],
    "Rumbai Pesisir": [-0.6079579, 101.5020752],
    "Rumbai": [-0.6453205, 101.4112049],
    "Lima Puluh": [-0.2490762, 100.6120232],
    "Sail": [-0.5176177, 101.4594696],
    "Bukit Raya": [-0.4689961, 101.4679893],
    "Marpoyan Damai": [-0.4736702, 101.4395931],
    "Tenayan Raya": [-0.4966231, 101.5475409],
    "Tampan": [-0.4691089, 101.3998518],
    "Payung Sekaki": [-0.5246769, 101.3998518]
}

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
uploaded_file = st.file_uploader("Unggah file CSV", type=["CSV"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data yang Diupload")
    st.dataframe(df.head())

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
                return 'Lakukan fogging massal, aktifkan posko siaga DBD, dan edukasi intensif berbasis RT/RW.'
            elif label == 'Sedang':
                return 'Lakukan pemantauan wilayah rawan, penguatan edukasi sekolah dan penyuluhan warga.'
            else:
                return 'Lanjutkan monitoring berkala, evaluasi lingkungan, dan edukasi ringan berbasis komunitas.'

        df['Rekomendasi'] = df['Prediksi Risiko DBD'].apply(rekomendasi)

        df['latitude'] = df['kecamatan'].map(lambda x: kecamatan_coords.get(x, [0, 0])[0])
        df['longitude'] = df['kecamatan'].map(lambda x: kecamatan_coords.get(x, [0, 0])[1])

        output = df[['kecamatan', 'latitude', 'longitude', 'Prediksi Risiko DBD', 'Rekomendasi']].copy()
        output.insert(0, 'No', range(1, len(output) + 1))

        st.subheader("Ringkasan Prediksi Risiko DBD per Kecamatan")
        for _, row in output.iterrows():
            with st.expander(f"{row['kecamatan']} — Risiko: {row['Prediksi Risiko DBD']}"):
                if st.button(f"Melihat Rekomendasi ({row['kecamatan']})", key=_):
                    st.markdown(f"""
                    **Rekomendasi Intervensi:**
                    {row['Rekomendasi']}

                    **Detail Data:**
                    - Jumlah Kasus DBD: {df.loc[_,'jumlah_kasus_dbd']}
                    - Curah Hujan: {df.loc[_,'curah_hujan']} mm
                    - Suhu Rata-rata: {df.loc[_,'suhu_rata_rata']} °C
                    - Genangan Air: {df.loc[_,'jumlah_genangan_air']}
                    - Pengangguran: {df.loc[_,'pengangguran']} %
                    - Pendidikan: {df.loc[_,'tingkat_pendidikan']} tahun rata-rata
                    
                    """)

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
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.DivIcon(html=f"""
                    <div style='font-size: 10pt; color: black'><b>{row['kecamatan']}</b></div>
                """),
            ).add_to(m)

        st_data = st_folium(m, width=800, height=500)

        csv = output.drop(columns=['latitude', 'longitude']).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Prediksi (CSV)",
            data=csv,
            file_name="hasil_prediksi_risiko_dbd.csv",
            mime="text/csv",
        )

    except KeyError:
        st.error("Kolom pada file CSV tidak sesuai. Pastikan menyertakan kolom: kecamatan dan fitur model.")
        st.markdown("### Kolom yang diperlukan:")
        st.code(", ".join(fitur + ['kecamatan']))

else:
    st.info("Silakan unggah data untuk memulai prediksi.")
