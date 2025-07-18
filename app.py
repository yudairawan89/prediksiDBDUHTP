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
# DEFAULT COORDINATES PER DISTRICT
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
st.set_page_config(page_title="Dengue Risk Prediction", layout="wide")
st.markdown("""
    <h1 style='color:#0056b3;'>📊 Dengue Risk Prediction Dashboard</h1>
    <p style='font-size:16px'>This tool is designed to assess the risk of Dengue outbreaks based on environmental, weather, and social data per district. Please upload a <b>.csv</b> file.</p>
""", unsafe_allow_html=True)

# ================================
# UPLOAD DATA
# ================================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Column alias for display
    column_aliases = {
        'tanggal': 'Date',
        'kecamatan': 'District',
        'jumlah_kasus_dbd': 'Dengue Cases',
        'jumlah_kematian': 'Deaths',
        'jumlah_fogging': 'Fogging Efforts',
        'curah_hujan': 'Rainfall (mm)',
        'kelembaban': 'Humidity (%)',
        'suhu_rata_rata': 'Avg. Temperature (°C)',
        'jumlah_tps_liar': 'Illegal Dump Sites',
        'jumlah_genangan_air': 'Water Puddles',
        'kepadatan_penduduk': 'Population Density',
        'pengangguran': 'Unemployment (%)',
        'tingkat_pendidikan': 'Education Level (yrs)'
    }

    df_display = df.copy()
    df_display.rename(columns=column_aliases, inplace=True)

    st.subheader("📂 Uploaded Data Preview")
    st.dataframe(df_display.head())

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
        df['Dengue Risk Prediction'] = prediksi_label

        # Map Indonesian labels to English
        risk_label_map = {
            'Tinggi': 'High',
            'Sedang': 'Medium',
            'Rendah': 'Low'
        }
        df['Risk Level (EN)'] = df['Dengue Risk Prediction'].map(risk_label_map)

        # Display table with English column names
        df_prediksi_only = df[['kecamatan', 'Risk Level (EN)']].copy()
        df_prediksi_only.rename(columns={'kecamatan': 'District', 'Risk Level (EN)': 'Dengue Risk Prediction'}, inplace=True)

        def recommendations(label):
            if label == 'Tinggi':
                return [
                    "Scheduled Mass Fogging: Conduct fogging at least twice a week in all districts under health authority supervision.",
                    "Community-Based Source Reduction: Encourage collective 3M Plus mosquito prevention.",
                    "Emergency Dengue Posts: Form RT/RW response teams with 24-hour fever symptom reporting.",
                    "Intensive Public Education: Disseminate key messages door-to-door and via social media.",
                    "Larvae Inspection Campaigns: Conduct inspections twice monthly by health workers or trained volunteers.",
                    "School Health Screening: Mandatory larvae inspections and dengue education brochures in schools.",
                    "Cross-Sector Coordination: Involve community leaders in weekly clean-up programs."
                ]
            elif label == 'Sedang':
                return [
                    "Selective Fogging: Focus on areas with recent cases or stagnant water.",
                    "RT/RW Health Education: Distribute leaflets and conduct early detection training.",
                    "Illegal Waste Monitoring: Inspect and relocate informal dump sites.",
                    "Drainage System Inspection: Clean and assess water flow barriers.",
                    "Active Surveillance: Strengthen Puskesmas and hospital-based case reporting.",
                    "School Collaboration: Promote cleanliness competitions and mosquito control in classrooms."
                ]
            else:
                return [
                    "Routine Monitoring: Continue weekly mosquito surveillance and digital reporting if available.",
                    "Preventive Messaging: Use local media and mosques to reinforce dengue prevention.",
                    "Community Readiness Surveys: Assess preparedness in case of outbreak.",
                    "Infrastructure Audit: Check for new waste hotspots or potential mosquito breeding sites.",
                    "Risk Communication Boards: Display dengue status at public offices and clinics."
                ]

        df['Recommendations'] = df['Dengue Risk Prediction'].apply(recommendations)
        df['latitude'] = df['kecamatan'].map(lambda x: kecamatan_coords.get(x, [0, 0])[0])
        df['longitude'] = df['kecamatan'].map(lambda x: kecamatan_coords.get(x, [0, 0])[1])

        output = df[['kecamatan', 'latitude', 'longitude', 'Risk Level (EN)', 'Recommendations']].copy()
        output.insert(0, 'No', range(1, len(output) + 1))

        st.subheader("📌 Dengue Risk Level Prediction Table")
        st.dataframe(df_prediksi_only)

        with st.expander("📋 Actionable Recommendations Based on Dengue Risk Level"):
            for _, row in output.iterrows():
                color = {
                    'High': '#d9534f',
                    'Medium': '#f0ad4e',
                    'Low': '#5cb85c'
                }.get(row['Risk Level (EN)'], 'gray')

                st.markdown(f"""
                <details>
                <summary><strong>{row['kecamatan']} — Risk Level: <span style='color:{color}; font-weight:bold'>{row['Risk Level (EN)']}</span></strong></summary>
                <div style='background-color:#f9f9f9; padding: 0.7rem 1rem; border-left: 5px solid {color}; border-radius: 6px; margin-top: 0.5rem'>
                    <b>🧾 Data Details:</b>
                    <ul>
                        <li>Dengue Cases: {df.loc[_,'jumlah_kasus_dbd']}</li>
                        <li>Rainfall: {df.loc[_,'curah_hujan']} mm</li>
                        <li>Avg. Temperature: {df.loc[_,'suhu_rata_rata']} °C</li>
                        <li>Water Puddles: {df.loc[_,'jumlah_genangan_air']}</li>
                        <li>Unemployment: {df.loc[_,'pengangguran']} %</li>
                        <li>Education Level: {df.loc[_,'tingkat_pendidikan']} years</li>
                    </ul>
                    <b>📌 Recommended Actions:</b>
                    <ol>
                        {''.join([f'<li>{s}</li>' for s in row['Recommendations']])}
                    </ol>
                </div>
                </details>
                """, unsafe_allow_html=True)

        st.markdown("### 🗺️ Dengue Risk Map Visualization")
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11)

        color_map = {
            'Low': 'green',
            'Medium': 'orange',
            'High': 'red'
        }

        for _, row in output.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=f"{row['kecamatan']}\nRisk: {row['Risk Level (EN)']}\n{'; '.join(row['Recommendations'])}",
                color=color_map.get(row['Risk Level (EN)'], 'blue'),
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.DivIcon(html=f"""
                    <div style='font-size: 10pt; color: black'><b>{row['kecamatan']}</b></div>
                """),
            ).add_to(m)

        st_folium(m, width=800, height=500)

        csv = output.drop(columns=['latitude', 'longitude']).rename(
            columns={'kecamatan': 'District', 'Risk Level (EN)': 'Dengue Risk Prediction'}
        ).to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Prediction Results (CSV)",
            data=csv,
            file_name="dengue_risk_prediction.csv",
            mime="text/csv",
        )

    except KeyError:
        st.error("The uploaded CSV is missing required columns.")
        st.markdown("### Required columns:")
        st.code(", ".join(fitur + ['kecamatan']))

else:
    st.info("Please upload your data file to begin prediction.")
