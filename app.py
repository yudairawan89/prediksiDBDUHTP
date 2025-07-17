# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="Dengue Risk Prediction", layout="wide")
st.markdown("""
    <h1 style='color:#0056b3;'>📊 Machine Learning-Based Dengue Risk Prediction Dashboard</h1>
    <p style='font-size:16px'>This tool is designed to detect the level of dengue risk based on environmental, weather, and social data per district. Please upload data in <b>.csv</b> format.</p>
""", unsafe_allow_html=True)

# ================================
# UPLOAD DATA
# ================================
uploaded_file = st.file_uploader("Upload CSV file", type=["CSV"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Uploaded Data")
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
        df['Dengue Risk Prediction'] = prediksi_label

        df_prediksi_only = df[['kecamatan', 'Dengue Risk Prediction']].copy()

        def rekomendasi(label):
            if label == 'Tinggi':
                return [
                    "Scheduled Mass Fogging: Conduct fogging at least twice a week in all subdistrict areas under health authority supervision.",
                    "Enhanced Mosquito Nest Eradication (PSN): Encourage communities to apply the 3M Plus method collectively.",
                    "Emergency Response Post: Establish neighborhood watch teams to report high fever symptoms within 24 hours.",
                    "Intensive Health Education: Implement door-to-door outreach and social media campaigns highlighting symptoms, prevention, and early treatment.",
                    "Regular Larvae Inspections: Conducted by health volunteers and public health centers at least twice a month.",
                    "School Health Screening: Mandatory mosquito larvae checks and distribution of dengue education brochures in schools.",
                    "Cross-Sector Coordination: Engage local leaders, army representatives, and community figures for weekly community clean-ups."
                ]
            elif label == 'Sedang':
                return [
                    "Selective Fogging: Conduct fogging in areas with new cases or water puddles.",
                    "RT/RW Education Strengthening: Distribute leaflets and conduct sessions on prevention and early detection.",
                    "Illegal Dumping Surveillance: Inspect and plan removal or relocation of illegal waste sites.",
                    "Puddle Monitoring: Evaluate and clean blocked drainage systems.",
                    "Active Surveillance: Optimize reporting from public health centers and hospitals.",
                    "School Collaboration: Promote environmental cleanliness competitions and larvae monitoring in classrooms."
                ]
            else:
                return [
                    "Regular Monitoring: Maintain weekly mosquito monitoring and digital reporting if available.",
                    "Light Preventive Campaigns: Use community media and religious facilities to remind people of dengue prevention.",
                    "Community Preparedness Surveys: Assess readiness of residents and health volunteers in case of case surges.",
                    "Infrastructure Evaluation: Ensure no potential new illegal dumps or blocked water flow.",
                    "Risk Communication Enhancement: Display dengue risk information boards at village and health office centers."
                ]

        df['Recommendations'] = df['Dengue Risk Prediction'].apply(rekomendasi)
        df['latitude'] = df['kecamatan'].map(lambda x: kecamatan_coords.get(x, [0, 0])[0])
        df['longitude'] = df['kecamatan'].map(lambda x: kecamatan_coords.get(x, [0, 0])[1])

        output = df[['kecamatan', 'latitude', 'longitude', 'Dengue Risk Prediction', 'Recommendations']].copy()
        output.insert(0, 'No', range(1, len(output) + 1))

        st.subheader("📌 Dengue Risk Prediction Table")
        st.dataframe(df_prediksi_only)

        with st.expander("📋 Recommended Actions Based on Dengue Risk Level by District"):
            for _, row in output.iterrows():
                warna = {
                    'Tinggi': '#d9534f',
                    'Sedang': '#f0ad4e',
                    'Rendah': '#5cb85c'
                }.get(row['Dengue Risk Prediction'], 'gray')

                st.markdown(f"""
                <details>
                <summary><strong>{row['kecamatan']} — Risk Level: <span style='color:{warna}; font-weight:bold'>{row['Dengue Risk Prediction']}</span></strong></summary>
                <div style='background-color:#f9f9f9; padding: 0.7rem 1rem; border-left: 5px solid {warna}; border-radius: 6px; margin-top: 0.5rem'>
                    <b>🧾 Data Details:</b>
                    <ul>
                        <li>Dengue Cases: {df.loc[_,'jumlah_kasus_dbd']}</li>
                        <li>Rainfall: {df.loc[_,'curah_hujan']} mm</li>
                        <li>Average Temperature: {df.loc[_,'suhu_rata_rata']} °C</li>
                        <li>Water Puddles: {df.loc[_,'jumlah_genangan_air']}</li>
                        <li>Unemployment: {df.loc[_,'pengangguran']} %</li>
                        <li>Education Level: {df.loc[_,'tingkat_pendidikan']} years on average</li>
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
            'Rendah': 'green',
            'Sedang': 'orange',
            'Tinggi': 'red'
        }

        for _, row in output.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=f"{row['kecamatan']}\nRisk: {row['Dengue Risk Prediction']}\n{'; '.join(row['Recommendations'])}",
                color=color_map.get(row['Dengue Risk Prediction'], 'blue'),
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
            label="📥 Download Prediction Results (CSV)",
            data=csv,
            file_name="dengue_risk_predictions.csv",
            mime="text/csv",
        )

    except KeyError:
        st.error("Column names in the CSV file do not match the required format. Ensure it includes: 'kecamatan' and all model features.")
        st.markdown("### Required Columns:")
        st.code(", ".join(fitur + ['kecamatan']))

else:
    st.info("Please upload a dataset to start the prediction process.")
