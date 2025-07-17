import streamlit as st
st.set_page_config(page_title="Dengue Risk Prediction", layout="wide")

import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# ================================
# LOAD MODEL & PREPROCESSORS
# ================================
model = joblib.load("model_stacking_dbd.pkl")
scaler = joblib.load("scaler_dbd.pkl")
le = joblib.load("label_encoder_dbd.pkl")

# ================================
# DEFAULT COORDINATES PER DISTRICT
# ================================
district_coords = {
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
st.markdown("""
    <h1 style='color:#0056b3;'>📊 Machine Learning-Based Dengue Risk Prediction Dashboard</h1>
    <p style='font-size:16px'>This tool helps detect dengue risk levels based on environmental, weather, and social indicators by district. Please upload data in <b>.csv</b> format.</p>
""", unsafe_allow_html=True)

# ================================
# UPLOAD CSV
# ================================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Uploaded Data Preview")
    st.dataframe(df.head())

    features = [
        'dengue_cases', 'rainfall', 'illegal_dump_sites',
        'avg_temperature', 'fogging_events', 'water_puddles',
        'humidity', 'unemployment_rate', 'education_years'
    ]

    try:
        X = df[features]
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        preds_label = le.inverse_transform(preds)

        # Convert model labels from Bahasa to English
        label_map = {'Tinggi': 'High', 'Sedang': 'Moderate', 'Rendah': 'Low'}
        df['Dengue Risk Prediction'] = [label_map.get(lbl, lbl) for lbl in preds_label]
        df_preds_only = df[['district', 'Dengue Risk Prediction']].copy()

        def recommendation(label):
            if label == 'High':
                return [
                    "Scheduled Mass Fogging: At least twice a week across the district with health authority supervision.",
                    "Enhanced Mosquito Nest Eradication (3M Plus) campaigns.",
                    "Emergency Response Posts in neighborhoods with 24-hour symptom reporting.",
                    "Intensive door-to-door and social media awareness.",
                    "Regular larval inspections twice a month by health volunteers.",
                    "School health inspections and educational materials distribution.",
                    "Cross-sector weekend cleaning involving local leaders and communities."
                ]
            elif label == 'Moderate':
                return [
                    "Selective fogging in recent case areas or with standing water.",
                    "Community awareness campaigns and leaflet distribution.",
                    "Inspection and management of illegal waste sites.",
                    "Drainage system inspections to prevent stagnant water.",
                    "Optimize case reporting from health centers.",
                    "School competitions on cleanliness and larval checks."
                ]
            else:
                return [
                    "Routine mosquito monitoring by community volunteers.",
                    "Light educational campaigns via community media.",
                    "Assess community readiness for potential outbreaks.",
                    "Ensure no new illegal dumping or blocked water flow.",
                    "Display dengue risk info boards at community centers."
                ]

        df['Recommendations'] = df['Dengue Risk Prediction'].apply(recommendation)
        df['latitude'] = df['district'].map(lambda x: district_coords.get(x, [0, 0])[0])
        df['longitude'] = df['district'].map(lambda x: district_coords.get(x, [0, 0])[1])

        output = df[['district', 'latitude', 'longitude', 'Dengue Risk Prediction', 'Recommendations']].copy()
        output.insert(0, 'No', range(1, len(output) + 1))

        st.subheader("📌 Dengue Risk Prediction Table")
        st.dataframe(df_preds_only)

        with st.expander("📋 Recommendations Based on Risk Level per District"):
            for _, row in output.iterrows():
                color = {
                    'High': '#d9534f',
                    'Moderate': '#f0ad4e',
                    'Low': '#5cb85c'
                }.get(row['Dengue Risk Prediction'], 'gray')

                st.markdown(f"""
                <details>
                <summary><strong>{row['district']} — Risk Level: <span style='color:{color}; font-weight:bold'>{row['Dengue Risk Prediction']}</span></strong></summary>
                <div style='background-color:#f9f9f9; padding: 0.7rem 1rem; border-left: 5px solid {color}; border-radius: 6px; margin-top: 0.5rem'>
                    <b>🧾 Data Summary:</b>
                    <ul>
                        <li>Dengue Cases: {df.loc[_,'dengue_cases']}</li>
                        <li>Rainfall: {df.loc[_,'rainfall']} mm</li>
                        <li>Average Temperature: {df.loc[_,'avg_temperature']} °C</li>
                        <li>Water Puddles: {df.loc[_,'water_puddles']}</li>
                        <li>Unemployment Rate: {df.loc[_,'unemployment_rate']} %</li>
                        <li>Education Level: {df.loc[_,'education_years']} yrs</li>
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
            'Moderate': 'orange',
            'High': 'red'
        }

        for _, row in output.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=f"{row['district']}\nRisk: {row['Dengue Risk Prediction']}\n{'; '.join(row['Recommendations'])}",
                color=color_map.get(row['Dengue Risk Prediction'], 'blue'),
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.DivIcon(html=f"""
                    <div style='font-size: 10pt; color: black'><b>{row['district']}</b></div>
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
        st.error("Missing or incorrect columns. Ensure your CSV includes: 'district' and all required model features.")
        st.markdown("### Required Columns:")
        st.code(", ".join(features + ['district']))

else:
    st.info("Please upload data to begin dengue risk prediction.")
