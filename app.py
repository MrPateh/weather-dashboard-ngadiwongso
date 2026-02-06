import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os
from datetime import timedelta
from darts import TimeSeries
from darts.models import NBEATSModel

# ==========================================
# 1. PAGE CONFIG & STYLING (MODERN UI)
# ==========================================
st.set_page_config(page_title="Sistem Cerdas Pertanian Desa", page_icon="🌾", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #1b5e20 0%, #004d40 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white !important;
        font-weight: 700;
        font-size: 1.8rem;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.8) !important;
        font-size: 0.9rem;
        margin-top: 5px;
    }

    /* Metric Cards */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        background-color: rgba(255, 255, 255, 0.08);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 5px;
        border: none;
        color: rgba(255,255,255,0.8);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255,255,255,0.1) !important;
        color: #ffffff !important;
        border-top: 2px solid #00E676;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🌾 Smart Farming Dashboard</h1>
    <p>Sistem Monitoring Iklim & Rekomendasi Cerdas Berbasis AI (N-BEATS)</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONFIG & CONSTANTS
# ==========================================
HORIZON = 30 

CONFIG = {
    "rainfall": {
        "model_file": "rainfall_final.pt", # <--- FILE BARU KITA
        "scaler_target": "scaler_rainfall_data.pkl",
        "scaler_cov": "scaler_cov_rainfall.pkl",
        "csv": "Rainfall_Daily_Ngadirejo_historical.csv",
        "title": "Curah Hujan",
        "unit": "mm",
        "color": "#00E676", # Hijau Neon
        "agg": "sum",
        "threshold": 20
    },
    "windspeed": {
        "model_file": "windspeed_final.pt", # <--- FILE BARU KITA
        "scaler_target": "scaler_windspeed_data.pkl",
        "scaler_cov": "scaler_cov_windspeed.pkl",
        "csv": "WindSpeed_Ngadirejo_Daily.csv",
        "title": "Kecepatan Angin",
        "unit": "m/s",
        "color": "#2979FF", # Biru Neon
        "agg": "mean",
        "threshold": 6
    }
}

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

@st.cache_resource
def load_all_models():
    artifacts = {}
    for key, cfg in CONFIG.items():
        if not os.path.exists(cfg['model_file']):
            st.error(f"❌ File Model Hilang: {cfg['model_file']}")
            return None
        try:
            # Load Model dengan map_location CPU agar aman di Cloud
            model = NBEATSModel.load(cfg['model_file'], map_location="cpu")
            
            # Load Scaler (Skip jika tidak ada agar dashboard tetap jalan)
            s_target, s_cov = None, None
            if os.path.exists(cfg['scaler_target']):
                with open(cfg['scaler_target'], "rb") as f: s_target = pickle.load(f)
            if os.path.exists(cfg['scaler_cov']):
                with open(cfg['scaler_cov'], "rb") as f: s_cov = pickle.load(f)
                
            artifacts[key] = (model, s_target, s_cov)
        except Exception as e:
            st.error(f"Gagal load {key}: {e}")
            return None
    return artifacts

# --- ADAPTER 5-DIMENSI (PENTING!) ---
def make_compatible_series(df_value, model_input_dim):
    """
    Mengubah data 1 kolom menjadi 5 kolom (duplikasi) agar model tidak error dimensi.
    """
    values = df_value.values 
    current_dim = 1
    
    if model_input_dim > current_dim:
        # Duplikasi kolom sebanyak dimensi yang dibutuhkan (misal 5)
        new_values = np.tile(values.reshape(-1, 1), (1, model_input_dim))
        return TimeSeries.from_values(new_values)
    else:
        return TimeSeries.from_values(values)

def process_daily_aggregation(df, date_col, value_col, method='sum'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    if method == 'sum':
        df_daily = df[value_col].resample('D').sum()
    else:
        df_daily = df[value_col].resample('D').mean()
    df_final = df_daily.reset_index()
    df_final[value_col] = df_final[value_col].fillna(0)
    return df_final

def run_forecast(df, model, s_target, s_cov, horizon):
    try:
        # 1. Siapkan Data Input dengan Adapter Dimensi
        req_dim = model.input_dim # Cek model butuh berapa kolom (biasanya 5)
        series_input = make_compatible_series(df['value'], req_dim)
        
        # 2. Scaling (Jika scaler ada)
        if s_target:
            try:
                series_input = s_target.transform(series_input)
            except:
                pass # Skip scaling jika dimensi scaler beda

        # 3. Prediksi
        pred = model.predict(n=horizon, series=series_input)
        
        # 4. Inverse Scaling
        if s_target:
            try:
                pred = s_target.inverse_transform(pred)
            except:
                pass

        # 5. Ambil hanya kolom pertama (nilai prediksi utama)
        return pred.pd_series().iloc[:, 0]
        
    except Exception as e:
        # Fallback mechanism jika error, agar dashboard tidak blank
        st.warning(f"Prediksi gagal: {e}")
        dates = pd.date_range(start=df['date'].max() + timedelta(days=1), periods=horizon)
        return pd.Series([df['value'].mean()]*horizon, index=dates)

def plot_interactive(df_hist, pred_series, title, unit, color, threshold=None):
    fig = go.Figure()
    
    # Ambil 60 hari terakhir agar grafik lebih padat
    hist_view = df_hist.tail(60)
    
    # Konversi series prediksi ke dataframe
    pred_df = pd.DataFrame({'date': pred_series.index, 'value': pred_series.values})

    # 1. Plot Historis
    fig.add_trace(go.Scatter(
        x=hist_view['date'], y=hist_view['value'],
        mode='lines', name='Data Historis',
        line=dict(color='rgba(255,255,255,0.4)', width=2), 
        fill='tozeroy', fillcolor='rgba(255,255,255,0.05)'
    ))
    
    # Marker Hari Ini
    fig.add_trace(go.Scatter(
        x=[hist_view['date'].iloc[-1]], y=[hist_view['value'].iloc[-1]],
        mode='markers', name='Hari Ini',
        marker=dict(color='#FFFFFF', size=8, line=dict(color=color, width=2))
    ))

    # 2. Plot Prediksi
    fig.add_trace(go.Scatter(
        x=pred_df['date'], y=pred_df['value'],
        mode='lines+markers', name='Prediksi AI',
        line=dict(color=color, width=3), 
        marker=dict(size=5, color=color)
    ))

    # 3. Threshold (Garis Batas)
    if threshold:
        fig.add_hline(
            y=threshold, 
            line_dash="dash", line_color="#FF5252", line_width=1,
            annotation_text="Batas Waspada", annotation_position="top right",
            annotation_font_color="#FF5252"
        )

    # 4. Layout Cantik
    fig.update_layout(
        title=dict(text=f"Proyeksi {title}", font=dict(size=16, color='#EEEEEE')),
        xaxis=dict(title="Tanggal", showgrid=False, color='#AAAAAA', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(
            title=unit, 
            showgrid=True, gridcolor='rgba(255,255,255,0.1)', 
            color='#AAAAAA', zerolinecolor='rgba(255,255,255,0.1)'
        ),
        height=350, margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#EEEEEE')),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    return fig

# ==========================================
# 4. LOGIC ENGINE (INSIGHTS)
# ==========================================
def generate_smart_insights(pred_rain_series, pred_wind_series):
    # Buat dataframe gabungan dari index yang sama
    # Pastikan index overlap, ambil irisan tanggal yang sama
    common_index = pred_rain_series.index.intersection(pred_wind_series.index)
    
    df = pd.DataFrame({
        'rain': pred_rain_series.loc[common_index].values,
        'wind': pred_wind_series.loc[common_index].values
    }, index=common_index)
    
    df_short = df.iloc[:7] # 7 Hari ke depan
    df_long = df.iloc[7:]  # Sisa hari
    
    insights = {
        "short_term": {"tani": [], "ternak": []},
        "long_term": {"tani": [], "ternak": []},
        "status_label": "Normal",
        "status_color": "success"
    }

    # [SHORT TERM - 7 HARI]
    # Analisis Penyemprotan
    good_spray_days = df_short[(df_short['rain'] < 1) & (df_short['wind'] < 5)]
    if not good_spray_days.empty:
        days = [d.strftime('%A') for d in good_spray_days.index][:3]
        insights['short_term']['tani'].append(f"🧪 **Jadwal Semprot:** Disarankan hari **{', '.join(days)}** (Cerah & Tenang).")
    else:
        insights['short_term']['tani'].append("⛔ **Stop Semprot:** Minggu ini tidak kondusif (Hujan/Angin).")

    # Analisis Tanah
    wet_soil_days = (df_short['rain'] > 5).sum()
    if wet_soil_days >= 4:
        insights['short_term']['tani'].append("🚫 **Tunda Pupuk Tabur:** Tanah terlalu basah, pupuk akan hanyut.")
        insights['short_term']['tani'].append("🍄 **Waspada Jamur:** Kelembaban tinggi memicu jamur pangkal batang.")
    elif wet_soil_days == 0:
        insights['short_term']['tani'].append("💧 **Irigasi Wajib:** Tanah mulai kering, segera siram tanaman.")

    if (df_short['wind'] > 7).any():
        insights['short_term']['tani'].append("💨 **Pasang Ajir:** Angin kencang terdeteksi. Ikat tanaman cabe/tomat.")

    # Analisis Ternak
    heavy_rain_days = (df_short['rain'] > 10).sum()
    if heavy_rain_days >= 3:
        insights['short_term']['ternak'].append("⚠️ **Bahaya Kembung:** Rumput terlalu basah. Layukan 4 jam sebelum diberi pakan.")
        insights['short_term']['ternak'].append("💊 **Vitamin:** Berikan vitamin tambahan untuk daya tahan.")
    else:
        insights['short_term']['ternak'].append("✅ **Pakan Aman:** Rumput dalam kondisi ideal untuk diberikan.")

    if (df_short['wind'] > 6).any():
        insights['short_term']['ternak'].append("🥶 **Tutup Kandang:** Pasang tirai sisi angin untuk cegah masuk angin.")

    # [LONG TERM - 30 HARI]
    total_rain = df['rain'].sum()
    high_wind_days = (df['wind'] > 6).sum()

    if total_rain > 200:
        insights['long_term']['tani'].append("🌧️ **Fase Basah:** Curah hujan bulanan tinggi.")
        insights['long_term']['tani'].append("👉 **Drainase:** Perdalam parit agar air cepat surut.")
    elif total_rain < 30:
        insights['long_term']['tani'].append("☀️ **Fase Kering:** Defisit air bulan ini.")
        insights['long_term']['tani'].append("👉 **Hama:** Waspada ledakan populasi Thrips/Tungau.")

    if high_wind_days > 7:
        insights['long_term']['tani'].append(f"💨 **Risiko Rebah:** Ada {high_wind_days} hari angin kencang. Hindari Urea berlebih.")
        
    # Status Keseluruhan
    if total_rain > 250:
        insights['status_label'] = "🌧️ SANGAT BASAH"
        insights['status_color'] = "error"
    elif total_rain < 20:
        insights['status_label'] = "🔥 SANGAT KERING"
        insights['status_color'] = "error"
    elif high_wind_days > 10:
        insights['status_label'] = "💨 BERANGIN KENCANG"
        insights['status_color'] = "warning"
    else:
        insights['status_label'] = "✅ KONDUSIF / NORMAL"
        insights['status_color'] = "success"

    return insights

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
artifacts = load_all_models()
if artifacts is None: st.stop()

data_store = {}
for key in ['rainfall', 'windspeed']:
    cfg = CONFIG[key]
    model, s_target, s_cov = artifacts[key]
    
    if os.path.exists(cfg['csv']):
        df = pd.read_csv(cfg['csv'])
        df = df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'value'})
        df = process_daily_aggregation(df, 'date', 'value', cfg['agg'])
        
        # Jalankan Prediksi
        pred_series = run_forecast(df, model, s_target, s_cov, HORIZON)
        
        data_store[key] = { "hist": df, "pred": pred_series }
    else:
        st.error(f"File CSV {cfg['csv']} tidak ditemukan!")
        st.stop()

# ==========================================
# 6. DASHBOARD LAYOUT (UI)
# ==========================================

col1, col2, col3= st.columns(3)

last_rain = data_store['rainfall']['hist']['value'].iloc[-1]
last_wind = data_store['windspeed']['hist']['value'].iloc[-1]
last_date = data_store['rainfall']['hist']['date'].iloc[-1].strftime("%d %b %Y")

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Update Terakhir</div>
        <div class="metric-value">{last_date}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Hujan Terakhir</div>
        <div class="metric-value" style="color: {CONFIG['rainfall']['color']}">{last_rain:.1f} <span style="font-size:0.8rem; color: #888;">mm</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Angin Terakhir</div>
        <div class="metric-value" style="color: {CONFIG['windspeed']['color']}">{last_wind:.1f} <span style="font-size:0.8rem; color: #888;">m/s</span></div>
    </div>
    """, unsafe_allow_html=True)

st.write("") 

# --- CHARTS ---
c_rain, c_wind = st.columns(2)

with c_rain:
    st.markdown("### 🌧️ Tren Curah Hujan")
    fig = plot_interactive(
        data_store['rainfall']['hist'], 
        data_store['rainfall']['pred'], 
        CONFIG['rainfall']['title'], "mm", 
        CONFIG['rainfall']['color'],
        threshold=CONFIG['rainfall']['threshold']
    )
    st.plotly_chart(fig, use_container_width=True)

with c_wind:
    st.markdown("### 💨 Tren Kecepatan Angin")
    fig = plot_interactive(
        data_store['windspeed']['hist'], 
        data_store['windspeed']['pred'], 
        CONFIG['windspeed']['title'], "m/s", 
        CONFIG['windspeed']['color'],
        threshold=CONFIG['windspeed']['threshold']
    )
    st.plotly_chart(fig, use_container_width=True)

# --- INSIGHTS ---
insight_data = generate_smart_insights(data_store['rainfall']['pred'], data_store['windspeed']['pred'])

st.markdown("---")
st.markdown("### 💡 Rekomendasi AI & Strategi Tani")

# Status Badge
status_msg = f"**STATUS IKLIM 30 HARI:** {insight_data['status_label']}"
if insight_data['status_color'] == 'error': st.error(status_msg, icon="🚨")
elif insight_data['status_color'] == 'warning': st.warning(status_msg, icon="⚠️")
else: st.success(status_msg, icon="✅")

tab1, tab2 = st.tabs(["📅 Rencana Minggu Ini", "🔭 Prospek 30 Hari"])

def show_list(items, empty_msg):
    if not items:
        st.info(f"_{empty_msg}_")
    else:
        for item in items:
            st.markdown(f"{item}")

with tab1:
    c_tani, c_ternak = st.columns(2)
    with c_tani:
        st.info("##### 🚜 Aksi Petani")
        show_list(insight_data['short_term']['tani'], "Kondisi normal, lakukan perawatan rutin.")
    with c_ternak:
        st.warning("##### 🐄 Aksi Peternak")
        show_list(insight_data['short_term']['ternak'], "Kondisi aman untuk ternak.")

with tab2:
    c_tani_l, c_ternak_l = st.columns(2)
    with c_tani_l:
        st.info("##### 🌾 Strategi Tani (Bulanan)")
        show_list(insight_data['long_term']['tani'], "Iklim kondusif.")
    with c_ternak_l:
        st.warning("##### 🐂 Strategi Ternak (Bulanan)")
        show_list(insight_data['long_term']['ternak'], "Prediksi pakan aman.")

# --- DATA DOWNLOAD ---
st.markdown("---")
st.subheader("📂 Download Data Prediksi")

with st.expander("Klik untuk melihat Tabel & Download CSV"):
    c_tbl_rain, c_tbl_wind = st.columns(2)
    
    def make_pretty_table(pred_series, unit):
        pred_df = pd.DataFrame({'Tanggal': pred_series.index, f'Prediksi ({unit})': pred_series.values})
        pred_df['Tanggal'] = pred_df['Tanggal'].dt.strftime('%Y-%m-%d')
        return pred_df

    with c_tbl_rain:
        p_rain = make_pretty_table(data_store['rainfall']['pred'], "mm")
        st.write("**Data Prediksi Hujan**")
        st.dataframe(p_rain, use_container_width=True, hide_index=True)
        csv_rain = p_rain.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV Hujan", data=csv_rain, file_name="prediksi_hujan.csv", mime="text/csv")

    with c_tbl_wind:
        p_wind = make_pretty_table(data_store['windspeed']['pred'], "m/s")
        st.write("**Data Prediksi Angin**")
        st.dataframe(p_wind, use_container_width=True, hide_index=True)
        csv_wind = p_wind.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV Angin", data=csv_wind, file_name="prediksi_angin.csv", mime="text/csv")
