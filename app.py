import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Sistem Cerdas Pertanian Desa", page_icon="🌾", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(90deg, #1b5e20 0%, #004d40 100%);
        padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .main-header h1 { color: white !important; font-weight: 700; font-size: 1.8rem; margin: 0; }
    
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px; padding: 15px; text-align: center;
    }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin: 5px 0; }
    .metric-label { font-size: 0.8rem; color: rgba(255, 255, 255, 0.7); text-transform: uppercase; letter-spacing: 1px; }

    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05); border-radius: 5px;
        border: none; color: rgba(255,255,255,0.8);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255,255,255,0.1) !important;
        color: #ffffff !important; border-top: 2px solid #00E676;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🌾 Smart Farming Dashboard</h1>
    <p>Sistem Monitoring Iklim & Rekomendasi Cerdas Berbasis AI (Live Auto-Update)</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONFIG
# ==========================================
CONFIG = {
    "rainfall": {
        "title": "Curah Hujan", "unit": "mm", "color": "#00E676", "threshold": 20,
        "hist_csv": "Rainfall_Daily_Ngadirejo_historical.csv",
        "pred_csv": "forecast_rainfall_2years.csv", "agg": "sum"
    },
    "windspeed": {
        "title": "Kecepatan Angin", "unit": "m/s", "color": "#2979FF", "threshold": 6,
        "hist_csv": "WindSpeed_Ngadirejo_Daily.csv",
        "pred_csv": "forecast_windspeed_2years.csv", "agg": "mean"
    }
}

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def plot_interactive(df_hist, df_pred, title, unit, color, threshold=None):
    fig = go.Figure()
    hist_view = df_hist.tail(30)
    pred_view = df_pred.head(30)
    
    visible_values = pd.concat([hist_view['value'], pred_view['value']])
    min_val, max_val = visible_values.min(), visible_values.max()
    value_range = max_val - min_val if max_val != min_val else 1
    
    padding = value_range * 0.15 
    y_min, y_max = max(0, min_val - padding), max_val + padding

    fig.add_trace(go.Scatter(x=hist_view['date'], y=hist_view['value'], mode='lines', name='Data Terakhir',
                             line=dict(color='rgba(255,255,255,0.3)', width=2), fill='tozeroy', fillcolor='rgba(255,255,255,0.05)'))
    
    today_val = pred_view['value'].iloc[0] if not pred_view.empty else 0
    fig.add_trace(go.Scatter(x=[pd.Timestamp.now().normalize()], y=[today_val], mode='markers', name='Hari Ini',
                             marker=dict(color='#FFFFFF', size=10, line=dict(color=color, width=2))))

    fig.add_trace(go.Scatter(x=pred_view['date'], y=pred_view['value'], mode='lines+markers', name='Prediksi AI',
                             line=dict(color=color, width=3), marker=dict(size=5, color=color)))

    if threshold and max_val >= (threshold * 0.8):
        fig.add_hline(y=threshold, line_dash="dash", line_color="#FF5252", annotation_text="Batas Waspada")

    fig.update_layout(title=dict(text=f"Proyeksi {title}", font=dict(size=16, color='#EEEEEE')),
                      xaxis=dict(title="Tanggal", showgrid=False, color='#AAAAAA'),
                      yaxis=dict(title=unit, range=[y_min, y_max], showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='#AAAAAA'),
                      height=320, margin=dict(l=20, r=20, t=50, b=20), hovermode="x unified",
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter, sans-serif"))
    return fig

# ==========================================
# 4. LOGIC ENGINE (PERSIS PERMINTAAN)
# ==========================================
def generate_smart_insights(df_rain_pred, df_wind_pred):
    # Gabungkan data berdasarkan tanggal (inner join)
    df = pd.merge(df_rain_pred, df_wind_pred, on='date', suffixes=('_rain', '_wind'))
    df = df.rename(columns={'value_rain': 'rain', 'value_wind': 'wind'})
    
    df_short = df.head(7)  # 7 Hari ke depan
    df_long = df.iloc[7:37] # 30 Hari ke depan (bulan depan)
    
    insights = {
        "short_term": {"tani": [], "ternak": []},
        "long_term": {"tani": [], "ternak": []},
        "status_label": "Normal",
        "status_color": "success"
    }

    # [SHORT TERM]
    good_spray_days = df_short[(df_short['rain'] < 1) & (df_short['wind'] < 5)]
    if not good_spray_days.empty:
        days = [d.strftime('%A') for d in good_spray_days['date']][:3]
        insights['short_term']['tani'].append(f"🧪 **Jadwal Semprot:** Disarankan hari **{', '.join(days)}** (Cerah & Tenang).")
    else:
        insights['short_term']['tani'].append("⛔ **Stop Semprot:** Minggu ini cuaca buruk.")

    wet_soil_days = (df_short['rain'] > 5).sum()
    if wet_soil_days >= 4:
        insights['short_term']['tani'].append("🚫 **Tunda Pupuk Tabur:** Tanah terlalu basah.")
        insights['short_term']['tani'].append("🍄 **Cek Jamur:** Risiko tinggi.")
    elif wet_soil_days == 0:
        insights['short_term']['tani'].append("💧 **Irigasi Wajib:** Tanah mulai kering.")

    if (df_short['wind'] > 7).any():
        insights['short_term']['tani'].append("💨 **Pasang Ajir:** Potensi angin kencang.")

    heavy_rain_days = (df_short['rain'] > 10).sum()
    if heavy_rain_days >= 3:
        insights['short_term']['ternak'].append("⚠️ **Bahaya Kembung:** Rumput basah. Layukan dulu.")
        insights['short_term']['ternak'].append("💊 **Vitamin:** Tambah suplemen.")
    else:
        insights['short_term']['ternak'].append("✅ **Pakan Aman:** Kondisi ideal.")

    if (df_short['wind'] > 6).any():
        insights['short_term']['ternak'].append("🥶 **Tutup Kandang:** Angin kencang.")

    # [LONG TERM]
    total_rain = df_long['rain'].sum()
    high_wind_days = (df_long['wind'] > 6).sum()

    if total_rain > 200:
        insights['long_term']['tani'].append("🌧️ **Fase Basah:** Bulan depan hujan tinggi. Perbaiki drainase.")
    elif total_rain < 30:
        insights['long_term']['tani'].append("☀️ **Fase Kering:** Defisit air. Siapkan mulsa.")

    if high_wind_days > 7:
        insights['long_term']['tani'].append(f"💨 **Waspada Rebah:** {high_wind_days} hari berangin.")
        
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
        insights['status_label'] = "✅ NORMAL / KONDUSIF"
        insights['status_color'] = "success"

    return insights

# ==========================================
# 5. MAIN EXECUTION (AUTO-SYNC LOGIC)
# ==========================================
today = pd.Timestamp.now().normalize()
data_store = {}

try:
    for key in ['rainfall', 'windspeed']:
        cfg = CONFIG[key]
        h_raw = pd.read_csv(cfg['hist_csv'])
        h_raw = h_raw.rename(columns={h_raw.columns[0]: 'date', h_raw.columns[1]: 'value'})
        h_raw['date'] = pd.to_datetime(h_raw['date'])
        
        p_raw = pd.read_csv(cfg['pred_csv'])
        p_raw = p_raw.rename(columns={p_raw.columns[0]: 'date', p_raw.columns[1]: 'value'})
        p_raw['date'] = pd.to_datetime(p_raw['date'])

        full = pd.concat([h_raw, p_raw]).drop_duplicates('date').sort_values('date')
        
        df_hist = full[full['date'] < today]
        df_pred = full[full['date'] >= today]

        # Agregasi harian
        df_hist = df_hist.set_index('date').resample('D').agg(cfg['agg']).reset_index().fillna(0)
        data_store[key] = {"hist": df_hist, "pred": df_pred}

except Exception as e:
    st.error(f"Error data: {e}")
    st.stop()

# ==========================================
# 6. DASHBOARD LAYOUT
# ==========================================
col1, col2, col3 = st.columns(3)
curr_rain = data_store['rainfall']['pred']['value'].iloc[0]
curr_wind = data_store['windspeed']['pred']['value'].iloc[0]

with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">📅 Hari Ini</div><div class="metric-value">{today.strftime("%d %b %Y")}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">🌧️ Prediksi Hujan</div><div class="metric-value" style="color:#00E676">{curr_rain:.1f} mm</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">💨 Prediksi Angin</div><div class="metric-value" style="color:#2979FF">{curr_wind:.1f} m/s</div></div>', unsafe_allow_html=True)

st.write("") 

c_rain, c_wind = st.columns(2)
with c_rain:
    st.plotly_chart(plot_interactive(data_store['rainfall']['hist'], data_store['rainfall']['pred'], "Hujan", "mm", "#00E676", 20), use_container_width=True)
with c_wind:
    st.plotly_chart(plot_interactive(data_store['windspeed']['hist'], data_store['windspeed']['pred'], "Angin", "m/s", "#2979FF", 6), use_container_width=True)

# Insights Section
insight_data = generate_smart_insights(data_store['rainfall']['pred'], data_store['windspeed']['pred'])
st.markdown("---")
st.markdown("### 💡 Rekomendasi AI")

if insight_data['status_color'] == 'error': st.error(f"**STATUS:** {insight_data['status_label']}")
elif insight_data['status_color'] == 'warning': st.warning(f"**STATUS:** {insight_data['status_label']}")
else: st.success(f"**STATUS:** {insight_data['status_label']}")

tab1, tab2 = st.tabs(["📅 Rencana Minggu Ini", "🔭 Prospek 30 Hari"])

def show_list(items, empty_msg):
    if not items: st.info(f"_{empty_msg}_")
    else:
        for item in items: st.markdown(f"{item}")

with tab1:
    ca, cb = st.columns(2)
    with ca:
        st.info("##### 🚜 Aksi Petani")
        show_list(insight_data['short_term']['tani'], "Kondisi normal.")
    with cb:
        st.warning("##### 🐄 Aksi Peternak")
        show_list(insight_data['short_term']['ternak'], "Kondisi aman.")

with tab2:
    ca, cb = st.columns(2)
    with ca:
        st.info("##### 🌾 Strategi Tani")
        show_list(insight_data['long_term']['tani'], "Iklim kondusif.")
    with cb:
        st.warning("##### 🐂 Strategi Ternak")
        show_list(insight_data['long_term']['ternak'], "Prediksi aman.")

# Table Section
with st.expander("📂 Detail Tabel Data"):
    ca, cb = st.columns(2)
    with ca:
        st.write("**30 Hari Prediksi Hujan**")
        st.dataframe(data_store['rainfall']['pred'].head(30), hide_index=True)
    with cb:
        st.write("**30 Hari Prediksi Angin**")
        st.dataframe(data_store['windspeed']['pred'].head(30), hide_index=True)
