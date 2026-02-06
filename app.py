import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. PAGE CONFIG & STYLING
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
    <p>Sistem Monitoring Iklim & Rekomendasi Cerdas Berbasis AI (Pre-Computed)</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONFIG
# ==========================================
# Kita hanya butuh membaca CSV sekarang
CONFIG = {
    "rainfall": {
        "title": "Curah Hujan",
        "unit": "mm",
        "color": "#00E676",
        "threshold": 20,
        "hist_csv": "Rainfall_Daily_Ngadirejo_historical.csv",
        "pred_csv": "forecast_rainfall_2years.csv", # File hasil generate
        "agg": "sum"
    },
    "windspeed": {
        "title": "Kecepatan Angin",
        "unit": "m/s",
        "color": "#2979FF",
        "threshold": 6,
        "hist_csv": "WindSpeed_Ngadirejo_Daily.csv",
        "pred_csv": "forecast_windspeed_2years.csv", # File hasil generate
        "agg": "mean"
    }
}

# ==========================================
# 3. HELPER FUNCTIONS (Versi Ringan Pandas Only)
# ==========================================

def load_data(hist_path, pred_path, agg_method):
    # 1. Load History
    df_hist = pd.read_csv(hist_path)
    # Standardize columns
    df_hist = df_hist.rename(columns={df_hist.columns[0]: 'date', df_hist.columns[1]: 'value'})
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    
    # Aggregation
    df_hist = df_hist.set_index('date')
    if agg_method == 'sum': df_hist = df_hist['value'].resample('D').sum()
    else: df_hist = df_hist['value'].resample('D').mean()
    df_hist = df_hist.reset_index().fillna(0)
    
    # 2. Load Forecast
    df_pred = pd.read_csv(pred_path)
    df_pred['date'] = pd.to_datetime(df_pred['date'])
    
    # Filter Forecast: Hanya ambil tanggal SETELAH data historis terakhir
    # Biar grafiknya nyambung rapi
    last_hist_date = df_hist['date'].max()
    df_pred = df_pred[df_pred['date'] > last_hist_date]
    
    return df_hist, df_pred

def plot_interactive(df_hist, df_pred, title, unit, color, threshold=None):
    fig = go.Figure()
    
    # Ambil 60 hari terakhir history agar grafik tidak terlalu padat
    hist_view = df_hist.tail(30)
    pred_view = df_pred.head(30)
    # 1. HITUNG DYNAMIC RANGE Y-AXIS
    visible_values = pd.concat([hist_view['value'], pred_view['value']])
    
    min_val = visible_values.min()
    max_val = visible_values.max()
    
    # Hitung padding agar grafik tidak menempel ke garis tepi
    value_range = max_val - min_val
    if value_range == 0: value_range = 1
    
    padding = value_range * 0.15 # Beri ruang 15%
    y_min = max(0, min_val - padding) # Pastikan tidak negatif untuk hujan/angin
    y_max = max_val + padding

    # 2. GAMBAR PLOT
    # Historis
    fig.add_trace(go.Scatter(
        x=hist_view['date'], y=hist_view['value'],
        mode='lines', name='Data Terakhir',
        line=dict(color='rgba(255,255,255,0.3)', width=2), 
        fill='tozeroy', fillcolor='rgba(255,255,255,0.05)'
    ))
    
    # Marker Hari Ini (Titik temu history dan prediksi)
    if not hist_view.empty:
        fig.add_trace(go.Scatter(
            x=[hist_view['date'].iloc[-1]], y=[hist_view['value'].iloc[-1]],
            mode='markers', name='Hari Ini',
            marker=dict(color='#FFFFFF', size=6)
        ))

    # Prediksi (Tampilkan 30 hari pertama saja secara default biar fokus)
    # User bisa zoom out kalau mau lihat 2 tahun
    pred_view = df_pred.head(30) 
    
    fig.add_trace(go.Scatter(
        x=pred_view['date'], y=pred_view['value'],
        mode='lines+markers', name='Prediksi AI',
        line=dict(color=color, width=3), 
        marker=dict(size=5, color=color)
    ))

    # 3. THRESHOLD DINAMIS
    if threshold:
        if max_val >= (threshold * 0.8):
            fig.add_hline(
                y=threshold, 
                line_dash="dash", line_color="#FF5252", 
                annotation_text="Batas Waspada", annotation_position="top right",
                annotation_font_color="#FF5252"
            )

    # 4. LAYOUT
    fig.update_layout(
        title=dict(text=f"Proyeksi {title}", font=dict(size=16, color='#EEEEEE')),
        xaxis=dict(title="Tanggal", showgrid=False, color='#AAAAAA', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(
            title=unit, 
            range=[y_min, y_max], 
            showgrid=True, gridcolor='rgba(255,255,255,0.1)', 
            color='#AAAAAA', zerolinecolor='rgba(255,255,255,0.1)'
        ),
        height=320, margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#EEEEEE')),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    return fig

# ==========================================
# 4. LOGIC ENGINE (VERSI PANDAS)
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

    # [SHORT TERM - Logic tetap sama, cuma sintaks pandas disesuaikan]
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
# 5. MAIN EXECUTION
# ==========================================

# Load Data dari CSV
data_store = {}
try:
    for key in ['rainfall', 'windspeed']:
        cfg = CONFIG[key]
        df_h, df_p = load_data(cfg['hist_csv'], cfg['pred_csv'], cfg['agg'])
        data_store[key] = {"hist": df_h, "pred": df_p}
except Exception as e:
    st.error(f"Gagal memuat data CSV. Pastikan file 'forecast_*.csv' sudah diupload. Error: {e}")
    st.stop()

# ==========================================
# 6. DASHBOARD LAYOUT (UI)
# ==========================================

col1, col2, col3 = st.columns(3)

last_rain = data_store['rainfall']['hist']['value'].iloc[-1]
last_wind = data_store['windspeed']['hist']['value'].iloc[-1]
last_date = data_store['rainfall']['hist']['date'].iloc[-1].strftime("%d %b %Y")

with col1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Tanggal Data</div><div class="metric-value">{last_date}</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Hujan Terakhir</div><div class="metric-value" style="color: {CONFIG['rainfall']['color']}">{last_rain:.1f} <span style="font-size:0.8rem; color: #888;">mm</span></div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Angin Terakhir</div><div class="metric-value" style="color: {CONFIG['windspeed']['color']}">{last_wind:.1f} <span style="font-size:0.8rem; color: #888;">m/s</span></div></div>""", unsafe_allow_html=True)

st.write("") 

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

# Generate Insights
insight_data = generate_smart_insights(data_store['rainfall']['pred'], data_store['windspeed']['pred'])

st.markdown("---")
st.markdown("### 💡 Rekomendasi dan Strategi dari AI")

status_msg = f"**STATUS IKLIM 30 HARI:** {insight_data['status_label']}"
if insight_data['status_color'] == 'error': st.error(status_msg, icon="🚨")
elif insight_data['status_color'] == 'warning': st.warning(status_msg, icon="⚠️")
else: st.success(status_msg, icon="✅")

tab1, tab2 = st.tabs(["📅 Rencana Minggu Ini", "🔭 Prospek 30 Hari"])

def show_list(items, empty_msg):
    if not items: st.info(f"_{empty_msg}_")
    else:
        for item in items: st.markdown(f"{item}")

with tab1:
    c_tani, c_ternak = st.columns(2)
    with c_tani:
        st.info("##### 🚜 Aksi Petani")
        show_list(insight_data['short_term']['tani'], "Kondisi normal.")
    with c_ternak:
        st.warning("##### 🐄 Aksi Peternak")
        show_list(insight_data['short_term']['ternak'], "Kondisi aman.")

with tab2:
    c_tani_l, c_ternak_l = st.columns(2)
    with c_tani_l:
        st.info("##### 🌾 Strategi Tani")
        show_list(insight_data['long_term']['tani'], "Iklim kondusif.")
    with c_ternak_l:
        st.warning("##### 🐂 Strategi Ternak")
        show_list(insight_data['long_term']['ternak'], "Prediksi aman.")

st.markdown("---")
st.subheader("📂 Detail Data")

with st.expander("Lihat Tabel Data & Download", expanded=False):
    c_tbl_rain, c_tbl_wind = st.columns(2)
    
    def make_pretty_table(df_hist, df_pred, unit):
        # Ambil 5 data terakhir history
        hist_5 = df_hist.tail(5).copy()
        hist_5['date'] = hist_5['date'].dt.strftime('%Y-%m-%d')
        hist_5.columns = ['Tanggal', f'Aktual ({unit})']
        
        # Ambil 30 hari prediksi
        pred_30 = df_pred.head(30).copy()
        pred_30['date'] = pred_30['date'].dt.strftime('%Y-%m-%d')
        pred_30.columns = ['Tanggal', f'Prediksi ({unit})']
        
        return hist_5, pred_30

    with c_tbl_rain:
        h_rain, p_rain = make_pretty_table(data_store['rainfall']['hist'], data_store['rainfall']['pred'], "mm")
        st.write("**Curah Hujan (mm)**")
        st.dataframe(p_rain.style.format(subset=[f'Prediksi (mm)'], formatter="{:.2f}"), use_container_width=True, hide_index=True)
        csv_rain = p_rain.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV Hujan", data=csv_rain, file_name="prediksi_hujan.csv", mime="text/csv")

    with c_tbl_wind:
        h_wind, p_wind = make_pretty_table(data_store['windspeed']['hist'], data_store['windspeed']['pred'], "m/s")
        st.write("**Kecepatan Angin (m/s)**")
        st.dataframe(p_wind.style.format(subset=[f'Prediksi (m/s)'], formatter="{:.2f}"), use_container_width=True, hide_index=True)
        csv_wind = p_wind.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV Angin", data=csv_wind, file_name="prediksi_angin.csv", mime="text/csv")
