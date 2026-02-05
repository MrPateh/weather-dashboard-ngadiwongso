import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
from datetime import timedelta
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts import concatenate

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
    <p>Sistem Monitoring Iklim & Rekomendasi Cerdas Berbasis AI</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. CONFIG & CONSTANTS
# ==========================================
HORIZON = 30 

CONFIG = {
    "rainfall": {
        "model_file": "rainfall_prediction_model.pt",
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
        "model_file": "windspeed_prediction_model.pt",
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
        try:
            model = NBEATSModel.load(cfg['model_file'])
            with open(cfg['scaler_target'], "rb") as f: s_target = pickle.load(f)
            with open(cfg['scaler_cov'], "rb") as f: s_cov = pickle.load(f)
            artifacts[key] = (model, s_target, s_cov)
        except Exception as e:
            st.error(f"Gagal load {key}: {e}")
            return None
    return artifacts

def generate_covariates(series_target, horizon=0):
    start_time = series_target.start_time()
    end_time = series_target.end_time() + timedelta(days=horizon)
    full_time_index = pd.date_range(start=start_time, end=end_time, freq='D')
    dummy_series = TimeSeries.from_times_and_values(full_time_index, values=range(len(full_time_index)))
    cov_month = datetime_attribute_timeseries(dummy_series, attribute="month", cyclic=True)
    cov_day = datetime_attribute_timeseries(dummy_series, attribute="day", cyclic=True)
    return concatenate([cov_month, cov_day], axis=1)

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

def auto_update_csv(df_hist, model, scaler_target, scaler_cov, csv_path):
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    last_date = df_hist['date'].iloc[-1]
    today = pd.Timestamp.now().normalize()
    
    if last_date < today:
        gap_days = (today - last_date).days
        series_hist = TimeSeries.from_dataframe(df_hist, time_col='date', value_cols='value')
        cov_full = generate_covariates(series_hist, horizon=gap_days)
        
        series_scaled = scaler_target.transform(series_hist)
        cov_scaled = scaler_cov.transform(cov_full)
        pred_gap = model.predict(n=gap_days, series=series_scaled, past_covariates=cov_scaled)
        pred_gap_real = scaler_target.inverse_transform(pred_gap) 
        
        df_gap = pred_gap_real.to_dataframe().reset_index()
        df_gap.columns = ['date', 'value']
        df_updated = pd.concat([df_hist, df_gap], ignore_index=True)
        df_updated.to_csv(csv_path, index=False)
        return df_updated, True
    return df_hist, False

def run_forecast(df, model, scaler_target, scaler_cov, horizon):
    series_input = TimeSeries.from_dataframe(df, time_col='date', value_cols='value')
    cov_future = generate_covariates(series_input, horizon=horizon)
    series_scaled = scaler_target.transform(series_input)
    cov_scaled = scaler_cov.transform(cov_future)
    pred_scaled = model.predict(n=horizon, series=series_scaled, past_covariates=cov_scaled)
    pred_final = scaler_target.inverse_transform(pred_scaled)
    return pred_final

# --- BAGIAN INI SUDAH DIUPDATE UNTUK MENGHILANGKAN SQUEEZING ---
def plot_interactive(df_hist, df_pred, title, unit, color, threshold=None):
    fig = go.Figure()
    
    # Ambil 30 hari terakhir
    hist_view = df_hist.tail(30)
    pred_df = df_pred.to_dataframe().reset_index()
    pred_df.columns = ['date', 'value']

    # 1. HITUNG DYNAMIC RANGE Y-AXIS
    # Gabungkan semua nilai (history + prediksi) untuk mencari Min dan Max
    all_values = pd.concat([hist_view['value'], pred_df['value']])
    min_val = all_values.min()
    max_val = all_values.max()
    
    # Beri padding 10% agar grafik tidak menempel di atas/bawah
    value_range = max_val - min_val
    if value_range == 0: value_range = 1 # Prevent division by zero
    
    padding = value_range * 0.1
    y_min = max(0, min_val - padding) # Pastikan tidak negatif
    y_max = max_val + padding

    # 2. GAMBAR PLOT
    # Historis
    fig.add_trace(go.Scatter(
        x=hist_view['date'], y=hist_view['value'],
        mode='lines', name='Data Terakhir',
        line=dict(color='rgba(255,255,255,0.3)', width=2), 
        fill='tozeroy', fillcolor='rgba(255,255,255,0.05)'
    ))
    
    # Marker Hari Ini
    fig.add_trace(go.Scatter(
        x=[hist_view['date'].iloc[-1]], y=[hist_view['value'].iloc[-1]],
        mode='markers', name='Hari Ini',
        marker=dict(color='#FFFFFF', size=6)
    ))

    # Prediksi
    fig.add_trace(go.Scatter(
        x=pred_df['date'], y=pred_df['value'],
        mode='lines+markers', name='Prediksi AI',
        line=dict(color=color, width=3), 
        marker=dict(size=5, color=color)
    ))

    # 3. THRESHOLD DINAMIS
    if threshold:
        # Hanya gambar jika nilai Max mendekati threshold (80%)
        if max_val >= (threshold * 0.8):
            fig.add_hline(
                y=threshold, 
                line_dash="dash", line_color="#FF5252", 
                annotation_text="Batas Waspada", annotation_position="top right",
                annotation_font_color="#FF5252"
            )

    # 4. UPDATE LAYOUT DENGAN RANGE Y CUSTOM
    fig.update_layout(
        title=dict(text=f"Proyeksi {title}", font=dict(size=16, color='#EEEEEE')),
        xaxis=dict(title="Tanggal", showgrid=False, color='#AAAAAA', gridcolor='rgba(255,255,255,0.1)'),
        
        # INI KUNCINYA: Pakai y_min dan y_max yang sudah dihitung
        yaxis=dict(
            title=unit, 
            range=[y_min, y_max], # SET RANGE MANUAL
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
# 4. LOGIC ENGINE
# ==========================================
def generate_smart_insights(pred_rain_series, pred_wind_series):
    df = pd.DataFrame({
        'rain': pred_rain_series.values().flatten(),
        'wind': pred_wind_series.values().flatten()
    }, index=pred_rain_series.time_index)
    
    df_short = df.iloc[:7]
    df_long = df.iloc[7:]
    
    insights = {
        "short_term": {"tani": [], "ternak": []},
        "long_term": {"tani": [], "ternak": []},
        "status_label": "Normal",
        "status_color": "success"
    }

    # [SHORT TERM]
    good_spray_days = df_short[(df_short['rain'] < 1) & (df_short['wind'] < 5)]
    if not good_spray_days.empty:
        days = [d.strftime('%A') for d in good_spray_days.index][:3]
        insights['short_term']['tani'].append(f"🧪 **Jadwal Semprot:** Disarankan melakukan penyemprotan pada hari **{', '.join(days)}** karena cuaca cerah dan angin tenang.")
    else:
        insights['short_term']['tani'].append("⛔ **Stop Semprot:** Minggu ini cuaca tidak kondusif (Hujan/Angin).")

    wet_soil_days = (df_short['rain'] > 5).sum()
    if wet_soil_days >= 4:
        insights['short_term']['tani'].append("🚫 **Tunda Pupuk Tabur:** Tanah terlalu basah, pupuk akan tercuci.")
        insights['short_term']['tani'].append("🍄 **Cek Jamur:** Kelembaban tinggi memicu jamur pada pangkal batang.")
    elif wet_soil_days == 0:
        insights['short_term']['tani'].append("💧 **Irigasi Wajib:** Tanah mulai kering. Lakukan penyiraman minggu ini.")

    if (df_short['wind'] > 7).any():
        insights['short_term']['tani'].append("💨 **Pasang Ajir:** Potensi angin kencang. Ikat tanaman cabe/tomat.")

    heavy_rain_days = (df_short['rain'] > 10).sum()
    if heavy_rain_days >= 3:
        insights['short_term']['ternak'].append("⚠️ **Bahaya Kembung:** Rumput basah/muda. Wajib layukan 4 jam sebelum diberi pakan.")
        insights['short_term']['ternak'].append("💊 **Vitamin:** Berikan vitamin untuk daya tahan tubuh.")
    else:
        insights['short_term']['ternak'].append("✅ **Pakan Aman:** Rumput dalam kondisi ideal.")

    if (df_short['wind'] > 6).any():
        insights['short_term']['ternak'].append("🥶 **Tutup Kandang:** Angin kencang. Pasang tirai sisi angin.")
        insights['short_term']['ternak'].append("🔥 **Tambah Energi:** Berikan pakan konsentrat lebih banyak.")

    if wet_soil_days > 4:
        insights['short_term']['ternak'].append("🧹 **Sanitasi:** Lantai lembab. Bersihkan kotoran 2x sehari.")

    # [LONG TERM]
    total_rain = df_long['rain'].sum()
    high_wind_days = (df_long['wind'] > 6).sum()

    if total_rain > 200:
        insights['long_term']['tani'].append("🌧️ **Fase Basah:** Curah hujan tinggi bulan depan.")
        insights['long_term']['tani'].append("👉 **Drainase:** Perdalam parit agar air cepat surut.")
        insights['long_term']['tani'].append("👉 **Penyakit:** Stok fungisida. Risiko patek/busuk batang.")
    elif total_rain < 30:
        insights['long_term']['tani'].append("☀️ **Fase Kering:** Defisit air bulan depan.")
        insights['long_term']['tani'].append("👉 **Hama:** Waspada Thrips/Tungau.")
        insights['long_term']['tani'].append("👉 **Mulsa:** Tutup tanah untuk tahan penguapan.")

    if high_wind_days > 7:
        insights['long_term']['tani'].append(f"💨 **Waspada Rebah:** {high_wind_days} hari angin kencang. Hindari Urea berlebih.")
        
    if total_rain > 100:
        insights['long_term']['ternak'].append("🌿 **Hijauan Melimpah:** Manfaatkan rumput liar.")
        insights['long_term']['ternak'].append("⛔ **Sulit Kering:** Jangan buat Hay (pakan kering).")
    elif total_rain < 30:
        insights['long_term']['ternak'].append("📉 **Stok Menipis:** Pertumbuhan rumput melambat.")
        insights['long_term']['ternak'].append("👉 **Beli Pakan:** Stok silase/konsentrat sekarang.")

    if high_wind_days > 10:
        insights['long_term']['ternak'].append("👀 **Kesehatan Mata:** Waspada debu (Pink Eye).")
        insights['long_term']['ternak'].append("🔨 **Perbaikan:** Cek paku atap seng kandang.")

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
        df, updated = auto_update_csv(df, model, s_target, s_cov, cfg['csv'])
        pred_series = run_forecast(df, model, s_target, s_cov, HORIZON)
        
        data_store[key] = { "hist": df, "pred": pred_series, "updated": updated }
    else:
        st.error(f"File {cfg['csv']} tidak ditemukan!")
        st.stop()

# ==========================================
# 6. DASHBOARD LAYOUT
# ==========================================

col1, col2, col3= st.columns(3)

last_rain = data_store['rainfall']['hist']['value'].iloc[-1]
last_wind = data_store['windspeed']['hist']['value'].iloc[-1]
last_date = data_store['rainfall']['hist']['date'].iloc[-1].strftime("%d %b %Y")

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Tanggal Data</div>
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

insight_data = generate_smart_insights(data_store['rainfall']['pred'], data_store['windspeed']['pred'])

st.markdown("---")
st.markdown("### 💡 Rekomendasi dan Strategi dari AI")

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
        hist_5 = df_hist.tail(5).copy()
        hist_5['date'] = hist_5['date'].dt.strftime('%Y-%m-%d')
        hist_5.columns = ['Tanggal', f'Aktual ({unit})']
        pred_df = df_pred.to_dataframe().reset_index()
        pred_df.columns = ['Tanggal', f'Prediksi ({unit})']
        pred_df['Tanggal'] = pred_df['Tanggal'].dt.strftime('%Y-%m-%d')
        return hist_5, pred_df

    with c_tbl_rain:
        h_rain, p_rain = make_pretty_table(data_store['rainfall']['hist'], data_store['rainfall']['pred'], "mm")
        st.write("**Curah Hujan (mm)**")
        st.dataframe(p_rain.style.format(subset=['Prediksi (mm)'], formatter="{:.2f}"), use_container_width=True, hide_index=True)
        csv_rain = p_rain.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV Hujan", data=csv_rain, file_name="prediksi_hujan.csv", mime="text/csv")

    with c_tbl_wind:
        h_wind, p_wind = make_pretty_table(data_store['windspeed']['hist'], data_store['windspeed']['pred'], "m/s")
        st.write("**Kecepatan Angin (m/s)**")
        st.dataframe(p_wind.style.format(subset=['Prediksi (m/s)'], formatter="{:.2f}"), use_container_width=True, hide_index=True)
        csv_wind = p_wind.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV Angin", data=csv_wind, file_name="prediksi_angin.csv", mime="text/csv")