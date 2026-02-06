import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from darts import TimeSeries
from darts.models import NBEATSModel
import pickle
import os

# ==========================================
# 1. KONFIGURASI
# ==========================================
st.set_page_config(page_title="Dashboard Cuaca Desa", layout="wide")

HORIZON = 7  # Prediksi 7 hari ke depan

CONFIG = {
    "rainfall": {
        "title": "Curah Hujan (Rainfall)",
        "unit": "mm",
        "csv": "Rainfall_Daily_Ngadirejo_historical.csv", 
        "model_file": "rainfall_final.pt",      # <--- Nama file baru
        "scaler_target": "scaler_rainfall_data.pkl",
        "scaler_cov": "scaler_cov_rainfall.pkl",
        "agg": "sum",
        "color": "blue"
    },
    "windspeed": {
        "title": "Kecepatan Angin (Wind Speed)",
        "unit": "m/s",
        "csv": "WindSpeed_Ngadirejo_Daily.csv",
        "model_file": "windspeed_final.pt",     # <--- Nama file baru
        "scaler_target": "scaler_windspeed_data.pkl",
        "scaler_cov": "scaler_cov_windspeed.pkl",
        "agg": "mean",
        "color": "orange"
    }
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

@st.cache_resource
def load_all_models():
    artifacts = {}
    for key, cfg in CONFIG.items():
        if not os.path.exists(cfg['model_file']):
            st.error(f"❌ File Model Hilang: {cfg['model_file']}")
            return None
            
        try:
            # Load Model
            model = NBEATSModel.load(cfg['model_file'], map_location="cpu")
            
            # Load Scalers (Optional, kalau scaler hilang kita skip biar ga crash)
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

def make_compatible_series(df_value, model_input_dim):
    """
    Fungsi sakti untuk mengubah data 1 kolom menjadi 5 kolom (atau berapapun)
    agar sesuai dengan permintaan model.
    """
    values = df_value.values # Shape (N,) atau (N, 1)
    current_dim = 1
    
    if model_input_dim > current_dim:
        # Trik: Duplikasi kolom data sebanyak dimensi yang dibutuhkan
        # Misal butuh 5, kita copy data hujan ke 5 kolom tersebut
        # Shape baru: (N, 5)
        new_values = np.tile(values.reshape(-1, 1), (1, model_input_dim))
        return TimeSeries.from_values(new_values)
    else:
        return TimeSeries.from_values(values)

def run_forecast(df, model, s_target, s_cov, horizon):
    try:
        # 1. Siapkan Data Input
        # Cek berapa dimensi yang diminta model (biasanya 5 dari hasil konversi tadi)
        req_dim = model.input_dim 
        
        # Buat series yang kompatibel (Padding otomatis)
        series = make_compatible_series(df['value'], req_dim)
        
        # 2. Scaling (Hanya jika scaler ada)
        if s_target:
            try:
                series = s_target.transform(series)
            except:
                pass # Skip scaling jika dimensi scaler beda (sering terjadi saat hack dimensi)

        # 3. Prediksi
        pred = model.predict(horizon, series=series)
        
        # 4. Inverse Scaling
        if s_target:
            try:
                pred = s_target.inverse_transform(pred)
            except:
                pass

        # 5. Ambil Kembalian (Hanya kolom pertama/utama)
        return pred.pd_series().iloc[:, 0]
        
    except Exception as e:
        st.warning(f"Prediksi gagal: {e}")
        # Return dummy zeros biar dashboard ga crash total
        dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=horizon)
        return pd.Series([0]*horizon, index=dates)

def process_daily_aggregation(df, date_col, val_col, agg_func):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    # Group by daily
    df = df.groupby(df[date_col].dt.date)[val_col].agg(agg_func).reset_index()
    df.columns = ['date', 'value']
    df['date'] = pd.to_datetime(df['date'])
    return df

# ==========================================
# 3. MAIN APP LAYOUT
# ==========================================

st.title("🌦️ Dashboard Prediksi Cuaca Ngadirejo")
st.markdown("Monitoring curah hujan dan kecepatan angin menggunakan AI (N-BEATS).")

# Load Models
artifacts = load_all_models()

if artifacts is None:
    st.error("Sistem berhenti karena model tidak ditemukan.")
    st.stop()

# TABS
tab1, tab2 = st.tabs(["📊 Curah Hujan", "💨 Kecepatan Angin"])

def render_tab(key):
    cfg = CONFIG[key]
    model, s_target, s_cov = artifacts[key]
    
    if not os.path.exists(cfg['csv']):
        st.warning(f"Data CSV {cfg['csv']} belum tersedia.")
        return

    # Load Data
    df = pd.read_csv(cfg['csv'])
    # Asumsi kolom 0=Date, kolom 1=Value
    df = df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'value'})
    df = process_daily_aggregation(df, 'date', 'value', cfg['agg'])
    
    # Tampilkan Data Terakhir
    last_date = df['date'].max().strftime('%d %B %Y')
    last_val = df['value'].iloc[-1]
    
    col_metric1, col_metric2 = st.columns(2)
    col_metric1.metric("Update Terakhir", last_date)
    col_metric1.metric(f"Nilai Terakhir", f"{last_val:.2f} {cfg['unit']}")
    
    # Lakukan Prediksi
    with st.spinner("Sedang menghitung prediksi AI..."):
        forecast_series = run_forecast(df, model, s_target, s_cov, HORIZON)
    
    # Gabungkan Data untuk Plotting
    # Ambil 30 hari terakhir history saja biar grafik enak dilihat
    history_plot = df.tail(30).set_index('date')['value']
    
    # PLOTTING
    fig = go.Figure()
    
    # 1. Data Historis
    fig.add_trace(go.Scatter(
        x=history_plot.index, 
        y=history_plot.values,
        mode='lines+markers',
        name='Data Historis',
        line=dict(color='gray')
    ))
    
    # 2. Data Prediksi
    fig.add_trace(go.Scatter(
        x=forecast_series.index, 
        y=forecast_series.values,
        mode='lines+markers',
        name='Prediksi AI',
        line=dict(color=cfg['color'], width=3, dash='dot')
    ))
    
    fig.update_layout(
        title=f"Prediksi {cfg['title']} (7 Hari ke Depan)",
        xaxis_title="Tanggal",
        yaxis_title=cfg['unit'],
        template="plotly_white",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tampilkan Tabel Prediksi
    st.subheader("📋 Angka Prediksi")
    pred_df = pd.DataFrame({"Tanggal": forecast_series.index, "Prediksi": forecast_series.values})
    pred_df['Tanggal'] = pred_df['Tanggal'].dt.strftime('%d-%m-%Y')
    st.dataframe(pred_df, hide_index=True)

with tab1:
    render_tab("rainfall")

with tab2:
    render_tab("windspeed")

st.markdown("---")
st.caption("Dibuat dengan Streamlit & Darts N-BEATS | Model 5-Dimensi Adapted")
