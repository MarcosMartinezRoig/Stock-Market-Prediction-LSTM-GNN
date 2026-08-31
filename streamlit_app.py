import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import yfinance as yf
import streamlit as st

from model.lstm_gat import CombinedLSTMGATWithStatic3Hop

# Configuración inicial de la página de Streamlit
st.set_page_config(
    page_title="Dashboard de Evaluación - LSTM-GAT",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_config(config_path="config.yaml"):
    """Carga la configuración por defecto desde el YAML."""
    default_args = {
        'use_returns': True,
        'use_rollout': False,
        'seq_len': 30,
        'base_horizon': 7,
        'rollout_steps': 1,
        'batch_size': 32,
        'save_dir': 'saved_models'
    }

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            if config_data:
                default_args.update(config_data)
    return default_args


@st.cache_data
def download_and_prepare_evaluation_data(seq_len=30, horizon=7, use_returns=True):
    """Prepara los datos de evaluación y precios base."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
    n_stations = len(tickers)
    
    raw_data = yf.download(tickers, period="5y", interval="1d", progress=False)['Close']
    raw_data = raw_data.ffill().dropna()
    prices_data = raw_data.copy()

    if use_returns:
        data = raw_data.pct_change().dropna()
        prices_data = prices_data.loc[data.index]
    else:
        data = raw_data

    split_date = data.index[-1] - pd.Timedelta(days=180)
    train_data = data[data.index < split_date]
    valid_data = data[data.index >= (split_date - pd.Timedelta(days=seq_len))]
    
    mean_val = data.mean()
    std_val = data.std()
    
    normalized = (valid_data - mean_val) / std_val
    values = normalized.values
    
    dyn_list, tgt_list, dates_list = [], [], []
    for i in range(len(values) - seq_len - horizon):
        dyn_list.append(values[i : i + seq_len, :])
        tgt_list.append(values[i + seq_len : i + seq_len + horizon, :])
        dates_list.append(valid_data.index[i + seq_len])
        
    dyn_valid = torch.tensor(np.array(dyn_list), dtype=torch.float32).permute(0, 2, 1).unsqueeze(-1)
    tgts_valid = torch.tensor(np.array(tgt_list), dtype=torch.float32).permute(0, 2, 1)
    stat_valid = torch.randn(dyn_valid.size(0), n_stations, 3)
    
    corr_matrix = ((train_data - mean_val) / std_val).corr().values
    edges_src, edges_dst = [], []
    for i in range(n_stations):
        for j in range(n_stations):
            if i != j and abs(corr_matrix[i, j]) > 0.4:
                edges_src.append(i)
                edges_dst.append(j)
                
    if len(edges_src) == 0:
        edges_src = [0, 1, 2, 3, 4, 5]
        edges_dst = [1, 2, 3, 4, 5, 0]
        
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return (dyn_valid, stat_valid, tgts_valid), edge_index, dates_list, mean_val, std_val, prices_data


def convert_returns_to_usd(preds_rets, dates_list, prices_data, std_arr, mean_arr, horizon, tickers):
    model_preds_usd = np.zeros_like(preds_rets)
    for i, d_date in enumerate(dates_list):
        base_idx = prices_data.index.get_loc(d_date) - 1
        base_prices = prices_data.iloc[base_idx].values.reshape(1, len(tickers))
        denorm_rets = preds_rets[i] * std_arr + mean_arr
        curr_prices = base_prices.copy()
        
        horizon_prices = np.zeros((len(tickers), horizon))
        for h in range(horizon):
            curr_prices = curr_prices * (1.0 + denorm_rets[:, h])
            horizon_prices[:, h] = curr_prices.flatten()
        model_preds_usd[i] = horizon_prices
    return model_preds_usd

# --- INTERFAZ DE STREAMLIT ---
st.title("📈 Dashboard de Evaluación: LSTM-GAT (Series Bursátiles)")
st.markdown("Visualización interactiva de predicciones, métricas de rendimiento y validación de modelos.")

# Sidebar para parámetros
st.sidebar.header("⚙️ Configuración de Ejecución")
config = load_config()

use_returns = st.sidebar.checkbox("Usar Retornos Porcentuales", value=config.get('use_returns', True))
use_rollout = st.sidebar.checkbox("Activar Rollout Autorregresivo", value=config.get('use_rollout', False))
seq_len = st.sidebar.number_input("Longitud de Secuencia (seq_len)", value=int(config.get('seq_len', 30)))
base_horizon = st.sidebar.number_input("Horizonte Base", value=int(config.get('base_horizon', 7)))
rollout_steps = st.sidebar.number_input("Pasos de Rollout", value=int(config.get('rollout_steps', 1)))
save_dir = st.sidebar.text_input("Directorio del Modelo", value=config.get('save_dir', 'saved_models'))

horizon = (base_horizon * rollout_steps) if use_rollout else base_horizon
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]

# Botón para ejecutar la inferencia
if st.sidebar.button("🚀 Ejecutar Evaluación", type="primary"):
    with st.spinner("Descargando datos y ejecutando inferencia..."):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Preparar datos
        valid_set, edge_index, dates_list, scaler_mean, scaler_std, prices_data = download_and_prepare_evaluation_data(
            seq_len=seq_len, horizon=horizon, use_returns=use_returns
        )
        dyn_valid, stat_valid, tgts_valid = valid_set
        edge_index = edge_index.to(device)
        dyn_valid, stat_valid = dyn_valid.to(device), stat_valid.to(device)

        if tgts_valid.ndim == 3 and tgts_valid.shape[-1] == 1:
            tgts_valid = tgts_valid.squeeze(-1)

        mean_arr = scaler_mean.values.reshape(len(tickers), 1)
        std_arr = scaler_std.values.reshape(len(tickers), 1)

        # 2. Reconstrucción de targets reales
        if use_returns:
            tgts_real_usd = np.zeros_like(tgts_valid.numpy())
            for i, d_date in enumerate(dates_list):
                base_idx = prices_data.index.get_loc(d_date) - 1
                base_prices = prices_data.iloc[base_idx].values.reshape(1, len(tickers))
                denorm_rets = tgts_valid.numpy()[i] * std_arr + mean_arr
                curr_prices = base_prices.copy()
                
                horizon_prices = np.zeros((len(tickers), horizon))
                for h in range(horizon):
                    curr_prices = curr_prices * (1.0 + denorm_rets[:, h])
                    horizon_prices[:, h] = curr_prices.flatten()
                tgts_real_usd[i] = horizon_prices
        else:
            tgts_real_usd = tgts_valid.numpy() * std_arr.reshape(1, len(tickers), 1) + scaler_mean.values.reshape(1, len(tickers), 1)

        # 3. Cargar modelo
        best_model_path = os.path.join(save_dir, "best_model.pth")
        if not os.path.exists(best_model_path):
            st.error(f"No se encontró el modelo en: {best_model_path}. Entrena el modelo primero.")
            st.stop()

        model = CombinedLSTMGATWithStatic3Hop(
            dynamic_input_dim=1, static_input_dim=3, lstm_hidden_dim=32, gnn_hidden_dim=32, output_dim=base_horizon
        ).to(device)
        
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        model.eval()

        # 4. Inferencia
        all_rollout_preds = []
        with torch.no_grad():
            for sample_idx in range(dyn_valid.size(0)):
                current_dyn = dyn_valid[sample_idx:sample_idx+1].clone()
                current_stat = stat_valid[sample_idx:sample_idx+1]
                
                if use_rollout:
                    rollout_preds = []
                    for step in range(rollout_steps):
                        pred_block = model(current_dyn, current_stat, edge_index)
                        if pred_block.ndim == 2:
                            pred_block = pred_block.unsqueeze(-1)
                        rollout_preds.append(pred_block)
                        pred_block_4d = pred_block.unsqueeze(-1)
                        current_dyn = torch.cat([current_dyn[:, :, base_horizon:, :], pred_block_4d], dim=2)
                    predictions = torch.cat(rollout_preds, dim=-1)
                else:
                    predictions = model(current_dyn, current_stat, edge_index)
                    if predictions.ndim == 2:
                        predictions = predictions.unsqueeze(-1)

                all_rollout_preds.append(predictions.cpu().numpy())
                
        preds_np = np.concatenate(all_rollout_preds, axis=0)
        
        if use_returns:
            preds_real_usd = convert_returns_to_usd(preds_np, dates_list, prices_data, std_arr, mean_arr, horizon, tickers)
        else:
            preds_real_usd = preds_np * std_arr.reshape(1, len(tickers), 1) + scaler_mean.values.reshape(1, len(tickers), 1)

        # Guardar en session_state para persistir entre interacciones
        st.session_state['preds'] = preds_real_usd
        st.session_state['tgts'] = tgts_real_usd
        st.session_state['dates'] = dates_list
        st.session_state['tickers'] = tickers
        st.success("¡Inferencia completada con éxito!")

# --- VISUALIZACIÓN SI LOS DATOS ESTÁN CARGADOS ---
if 'preds' in st.session_state:
    preds = st.session_state['preds']
    tgts = st.session_state['tgts']
    dates = st.session_state['dates']
    tickers = st.session_state['tickers']

    tab1, tab2 = st.tabs(["📊 Gráficos por Ticker", "📋 Tabla de Métricas"])

    with tab1:
        selected_ticker = st.selectbox("Selecciona un activo para visualizar:", tickers)
        t_idx = tickers.index(selected_ticker)

        fig, ax = plt.subplots(figsize=(10, 5))
        p_ticker = preds[:, t_idx, 0]
        t_ticker = tgts[:, t_idx, 0]

        rmse = np.sqrt(np.mean((p_ticker - t_ticker) ** 2))
        corr, _ = pearsonr(p_ticker, t_ticker)
        mape = np.mean(np.abs((t_ticker - p_ticker) / t_ticker)) * 100

        ax.plot(dates, t_ticker, label='Real (USD)', color='black', linewidth=1.5)
        ax.plot(dates, p_ticker, label='Predicción (USD)', color='tab:blue', linestyle='--', linewidth=1.5)
        ax.set_title(f"Evolución para {selected_ticker} (Horizonte t+1)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Precio (USD)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right")

        st.pyplot(fig)
        
        st.metric(label=f"RMSE ({selected_ticker})", value=f"{rmse:.4f}")
        st.metric(label=f"Correlación ({selected_ticker})", value=f"{corr:.4f}")

    with tab2:
        st.subheader("Resumen General de Métricas (Paso t+1)")
        metrics_summary = []
        for idx, ticker in enumerate(tickers):
            p_ticker = preds[:, idx, 0]
            t_ticker = tgts[:, idx, 0]
            rmse = np.sqrt(np.mean((p_ticker - t_ticker) ** 2))
            corr, _ = pearsonr(p_ticker, t_ticker)
            mape = np.mean(np.abs((t_ticker - p_ticker) / t_ticker)) * 100
            
            metrics_summary.append({
                "Ticker": ticker,
                "RMSE": round(rmse, 4),
                "Correlación": round(corr, 4),
                "MAPE (%)": round(mape, 2)
            })
        
        df_metrics = pd.DataFrame(metrics_summary)
        st.dataframe(df_metrics, use_container_width=True)
else:
    st.info("👈 Configura los parámetros en la barra lateral y haz clic en **Ejecutar Evaluación** para iniciar.")