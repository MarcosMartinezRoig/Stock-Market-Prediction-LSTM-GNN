import argparse
import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import yfinance as yf

from model.lstm_gat import CombinedLSTMGATWithStatic3Hop

def download_and_prepare_evaluation_data(seq_len=30, horizon=7, use_returns=True):
    """
    Prepara los datos de evaluación manejando correctamente la lógica de retornos 
    porcentuales y conservando los precios base en USD para la reconstrucción.
    """
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
    n_stations = len(tickers)
    
    print(f"Descargando datos bursátiles para evaluación: {tickers}...")
    raw_data = yf.download(tickers, period="5y", interval="1d", progress=False)['Close']
    raw_data = raw_data.ffill().dropna()

    prices_data = raw_data.copy()

    if use_returns:
        print("-> Activado: Transformando precios absolutos en retornos porcentuales diarios para evaluación.")
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
        target_date = valid_data.index[i + seq_len]
        dates_list.append(target_date)
        
    dyn_valid = torch.tensor(np.array(dyn_list), dtype=torch.float32).permute(0, 2, 1).unsqueeze(-1)
    tgts_valid = torch.tensor(np.array(tgt_list), dtype=torch.float32).permute(0, 2, 1)
    
    stat_valid = torch.randn(dyn_valid.size(0), n_stations, 3)
    
    corr_matrix = ((train_data - mean_val) / std_val).corr().values
    edges_src, edges_dst = [], []
    for i in range(n_stations):
        for j in range(n_stations):
            # Umbral de correlación flexible adaptado
            if i != j and abs(corr_matrix[i, j]) > 0.4:
                edges_src.append(i)
                edges_dst.append(j)
                
    if len(edges_src) == 0:
        edges_src = [0, 1, 2, 3, 4, 5]
        edges_dst = [1, 2, 3, 4, 5, 0]
        
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    
    return (dyn_valid, stat_valid, tgts_valid), edge_index, dates_list, mean_val, std_val, prices_data


def convert_returns_to_usd(preds_rets, dates_list, prices_data, std_arr, mean_arr, horizon, tickers):
    """
    Reconstruye los precios acumulados en USD a partir de los retornos predichos normalizados.
    """
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


def evaluate_model_performance(preds_real, tgts_real, tickers, dates_list, horizon, use_returns=False):
    """
    Calcula métricas avanzadas (RMSE, Corr, MAPE, MDA) y genera gráficos de validación en USD o Retornos.
    """
    n_stations = len(tickers)
    unit_label = "Retorno (%)" if use_returns else "Precio (USD)"
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    axes = axes.flatten()
    
    print("\n" + "="*50)
    print(f"MÉTRICAS DE VALIDACIÓN AVANZADAS (Paso t+1) [{unit_label}]")
    print("="*50)
    
    for idx, ticker in enumerate(tickers):
        ax = axes[idx]
        p_ticker = preds_real[:, idx, 0]
        t_ticker = tgts_real[:, idx, 0]
        
        rmse = np.sqrt(np.mean((p_ticker - t_ticker) ** 2))
        corr, _ = pearsonr(p_ticker, t_ticker)
        
        if use_returns:
            mape = np.mean(np.abs((t_ticker - p_ticker) / (np.abs(t_ticker) + 1e-8))) * 100
        else:
            mape = np.mean(np.abs((t_ticker - p_ticker) / t_ticker)) * 100
        
        t_prev = np.roll(t_ticker, 1)
        t_prev[0] = t_ticker[0]
        actual_direction = np.sign(t_ticker - t_prev)
        pred_direction = np.sign(p_ticker - t_prev)
        mda = np.mean(actual_direction == pred_direction) * 100
        
        print(f"[{ticker}]")
        print(f"  -> RMSE : {rmse:.4f}     | Correlación: {corr:.4f}")
        print(f"  -> MAPE : {mape:.2f}%     | MDA (Dirección): {mda:.1f}%")
        
        ax.plot(dates_list, t_ticker, label='Real', color='black', linewidth=1.5)
        ax.plot(dates_list, p_ticker, label='Predicción', color='tab:blue', linestyle='--', linewidth=1.5)
        
        ax.set_title(f"{ticker}", fontsize=12, fontweight='bold')
        ax.set_ylabel(unit_label, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        metrics_text = f"RMSE: {rmse:.3f}\nCorr: {corr:.2f}\nMAPE: {mape:.1f}%\nMDA: {mda:.1f}%"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.6)
        ax.text(0.03, 0.95, metrics_text, transform=ax.transAxes, fontsize=8.5,
                verticalalignment='top', bbox=props)
        ax.legend(loc="lower right", fontsize=8)

    plt.suptitle(f"Validación Avanzada: Predicciones vs Reales ({unit_label} - Horizonte t+1)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación del modelo LSTM-GAT.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Ruta al archivo de configuración YAML.')
    parser.add_argument('--use_returns', action='store_true', help='Usar retornos porcentuales.')
    parser.add_argument('--use_rollout', action='store_true', help='Activa evaluación autorregresiva por rollout.')
    parser.add_argument('--seq_len', type=int, help='Longitud de la secuencia de entrada.')
    parser.add_argument('--base_horizon', type=int, help='Horizonte base de salida.')
    parser.add_argument('--rollout_steps', type=int, help='Pasos de rollout.')
    parser.add_argument('--batch_size', type=int, help='Tamaño del lote.')
    parser.add_argument('--save_dir', type=str, help='Directorio de guardado del modelo.')

    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            if config_data:
                for key, value in config_data.items():
                    parser.set_defaults(**{key: value})
        args = parser.parse_args()
    else:
        print(f"[Aviso] No se encontró el archivo '{args.config}'. Usando valores por defecto o CLI.")

    return args


if __name__ == "__main__":
    args = parse_args()

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
    seq_len = args.seq_len
    base_horizon = args.base_horizon
    USE_ROLLOUT = args.use_rollout
    ROLLOUT_STEPS = args.rollout_steps
    horizon = (base_horizon * ROLLOUT_STEPS) if USE_ROLLOUT else base_horizon
    USE_RETURNS = args.use_returns
    BATCH_SIZE = args.batch_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # 1. Preparar datos coherentes con retornos y precios
    valid_set, edge_index, dates_list, scaler_mean, scaler_std, prices_data = download_and_prepare_evaluation_data(
        seq_len=seq_len, horizon=horizon, use_returns=USE_RETURNS
    )
    dyn_valid, stat_valid, tgts_valid = valid_set
    edge_index = edge_index.to(device)
    dyn_valid, stat_valid = dyn_valid.to(device), stat_valid.to(device)

    if tgts_valid.ndim == 3 and tgts_valid.shape[-1] == 1:
        tgts_valid = tgts_valid.squeeze(-1)

    mean_arr = scaler_mean.values.reshape(len(tickers), 1)
    std_arr = scaler_std.values.reshape(len(tickers), 1)

    # 2. Reconstrucción de los targets reales en USD acumulativos si se usan retornos
    if USE_RETURNS:
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

    # 3. Cargar modelo guardado
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    print(f"Cargando pesos desde '{best_model_path}'...")
    
    model = CombinedLSTMGATWithStatic3Hop(
        dynamic_input_dim=1, 
        static_input_dim=3, 
        lstm_hidden_dim=32, 
        gnn_hidden_dim=32, 
        output_dim=base_horizon
    ).to(device)
    
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()

    # 4. Inferencia compatible con Rollout iterativo (muestra a muestra para preservar la autorregresión limpia)
    all_rollout_preds = []
    
    with torch.no_grad():
        for sample_idx in range(dyn_valid.size(0)):
            current_dyn = dyn_valid[sample_idx:sample_idx+1].clone()
            current_stat = stat_valid[sample_idx:sample_idx+1]
            
            if USE_ROLLOUT:
                rollout_preds = []
                for step in range(ROLLOUT_STEPS):
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
            
    # 5. Conversión de predicciones a USD si está activado use_returns
    if USE_RETURNS:
        preds_real_usd = convert_returns_to_usd(preds_np, dates_list, prices_data, std_arr, mean_arr, horizon, tickers)
    else:
        preds_real_usd = preds_np * std_arr.reshape(1, len(tickers), 1) + scaler_mean.values.reshape(1, len(tickers), 1)

    # 6. Evaluar y graficar resultados
    evaluate_model_performance(preds_real_usd, tgts_real_usd, tickers, dates_list, horizon, use_returns=False)