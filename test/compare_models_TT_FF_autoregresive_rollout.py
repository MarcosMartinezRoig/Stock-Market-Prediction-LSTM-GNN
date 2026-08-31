import torch
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from model.lstm_gat import CombinedLSTMGATWithStatic3Hop

def prepare_data_for_mode(use_returns=True, seq_len=30, horizon=7):
    """
    Prepara los datos y calcula las estadísticas de normalización 
    específicas para el entorno de retornos.
    """
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
    n_stations = len(tickers)
    
    raw_data = yf.download(tickers, period="5y", interval="1d", progress=False)['Close']
    raw_data = raw_data.ffill().dropna()

    prices_data = raw_data.copy()

    data = raw_data.pct_change().dropna()
    prices_data = prices_data.loc[data.index]

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
            if i != j and abs(corr_matrix[i, j]) > 0.4:
                edges_src.append(i)
                edges_dst.append(j)
                
    if len(edges_src) == 0:
        edges_src = [0, 1, 2, 3, 4, 5]
        edges_dst = [1, 2, 3, 4, 5, 0]
        
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    
    return (dyn_valid, stat_valid, tgts_valid), edge_index, dates_list, mean_val, std_val, prices_data


def compare_models_performance(models_preds_real, tgts_real, tickers, dates_list, horizon):
    """
    Compara los modelos graficando sus curvas en USD y evaluando RMSE y Correlación.
    """
    n_stations = len(tickers)
    model_names = list(models_preds_real.keys())
    
    colors = {
        'saved_models_h1_TT_NR': 'tab:blue',   # H1 No Rollout (autorregresivo en test)
        'saved_models_h1_TT_R': 'tab:orange',  # H1 Rollout (autorregresivo en test)
        'saved_models_h7_TT': 'tab:red'        # H7 Retornos + Híbrida (Directo)
    }
    
    # ==========================================
    # 1. GRÁFICO COMPARATIVO (Paso t+1)
    # ==========================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    print("\n" + "="*60)
    print("COMPARATIVA DE MÉTRICAS DE VALIDACIÓN (Paso t+1)")
    print("="*60)
    
    for idx, ticker in enumerate(tickers):
        ax = axes[idx]
        t_ticker = tgts_real[:, idx, 0]
        
        ax.plot(dates_list, t_ticker, label='Real', color='black', linewidth=1.5, zorder=5)
        
        print(f"\n[{ticker}]")
        for name in model_names:
            p_ticker = models_preds_real[name][:, idx, 0]
            
            rmse = np.sqrt(np.mean((p_ticker - t_ticker) ** 2))
            corr, _ = pearsonr(p_ticker, t_ticker)
            mape = np.mean(np.abs((t_ticker - p_ticker) / t_ticker)) * 100
            
            print(f"  -> {name} | RMSE: {rmse:.2f} USD | Corr: {corr:.4f} | MAPE: {mape:.2f}%")
            
            ax.plot(dates_list, p_ticker, label=f'Pred {name}', color=colors.get(name, 'gray'), linestyle='--', linewidth=1.2)
        
        ax.set_title(f"{ticker}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Precio (USD)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=6.5)

    plt.suptitle("Comparativa Modelos (True, True) - Horizonte t+1", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 2. ANÁLISIS DE DEGRADACIÓN POR HORIZONTE
    # ==========================================
    print("\n" + "="*60)
    print("EVOLUCIÓN DEL ERROR SEGÚN EL HORIZONTE POR MODELO")
    print("="*60)
    
    horizon_steps = list(range(1, horizon + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for name in model_names:
        preds_real = models_preds_real[name]
        mean_corrs, std_corrs = [], []
        mean_rmses, std_rmses = [], []

        for h in range(horizon):
            step_corrs, step_rmses = [], []
            for idx in range(n_stations):
                p_ticker = preds_real[:, idx, h]
                t_ticker = tgts_real[:, idx, h]
                
                rmse = np.sqrt(np.mean((p_ticker - t_ticker) ** 2))
                corr, _ = pearsonr(p_ticker, t_ticker)
                
                step_rmses.append(rmse)
                if not np.isnan(corr):
                    step_corrs.append(corr)
                    
            mean_corrs.append(np.mean(step_corrs))
            std_corrs.append(np.std(step_corrs))
            mean_rmses.append(np.mean(step_rmses))
            std_rmses.append(np.std(step_rmses))
            
        print(f"\nModelo: {name}")
        for h_idx, step in enumerate(horizon_steps):
            print(f"  Horizonte t+{step} -> Corr Media: {mean_corrs[h_idx]:.4f} | RMSE Medio: {mean_rmses[h_idx]:.2f} USD")

        ax1.errorbar(horizon_steps, mean_corrs, yerr=std_corrs, fmt='-o', label=name, color=colors.get(name, 'gray'), elinewidth=1.2, capsize=2)
        ax2.errorbar(horizon_steps, mean_rmses, yerr=std_rmses, fmt='-s', label=name, color=colors.get(name, 'gray'), elinewidth=1.2, capsize=2)

    ax1.set_title("Correlación Media (± Std) vs Horizonte", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Paso en el Horizonte (Días)", fontsize=10)
    ax1.set_ylabel("Correlación de Pearson", fontsize=10)
    ax1.set_xticks(horizon_steps)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=7.5)
    
    ax2.set_title("RMSE Medio (± Std) vs Horizonte", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Paso en el Horizonte (Días)", fontsize=10)
    ax2.set_ylabel("RMSE Medio (USD)", fontsize=10)
    ax2.set_xticks(horizon_steps)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=7.5)

    plt.suptitle("Comparativa de Degradación del Rendimiento por Horizonte Temporal", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
    seq_len = 30
    horizon = 7  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    models_preds_real = {}

    # =========================================================================
    # CARGA DE ENTORNO DE RETORNOS (Común para los 3 modelos True, True)
    # =========================================================================
    v_set_t, edge_idx_t, dates_list, mean_t, std_t, prices_data_t = prepare_data_for_mode(use_returns=True, seq_len=seq_len, horizon=horizon)
    dyn_t, stat_t, tgts_t = v_set_t
    dyn_t, stat_t, edge_idx_t = dyn_t.to(device), stat_t.to(device), edge_idx_t.to(device)
    
    if tgts_t.ndim == 3 and tgts_t.shape[-1] == 1:
        tgts_t = tgts_t.squeeze(-1)
        
    mean_arr_t = mean_t.values.reshape(len(tickers), 1)
    std_arr_t = std_t.values.reshape(len(tickers), 1)

    # Reconstrucción de los targets reales a precios USD acumulativos para la evaluación
    tgts_real_usd = np.zeros_like(tgts_t.numpy())
    for i, d_date in enumerate(dates_list):
        base_idx = prices_data_t.index.get_loc(d_date) - 1
        base_prices = prices_data_t.iloc[base_idx].values.reshape(1, len(tickers))
        
        denorm_rets = tgts_t.numpy()[i] * std_arr_t + mean_arr_t
        curr_prices = base_prices.copy()
        
        horizon_prices = np.zeros((len(tickers), horizon))
        for h in range(horizon):
            curr_prices = curr_prices * (1.0 + denorm_rets[:, h])
            horizon_prices[:, h] = curr_prices.flatten()
            
        tgts_real_usd[i] = horizon_prices


    # Función auxiliar para convertir retornos predichos a precios absolutos en USD acumulativos
    def convert_returns_to_usd(preds_rets, dates_list, prices_data, std_arr, mean_arr, base_horizon):
        model_preds_usd = np.zeros_like(preds_rets)
        for i, d_date in enumerate(dates_list):
            base_idx = prices_data.index.get_loc(d_date) - 1
            base_prices = prices_data.iloc[base_idx].values.reshape(1, len(tickers))
            
            denorm_rets = preds_rets[i] * std_arr + mean_arr
            curr_prices = base_prices.copy()
            
            horizon_prices = np.zeros((len(tickers), base_horizon))
            for h in range(base_horizon):
                curr_prices = curr_prices * (1.0 + denorm_rets[:, h])
                horizon_prices[:, h] = curr_prices.flatten()
                
            model_preds_usd[i] = horizon_prices
        return model_preds_usd


    # =========================================================================
    # 1 & 2. MODELOS H1 (saved_models_h1_TT_NR y saved_models_h1_TT_R) -> Rollout Autorregresivo
    # =========================================================================
    ROLLOUT_STEPS = 7  
    base_horizon = 1

    for model_key, model_folder in [('saved_models_h1_TT_NR', 'saved_models_h1_TT_NR'), ('saved_models_h1_TT_R', 'saved_models_h1_TT_R')]:
        print(f"\n--- Procesando {model_key} (Autorregresivo / Rollout) ---")
        model_h1 = CombinedLSTMGATWithStatic3Hop(
            dynamic_input_dim=1, static_input_dim=3, lstm_hidden_dim=32, gnn_hidden_dim=32, output_dim=base_horizon
        ).to(device)
        model_h1.load_state_dict(torch.load(f"{model_folder}/best_model.pth", map_location=device, weights_only=True))
        model_h1.eval()

        all_rollout_preds = []
        with torch.no_grad():
            for sample_idx in range(dyn_t.size(0)):
                current_dyn = dyn_t[sample_idx:sample_idx+1].clone() 
                current_stat = stat_t[sample_idx:sample_idx+1]
                
                rollout_preds = []
                for step in range(ROLLOUT_STEPS):
                    pred_block = model_h1(current_dyn, current_stat, edge_idx_t) 
                    if pred_block.ndim == 2:
                        pred_block = pred_block.unsqueeze(-1) 
                    
                    rollout_preds.append(pred_block)
                    
                    pred_block_4d = pred_block.unsqueeze(-1) 
                    current_dyn = torch.cat([current_dyn[:, :, base_horizon:, :], pred_block_4d], dim=2)
                
                full_pred_seq = torch.cat(rollout_preds, dim=-1)
                all_rollout_preds.append(full_pred_seq.cpu().numpy())

        preds_h1_rets = np.concatenate(all_rollout_preds, axis=0) 
        models_preds_real[model_key] = convert_returns_to_usd(preds_h1_rets, dates_list, prices_data_t, std_arr_t, mean_arr_t, horizon)


    # =========================================================================
    # 3. MODELO H7 (saved_models_h7_TT) -> Predicción directa de 7 pasos
    # =========================================================================
    print(f"\n--- Procesando saved_models_h7_TT (Horizonte 7 Directo) ---")
    model_h7 = CombinedLSTMGATWithStatic3Hop(
        dynamic_input_dim=1, static_input_dim=3, lstm_hidden_dim=32, gnn_hidden_dim=32, output_dim=horizon
    ).to(device)
    model_h7.load_state_dict(torch.load("saved_models_h7_TT/best_model.pth", map_location=device, weights_only=True))
    model_h7.eval()

    with torch.no_grad():
        preds_h7_rets = model_h7(dyn_t, stat_t, edge_idx_t).cpu().numpy()
        if preds_h7_rets.ndim == 3 and preds_h7_rets.shape[-1] == 1:
            preds_h7_rets = preds_h7_rets.squeeze(-1)

    models_preds_real['saved_models_h7_TT'] = convert_returns_to_usd(preds_h7_rets, dates_list, prices_data_t, std_arr_t, mean_arr_t, horizon)


    # =========================================================================
    # 4. LANZAR COMPARATIVA COMPLETA
    # =========================================================================
    compare_models_performance(models_preds_real, tgts_real_usd, tickers, dates_list, horizon)