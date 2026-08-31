import torch
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from model.lstm_gat import CombinedLSTMGATWithStatic3Hop

def download_and_prepare_evaluation_data(seq_len=30, horizon=7):
    """
    Descarga y prepara los datos exactamente igual que en el entrenamiento 
    para aislar el conjunto de validación y extraer las fechas de los targets.
    """
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
    n_stations = len(tickers)
    
    print(f"Descargando datos bursátiles para evaluación: {tickers}...")
    data = yf.download(tickers, period="5y", interval="1d", progress=False)['Close']
    data = data.ffill().dropna()

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
    
    return (dyn_valid, stat_valid, tgts_valid), edge_index, dates_list, mean_val, std_val


def compare_models_performance(models_preds_real, tgts_real, tickers, dates_list, horizon):
    """
    Compara múltiples modelos graficando sus curvas para t+1 y evaluando 
    la degradación del RMSE y la Correlación frente al horizonte para cada uno.
    """
    n_stations = len(tickers)
    model_names = list(models_preds_real.keys())
    
    # Paleta de colores extendida para los 7 modelos (0.3 a 0.9)
    colors = {
        'saved_models_0p3': 'tab:blue', 
        'saved_models_0p4': 'tab:orange', 
        'saved_models_0p5': 'tab:green',
        'saved_models_0p6': 'tab:red',
        'saved_models_0p7': 'tab:purple',
        'saved_models_0p8': 'tab:brown',
        'saved_models_0p9': 'tab:pink',
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
        
        # Pintar valor real una sola vez
        ax.plot(dates_list, t_ticker, label='Real', color='black', linewidth=1.5, zorder=5)
        
        print(f"\n[{ticker}]")
        for name in model_names:
            p_ticker = models_preds_real[name][:, idx, 0]
            
            rmse = np.sqrt(np.mean((p_ticker - t_ticker) ** 2))
            corr, _ = pearsonr(p_ticker, t_ticker)
            mape = np.mean(np.abs((t_ticker - p_ticker) / t_ticker)) * 100
            
            print(f"  -> {name} | RMSE: {rmse:.2f} USD | Corr: {corr:.4f} | MAPE: {mape:.2f}%")
            
            # Línea de predicción de cada modelo
            ax.plot(dates_list, p_ticker, label=f'Pred {name}', color=colors.get(name, 'gray'), linestyle='--', linewidth=1.2)
        
        ax.set_title(f"{ticker}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Precio (USD)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=6.5)

    plt.suptitle("Comparativa de Modelos (Thresholds 0.3 a 0.9) - Horizonte t+1", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 2. ANÁLISIS DE DEGRADACIÓN POR HORIZONTE COMPARADO
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
            step_corrs = []
            step_rmses = []
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

        # Gráficas de error con barras de desviación estándar
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

    plt.suptitle("Comparativa de Degradación del Rendimiento por Horizonte Temporal (7 Modelos)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
    seq_len = 30
    horizon = 7  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # 1. Preparar datos
    valid_set, edge_index, dates_list, mean_val, std_val = download_and_prepare_evaluation_data(seq_len=seq_len, horizon=horizon)
    dyn_valid, stat_valid, tgts_valid = valid_set
    dyn_valid = dyn_valid.to(device)
    stat_valid = stat_valid.to(device)
    edge_index = edge_index.to(device)
    
    if tgts_valid.ndim == 3 and tgts_valid.shape[-1] == 1:
        tgts_valid = tgts_valid.squeeze(-1)
    tgts_np = tgts_valid.numpy()
    
    # Des-normalizar targets reales
    mean_arr = mean_val.values.reshape(1, len(tickers), 1)
    std_arr = std_val.values.reshape(1, len(tickers), 1)
    tgts_real = tgts_np * std_arr + mean_arr

    # 2. Definir los 7 modelos a evaluar (de 0.3 a 0.9)
    model_paths = {
        #'saved_models_0p3': "saved_models_0p3/best_model.pth",
        #'saved_models_0p4': "saved_models_0p4/best_model.pth",
        #'saved_models_0p5': "saved_models_0p5/best_model.pth",
        #'saved_models_0p6': "saved_models_0p6/best_model.pth",
        'saved_models_0p7': "saved_models_0p7/best_model.pth",
        #'saved_models_0p8': "saved_models_0p8/best_model.pth",
        #'saved_models_0p9': "saved_models_0p9/best_model.pth"
    }
    
    models_preds_real = {}

    # 3. Iterar, cargar e inferir cada modelo
    for model_name, path in model_paths.items():
        print(f"Cargando y evaluando modelo '{model_name}' desde '{path}'...")
        
        model = CombinedLSTMGATWithStatic3Hop(
            dynamic_input_dim=1, 
            static_input_dim=3, 
            lstm_hidden_dim=32, 
            gnn_hidden_dim=32, 
            output_dim=horizon
        ).to(device)
        
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()

        with torch.no_grad():
            predictions = model(dyn_valid, stat_valid, edge_index)
            
        preds_np = predictions.cpu().numpy()
        models_preds_real[model_name] = preds_np * std_arr + mean_arr

    # 4. Lanzar la función de comparación unificada
    compare_models_performance(models_preds_real, tgts_real, tickers, dates_list, horizon)