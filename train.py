import argparse
import yaml
import os 
import time
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import mlflow

from model.lstm_gat import CombinedLSTMGATWithStatic3Hop
from early_stopping import EarlyStopping  

#os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.enable_system_metrics_logging()

def download_and_prepare_stock_data(seq_len=30, horizon=7, print_data=True, use_returns=False):
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
    n_stations = len(tickers)
    
    print(f"Descargando datos bursátiles para: {tickers}...")
    data = yf.download(tickers, period="5y", interval="1d", progress=False)['Close']
    data = data.ffill().dropna()

    if use_returns:
        print("-> Activado: Transformando precios absolutos en retornos porcentuales diarios.")
        data = data.pct_change().dropna()

    if print_data:
        plt.figure(figsize=(12, 6))
        for ticker in tickers:
            plt.plot(data.index, data[ticker], label=ticker, linewidth=2)
            
        title_suffix = "Retornos Porcentuales" if use_returns else "Precio de Cierre (USD)"
        plt.title(f"Evolución de los Datos ({title_suffix}) - Últimos 5 años", fontsize=14, fontweight='bold')
        plt.xlabel("Fecha", fontsize=12)
        plt.ylabel(title_suffix, fontsize=12)
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        print("Mostrando gráfico de las series temporales...")
        plt.show()
    
    split_date = data.index[-1] - pd.Timedelta(days=180)
    
    train_data = data[data.index < split_date]
    valid_data = data[data.index >= (split_date - pd.Timedelta(days=seq_len))] 
    
    def process_split(df_subset):
        normalized = (df_subset - data.mean()) / data.std()
        values = normalized.values
        
        dyn_list, tgt_list = [], []
        for i in range(len(values) - seq_len - horizon):
            dyn_list.append(values[i : i + seq_len, :])
            tgt_list.append(values[i + seq_len : i + seq_len + horizon, :])
            
        if len(dyn_list) == 0:
            return None, None
            
        dyn_tensor = torch.tensor(np.array(dyn_list), dtype=torch.float32).permute(0, 2, 1).unsqueeze(-1)
        tgt_tensor = torch.tensor(np.array(tgt_list), dtype=torch.float32).permute(0, 2, 1)
        return dyn_tensor, tgt_tensor

    print("Procesando conjuntos de entrenamiento y validación...")
    dyn_train, tgts_train = process_split(train_data)
    dyn_valid, tgts_valid = process_split(valid_data)
    
    stat_train = torch.randn(dyn_train.size(0), n_stations, 3)
    stat_valid = torch.randn(dyn_valid.size(0), n_stations, 3)
    
    corr_matrix = ((train_data - data.mean()) / data.std()).corr().values
    edges_src, edges_dst = [], []
    for i in range(n_stations):
        for j in range(n_stations):
            if i != j and abs(corr_matrix[i, j]) > 0.7:
                edges_src.append(i)
                edges_dst.append(j)
                
    if len(edges_src) == 0:
        edges_src = [0, 1, 2, 3, 4, 5]
        edges_dst = [1, 2, 3, 4, 5, 0]
        
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    
    return (dyn_train, stat_train, tgts_train), (dyn_valid, stat_valid, tgts_valid), edge_index


class DirectionalMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(DirectionalMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, preds, targets):
        mse_loss = self.mse(preds, targets)
        sign_agreement = torch.sign(preds) * torch.sign(targets)
        directional_penalty = torch.mean(torch.relu(-sign_agreement))
        total_loss = mse_loss + self.alpha * directional_penalty
        return total_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo LSTM-GAT para series temporales bursátiles.")
    
    parser.add_argument('--config', type=str, default='config.yaml', help='Ruta al archivo de configuración YAML.')
    parser.add_argument('--experiment_name', type=str, default='LSTM-GAT-Stock-Prediction', help='Nombre del experimento en MLflow.')
    parser.add_argument('--use_returns', action='store_true', help='Transforma precios absolutos en retornos.')
    parser.add_argument('--use_directional_loss', action='store_true', help='Usa pérdida híbrida.')
    parser.add_argument('--use_rollout', action='store_true', help='Activa entrenamiento autorregresivo.')
    parser.add_argument('--seq_len', type=int, help='Longitud de la ventana de entrada.')
    parser.add_argument('--base_horizon', type=int, help='Horizonte base de salida.')
    parser.add_argument('--rollout_steps', type=int, help='Pasos de rollout.')
    parser.add_argument('--batch_size', type=int, help='Tamaño del lote.')
    parser.add_argument('--epochs', type=int, help='Número máximo de épocas.')
    parser.add_argument('--patience', type=int, help='Paciencia para Early Stopping.')
    parser.add_argument('--lr', type=float, help='Tasa de aprendizaje.')
    parser.add_argument('--alpha_loss', type=float, help='Peso de la penalización direccional.')
    parser.add_argument('--save_dir', type=str, help='Directorio de guardado.')

    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        for key, value in config_data.items():
            parser.set_defaults(**{key: value})
            
        args = parser.parse_args()
    else:
        print(f"[Aviso] No se encontró el archivo '{args.config}'. Usando valores por defecto o de la línea de comandos.")

    return args


if __name__ == "__main__":
    args = parse_args()

    USE_RETURNS = args.use_returns
    USE_DIRECTIONAL_LOSS = args.use_directional_loss
    USE_ROLLOUT = args.use_rollout
    ROLLOUT_STEPS = args.rollout_steps
    base_horizon = args.base_horizon
    
    seq_len = args.seq_len
    horizon = (base_horizon * ROLLOUT_STEPS) if USE_ROLLOUT else base_horizon
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    PATIENCE = args.patience

    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    last_model_path = os.path.join(args.save_dir, "last_model.pth")
    
    # Inicializar el gestor de Early Stopping externo
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=best_model_path)
    
    # Configuración robusta de MLflow
    #mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        
        train_set, valid_set, edge_index = download_and_prepare_stock_data(
            seq_len=seq_len, horizon=horizon, print_data=False, use_returns=USE_RETURNS
        )
        dyn_train, stat_train, tgts_train = train_set
        dyn_valid, stat_valid, tgts_valid = valid_set
        
        print(f"\nDimensiones de los datos:")
        print(f" - Train samples: {dyn_train.size(0)} | Valid samples: {dyn_valid.size(0)}")
        
        train_dataset = TensorDataset(dyn_train, stat_train, tgts_train)
        valid_dataset = TensorDataset(dyn_valid, stat_valid, tgts_valid)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {device}")
        
        model = CombinedLSTMGATWithStatic3Hop(
            dynamic_input_dim=1, static_input_dim=3, lstm_hidden_dim=32, gnn_hidden_dim=32, output_dim=base_horizon
        ).to(device)
        
        edge_index = edge_index.to(device)
        
        if USE_DIRECTIONAL_LOSS:
            print(f"-> Activado: Usando función de pérdida híbrida (MSE + Direccional con alpha={args.alpha_loss}).\n")
            criterion = DirectionalMSELoss(alpha=args.alpha_loss)
        else:
            print("-> Usando función de pérdida estándar (MSELoss puro).\n")
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        history_train_loss = []
        history_valid_loss = []
        
        for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            
            # --- FASE DE ENTRENAMIENTO ---
            model.train()
            train_loss = 0.0

            train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] [Train]", leave=False)
            for batch_dyn, batch_stat, batch_tgt in train_bar:
                batch_dyn = batch_dyn.to(device)
                batch_stat = batch_stat.to(device)
                batch_tgt = batch_tgt.to(device)

                optimizer.zero_grad()

                if USE_ROLLOUT:
                    current_dyn = batch_dyn.clone()
                    rollout_preds = []

                    for step in range(ROLLOUT_STEPS):
                        pred_block = model(current_dyn, batch_stat, edge_index)

                        if pred_block.ndim == 2:
                            pred_block = pred_block.unsqueeze(-1)

                        rollout_preds.append(pred_block)
                        pred_block_4d = pred_block.unsqueeze(-1)
                        current_dyn = torch.cat([current_dyn[:, :, base_horizon:, :], pred_block_4d], dim=2)

                    predictions = torch.cat(rollout_preds, dim=-1)

                    if batch_tgt.ndim == 4:
                        batch_tgt = batch_tgt.squeeze(-1)
                    
                    total_target_steps = base_horizon * ROLLOUT_STEPS
                    if batch_tgt.shape[-1] != total_target_steps and batch_tgt.shape[1] == total_target_steps:
                        batch_tgt = batch_tgt.permute(0, 2, 1)

                    loss = criterion(predictions, batch_tgt[..., :total_target_steps])

                else:
                    predictions = model(batch_dyn, batch_stat, edge_index)

                    if batch_tgt.ndim == 3 and batch_tgt.shape[-1] == 1:
                        batch_tgt = batch_tgt.squeeze(-1)

                    target_slice = batch_tgt[..., :base_horizon] if batch_tgt.ndim == 3 else batch_tgt
                    loss = criterion(predictions, target_slice)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_dyn.size(0)
                train_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss /= len(train_loader.dataset)
            history_train_loss.append(train_loss)
            
            # --- FASE DE VALIDACIÓN ---
            model.eval()
            valid_loss = 0.0
            
            valid_bar = tqdm(valid_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] [Valid]", leave=False)
            with torch.no_grad():
                for batch_dyn, batch_stat, batch_tgt in valid_bar:
                    batch_dyn = batch_dyn.to(device)
                    batch_stat = batch_stat.to(device)
                    batch_tgt = batch_tgt.to(device)
                    
                    if USE_ROLLOUT:
                        current_dyn = batch_dyn.clone()
                        rollout_preds = []

                        for step in range(ROLLOUT_STEPS):
                            pred_block = model(current_dyn, batch_stat, edge_index)

                            if pred_block.ndim == 2:
                                pred_block = pred_block.unsqueeze(-1)

                            rollout_preds.append(pred_block)
                            pred_block_4d = pred_block.unsqueeze(-1)
                            current_dyn = torch.cat([current_dyn[:, :, base_horizon:, :], pred_block_4d], dim=2)

                        predictions = torch.cat(rollout_preds, dim=-1)

                        if batch_tgt.ndim == 4:
                            batch_tgt = batch_tgt.squeeze(-1)
                        
                        total_target_steps = base_horizon * ROLLOUT_STEPS
                        if batch_tgt.shape[-1] != total_target_steps and batch_tgt.shape[1] == total_target_steps:
                            batch_tgt = batch_tgt.permute(0, 2, 1)

                        loss = criterion(predictions, batch_tgt[..., :total_target_steps])
                    else:
                        predictions = model(batch_dyn, batch_stat, edge_index)

                        if batch_tgt.ndim == 3 and batch_tgt.shape[-1] == 1:
                            batch_tgt = batch_tgt.squeeze(-1)

                        target_slice = batch_tgt[..., :base_horizon] if batch_tgt.ndim == 3 else batch_tgt
                        loss = criterion(predictions, target_slice)

                    valid_loss += loss.item() * batch_dyn.size(0)
                    valid_bar.set_postfix(loss=f"{loss.item():.4f}")
                    
            valid_loss /= len(valid_loader.dataset)
            history_valid_loss.append(valid_loss)
            
            epoch_duration = time.time() - epoch_start_time
            loss_ratio = valid_loss / (train_loss + 1e-8)
            
            # --- REGISTRO EXTENDIDO EN MLFLOW ---
            mlflow.log_metrics({
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "loss_ratio": loss_ratio,
                "epoch_duration_sec": epoch_duration
            }, step=epoch)
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] Summary -> Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Ratio: {loss_ratio:.2f}")

            # --- GESTIÓN DE EARLY STOPPING (Clase Externa) ---
            early_stopping(valid_loss, model)
            
            if early_stopping.early_stop:
                print(f"\n[Early Stopping] El modelo no ha mejorado en {PATIENCE} épocas consecutivas. Deteniendo entrenamiento.")
                break

        torch.save(model.state_dict(), last_model_path)
        
        # Registrar artefactos finales en MLflow
        mlflow.log_artifact(best_model_path, artifact_path="models")
        mlflow.log_artifact(last_model_path, artifact_path="models")
        
        print(f"Modelos guardados localmente y registrados en MLflow.")
        print("\n¡Entrenamiento y validación completados con éxito!")

        # 5. GRAFICAR RESULTADOS DE PÉRDIDA
        active_epochs = len(history_train_loss)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, active_epochs + 1), history_train_loss, label='Train Loss', marker='o', linewidth=2)
        plt.plot(range(1, active_epochs + 1), history_valid_loss, label='Valid Loss', marker='s', linewidth=2)
        plt.title("Evolución de la Pérdida (Loss) por Época", fontsize=14, fontweight='bold')
        plt.xlabel("Épocas", fontsize=12)
        plt.ylabel("Pérdida Global", fontsize=12)
        plt.xticks(range(1, active_epochs + 1))
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        plot_path = os.path.join(args.save_dir, "loss_curve.png")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        
        print("Mostrando gráfico de entrenamiento y validación...")
        plt.show()