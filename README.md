LSTM-GNN model taken from: **"A GNN Routing Module Is All You Need for LSTM Rainfall-Runoff Models, Hamidreza Mosaffa, Florian Pappenberger, Christel Prudhomme, Matthew Chantry, Christoph Rüdiger, Hannah Cloke"**

## 📋 Overview

This repository contains the implementation of a novel LSTM-Graph Neural Network (GNN) framework for rainfall-runoff modeling that explicitly integrates runoff generation and spatial flow routing. The framework combines:

- **LSTM networks** for local temporal runoff generation at each subbasin
- **Graph Neural Networks** for spatial flow routing across the river network topology

## 🏗️ Architecture

```
Input: [Precipitation, Temperature, Soil Moisture] + Static Catchment Attributes
    ↓
LSTM Encoder (per subbasin)
    ↓
Node Embeddings [temporal + static features]
    ↓
GNN Module (GAT/GCN/GraphSAGE/ChebNet)
    ↓
Output: Daily Discharge Predictions
```

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.0+
- CUDA 12.0+ (optional, for GPU acceleration)

```

---

## 📁 Repository Structure

```text
GNN_flow_routing/
│
├── model/                               # Neural network architectures and components
├── saved_models_*/                      # Checkpoints and trained model weights for different configurations
├── mlruns/ & mlflow.db                  # MLflow tracking database and experiment logs
│
├── config.yaml                          # Hyperparameters and run configuration settings
├── train.py                             # Main script for model training
├── eval.py                              # Evaluation script for model performance metrics
├── test                                 # Different test scripts
├── early_stopping.py                    # Early stopping utility for training optimization
├── streamlit_app.py                     # Interactive Streamlit dashboard for visualization
└── README.md
```



## ⚙️ Configuration (`config.yaml`)

The `config.yaml` file contains all the hyperparameters that define the model architecture, training configuration, and run characteristics:

```yaml
# General configuration for LSTM-GAT training
use_returns: true
use_directional_loss: true
use_rollout: false

seq_len: 30
base_horizon: 7
rollout_steps: 7
batch_size: 32
epochs: 30
patience: 7
lr: 0.001
alpha_loss: 0.3
save_dir: "saved_models_h7_TT_NR_test"
experiment_name: "LSTM-GAT_h7_TT_NR_test"
```

---

## 🚀 Installation & Usage Guide

### 1. Environment Setup & Dependencies

First, ensure you have Python and `uv` installed. To install PyTorch with GPU support, run:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Then, install the rest of the required dependencies using the requirements file:

```bash
uv pip install -r requirements.txt
```

### 2. Training the Model

To train the model using the hyperparameters defined in `config.yaml`:

```bash
python train.py
```

### 3. Evaluating the Model

To evaluate the trained model on test data:

```bash
python eval.py
```

### 4. Tracking Experiments with MLflow

To view and analyze your trained models, parameters, and metrics through the MLflow web interface:

```bash
mlflow ui
```

### 5. Deploying the Interactive Dashboard

To launch the Streamlit application for visual exploration and model deployment:

```bash
streamlit run streamlit_app.py
```

---

## 📄 License

This project is open-source and available under the terms of the [MIT License](LICENSE).