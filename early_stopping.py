import os
import torch

class EarlyStopping:
    """
    Gestiona el criterio de parada temprana (Early Stopping) y el guardado 
    del mejor modelo basado en la pérdida de validación.
    """
    def __init__(self, patience=7, verbose=True, delta=0, path='checkpoints/best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.delta = delta
        self.path = path
        
        # Asegurar que el directorio de guardado exista
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f" -> Sin mejora ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Guarda el modelo cuando la pérdida de validación disminuye."""
        if self.verbose:
            print(f" -> ¡Mejora detectada! Guardado mejor modelo en '{self.path}'")
        torch.save(model.state_dict(), self.path)