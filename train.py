"""
train.py - Módulo de entrenamiento
==================================

Este módulo implementa:
1. Loop de entrenamiento con early stopping
2. Validación durante el entrenamiento
3. Guardado del mejor modelo (según AUC en validación)

El entrenamiento es el paso MAP del enfoque bayesiano:
    ω* = argmax_ω [log p(D|ω) + log p(ω)]
    
donde:
    - log p(D|ω) es la log-likelihood (cross-entropy negativa)
    - log p(ω) es el log-prior (regularización L2 / weight decay)

Por tanto, el entrenamiento estándar con cross-entropy + weight decay
nos da exactamente el estimador MAP, que es el punto de partida para Laplace.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

from sklearn.metrics import roc_auc_score

from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, MODELS_DIR, RESULTS_DIR
)
from models import create_deterministic_model, create_mc_dropout_model


class EarlyStopping:
    """
    Early stopping para prevenir overfitting.
    
    Monitorea una métrica de validación y detiene el entrenamiento
    si no mejora después de 'patience' épocas.
    
    Args:
        patience: Épocas a esperar antes de detener
        min_delta: Mejora mínima para considerar progreso
        mode: 'max' para métricas a maximizar (AUC), 'min' para minimizar (loss)
    """
    
    def __init__(
        self,
        patience: int = PATIENCE,
        min_delta: float = 1e-4,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Actualiza el estado y retorna True si debe detenerse.
        
        Args:
            score: Métrica actual de validación
            
        Returns:
            True si se debe detener el entrenamiento
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """
    Clase para manejar el entrenamiento de modelos.
    
    Encapsula todo el proceso de entrenamiento incluyendo:
    - Optimización con Adam
    - Learning rate scheduling
    - Logging de métricas
    - Early stopping
    - Guardado del mejor modelo
    
    El entrenamiento minimiza:
        L(ω) = -log p(D|ω) + (λ/2)||ω||²
        
    que es equivalente a maximizar log p(ω|D) (MAP estimation).
    
    Args:
        model: Modelo PyTorch a entrenar
        device: Dispositivo (CPU/GPU)
        learning_rate: Tasa de aprendizaje inicial
        weight_decay: Regularización L2 (prior Gaussiano)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = DEVICE,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY
    ):
        self.model = model.to(device)
        self.device = device
        self.use_cuda = device.type == 'cuda'

        # Cross-entropy loss: -log p(y|x, ω)
        self.criterion = nn.CrossEntropyLoss()

        # Adam optimizer con weight decay
        # Weight decay λ implementa el prior: p(ω) = N(0, λ^-1 * I)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler: reduce LR cuando la validación deja de mejorar
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )

        # Automatic Mixed Precision (AMP): usa float16 en GPU para 2-3x speedup
        # en hardware con Tensor Cores (RTX, A100, etc.). En CPU no tiene efecto.
        self.scaler = torch.amp.GradScaler(enabled=self.use_cuda)

        # Historial de entrenamiento
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Entrena una época completa.
        
        Una época = una pasada completa por todos los datos de entrenamiento.
        
        Args:
            train_loader: DataLoader de entrenamiento
            
        Returns:
            Tupla (loss_promedio, auc)
        """
        self.model.train()
        
        total_loss = 0.0
        all_probs = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        
        for images, labels in progress_bar:
            # non_blocking=True: transferencia asíncrona CPU→GPU cuando pin_memory=True
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # autocast: usa float16 automáticamente en GPU (float32 en CPU)
            with torch.amp.autocast(enabled=self.use_cuda, device_type=self.device.type):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            # GradScaler escala el loss para evitar underflow en float16
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Acumular estadísticas
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
            # Actualizar barra de progreso
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calcular métricas de la época
        avg_loss = total_loss / len(train_loader.dataset)
        auc = roc_auc_score(all_labels, all_probs)
        
        return avg_loss, auc
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evalúa el modelo en el conjunto de validación.
        
        Args:
            val_loader: DataLoader de validación
            
        Returns:
            Tupla (loss_promedio, auc)
        """
        self.model.eval()
        
        total_loss = 0.0
        all_probs = []
        all_labels = []
        
        for images, labels in tqdm(val_loader, desc='Validation', leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(enabled=self.use_cuda, device_type=self.device.type):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader.dataset)
        auc = roc_auc_score(all_labels, all_probs)
        
        return avg_loss, auc
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = NUM_EPOCHS,
        save_path: Optional[Path] = None
    ) -> Dict[str, list]:
        """
        Entrena el modelo completo.
        
        Implementa el loop de entrenamiento con:
        - Evaluación en validación cada época
        - Early stopping basado en AUC de validación
        - Guardado del mejor modelo
        
        Args:
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            num_epochs: Número máximo de épocas
            save_path: Ruta para guardar el mejor modelo
            
        Returns:
            Diccionario con historial de entrenamiento
        """
        early_stopping = EarlyStopping(patience=PATIENCE, mode='max')
        best_auc = 0.0
        
        if save_path is None:
            save_path = MODELS_DIR / 'best_model.pt'
        
        print(f"Iniciando entrenamiento por {num_epochs} épocas máximo...")
        print(f"Dispositivo: {self.device}")
        print(f"Early stopping patience: {PATIENCE}")
        print("-" * 50)
        
        for epoch in range(1, num_epochs + 1):
            # Entrenar una época
            train_loss, train_auc = self.train_epoch(train_loader)
            
            # Validar
            val_loss, val_auc = self.validate(val_loader)
            
            # Actualizar scheduler
            self.scheduler.step(val_auc)
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            # Logging
            print(f"Época {epoch:3d}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
            
            # Guardar mejor modelo
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ Nuevo mejor modelo guardado (AUC: {best_auc:.4f})")
            
            # Early stopping
            if early_stopping(val_auc):
                print(f"\nEarly stopping en época {epoch}")
                break
            
            print("-" * 50)
        
        print(f"\nEntrenamiento completado. Mejor AUC: {best_auc:.4f}")
        
        # Guardar historial
        history_path = RESULTS_DIR / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def load_best_model(self, model_path: Optional[Path] = None):
        """
        Carga el mejor modelo guardado.
        
        Args:
            model_path: Ruta al modelo (usa default si None)
        """
        if model_path is None:
            model_path = MODELS_DIR / 'best_model.pt'
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Modelo cargado desde {model_path}")


def train_deterministic_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    save_path: Optional[Path] = None
) -> Tuple[nn.Module, Dict]:
    """
    Función de conveniencia para entrenar modelo determinista.

    Args:
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        num_epochs: Número de épocas
        save_path: Ruta para guardar el mejor modelo

    Returns:
        Tupla (modelo_entrenado, historial)
    """
    if save_path is None:
        save_path = MODELS_DIR / 'deterministic_best.pt'

    model = create_deterministic_model()
    trainer = Trainer(model)
    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        save_path=save_path
    )

    # Cargar mejor modelo
    trainer.load_best_model(save_path)

    return trainer.model, history


def train_mc_dropout_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    save_path: Optional[Path] = None
) -> Tuple[nn.Module, Dict]:
    """
    Función de conveniencia para entrenar modelo MC Dropout.

    El entrenamiento es idéntico al determinista.
    La diferencia es que en inferencia mantenemos dropout activo.

    Args:
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        num_epochs: Número de épocas
        save_path: Ruta para guardar el mejor modelo

    Returns:
        Tupla (modelo_entrenado, historial)
    """
    if save_path is None:
        save_path = MODELS_DIR / 'mc_dropout_best.pt'

    model = create_mc_dropout_model()
    trainer = Trainer(model)
    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        save_path=save_path
    )

    # Cargar mejor modelo
    trainer.load_best_model(save_path)

    return trainer.model, history
