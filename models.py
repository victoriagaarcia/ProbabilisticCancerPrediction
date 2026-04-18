# Latest version
"""
models.py - Definición de modelos
=================================

Este módulo contiene:
1. CNN Determinista (baseline): ResNet18 con fine-tuning
2. Wrapper para Laplace Approximation
3. Modelo con MC Dropout

Flujo del proyecto:
1. Entrenar CNN determinista → obtener MAP ω*
2. Aplicar Laplace sobre ω* → obtener q(ω) = N(ω | ω*, Σ)
3. Entrenar CNN con dropout → obtener modelo para MC Dropout

Fundamento teórico (BNN02_Laplace):
- El entrenamiento con weight decay equivale a maximizar log p(ω|D)
- Weight decay λ corresponde a prior p(ω) = N(0, λ^-1 * I)
- Por tanto, la solución es el MAP: ω* = argmax_ω [log p(D|ω) + log p(ω)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional, Union, Any
from pathlib import Path
import numpy as np
import pickle

try:
    from laplace import Laplace
except ImportError:
    Laplace = None  # Se comprobará en tiempo de uso

from config import (
    NUM_CLASSES, DEVICE, DROPOUT_RATE, MC_SAMPLES,
    LAPLACE_SUBSET, LAPLACE_HESSIAN, LAPLACE_SAMPLES
)


class DeterministicCNN(nn.Module):
    """
    CNN determinista basada en ResNet18 preentrenada.
    
    Arquitectura:
    - Backbone: ResNet18 (primeras capas congeladas)
    - Clasificador: Capa fully-connected personalizada
    
    El fine-tuning se hace solo en las últimas capas porque:
    1. Las primeras capas capturan features genéricos (bordes, texturas)
    2. Las últimas capas capturan features específicos del dominio
    3. Menos parámetros a entrenar = menos overfitting
    
    Args:
        pretrained: Si True, carga pesos preentrenados de ImageNet
        freeze_backbone: Si True, congela las primeras capas
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        # Cargar ResNet18 preentrenada
        # weights='IMAGENET1K_V1' es equivalente a pretrained=True en versiones nuevas
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Obtener dimensión de features antes del clasificador original
        num_features = self.backbone.fc.in_features  # 512 para ResNet18
        
        # Reemplazar el clasificador original por uno nuevo
        # El clasificador original fue entrenado para 1000 clases de ImageNet
        self.backbone.fc = nn.Linear(num_features, NUM_CLASSES)
        
        # Congelar capas del backbone (opcional)
        if freeze_backbone:
            self._freeze_backbone_layers()
    
    def _freeze_backbone_layers(self):
        """
        Congela las primeras capas del backbone.
        
        Congelamos: conv1, bn1, layer1, layer2
        Entrenamos: layer3, layer4, fc
        
        Esto reduce el número de parámetros a optimizar y
        previene el olvido catastrófico de features útiles.
        """
        # Capas a congelar
        layers_to_freeze = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.layer1,
            self.backbone.layer2
        ]
        
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Contar parámetros
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Parámetros totales: {total_params:,}")
        print(f"Parámetros entrenables: {trainable_params:,}")
        print(f"Parámetros congelados: {total_params - trainable_params:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de imágenes [B, 3, H, W]
            
        Returns:
            Logits [B, NUM_CLASSES]
        """
        return self.backbone(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtiene probabilidades (aplicando softmax).
        
        Para clasificación binaria, usamos softmax en lugar de sigmoid
        para mantener consistencia con modelos multiclase.
        
        Args:
            x: Tensor de imágenes [B, 3, H, W]
            
        Returns:
            Probabilidades [B, NUM_CLASSES]
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class MCDropoutCNN(nn.Module):
    """
    CNN con MC Dropout para estimación de incertidumbre.
    
    Fundamento teórico (BNN04_MCDropoutDeepEnsembles):
    - Dropout define una familia variacional q(ω) donde cada máscara
      de dropout corresponde a una muestra del posterior aproximado
    - En test time, hacemos T forward passes con dropout activo
    - La predicción final es: p(y|x,D) ≈ (1/T) Σ_t p(y|x, ω_t)
    - La varianza entre predicciones captura la incertidumbre epistémica
    
    Diferencia con modelo determinista:
    - Añadimos capas de Dropout en el clasificador
    - En inferencia, mantenemos dropout activo (model.train())
    
    Args:
        pretrained: Si True, carga pesos preentrenados
        freeze_backbone: Si True, congela primeras capas
        dropout_rate: Probabilidad de dropout (default: 0.3)
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = DROPOUT_RATE
    ):
        super().__init__()
        
        # Cargar ResNet18 preentrenada
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Obtener dimensión de features
        num_features = self.backbone.fc.in_features  # 512
        
        # Reemplazar clasificador por uno con dropout
        # Arquitectura: Linear -> ReLU -> Dropout -> Linear
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),  # Dropout para MC
            nn.Linear(256, NUM_CLASSES)
        )
        
        self.dropout_rate = dropout_rate
        
        # Congelar capas del backbone
        if freeze_backbone:
            self._freeze_backbone_layers()
    
    def _freeze_backbone_layers(self):
        """Congela las primeras capas del backbone."""
        layers_to_freeze = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.layer1,
            self.backbone.layer2
        ]
        
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass estándar."""
        return self.backbone(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = MC_SAMPLES
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predicción con estimación de incertidumbre usando MC Dropout.
        
        Algoritmo:
        1. Activar modo entrenamiento solo para las capas de dropout
        2. Hacer T forward passes, cada uno con diferente máscara
        3. Promediar para obtener predicción media
        4. Calcular varianza como medida de incertidumbre
        
        Interpretación bayesiana:
        - Cada forward pass usa una configuración diferente de pesos
        - Esto es equivalente a muestrear de q(ω)
        - La media aproxima la integral predictiva bayesiana
        
        Args:
            x: Tensor de imágenes [B, 3, H, W]
            n_samples: Número de forward passes
            
        Returns:
            Tupla (mean_probs, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty, all_probs):
            - mean_probs: Probabilidades medias [B, NUM_CLASSES]
            - epistemic_uncertainty: Varianza de las predicciones [B]
            - aleatoric_uncertainty: Incertidumbre aleatoria [B]
            - total_uncertainty: Incertidumbre total [B]
            - all_probs: Todas las muestras [n_samples, B, NUM_CLASSES]
        """
        was_training = self.training
        
        # Ponemos el modelo en modo evaluación para que BatchNorm no actualice sus estadísticas,
        # pero activamos solo las capas de Dropout para que sigan siendo estocásticas
        self.eval()  # BatchNorm en eval
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Dropout en train para muestreo
        
        all_probs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)
        
        # Restaurar modo original
        if not was_training:
            self.eval()
        
        # Stack: [n_samples, B, NUM_CLASSES]
        all_probs = torch.stack(all_probs, dim=0)
        
        # Media sobre muestras: [B, NUM_CLASSES]
        mean_probs = all_probs.mean(dim=0)

        # Descomposición de incertidumbre (ley de la varianza total):
        # p_t = probabilidad de clase positiva en la muestra t: [n_samples, B]
        p_positive = all_probs[:, :, 1]  # [n_samples, B]

        # Epistémica: debida a la incertidumbre en los pesos, se refleja en la varianza entre las muestras
        # (alta cuando las máscaras de dropout generan predicciones muy diferentes)
        epistemic_uncertainty = p_positive.var(dim=0)  # [B]
        
        # Aleatoria: debida al ruido inherente en los datos, se refleja en la media de p(1-p) sobre las muestras
        # (alta cuando la media está cerca de 0.5, indicando incertidumbre intrínseca)
        aleatoric_uncertainty = (p_positive * (1 - p_positive)).mean(dim=0)  # [B]

        # Incertidumbre total: suma de ambas componentes
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty  # [B]
        
        return mean_probs, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty, all_probs


class LaplaceWrapper:
    """
    Wrapper para aplicar Laplace Approximation a un modelo entrenado.
    
    Fundamento teórico (BNN02_Laplace):
    
    La idea central es aproximar el posterior p(ω|D) con una Gaussiana:
    
        q(ω) = N(ω | ω*, Σ)
    
    donde:
        - ω* es el MAP (pesos del modelo entrenado)
        - Σ = H^{-1} donde H es el Hessiano de la log-posterior en ω*
    
    Para clasificación, el Hessiano se aproxima con el GGN
    (Generalized Gauss-Newton), que es siempre semi-definido positivo.
    
    La librería `laplace-torch` implementa esto eficientemente:
    - Usa aproximaciones Kronecker-factored del Hessiano
    - Permite aplicar Laplace solo a la última capa (eficiente)
    
    Limitación importante:
    - Laplace es una aproximación LOCAL (solo ve la curvatura en ω*)
    - Si el posterior es multimodal, la Gaussiana lo subestimará
    
    Args:
        model: Modelo PyTorch entrenado (MAP)
        likelihood: Tipo de problema ('classification' o 'regression')
    """
    
    def __init__(
        self,
        model: nn.Module,
        likelihood: str = 'classification'
    ):
        self.model = model
        self.likelihood = likelihood
        self.la = None
        self.fitted = False
    
    def fit(self, train_loader) -> 'LaplaceWrapper':
        """
        Ajusta la aproximación de Laplace.
        
        Pasos:
        1. Calcular el Hessiano de la log-posterior en ω*
        2. Invertir el Hessiano para obtener la covarianza Σ
        3. Optimizar la precisión del prior (hiperparámetro)
        
        Args:
            train_loader: DataLoader de entrenamiento
            
        Returns:
            self (para encadenar llamadas)
        """
        if Laplace is None:
            raise ImportError("Instala laplace-torch: pip install laplace-torch")

        # Crear objeto Laplace
        # subset_of_weights='last_layer' es CRUCIAL para eficiencia
        # El Hessiano completo de ResNet sería intratable (millones de params)
        self.la = Laplace(
            self.model,
            self.likelihood,
            subset_of_weights=LAPLACE_SUBSET,
            hessian_structure=LAPLACE_HESSIAN
        )
        
        print("Calculando Hessiano de la log-posterior...")
        
        # fit() calcula H = ∇²_ω L(ω*) usando el training set
        self.la.fit(train_loader)
        
        print("Optimizando precisión del prior...")
        
        # Optimiza la precisión del prior usando marginal likelihood
        # Esto es una forma de model selection bayesiana
        self.la.optimize_prior_precision(method='marglik')
        
        self.fitted = True
        print("Laplace Approximation ajustada correctamente.")
        
        return self
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = LAPLACE_SAMPLES
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicción con incertidumbre usando Laplace.
        
        El proceso es:
        1. Muestrear pesos de q(ω) = N(ω | ω*, Σ)
        2. Para cada muestra, hacer forward pass
        3. Promediar las predicciones
        
        La varianza predictiva tiene dos componentes:
        - Incertidumbre aleatórica: inherente al ruido de los datos
        - Incertidumbre epistémica: debido a incertidumbre en los pesos
        
        Args:
            x: Tensor de imágenes [B, 3, H, W]
            n_samples: Número de muestras de pesos
            
        Returns:
            Tupla (mean_probs, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty, all_probs):
            - mean_probs: Probabilidades medias [B, NUM_CLASSES]
            - epistemic_uncertainty: Incertidumbre epistémica [B]
            - aleatoric_uncertainty: Incertidumbre aleatoria [B]
            - total_uncertainty: Incertidumbre total [B]
            - all_probs: Todas las muestras [n_samples, B, NUM_CLASSES]
        """
        if not self.fitted:
            raise RuntimeError("Debe llamar fit() antes de predict_with_uncertainty()")

        # predictive_samples muestrea pesos del posterior q(ω) = N(ω*, Σ)
        # y hace un forward pass por cada muestra.
        # Retorna shape: [n_samples, B, num_classes]
        with torch.no_grad():
            all_logits = self.la.predictive_samples(
                x, pred_type='nn', n_samples=n_samples
            )
        
        # Convertir logits a probabilidades
        all_probs = torch.softmax(all_logits, dim=-1)  # [n_samples, B, num_classes]

        # all_probs: [n_samples, B, num_classes]
        mean_probs = all_probs.mean(dim=0)                  # [B, num_classes]

        # En el modelo de Laplace, también descomponemos la incertidumbre:
        # - Epistémica: varianza entre las muestras de predicción
        # - Aleatoria: media de p(1-p) sobre las muestras

        # Probabilidad de clase positiva
        p_positive = all_probs[:, :, 1]  # [n_samples, B]

        # Incertidumbre epistémica (igual que en MC Dropout): varianza entre las muestras
        epistemic_uncertainty = p_positive.var(dim=0)  # [B]

        # Incertidumbre aleatoria: media de p(1-p) sobre las muestras
        aleatoric_uncertainty = (p_positive * (1 - p_positive)).mean(dim=0)  # [B]

        # Incertidumbre total: suma de las dos componentes
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty  # [B]

        return mean_probs, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty, all_probs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Predicción directa usando la media del posterior."""
        if not self.fitted:
            raise RuntimeError("Debe llamar fit() primero")
        
        with torch.no_grad():
            return self.la(x, pred_type='glm', link_approx='probit')
    
    def load(self, path: Union[str, Path]) -> 'LaplaceWrapper':
        """
    Carga un objeto Laplace previamente serializado con pickle.
    
    Args:
        path: Ruta al archivo .pkl generado durante el entrenamiento
        
    Returns:
        self (para encadenar llamadas)
    """
    with open(path, "rb") as f:
        self.la = pickle.load(f)
    self.fitted = True
    return self


def create_deterministic_model(pretrained: bool = True) -> DeterministicCNN:
    """
    Factory function para crear modelo determinista.
    
    Args:
        pretrained: Si True, usa pesos de ImageNet
        
    Returns:
        Modelo en el dispositivo correcto
    """
    model = DeterministicCNN(pretrained=pretrained)
    return model.to(DEVICE)


def create_mc_dropout_model(
    pretrained: bool = True,
    dropout_rate: float = DROPOUT_RATE
) -> MCDropoutCNN:
    """
    Factory function para crear modelo MC Dropout.
    
    Args:
        pretrained: Si True, usa pesos de ImageNet
        dropout_rate: Probabilidad de dropout
        
    Returns:
        Modelo en el dispositivo correcto
    """
    model = MCDropoutCNN(pretrained=pretrained, dropout_rate=dropout_rate)
    return model.to(DEVICE)


def load_model(model_path: Union[str, Path], model_type: str = 'deterministic') -> nn.Module:
    """
    Carga un modelo guardado.
    
    Args:
        model_path: Ruta al archivo .pt
        model_type: 'deterministic' o 'mc_dropout'
        
    Returns:
        Modelo cargado
    """
    if model_type == 'deterministic':
        model = DeterministicCNN(pretrained=False)
    else:
        model = MCDropoutCNN(pretrained=False)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE)

# # Función para cargar un modelo Laplace
# def load_laplace_model(model_path: Union[str, Path]) -> LaplaceWrapper:
#     """
#     Load a fitted Laplace model from disk.

#     The checkpoint must contain:
#     - base_model_state: state_dict of the deterministic base model
#     - laplace_obj: fitted Laplace object
#     - laplace_fitted: whether Laplace was fitted
#     """
#     checkpoint = torch.load(model_path, map_location=DEVICE)

#     base_model = DeterministicCNN(pretrained=False)
#     base_model.load_state_dict(checkpoint['base_model_state'])
#     base_model = base_model.to(DEVICE)
#     base_model.eval()

#     laplace_wrapper = LaplaceWrapper(base_model)
#     laplace_wrapper.la = checkpoint['laplace_obj']
#     laplace_wrapper.fitted = checkpoint.get('laplace_fitted', True)

#     return laplace_wrapper
