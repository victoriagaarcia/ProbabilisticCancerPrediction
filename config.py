"""
config.py - Configuración centralizada del proyecto
====================================================

Este archivo contiene todos los hiperparámetros y configuraciones del proyecto.
Centralizar la configuración facilita la experimentación y reproducibilidad.

Referencia teórica:
- El weight_decay actúa como prior Gaussiano isótropo sobre los pesos (BNN02_Laplace)
- El dropout_rate define la probabilidad de "apagar" neuronas en MC Dropout (BNN04)
"""

import torch
from pathlib import Path

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================
# Estructura de directorios para organizar datos, modelos y resultados
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Crear directorios si no existen
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONFIGURACIÓN DEL DISPOSITIVO
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 0  # Workers para DataLoader

# =============================================================================
# CONFIGURACIÓN DEL DATASET
# =============================================================================
# El dataset de Kaggle tiene imágenes de 96x96 píxeles
IMAGE_SIZE = 96
NUM_CLASSES = 2  # Clasificación binaria: tumor vs no tumor

# División del dataset (como se indica en el documento)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Normalización con estadísticas de ImageNet (para modelos preentrenados)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# =============================================================================
BATCH_SIZE = 64
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4

# Weight decay: Actúa como prior Gaussiano isótropo p(ω) = N(0, λ^-1 * I)
# Matemáticamente: log p(ω) ∝ -λ/2 * ||ω||²
# Esto es CRUCIAL para que Laplace Approximation sea válida
WEIGHT_DECAY = 1e-4

# Early stopping para evitar overfitting
PATIENCE = 5

# =============================================================================
# CONFIGURACIÓN DE MC DROPOUT
# =============================================================================
# Probabilidad de dropout: cada peso se "apaga" con probabilidad p
# El posterior aproximado es q(ω) = Bernoulli mixture
DROPOUT_RATE = 0.3

# Número de forward passes en inferencia
# Más muestras = mejor aproximación de la integral predictiva
# p(y|x,D) ≈ (1/T) * Σ p(y|x, ω_t) donde ω_t ~ q(ω)
MC_SAMPLES = 50

# =============================================================================
# CONFIGURACIÓN DE LAPLACE APPROXIMATION
# =============================================================================
# 'last_layer': Aplica Laplace solo a la última capa (eficiente)
# 'all': Aplica a toda la red (intractable para redes grandes)
LAPLACE_SUBSET = 'last_layer'

# Estructura del Hessiano:
# 'kron': Aproximación Kronecker-factored (eficiente, buena aproximación)
# 'diag': Solo diagonal del Hessiano (muy eficiente, peor aproximación)
# 'full': Hessiano completo (solo viable para redes pequeñas)
LAPLACE_HESSIAN = 'kron'

# Número de muestras para predicción probabilística
LAPLACE_SAMPLES = 100

# =============================================================================
# CONFIGURACIÓN DE MÉTRICAS DE CALIBRACIÓN
# =============================================================================
# Expected Calibration Error (ECE): número de bins para agrupar predicciones
# Más bins = medida más granular, pero necesitas más datos por bin
ECE_NUM_BINS = 15

# =============================================================================
# SEMILLA PARA REPRODUCIBILIDAD
# =============================================================================
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    """
    Fija todas las semillas para reproducibilidad.
    
    Es importante fijar semillas en:
    - Python random
    - NumPy
    - PyTorch CPU y CUDA
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinismo adicional (puede reducir rendimiento)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
