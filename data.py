# Latest version
"""
data.py - Módulo de carga y preprocesamiento de datos
======================================================

Este módulo maneja:
1. Descarga del dataset de Kaggle (si es necesario)
2. División en train/val/test
3. Augmentaciones de datos
4. Creación de DataLoaders

Dataset: Histopathologic Cancer Detection
- Imágenes de 96x96 píxeles
- Etiquetas binarias: 0 (no tumor) / 1 (tumor)
- Las imágenes son parches de tejido de biopsias

Referencia: https://www.kaggle.com/c/histopathologic-cancer-detection
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED
)


class HistopathDataset(Dataset):
    """
    Dataset personalizado para imágenes histopatológicas.
    
    Implementa el protocolo de PyTorch Dataset:
    - __len__: retorna el número de muestras
    - __getitem__: retorna una muestra (imagen, etiqueta)
    
    Args:
        image_dir: Directorio con las imágenes
        labels_df: DataFrame con columnas 'id' y 'label'
        transform: Transformaciones a aplicar a las imágenes
    """
    
    def __init__(
        self,
        image_dir: Path,
        labels_df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None
    ):
        self.image_dir = Path(image_dir)
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.labels_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Obtener ID y etiqueta
        row = self.labels_df.iloc[idx]
        image_id = row['id']
        label = int(row['label'])
        
        # Cargar imagen
        image_path = self.image_dir / f"{image_id}.tif"
        image = Image.open(image_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(mode: str = 'train') -> transforms.Compose:
    """
    Define las transformaciones para cada fase.
    
    Augmentaciones de entrenamiento:
    - RandomHorizontalFlip: Volteo horizontal aleatorio
    - RandomVerticalFlip: Volteo vertical aleatorio  
    - RandomRotation: Rotación aleatoria (las células no tienen orientación fija)
    - ColorJitter: Variaciones en brillo/contraste (simula variabilidad de tinción)
    
    Para validación/test: Solo normalización (queremos evaluar sin augmentaciones)
    
    Normalizamos con media/std de ImageNet porque usamos modelos preentrenados.
    
    Args:
        mode: 'train', 'val', o 'test'
        
    Returns:
        Compose con las transformaciones
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Validación y test: solo normalización
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def create_data_splits(
    labels_path: Optional[Path] = None,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en train/val/test manteniendo la proporción de clases.

    La estratificación es importante en problemas desbalanceados para asegurar
    que cada split tenga la misma proporción de clases positivas/negativas.

    Args:
        labels_path: Ruta al CSV con etiquetas (por defecto: DATA_DIR/train_labels.csv)
        stratify: Si True, mantiene proporción de clases en cada split

    Returns:
        Tupla de DataFrames (train, val, test)
    """
    if labels_path is None:
        labels_path = DATA_DIR / "train_labels.csv"

    # Cargar etiquetas
    labels_df = pd.read_csv(labels_path)
    
    # Calcular tamaños de splits
    # Primero separamos test, luego dividimos el resto en train/val
    test_size = TEST_RATIO
    val_size = VAL_RATIO / (1 - TEST_RATIO)  # Proporción relativa al restante
    
    stratify_col = labels_df['label'] if stratify else None
    
    # Primera división: train+val vs test
    train_val_df, test_df = train_test_split(
        labels_df,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=stratify_col
    )
    
    # Segunda división: train vs val
    stratify_col = train_val_df['label'] if stratify else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=RANDOM_SEED,
        stratify=stratify_col
    )
    
    print(f"División del dataset:")
    print(f"  - Train: {len(train_df)} muestras ({len(train_df)/len(labels_df)*100:.1f}%)")
    print(f"  - Val:   {len(val_df)} muestras ({len(val_df)/len(labels_df)*100:.1f}%)")
    print(f"  - Test:  {len(test_df)} muestras ({len(test_df)/len(labels_df)*100:.1f}%)")
    
    # Verificar balance de clases
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        pos_ratio = df['label'].mean()
        print(f"  - {name} clase positiva: {pos_ratio*100:.1f}%")
    
    return train_df, val_df, test_df


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea los DataLoaders para train, val y test.

    DataLoader maneja:
    - Batching: agrupa muestras en batches
    - Shuffling: aleatoriza el orden (solo en train)
    - Prefetching: carga datos en paralelo con num_workers
    - Pin memory: acelera transferencia CPU->GPU

    Args:
        train_df: DataFrame de entrenamiento con columnas 'id' y 'label'
        val_df: DataFrame de validación
        test_df: DataFrame de test
        batch_size: Tamaño del batch
        num_workers: Workers para carga paralela

    Returns:
        Tupla de DataLoaders (train, val, test)
    """
    image_dir = DATA_DIR / "train"

    # Crear datasets con transformaciones apropiadas
    train_dataset = HistopathDataset(
        image_dir=image_dir,
        labels_df=train_df,
        transform=get_transforms('train')
    )

    val_dataset = HistopathDataset(
        image_dir=image_dir,
        labels_df=val_df,
        transform=get_transforms('val')
    )

    test_dataset = HistopathDataset(
        image_dir=image_dir,
        labels_df=test_df,
        transform=get_transforms('test')
    )
    
    # pin_memory solo es útil con CUDA: transfiere datos a memoria fija del host
    # para acelerar la copia asíncrona CPU→GPU con non_blocking=True
    pin = torch.cuda.is_available()

    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Importante para SGD
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True  # Evita batches pequeños al final
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin
    )
    
    return train_loader, val_loader, test_loader


def get_sample_for_visualization(
    dataloader: DataLoader,
    n_samples: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Obtiene una muestra de imágenes para visualización.
    
    Útil para:
    - Verificar que las augmentaciones son correctas
    - Visualizar ejemplos con alta/baja incertidumbre
    
    Args:
        dataloader: DataLoader del cual extraer muestras
        n_samples: Número de muestras a extraer
        
    Returns:
        Tupla (imágenes, etiquetas)
    """
    images, labels = next(iter(dataloader))
    return images[:n_samples], labels[:n_samples]


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Desnormaliza una imagen para visualización.
    
    Invierte la normalización de ImageNet:
    x_original = x_normalized * std + mean
    
    Args:
        image: Tensor normalizado [C, H, W]
        
    Returns:
        Tensor en rango [0, 1] para visualización
    """
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    
    image = image.cpu() * std + mean
    return torch.clamp(image, 0, 1)
