"""
metrics.py - Métricas de evaluación
===================================

Este módulo implementa las métricas requeridas según el proyecto:

1. Métricas de clasificación:
   - AUC-ROC: Área bajo la curva ROC
   - F1-Score: Media armónica de precisión y recall
   - Accuracy: Proporción de predicciones correctas

2. Métricas de calibración (FUNDAMENTALES para modelos probabilísticos):
   - ECE (Expected Calibration Error): Mide si la confianza coincide con la accuracy
   - Brier Score: Error cuadrático medio entre probabilidad y etiqueta
   - Reliability Diagram: Visualización de la calibración

Referencia teórica:
La calibración mide si las probabilidades predichas son "honestas":
- Un modelo calibrado con p(y=1|x) = 0.8 debería acertar el 80% de las veces
- ECE = 0 significa calibración perfecta
- Brier Score combina calibración + discriminación

El proyecto requiere comparar la calibración de:
- Modelo determinista (sin incertidumbre)
- Laplace Approximation
- MC Dropout
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
from pathlib import Path

from config import ECE_NUM_BINS, FIGURES_DIR


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calcula métricas de clasificación binaria.
    
    Args:
        y_true: Etiquetas reales [N]
        y_pred_proba: Probabilidades predichas para clase positiva [N]
        threshold: Umbral para convertir probabilidades a clases
        
    Returns:
        Diccionario con métricas
    """
    # Convertir probabilidades a predicciones binarias
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }
    
    return metrics


def compute_ece(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = ECE_NUM_BINS
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula el Expected Calibration Error (ECE).
    
    El ECE mide la diferencia entre la confianza del modelo y la accuracy real:
    
        ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    
    donde:
        - B_b es el conjunto de predicciones en el bin b
        - acc(B_b) es la accuracy real en ese bin
        - conf(B_b) es la confianza media en ese bin
    
    Interpretación:
        - ECE = 0: Calibración perfecta
        - ECE alto: El modelo está sobre/sub-confiado
    
    Args:
        y_true: Etiquetas reales [N]
        y_pred_proba: Probabilidades predichas [N]
        n_bins: Número de bins
        
    Returns:
        Tupla (ECE, accuracies_por_bin, confidences_por_bin, proporciones)
    """
    # Crear bins de confianza [0, 1/n_bins, 2/n_bins, ..., 1]
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    # Asignar cada predicción a un bin
    bin_indices = np.digitize(y_pred_proba, bin_boundaries[1:-1])
    
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for b in range(n_bins):
        # Predicciones en este bin
        mask = bin_indices == b
        bin_counts[b] = mask.sum()
        
        if bin_counts[b] > 0:
            # Accuracy real: proporción de aciertos
            bin_accuracies[b] = y_true[mask].mean()
            # Confianza media: probabilidad predicha media
            bin_confidences[b] = y_pred_proba[mask].mean()
    
    # Calcular ECE: media ponderada de |accuracy - confidence|
    weights = bin_counts / bin_counts.sum()
    ece = np.sum(weights * np.abs(bin_accuracies - bin_confidences))
    
    return ece, bin_accuracies, bin_confidences, weights


def compute_brier_score(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> float:
    """
    Calcula el Brier Score.
    
    El Brier Score es el MSE entre probabilidades y etiquetas:
    
        BS = (1/N) * Σ (p_i - y_i)²
    
    Propiedades:
        - Rango: [0, 1]
        - BS = 0: Predicción perfecta
        - Combina calibración y discriminación (sharpness)
    
    Descomposición (no implementada aquí pero relevante):
        BS = Calibración + Resolución - Incertidumbre
    
    Args:
        y_true: Etiquetas reales [N]
        y_pred_proba: Probabilidades predichas [N]
        
    Returns:
        Brier Score
    """
    return np.mean((y_pred_proba - y_true) ** 2)


def compute_negative_log_likelihood(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    eps: float = 1e-15
) -> float:
    """
    Calcula la Negative Log-Likelihood (NLL).
    
    NLL = -(1/N) * Σ [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
    
    Es la métrica natural para evaluar distribuciones predictivas.
    También conocida como cross-entropy loss.
    
    Args:
        y_true: Etiquetas reales [N]
        y_pred_proba: Probabilidades predichas [N]
        eps: Epsilon para estabilidad numérica
        
    Returns:
        NLL promedio
    """
    # Clip para evitar log(0)
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    
    nll = -np.mean(
        y_true * np.log(y_pred_proba) + 
        (1 - y_true) * np.log(1 - y_pred_proba)
    )
    
    return nll


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calcula todas las métricas de evaluación.
    
    Args:
        y_true: Etiquetas reales
        y_pred_proba: Probabilidades predichas
        
    Returns:
        Diccionario con todas las métricas
    """
    # Métricas de clasificación
    metrics = compute_classification_metrics(y_true, y_pred_proba)
    
    # Métricas de calibración
    ece, _, _, _ = compute_ece(y_true, y_pred_proba)
    metrics['ece'] = ece
    metrics['brier_score'] = compute_brier_score(y_true, y_pred_proba)
    metrics['nll'] = compute_negative_log_likelihood(y_true, y_pred_proba)
    
    return metrics


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = ECE_NUM_BINS,
    model_name: str = 'Model',
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Genera el Reliability Diagram (diagrama de calibración).
    
    El reliability diagram muestra:
    - Eje X: Confianza del modelo (probabilidad predicha)
    - Eje Y: Accuracy real (fracción de positivos reales)
    - Línea diagonal: Calibración perfecta
    
    Un modelo bien calibrado tiene barras cercanas a la diagonal.
    
    Args:
        y_true: Etiquetas reales
        y_pred_proba: Probabilidades predichas
        n_bins: Número de bins
        model_name: Nombre del modelo para el título
        ax: Axes existente (opcional)
        save_path: Ruta para guardar la figura
        
    Returns:
        Figura de matplotlib
    """
    ece, bin_accuracies, bin_confidences, weights = compute_ece(
        y_true, y_pred_proba, n_bins
    )
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Centros de los bins
    bin_centers = np.linspace(1/(2*n_bins), 1 - 1/(2*n_bins), n_bins)
    bin_width = 1 / n_bins
    
    # Barras de accuracy
    ax.bar(
        bin_centers,
        bin_accuracies,
        width=bin_width * 0.8,
        alpha=0.7,
        color='steelblue',
        edgecolor='black',
        label='Accuracy'
    )
    
    # Línea de calibración perfecta
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')
    
    # Configuración
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(f'Reliability Diagram - {model_name}\nECE = {ece:.4f}', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    # Agregar histograma de predicciones (subplot inferior)
    # ax2 = ax.twinx()
    # ax2.bar(bin_centers, weights, width=bin_width*0.8, alpha=0.3, color='gray')
    # ax2.set_ylabel('Fraction of Samples', color='gray')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Reliability diagram guardado en {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = 'Model',
    ax: Optional[plt.Axes] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Genera la curva ROC.
    
    Args:
        y_true: Etiquetas reales
        y_pred_proba: Probabilidades predichas
        model_name: Nombre del modelo
        ax: Axes existente
        save_path: Ruta para guardar
        
    Returns:
        Figura
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_uncertainty_histogram(
    uncertainties: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza la distribución de incertidumbre separada por aciertos/errores.
    
    Idealmente, el modelo debería tener:
    - Baja incertidumbre en predicciones correctas
    - Alta incertidumbre en predicciones incorrectas
    
    Args:
        uncertainties: Incertidumbre epistémica por muestra
        y_true: Etiquetas reales
        y_pred: Predicciones binarias
        model_name: Nombre del modelo
        save_path: Ruta para guardar
        
    Returns:
        Figura
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separar por correctas/incorrectas
    correct_mask = y_true == y_pred
    
    ax.hist(
        uncertainties[correct_mask],
        bins=50,
        alpha=0.6,
        label=f'Correct ({correct_mask.sum()})',
        color='green',
        density=True
    )
    ax.hist(
        uncertainties[~correct_mask],
        bins=50,
        alpha=0.6,
        label=f'Incorrect ({(~correct_mask).sum()})',
        color='red',
        density=True
    )
    
    ax.set_xlabel('Epistemic Uncertainty', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Uncertainty Distribution - {model_name}', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compare_models_metrics(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Compara métricas de múltiples modelos en un gráfico de barras.
    
    Args:
        results: Dict {model_name: {metric_name: value}}
        save_path: Ruta para guardar
        
    Returns:
        Figura
    """
    models = list(results.keys())
    metrics = ['auc_roc', 'f1_score', 'accuracy', 'ece', 'brier_score']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
    
    for idx, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        
        # Para ECE y Brier, menor es mejor
        colors = ['steelblue'] * len(models)
        
        axes[idx].bar(models, values, color=colors, alpha=0.8, edgecolor='black')
        axes[idx].set_title(metric.upper().replace('_', ' '), fontsize=12)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Añadir valores en las barras
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def print_metrics_table(results: Dict[str, Dict[str, float]]):
    """
    Imprime una tabla formateada con las métricas de todos los modelos.
    
    Args:
        results: Dict {model_name: {metric_name: value}}
    """
    print("\n" + "=" * 80)
    print("COMPARACIÓN DE MÉTRICAS")
    print("=" * 80)
    
    # Métricas a mostrar
    metrics = ['auc_roc', 'f1_score', 'accuracy', 'ece', 'brier_score', 'nll']
    
    # Encabezado
    header = f"{'Modelo':<20}"
    for m in metrics:
        header += f"{m.upper():<12}"
    print(header)
    print("-" * 80)
    
    # Filas
    for model_name, model_metrics in results.items():
        row = f"{model_name:<20}"
        for m in metrics:
            if m in model_metrics:
                row += f"{model_metrics[m]:<12.4f}"
            else:
                row += f"{'N/A':<12}"
        print(row)
    
    print("=" * 80)
    print("\nNOTA: Para ECE, Brier Score y NLL, menor es mejor.")
    print("      Para AUC-ROC, F1-Score y Accuracy, mayor es mejor.")
