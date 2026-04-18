# Latest version
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
    Compute Expected Calibration Error (ECE) using confidence vs empirical accuracy.

    This version implements the standard classification-calibration view:
    - confidence(B_m): mean confidence of the predicted class in bin m
    - accuracy(B_m): empirical accuracy in bin m

    For binary classification, if p = P(y=1|x):
    - predicted class = 1 if p >= 0.5 else 0
    - confidence = p if predicted class is 1, else 1 - p

    Args:
        y_true: True binary labels of shape [N]
        y_pred_proba: Predicted probability for the positive class of shape [N]
        n_bins: Number of bins

    Returns:
        Tuple:
            - ece: scalar ECE
            - bin_accuracies: empirical accuracy per bin
            - bin_confidences: mean confidence per bin
            - weights: fraction of samples per bin
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)

    # Safety clip in case probabilities are numerically unstable
    y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)

    # Predicted class
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Confidence of the predicted class
    confidences = np.where(y_pred == 1, y_pred_proba, 1.0 - y_pred_proba)

    # Whether the prediction is correct
    correctness = (y_pred == y_true).astype(float)

    # Uniform bins over confidence
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

    bin_accuracies = np.zeros(n_bins, dtype=float)
    bin_confidences = np.zeros(n_bins, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=float)

    for b in range(n_bins):
        mask = bin_indices == b
        bin_counts[b] = mask.sum()

        if bin_counts[b] > 0:
            bin_accuracies[b] = correctness[mask].mean()
            bin_confidences[b] = confidences[mask].mean()

    weights = bin_counts / max(bin_counts.sum(), 1.0)
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
    Plot a reliability diagram for confidence vs empirical accuracy.

    X-axis: mean confidence in each bin
    Y-axis: empirical accuracy in each bin

    A perfectly calibrated model lies on the diagonal.
    """
    ece, bin_accuracies, bin_confidences, weights = compute_ece(
        y_true, y_pred_proba, n_bins
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Use actual mean confidence as bar positions when bins are non-empty
    non_empty = weights > 0
    bin_width = 1 / n_bins

    ax.bar(
        bin_confidences[non_empty],
        bin_accuracies[non_empty],
        width=bin_width * 0.8,
        alpha=0.7,
        edgecolor='black',
        label='Empirical accuracy'
    )

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')

    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Reliability Diagram - {model_name}\nECE = {ece:.4f}', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

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
    epistemic: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model',
    aleatoric: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza la distribución de incertidumbre separada por aciertos/errores.

    Muestra la incertidumbre epistémica (y opcionalmente la aleatórica)
    para predicciones correctas vs incorrectas.

    Un modelo bien calibrado debería tener:
    - Distribución epistémica desplazada a la derecha en incorrectas
    - Distribución aleatórica similar en ambos grupos (es ruido inherente)

    Args:
        epistemic: Incertidumbre epistémica [N]
        y_true: Etiquetas reales [N]
        y_pred: Predicciones binarias [N]
        model_name: Nombre del modelo para el título
        aleatoric: Incertidumbre aleatórica [N] (opcional)
        save_path: Ruta para guardar
    """
    n_plots = 2 if aleatoric is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    correct_mask = (y_true == y_pred)

    def _plot_one(ax, values, title):
        ax.hist(values[correct_mask],  bins=50, alpha=0.6, density=True,
                color='green', label=f'Correct (n={correct_mask.sum()})')
        ax.hist(values[~correct_mask], bins=50, alpha=0.6, density=True,
                color='red',   label=f'Incorrect (n={(~correct_mask).sum()})')
        ax.set_xlabel('Uncertainty', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{title}\n{model_name}', fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)

    _plot_one(axes[0], epistemic, 'Epistemic Uncertainty')
    if aleatoric is not None:
        _plot_one(axes[1], aleatoric, 'Aleatoric Uncertainty')

    plt.tight_layout()

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


def triage_decision(
    mean_prob: float,
    epistemic_uncertainty: float,
    uncertainty_threshold: float = None,
    confidence_threshold: float = None
) -> dict:
    """
    Traduce incertidumbre en una decisión clínica accionable.

    Lógica:
    - Si la incertidumbre epistémica supera el umbral → derivar a patólogo
    - Si no, clasificar según la probabilidad media

    Esta función implementa el requisito de 'valor de negocio':
    la incertidumbre deja de ser un número abstracto y se convierte
    en una acción concreta del sistema.

    Args:
        mean_prob: Probabilidad media de cáncer p(y=1|x)
        epistemic_uncertainty: Varianza entre muestras del posterior
        uncertainty_threshold: Umbral de derivación (default: config)
        confidence_threshold: Umbral de clasificación (default: config)

    Returns:
        dict con keys: decision, confidence, action, color
    """
    from config import UNCERTAINTY_THRESHOLD, CONFIDENCE_THRESHOLD

    if uncertainty_threshold is None:
        uncertainty_threshold = UNCERTAINTY_THRESHOLD
    if confidence_threshold is None:
        confidence_threshold = CONFIDENCE_THRESHOLD

    if epistemic_uncertainty > uncertainty_threshold:
        return {
            "decision": "UNCERTAIN",
            "confidence": mean_prob,
            "action": "⚠️  High epistemic uncertainty — refer to human pathologist review",
            "color": "orange"
        }
    elif mean_prob >= confidence_threshold:
        return {
            "decision": "CANCER",
            "confidence": mean_prob,
            "action": "🔴  Malignant tissue detected — flag for clinical follow-up",
            "color": "red"
        }
    else:
        return {
            "decision": "BENIGN",
            "confidence": 1 - mean_prob,
            "action": "🟢  No malignancy detected — routine monitoring",
            "color": "green"
        }


def compute_triage_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    epistemic: np.ndarray,
    uncertainty_threshold: float = None
) -> dict:
    """
    Calcula métricas de negocio basadas en el sistema de triaje.

    Métricas:
    - referral_rate: fracción de casos derivados a revisión humana
    - accuracy_on_confident: accuracy solo en casos NO derivados
    - coverage: fracción de casos resueltos automáticamente

    Args:
        y_true: Etiquetas reales [N]
        y_pred_proba: Probabilidades medias [N]
        epistemic: Incertidumbre epistémica [N]
        uncertainty_threshold: Umbral de derivación

    Returns:
        dict con métricas de negocio
    """
    from config import UNCERTAINTY_THRESHOLD, CONFIDENCE_THRESHOLD

    if uncertainty_threshold is None:
        uncertainty_threshold = UNCERTAINTY_THRESHOLD

    uncertain_mask = epistemic > uncertainty_threshold
    confident_mask = ~uncertain_mask

    referral_rate = uncertain_mask.mean()
    coverage = confident_mask.mean()

    if confident_mask.sum() > 0:
        y_pred_confident = (y_pred_proba[confident_mask] >= CONFIDENCE_THRESHOLD).astype(int)
        acc_confident = accuracy_score(y_true[confident_mask], y_pred_confident)
    else:
        acc_confident = float('nan')

    return {
        "referral_rate": float(referral_rate),
        "coverage": float(coverage),
        "accuracy_on_confident": float(acc_confident),
    }
