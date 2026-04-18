"""
visualize.py - Módulo de visualización
======================================

Este módulo genera las figuras necesarias para la memoria:
1. Curvas de entrenamiento (loss, accuracy)
2. Ejemplos de imágenes con predicciones
3. Mapas de incertidumbre
4. Comparaciones entre modelos

Todas las figuras se guardan en FIGURES_DIR para incluir en el paper.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
import json

from config import FIGURES_DIR, IMAGENET_MEAN, IMAGENET_STD, DEVICE
from data import denormalize_image


def setup_plotting_style():
    """
    Configura el estilo de las gráficas para publicación.
    
    Usa un estilo limpio apropiado para papers académicos.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str = 'Model',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza las curvas de entrenamiento.
    
    Muestra:
    - Loss de entrenamiento y validación
    - AUC de entrenamiento y validación
    
    Útil para detectar overfitting (divergencia train/val).
    
    Args:
        history: Dict con 'train_loss', 'val_loss', 'train_auc', 'val_auc'
        model_name: Nombre para el título
        save_path: Ruta para guardar
        
    Returns:
        Figura
    """
    setup_plotting_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC
    axes[1].plot(epochs, history['train_auc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_auc'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC-ROC')
    axes[1].set_title(f'{model_name} - AUC-ROC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Curvas de entrenamiento guardadas en {save_path}")
    
    return fig


def plot_sample_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    probs: torch.Tensor,
    uncertainties: Optional[torch.Tensor] = None,
    n_samples: int = 16,
    model_name: str = 'Model',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza predicciones en imágenes de ejemplo.
    
    Muestra:
    - Imagen original
    - Etiqueta real
    - Probabilidad predicha
    - Incertidumbre (si disponible)
    
    Colorea el borde según si la predicción es correcta (verde) o no (rojo).
    
    Args:
        images: Tensor de imágenes [N, C, H, W]
        labels: Etiquetas reales [N]
        probs: Probabilidades predichas [N]
        uncertainties: Incertidumbre epistémica [N] (opcional)
        n_samples: Número de imágenes a mostrar
        model_name: Nombre del modelo
        save_path: Ruta para guardar
        
    Returns:
        Figura
    """
    setup_plotting_style()
    
    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_samples):
        # Desnormalizar imagen
        img = denormalize_image(images[i])
        img_np = img.permute(1, 2, 0).numpy()
        
        # Obtener valores
        label = labels[i].item()
        prob = probs[i].item() if isinstance(probs[i], torch.Tensor) else probs[i]
        pred = 1 if prob >= 0.5 else 0
        correct = pred == label
        
        # Mostrar imagen
        axes[i].imshow(img_np)
        
        # Borde según correctitud
        color = 'green' if correct else 'red'
        for spine in axes[i].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        
        # Título con información
        title = f"True: {label}, Pred: {prob:.2f}"
        if uncertainties is not None:
            unc = uncertainties[i].item() if isinstance(uncertainties[i], torch.Tensor) else uncertainties[i]
            title += f"\nUnc: {unc:.4f}"
        
        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')
    
    # Ocultar axes vacíos
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'{model_name} - Sample Predictions', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Predicciones de ejemplo guardadas en {save_path}")
    
    return fig


def plot_high_uncertainty_samples(
    images: torch.Tensor,
    labels: torch.Tensor,
    probs: np.ndarray,
    uncertainties: np.ndarray,
    top_k: int = 12,
    model_name: str = 'Model',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza los ejemplos con mayor incertidumbre epistémica.
    
    Esto es clave para el valor clínico del proyecto:
    estos ejemplos representan casos que el modelo considera ambiguos
    y que deberían ser revisados por un experto.
    
    Args:
        images: Tensor de imágenes
        labels: Etiquetas reales
        probs: Probabilidades predichas
        uncertainties: Incertidumbre epistémica
        top_k: Número de ejemplos a mostrar
        model_name: Nombre del modelo
        save_path: Ruta para guardar
        
    Returns:
        Figura
    """
    setup_plotting_style()
    
    # Encontrar los top_k más inciertos
    sorted_indices = np.argsort(uncertainties)[::-1][:top_k]
    
    n_cols = 4
    n_rows = (top_k + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    for i, idx in enumerate(sorted_indices):
        # Desnormalizar
        img = denormalize_image(images[idx])
        img_np = img.permute(1, 2, 0).numpy()
        
        label = labels[idx].item()
        prob = probs[idx]
        unc = uncertainties[idx]
        pred = 1 if prob >= 0.5 else 0
        correct = pred == label
        
        axes[i].imshow(img_np)
        
        # Borde
        color = 'green' if correct else 'red'
        for spine in axes[i].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        
        title = f"Rank #{i+1}\nTrue: {label}, Pred: {prob:.2f}\nUncertainty: {unc:.4f}"
        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')
    
    for i in range(len(sorted_indices), len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'{model_name} - Highest Uncertainty Samples', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Ejemplos de alta incertidumbre guardados en {save_path}")
    
    return fig


def plot_uncertainty_vs_error(
    uncertainties: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualiza la relación entre incertidumbre y error de predicción.
    
    Muestra:
    - Scatter plot: incertidumbre vs error absoluto
    - Curva de retención: accuracy vs fracción de datos retenidos
    
    La curva de retención es importante: si descartamos las predicciones
    más inciertas, ¿mejora la accuracy?
    
    Args:
        uncertainties: Incertidumbre por muestra
        y_true: Etiquetas reales
        y_pred: Probabilidades predichas
        model_name: Nombre del modelo
        save_path: Ruta para guardar
        
    Returns:
        Figura
    """
    setup_plotting_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Error absoluto
    errors = np.abs(y_pred - y_true)
    
    # Scatter plot
    axes[0].scatter(uncertainties, errors, alpha=0.3, s=10)
    axes[0].set_xlabel('Epistemic Uncertainty')
    axes[0].set_ylabel('Absolute Error |p - y|')
    axes[0].set_title(f'{model_name} - Uncertainty vs Error')
    
    # Añadir línea de tendencia
    z = np.polyfit(uncertainties, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(uncertainties.min(), uncertainties.max(), 100)
    axes[0].plot(x_line, p(x_line), 'r--', label=f'Trend (slope={z[0]:.3f})')
    axes[0].legend()
    
    # Curva de retención
    # Ordenar por incertidumbre (ascendente = menos incierto primero)
    sorted_indices = np.argsort(uncertainties)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    correct = (y_true == y_pred_binary).astype(int)
    
    fractions = np.linspace(0.1, 1.0, 20)
    accuracies = []
    
    for frac in fractions:
        n_keep = int(len(correct) * frac)
        kept_indices = sorted_indices[:n_keep]
        acc = correct[kept_indices].mean()
        accuracies.append(acc)
    
    axes[1].plot(fractions * 100, accuracies, 'b-o', linewidth=2)
    axes[1].axhline(y=correct.mean(), color='r', linestyle='--', 
                    label=f'Full dataset acc: {correct.mean():.3f}')
    axes[1].set_xlabel('Fraction of Data Retained (%)')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy vs Coverage (Retention Curve)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Análisis de incertidumbre guardado en {save_path}")
    
    return fig


def create_summary_figure(
    results: Dict[str, Dict],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Crea una figura resumen para el paper.
    
    Incluye:
    - Comparación de AUC, F1, ECE, Brier
    - Destacar el mejor modelo en cada métrica
    
    Args:
        results: Dict con resultados de cada modelo
        save_path: Ruta para guardar
        
    Returns:
        Figura
    """
    setup_plotting_style()
    
    models = list(results.keys())
    metrics_to_plot = ['auc_roc', 'f1_score', 'ece', 'brier_score']
    metric_labels = ['AUC-ROC ↑', 'F1-Score ↑', 'ECE ↓', 'Brier Score ↓']
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Verde, Azul, Rojo
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [results[m][metric] for m in models]
        
        bars = axes[idx].bar(models, values, color=colors, edgecolor='black', alpha=0.8)
        axes[idx].set_title(label, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(metric.upper())
        
        # Añadir valores sobre las barras
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].annotate(f'{val:.3f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=10)
        
        axes[idx].tick_params(axis='x', rotation=15)
        
        # Destacar el mejor
        if '↑' in label:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
    
    fig.suptitle('Model Comparison Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        print(f"Figura resumen guardada en {save_path}")
    
    return fig


def generate_all_figures(
    results: Dict,
    histories: Optional[Dict] = None,
    test_loader=None,
    save_dir: Optional[Path] = None
):
    """
    Genera todas las figuras necesarias para la memoria.

    Args:
        results: Dict con resultados de evaluación (output de full_evaluation).
                 Claves esperadas: 'deterministic', 'mc_dropout', 'laplace'.
                 Cada entrada debe incluir 'y_true', 'y_pred', 'uncertainty' (opcionales
                 pero necesarios para las figuras por muestra).
        histories: Dict con historiales de entrenamiento:
                   {'deterministic': history_dict, 'mc_dropout': history_dict}
        test_loader: DataLoader de test (reservado para uso futuro)
        save_dir: Directorio donde guardar las figuras
    """
    from metrics import (
        plot_roc_curve, plot_reliability_diagram,
        plot_uncertainty_histogram, compare_models_metrics
    )

    if save_dir is None:
        save_dir = FIGURES_DIR

    print("\n" + "=" * 60)
    print("GENERANDO FIGURAS PARA LA MEMORIA")
    print("=" * 60)

    # 1. Curvas de entrenamiento
    if histories is not None:
        for key, name in [('deterministic', 'Deterministic'), ('mc_dropout', 'MC Dropout')]:
            if key in histories:
                plot_training_history(
                    histories[key],
                    model_name=name,
                    save_path=save_dir / f'training_{key}.png'
                )

    # Modelos disponibles con datos por muestra
    model_map = [
        ('deterministic', 'Deterministic'),
        ('mc_dropout', 'MC Dropout'),
        ('laplace', 'Laplace'),
    ]
    available = [(k, n) for k, n in model_map if k in results and '_y_true' in results[k]]

    # 2. Curvas ROC
    if available:
        fig, ax = plt.subplots(figsize=(8, 6))
        for key, name in available:
            plot_roc_curve(results[key]['_y_true'], results[key]['_y_pred'], name, ax=ax)
        ax.legend()
        fig.savefig(save_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Reliability diagrams
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
        if len(available) == 1:
            axes = [axes]
        for idx, (key, name) in enumerate(available):
            plot_reliability_diagram(
                results[key]['_y_true'], results[key]['_y_pred'],
                model_name=name, ax=axes[idx]
            )
        plt.tight_layout()
        fig.savefig(save_dir / 'reliability_diagrams.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Histogramas de incertidumbre
    for key, name in [('mc_dropout', 'MC Dropout'), ('laplace', 'Laplace')]:
        if key in results and '_uncertainty' in results[key]:
            y_pred_binary = (results[key]['_y_pred'] >= 0.5).astype(int)
            plot_uncertainty_histogram(
                results[key]['_uncertainty'],
                results[key]['_y_true'],
                y_pred_binary,
                model_name=name,
                save_path=save_dir / f'uncertainty_hist_{key}.png'
            )
            plt.close()

    # 5. Incertidumbre vs error
    for key, name in [('mc_dropout', 'MC Dropout'), ('laplace', 'Laplace')]:
        if key in results and '_uncertainty' in results[key]:
            plot_uncertainty_vs_error(
                results[key]['_uncertainty'],
                results[key]['_y_true'],
                results[key]['_y_pred'],
                model_name=name,
                save_path=save_dir / f'uncertainty_vs_error_{key}.png'
            )
            plt.close()

    # 6. Figura resumen y comparación de métricas
    # Filtrar solo valores escalares para las figuras de comparación
    metrics_only = {}
    for key, name in model_map:
        if key in results:
            metrics_only[name] = {
                k: float(v) for k, v in results[key].items()
                if isinstance(v, (int, float, np.floating))
            }

    if metrics_only:
        create_summary_figure(metrics_only, save_path=save_dir / 'summary_comparison.png')
        plt.close()
        compare_models_metrics(metrics_only, save_path=save_dir / 'metrics_comparison.png')
        plt.close()
    
    # 7. Figura con predicciones de muestras (sample predictions)
    if test_loader is not None:
        images, labels = next(iter(test_loader))
        n = len(images)

        for key, name in [('mc_dropout', 'MC Dropout'), ('laplace', 'Laplace')]:
            if key in results and '_y_pred' in results[key]:
                probs = torch.tensor(results[key]['_y_pred'][:n])
                uncertainties = torch.tensor(results[key]['_uncertainty'][:n])
                plot_sample_predictions(
                    images, labels, probs, uncertainties,
                    n_samples=16, model_name=name,
                    save_path=save_dir / f'sample_predictions_{key}.png'
                )
                plt.close()
    
    # 8. Figura con ejemplos de alta incertidumbre (top 12 en primeras 500 imágenes)
    if test_loader is not None:
        all_images = []
        all_labels = []
        for images, labels in test_loader: # Recolectar imágenes y etiquetas para las primeras 500 muestras
            all_images.append(images)
            all_labels.append(labels)
            if sum(len(x) for x in all_images) >= 500:
                break
        
        all_images = torch.cat(all_images)[:500]
        all_labels = torch.cat(all_labels)[:500]

        for key, name in [('mc_dropout', 'MC Dropout'), ('laplace', 'Laplace')]:
            if key in results and '_uncertainty' in results[key]:
                probs = np.array(results[key]['_y_pred'][:500])
                uncertainties = np.array(results[key]['_uncertainty'][:500])
                plot_high_uncertainty_samples(
                    all_images, all_labels, probs, uncertainties,
                    top_k=12, model_name=name,
                    save_path=save_dir / f'high_uncertainty_samples_{key}.png'
                )
                plt.close()

    print(f"\nTodas las figuras guardadas en {save_dir}")
