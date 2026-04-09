"""
evaluate.py - Módulo de evaluación e inferencia
===============================================

Este módulo implementa:
1. Inferencia en test set para cada modelo
2. Comparación cuantitativa de métricas
3. Análisis cualitativo de incertidumbre
4. Identificación de ejemplos ambiguos

El análisis de incertidumbre es clave para el valor clínico del proyecto:
- Los ejemplos con alta incertidumbre epistémica son candidatos a revisión humana
- Esto permite que el modelo "sepa cuándo no sabe"
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from pathlib import Path
import json
import matplotlib.pyplot as plt

from config import DEVICE, MC_SAMPLES, FIGURES_DIR, RESULTS_DIR
from models import DeterministicCNN, MCDropoutCNN, LaplaceWrapper
from metrics import (
    compute_all_metrics,
    plot_reliability_diagram,
    plot_roc_curve,
    plot_uncertainty_histogram,
    compare_models_metrics,
    print_metrics_table
)


@torch.no_grad()
def evaluate_deterministic(
    model: DeterministicCNN,
    test_loader: DataLoader,
    device: torch.device = DEVICE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evalúa el modelo determinista en el test set.
    
    El modelo determinista produce una única predicción sin incertidumbre.
    Usamos la máxima probabilidad como medida de "confianza" (no es
    incertidumbre epistémica real, solo refleja la seguridad del clasificador).
    
    Args:
        model: Modelo determinista entrenado
        test_loader: DataLoader de test
        device: Dispositivo
        
    Returns:
        Tupla (y_true, y_pred_proba, confidence):
        - y_true: Etiquetas reales
        - y_pred_proba: P(y=1|x)
        - confidence: max(P(y|x)) como pseudo-incertidumbre
    """
    model.eval()
    model = model.to(device)
    
    all_labels = []
    all_probs = []
    all_confidence = []
    
    use_cuda = device.type == 'cuda'
    for images, labels in tqdm(test_loader, desc='Evaluating Deterministic'):
        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_cuda):
            logits = model(images)
            probs = F.softmax(logits, dim=1)
        
        # Probabilidad de clase positiva
        prob_positive = probs[:, 1].cpu().numpy()
        
        # Confianza: max prob (no es incertidumbre real)
        confidence = probs.max(dim=1)[0].cpu().numpy()
        
        all_labels.extend(labels.numpy())
        all_probs.extend(prob_positive)
        all_confidence.extend(confidence)
    
    return (
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_confidence)
    )


def evaluate_mc_dropout(
    model: MCDropoutCNN,
    test_loader: DataLoader,
    n_samples: int = MC_SAMPLES,
    device: torch.device = DEVICE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evalúa el modelo MC Dropout con cuantificación de incertidumbre.
    
    Realiza T forward passes con dropout activo para aproximar:
        p(y|x, D) ≈ (1/T) Σ_t p(y|x, ω_t)
    
    La varianza entre las T predicciones captura la incertidumbre epistémica.
    
    Args:
        model: Modelo MC Dropout entrenado
        test_loader: DataLoader de test
        n_samples: Número de forward passes
        device: Dispositivo
        
    Returns:
        Tupla (y_true, y_pred_proba, epistemic_uncertainty)
    """
    model = model.to(device)
    
    all_labels = []
    all_probs = []
    all_uncertainties = []
    
    for images, labels in tqdm(test_loader, desc='Evaluating MC Dropout'):
        images = images.to(device, non_blocking=True)
        
        # Predicción con incertidumbre
        mean_probs, epistemic_unc, _ = model.predict_with_uncertainty(
            images, n_samples=n_samples
        )
        
        # Probabilidad de clase positiva
        prob_positive = mean_probs[:, 1].cpu().numpy()
        
        all_labels.extend(labels.numpy())
        all_probs.extend(prob_positive)
        all_uncertainties.extend(epistemic_unc.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_uncertainties)
    )


def evaluate_laplace(
    laplace_model: LaplaceWrapper,
    test_loader: DataLoader,
    device: torch.device = DEVICE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evalúa el modelo con Laplace Approximation.
    
    Laplace aproxima el posterior como:
        q(ω) = N(ω | ω*, Σ)
    
    donde Σ = H^{-1} y H es el Hessiano de la log-posterior.
    
    La predicción integra sobre esta distribución aproximada.
    
    Args:
        laplace_model: Wrapper de Laplace ajustado
        test_loader: DataLoader de test
        device: Dispositivo
        
    Returns:
        Tupla (y_true, y_pred_proba, epistemic_uncertainty)
    """
    all_labels = []
    all_probs = []
    all_uncertainties = []
    
    for images, labels in tqdm(test_loader, desc='Evaluating Laplace'):
        images = images.to(device, non_blocking=True)
        
        # Predicción con Laplace
        mean_probs, epistemic_unc = laplace_model.predict_with_uncertainty(images)
        
        # Probabilidad de clase positiva
        prob_positive = mean_probs[:, 1].cpu().numpy()
        
        all_labels.extend(labels.numpy())
        all_probs.extend(prob_positive)
        all_uncertainties.extend(epistemic_unc.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_uncertainties)
    )


def identify_high_uncertainty_samples(
    uncertainties: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int = 20
) -> Dict:
    """
    Identifica los ejemplos con mayor incertidumbre epistémica.
    
    Estos ejemplos son candidatos a revisión por expertos humanos.
    En el contexto clínico, representan casos ambiguos donde el
    modelo necesita supervisión.
    
    Args:
        uncertainties: Incertidumbre por muestra
        y_true: Etiquetas reales
        y_pred: Predicciones
        top_k: Número de ejemplos a identificar
        
    Returns:
        Dict con índices y estadísticas de los ejemplos más inciertos
    """
    # Ordenar por incertidumbre descendente
    sorted_indices = np.argsort(uncertainties)[::-1]
    top_indices = sorted_indices[:top_k]
    
    # Analizar estos ejemplos
    top_uncertainties = uncertainties[top_indices]
    top_labels = y_true[top_indices]
    top_preds = y_pred[top_indices]
    
    # Calcular estadísticas
    correct_in_top = (top_labels == top_preds).sum()
    
    results = {
        'indices': top_indices.tolist(),
        'uncertainties': top_uncertainties.tolist(),
        'true_labels': top_labels.tolist(),
        'predictions': top_preds.tolist(),
        'accuracy_in_top_k': correct_in_top / top_k,
        'mean_uncertainty_top_k': top_uncertainties.mean(),
        'mean_uncertainty_all': uncertainties.mean()
    }
    
    return results


def analyze_uncertainty_by_correctness(
    uncertainties: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Analiza si la incertidumbre es mayor en predicciones incorrectas.
    
    Un buen modelo bayesiano debería:
    - Tener baja incertidumbre cuando acierta
    - Tener alta incertidumbre cuando falla
    
    Args:
        uncertainties: Incertidumbre por muestra
        y_true: Etiquetas reales
        y_pred: Predicciones
        
    Returns:
        Estadísticas de incertidumbre por grupo
    """
    correct_mask = y_true == y_pred
    
    return {
        'mean_uncertainty_correct': uncertainties[correct_mask].mean(),
        'std_uncertainty_correct': uncertainties[correct_mask].std(),
        'mean_uncertainty_incorrect': uncertainties[~correct_mask].mean(),
        'std_uncertainty_incorrect': uncertainties[~correct_mask].std(),
        'ratio': uncertainties[~correct_mask].mean() / (uncertainties[correct_mask].mean() + 1e-10)
    }


def full_evaluation(
    det_model: DeterministicCNN,
    mc_model: MCDropoutCNN,
    laplace_model: LaplaceWrapper,
    test_loader: DataLoader,
    train_loader: Optional[DataLoader] = None,
    save_dir: Optional[Path] = None
) -> Dict[str, Dict]:
    """
    Ejecuta la evaluación completa de todos los modelos.

    Esta función:
    1. Evalúa cada modelo en el test set
    2. Calcula métricas de clasificación y calibración
    3. Genera visualizaciones comparativas
    4. Analiza la incertidumbre

    Args:
        det_model: Modelo determinista
        mc_model: Modelo MC Dropout
        laplace_model: Modelo con Laplace
        test_loader: DataLoader de test
        train_loader: DataLoader de entrenamiento (no usado, por compatibilidad)
        save_dir: Directorio donde guardar resultados JSON

    Returns:
        Dict con todos los resultados (claves: 'deterministic', 'mc_dropout', 'laplace')
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    results = {}

    # =========================================================================
    # 1. Evaluar modelo determinista
    # =========================================================================
    print("\n" + "=" * 60)
    print("EVALUANDO MODELO DETERMINISTA")
    print("=" * 60)

    y_true, y_pred_det, conf_det = evaluate_deterministic(det_model, test_loader)

    metrics_det = compute_all_metrics(y_true, y_pred_det)
    # Aliases para compatibilidad con main.py
    metrics_det['auc'] = metrics_det['auc_roc']
    metrics_det['brier'] = metrics_det['brier_score']
    # Datos por muestra con prefijo _ para que main.py los excluya del JSON
    metrics_det['_y_true'] = y_true
    metrics_det['_y_pred'] = y_pred_det
    metrics_det['_uncertainty'] = 1.0 - conf_det
    results['deterministic'] = metrics_det

    print(f"AUC-ROC: {metrics_det['auc_roc']:.4f}")
    print(f"ECE: {metrics_det['ece']:.4f}")

    # =========================================================================
    # 2. Evaluar MC Dropout
    # =========================================================================
    print("\n" + "=" * 60)
    print("EVALUANDO MC DROPOUT")
    print("=" * 60)

    y_true, y_pred_mc, unc_mc = evaluate_mc_dropout(mc_model, test_loader)

    metrics_mc = compute_all_metrics(y_true, y_pred_mc)
    metrics_mc['auc'] = metrics_mc['auc_roc']
    metrics_mc['brier'] = metrics_mc['brier_score']
    metrics_mc['_y_true'] = y_true
    metrics_mc['_y_pred'] = y_pred_mc
    metrics_mc['_uncertainty'] = unc_mc

    y_pred_binary_mc = (y_pred_mc >= 0.5).astype(int)
    unc_analysis_mc = analyze_uncertainty_by_correctness(unc_mc, y_true, y_pred_binary_mc)
    metrics_mc['_uncertainty_analysis'] = unc_analysis_mc
    results['mc_dropout'] = metrics_mc

    print(f"AUC-ROC: {metrics_mc['auc_roc']:.4f}")
    print(f"ECE: {metrics_mc['ece']:.4f}")
    print(f"Incertidumbre media (correctas): {unc_analysis_mc['mean_uncertainty_correct']:.4f}")
    print(f"Incertidumbre media (incorrectas): {unc_analysis_mc['mean_uncertainty_incorrect']:.4f}")

    # =========================================================================
    # 3. Evaluar Laplace
    # =========================================================================
    print("\n" + "=" * 60)
    print("EVALUANDO LAPLACE APPROXIMATION")
    print("=" * 60)

    y_true, y_pred_la, unc_la = evaluate_laplace(laplace_model, test_loader)

    metrics_la = compute_all_metrics(y_true, y_pred_la)
    metrics_la['auc'] = metrics_la['auc_roc']
    metrics_la['brier'] = metrics_la['brier_score']
    metrics_la['_y_true'] = y_true
    metrics_la['_y_pred'] = y_pred_la
    metrics_la['_uncertainty'] = unc_la

    y_pred_binary_la = (y_pred_la >= 0.5).astype(int)
    unc_analysis_la = analyze_uncertainty_by_correctness(unc_la, y_true, y_pred_binary_la)
    metrics_la['_uncertainty_analysis'] = unc_analysis_la
    results['laplace'] = metrics_la

    print(f"AUC-ROC: {metrics_la['auc_roc']:.4f}")
    print(f"ECE: {metrics_la['ece']:.4f}")
    print(f"Incertidumbre media (correctas): {unc_analysis_la['mean_uncertainty_correct']:.4f}")
    print(f"Incertidumbre media (incorrectas): {unc_analysis_la['mean_uncertainty_incorrect']:.4f}")

    # =========================================================================
    # 4. Imprimir tabla comparativa
    # =========================================================================
    print_metrics_table({
        'Deterministic': metrics_det,
        'MC Dropout': metrics_mc,
        'Laplace': metrics_la
    })

    # =========================================================================
    # 5. Generar visualizaciones
    # =========================================================================
    print("\nGenerando visualizaciones...")

    # Reliability diagrams
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_reliability_diagram(y_true, y_pred_det, model_name='Deterministic', ax=axes[0])
    plot_reliability_diagram(y_true, y_pred_mc, model_name='MC Dropout', ax=axes[1])
    plot_reliability_diagram(y_true, y_pred_la, model_name='Laplace', ax=axes[2])
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'reliability_diagrams.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_roc_curve(y_true, y_pred_det, 'Deterministic', ax=ax)
    plot_roc_curve(y_true, y_pred_mc, 'MC Dropout', ax=ax)
    plot_roc_curve(y_true, y_pred_la, 'Laplace', ax=ax)
    ax.legend()
    fig.savefig(FIGURES_DIR / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Histogramas de incertidumbre
    plot_uncertainty_histogram(unc_mc, y_true, y_pred_binary_mc, model_name='MC Dropout')
    plt.savefig(FIGURES_DIR / 'uncertainty_hist_mc.png', dpi=150, bbox_inches='tight')
    plt.close()

    plot_uncertainty_histogram(unc_la, y_true, y_pred_binary_la, model_name='Laplace')
    plt.savefig(FIGURES_DIR / 'uncertainty_hist_laplace.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Comparación de métricas
    compare_models_metrics(
        {'Deterministic': metrics_det, 'MC Dropout': metrics_mc, 'Laplace': metrics_la},
        save_path=FIGURES_DIR / 'metrics_comparison.png'
    )
    plt.close()

    # Guardar resultados en JSON (solo valores escalares, sin arrays ni dicts)
    results_json = {}
    for model_name, model_results in results.items():
        results_json[model_name] = {
            k: float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v
            for k, v in model_results.items()
            if isinstance(v, (np.floating, np.integer, float, int))
        }

    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"Resultados guardados en {save_dir}")
    print(f"Figuras guardadas en {FIGURES_DIR}")

    return results


def generate_uncertainty_report(
    results: Dict[str, Dict],
    save_path: Optional[Path] = None,
    top_k: int = 20
) -> str:
    """
    Genera un reporte textual de análisis de incertidumbre desde el dict de resultados.

    Útil para la memoria del proyecto: identifica casos ambiguos
    que justifican el uso de modelos bayesianos en contexto clínico.

    Args:
        results: Dict con resultados de evaluación (output de full_evaluation)
        save_path: Ruta para guardar el reporte en disco (opcional)
        top_k: Número de ejemplos más inciertos a reportar

    Returns:
        Reporte formateado como string
    """
    report_parts = []

    for model_key, model_display in [('mc_dropout', 'MC DROPOUT'), ('laplace', 'LAPLACE')]:
        if model_key not in results:
            continue

        model_results = results[model_key]

        if '_uncertainty' not in model_results or '_y_true' not in model_results:
            continue

        uncertainties = model_results['_uncertainty']
        y_true = model_results['_y_true']
        y_pred = (model_results['_y_pred'] >= 0.5).astype(int)

        high_unc = identify_high_uncertainty_samples(uncertainties, y_true, y_pred, top_k)
        unc_by_correct = analyze_uncertainty_by_correctness(uncertainties, y_true, y_pred)

        part = f"""
================================================================================
REPORTE DE ANÁLISIS DE INCERTIDUMBRE - {model_display}
================================================================================

ESTADÍSTICAS GENERALES
----------------------
- Incertidumbre media global: {uncertainties.mean():.6f}
- Desviación estándar: {uncertainties.std():.6f}
- Mínimo: {uncertainties.min():.6f}
- Máximo: {uncertainties.max():.6f}

INCERTIDUMBRE POR CORRECTITUD DE PREDICCIÓN
-------------------------------------------
- Predicciones correctas:
  * Media: {unc_by_correct['mean_uncertainty_correct']:.6f}
  * Std: {unc_by_correct['std_uncertainty_correct']:.6f}

- Predicciones incorrectas:
  * Media: {unc_by_correct['mean_uncertainty_incorrect']:.6f}
  * Std: {unc_by_correct['std_uncertainty_incorrect']:.6f}

- Ratio (incorrectas/correctas): {unc_by_correct['ratio']:.2f}x

Un ratio > 1 indica que el modelo tiene más incertidumbre cuando falla,
lo cual es el comportamiento deseado de un modelo bayesiano bien calibrado.

TOP {top_k} EJEMPLOS MÁS INCIERTOS
----------------------------------
- Accuracy en estos ejemplos: {high_unc['accuracy_in_top_k']*100:.1f}%
- Incertidumbre media en top-{top_k}: {high_unc['mean_uncertainty_top_k']:.6f}

Estos ejemplos representan casos ambiguos que deberían ser revisados
por un experto humano en un entorno clínico real.

CONCLUSIÓN
----------
{"El modelo muestra buena separación de incertidumbre entre aciertos y errores."
if unc_by_correct['ratio'] > 1.5 else
"El modelo podría beneficiarse de más datos o una mejor arquitectura para separar incertidumbre."}

================================================================================"""
        report_parts.append(part)

    if not report_parts:
        report = "No hay datos de incertidumbre disponibles para generar el reporte."
    else:
        report = "\n".join(report_parts)

    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Reporte guardado en {save_path}")

    return report
