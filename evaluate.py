# Latest version
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
import time

from config import (DEVICE, MC_SAMPLES, 
                    FIGURES_DIR, RESULTS_DIR, TRIAGE_PERCENTILE
                    )
from models import DeterministicCNN, MCDropoutCNN, LaplaceWrapper
from metrics import (
    compute_all_metrics,
    plot_reliability_diagram,
    plot_roc_curve,
    plot_uncertainty_histogram,
    compare_models_metrics,
    print_metrics_table,
    compute_triage_metrics,
    calibrate_uncertainty_threshold
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

        with torch.amp.autocast(enabled=use_cuda, device_type=device.type):
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
    all_epistemic = []
    all_aleatoric = []
    all_total = []
    
    for images, labels in tqdm(test_loader, desc='Evaluating MC Dropout'):
        images = images.to(device, non_blocking=True)
        
        # Predicción con incertidumbre
        mean_probs, epistemic_unc, aleatoric_unc, total_unc, _ = model.predict_with_uncertainty(
            images, n_samples=n_samples
        )
        
        # Probabilidad de clase positiva
        prob_positive = mean_probs[:, 1].cpu().numpy()
        
        all_labels.extend(labels.numpy())
        all_probs.extend(prob_positive)
        all_epistemic.extend(epistemic_unc.cpu().numpy())
        all_aleatoric.extend(aleatoric_unc.cpu().numpy())
        all_total.extend(total_unc.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_epistemic),
        np.array(all_aleatoric),
        np.array(all_total)
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
    all_epistemic = []
    all_aleatoric = []
    all_total = []
    
    for images, labels in tqdm(test_loader, desc='Evaluating Laplace'):
        images = images.to(device, non_blocking=True)
        
        # Predicción con Laplace
        mean_probs, epistemic_unc, aleatoric_unc, total_unc, _ = laplace_model.predict_with_uncertainty(images)
        
        # Probabilidad de clase positiva
        prob_positive = mean_probs[:, 1].cpu().numpy()
        
        all_labels.extend(labels.numpy())
        all_probs.extend(prob_positive)
        all_epistemic.extend(epistemic_unc.cpu().numpy())
        all_aleatoric.extend(aleatoric_unc.cpu().numpy())
        all_total.extend(total_unc.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_epistemic),
        np.array(all_aleatoric),
        np.array(all_total)
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
    Analiza si la incertidumbre epistémica es mayor en predicciones incorrectas.

    Un modelo bayesiano bien calibrado debería asignar:
    - Baja incertidumbre cuando acierta (el posterior está concentrado)
    - Alta incertidumbre cuando falla (el posterior está difuso)

    El ratio incorrectas/correctas es la métrica clave:
    - ratio > 1: el modelo "sabe cuándo no sabe" → comportamiento deseable
    - ratio ≈ 1: la incertidumbre no discrimina entre aciertos y errores
    - ratio < 1: el modelo es más inseguro cuando acierta (señal de mal calibrado)

    Args:
        uncertainties: Incertidumbre epistémica por muestra [N]
        y_true: Etiquetas reales [N]
        y_pred: Predicciones binarias [N]  ← debe ser (proba >= 0.5).astype(int)

    Returns:
        Dict con estadísticas por grupo y ratio comparativo
    """
    correct_mask = (y_true == y_pred)
    incorrect_mask = ~correct_mask

    def _safe_stats(arr):
        """Estadísticas seguras para arrays potencialmente vacíos."""
        if len(arr) == 0:
            return {'mean': float('nan'), 'std': float('nan'), 'count': 0}
        return {'mean': float(arr.mean()), 'std': float(arr.std()), 'count': int(len(arr))}

    correct_stats   = _safe_stats(uncertainties[correct_mask])
    incorrect_stats = _safe_stats(uncertainties[incorrect_mask])

    # Ratio: cuántas veces más incierto está el modelo cuando falla
    # Añadimos eps para evitar división por cero
    eps = 1e-10
    ratio = incorrect_stats['mean'] / (correct_stats['mean'] + eps)

    return {
        'mean_uncertainty_correct':   correct_stats['mean'],
        'std_uncertainty_correct':    correct_stats['std'],
        'n_correct':                  correct_stats['count'],
        'mean_uncertainty_incorrect': incorrect_stats['mean'],
        'std_uncertainty_incorrect':  incorrect_stats['std'],
        'n_incorrect':                incorrect_stats['count'],
        'ratio':                      float(ratio),
    }


def full_evaluation(
    det_model: DeterministicCNN,
    mc_model: MCDropoutCNN,
    laplace_model: LaplaceWrapper,
    val_loader: DataLoader,
    test_loader: DataLoader,
    save_dir: Optional[Path] = None
) -> Dict[str, Dict]:
    """
    Ejecuta la evaluación completa de todos los modelos.

    Flujo:
    1. Calibra umbrales de triaje en validation usando percentil fijo
    2. Evalúa cada modelo en test
    3. Calcula métricas de clasificación/calibración
    4. Calcula métricas de triaje con umbrales calibrados
    5. Guarda thresholds y resultados

    Args:
        det_model: Modelo determinista
        mc_model: Modelo MC Dropout
        laplace_model: Modelo con Laplace
        val_loader: DataLoader de validación
        test_loader: DataLoader de test
        save_dir: Directorio donde guardar resultados JSON

    Returns:
        Dict con todos los resultados
    """
    if save_dir is None:
        save_dir = RESULTS_DIR

    results = {}

    # ==============================================================
    # 0. Calibración de thresholds en validation
    # ==============================================================
    print("\n" + "=" * 60)
    print("CALIBRANDO UMBRALES DE TRIAJE EN VALIDATION")
    print("=" * 60)

    # MC Dropout en validation
    _, _, epis_mc_val, _, total_unc_mc_val = evaluate_mc_dropout(mc_model, val_loader)
    mc_thr = calibrate_uncertainty_threshold(epis_mc_val, percentile=TRIAGE_PERCENTILE)

    # Laplace en validation
    _, _, epis_la_val, _, total_unc_la_val = evaluate_laplace(laplace_model, val_loader)
    la_thr = calibrate_uncertainty_threshold(epis_la_val, percentile=TRIAGE_PERCENTILE)

    thresholds = {
        "triage_percentile": TRIAGE_PERCENTILE,
        "mc_dropout": {
            "epistemic_threshold": float(mc_thr),
            "epistemic_p50": float(np.percentile(epis_mc_val, 50)),
            "epistemic_p85": float(np.percentile(epis_mc_val, 85)),
            "epistemic_p90": float(np.percentile(epis_mc_val, 90)),
            "epistemic_p95": float(np.percentile(epis_mc_val, 95)),
            "total_std_p50": float(np.percentile(np.sqrt(np.maximum(total_unc_mc_val, 0.0)), 50)),
            "total_std_p85": float(np.percentile(np.sqrt(np.maximum(total_unc_mc_val, 0.0)), 85)),
            "total_std_p90": float(np.percentile(np.sqrt(np.maximum(total_unc_mc_val, 0.0)), 90)),
            "total_std_p95": float(np.percentile(np.sqrt(np.maximum(total_unc_mc_val, 0.0)), 95)),
        },
        "laplace": {
            "epistemic_threshold": float(la_thr),
            "epistemic_p50": float(np.percentile(epis_la_val, 50)),
            "epistemic_p85": float(np.percentile(epis_la_val, 85)),
            "epistemic_p90": float(np.percentile(epis_la_val, 90)),
            "epistemic_p95": float(np.percentile(epis_la_val, 95)),
            "total_std_p50": float(np.percentile(np.sqrt(np.maximum(total_unc_la_val, 0.0)), 50)),
            "total_std_p85": float(np.percentile(np.sqrt(np.maximum(total_unc_la_val, 0.0)), 85)),
            "total_std_p90": float(np.percentile(np.sqrt(np.maximum(total_unc_la_val, 0.0)), 90)),
            "total_std_p95": float(np.percentile(np.sqrt(np.maximum(total_unc_la_val, 0.0)), 95)),
        }
    }

    with open(save_dir / "uncertainty_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"MC Dropout threshold (p{TRIAGE_PERCENTILE}): {mc_thr:.6f}")
    print(f"Laplace threshold    (p{TRIAGE_PERCENTILE}): {la_thr:.6f}")

    # ==============================================================
    # 1. Deterministic
    # ==============================================================
    print("\n" + "=" * 60)
    print("EVALUANDO MODELO DETERMINISTA")
    print("=" * 60)

    start_time = time.perf_counter()
    mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    y_true_det, y_pred_det, conf_det = evaluate_deterministic(det_model, test_loader)

    end_time = time.perf_counter()
    mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    metrics_det = compute_all_metrics(y_true_det, y_pred_det)
    metrics_det['auc'] = metrics_det['auc_roc']
    metrics_det['brier'] = metrics_det['brier_score']
    metrics_det['_y_true'] = y_true_det
    metrics_det['_y_pred'] = y_pred_det
    metrics_det['_uncertainty'] = 1.0 - conf_det
    metrics_det['inference_time_sec'] = end_time - start_time
    metrics_det['inference_memory_delta_mb'] = (mem_after - mem_before) / 1e6

    results['deterministic'] = metrics_det

    # ==============================================================
    # 2. MC Dropout
    # ==============================================================
    print("\n" + "=" * 60)
    print("EVALUANDO MC DROPOUT")
    print("=" * 60)

    start_time = time.perf_counter()
    mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    y_true_mc, y_pred_mc, epis_mc, ale_mc, total_unc_mc = evaluate_mc_dropout(mc_model, test_loader)

    end_time = time.perf_counter()
    mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    metrics_mc = compute_all_metrics(y_true_mc, y_pred_mc)
    metrics_mc['auc'] = metrics_mc['auc_roc']
    metrics_mc['brier'] = metrics_mc['brier_score']
    metrics_mc['_y_true'] = y_true_mc
    metrics_mc['_y_pred'] = y_pred_mc
    metrics_mc['_epistemic_uncertainty'] = epis_mc
    metrics_mc['_aleatoric_uncertainty'] = ale_mc
    metrics_mc['_total_uncertainty'] = total_unc_mc
    metrics_mc['_epistemic_std'] = np.sqrt(np.maximum(epis_mc, 0.0))
    metrics_mc['_total_std'] = np.sqrt(np.maximum(total_unc_mc, 0.0))
    metrics_mc['inference_time_sec'] = end_time - start_time
    metrics_mc['inference_memory_delta_mb'] = (mem_after - mem_before) / 1e6

    y_pred_binary_mc = (y_pred_mc >= 0.5).astype(int)
    unc_analysis_mc = analyze_uncertainty_by_correctness(epis_mc, y_true_mc, y_pred_binary_mc)
    metrics_mc['_uncertainty_analysis'] = unc_analysis_mc

    triage_mc = compute_triage_metrics(
        y_true_mc, y_pred_mc, epis_mc,
        uncertainty_threshold=mc_thr
    )
    metrics_mc.update(triage_mc)
    metrics_mc['triage_percentile'] = TRIAGE_PERCENTILE

    results['mc_dropout'] = metrics_mc

    print(f"AUC-ROC: {metrics_mc['auc_roc']:.4f}")
    print(f"ECE: {metrics_mc['ece']:.4f}")
    print(f"MC threshold: {mc_thr:.6f}")
    print(f"Incertidumbre media (correctas): {unc_analysis_mc['mean_uncertainty_correct']:.6f}")
    print(f"Incertidumbre media (incorrectas): {unc_analysis_mc['mean_uncertainty_incorrect']:.6f}")
    print(f"Referral rate: {metrics_mc['referral_rate']:.2%}")
    print(f"FN no derivados: {metrics_mc['false_negatives_non_referred']}")

    # ==============================================================
    # 3. Laplace
    # ==============================================================
    print("\n" + "=" * 60)
    print("EVALUANDO LAPLACE APPROXIMATION")
    print("=" * 60)

    start_time = time.perf_counter()
    mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    y_true_la, y_pred_la, epis_la, ale_la, total_unc_la = evaluate_laplace(laplace_model, test_loader)

    end_time = time.perf_counter()
    mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    metrics_la = compute_all_metrics(y_true_la, y_pred_la)
    metrics_la['auc'] = metrics_la['auc_roc']
    metrics_la['brier'] = metrics_la['brier_score']
    metrics_la['_y_true'] = y_true_la
    metrics_la['_y_pred'] = y_pred_la
    metrics_la['_epistemic_uncertainty'] = epis_la
    metrics_la['_aleatoric_uncertainty'] = ale_la
    metrics_la['_total_uncertainty'] = total_unc_la
    metrics_la['_epistemic_std'] = np.sqrt(np.maximum(epis_la, 0.0))
    metrics_la['_total_std'] = np.sqrt(np.maximum(total_unc_la, 0.0))
    metrics_la['inference_time_sec'] = end_time - start_time
    metrics_la['inference_memory_delta_mb'] = (mem_after - mem_before) / 1e6

    y_pred_binary_la = (y_pred_la >= 0.5).astype(int)
    unc_analysis_la = analyze_uncertainty_by_correctness(epis_la, y_true_la, y_pred_binary_la)
    metrics_la['_uncertainty_analysis'] = unc_analysis_la

    triage_la = compute_triage_metrics(
        y_true_la, y_pred_la, epis_la,
        uncertainty_threshold=la_thr
    )
    metrics_la.update(triage_la)
    metrics_la['triage_percentile'] = TRIAGE_PERCENTILE

    results['laplace'] = metrics_la

    print(f"AUC-ROC: {metrics_la['auc_roc']:.4f}")
    print(f"ECE: {metrics_la['ece']:.4f}")
    print(f"Laplace threshold: {la_thr:.6f}")
    print(f"Incertidumbre media (correctas): {unc_analysis_la['mean_uncertainty_correct']:.6f}")
    print(f"Incertidumbre media (incorrectas): {unc_analysis_la['mean_uncertainty_incorrect']:.6f}")
    print(f"Referral rate: {metrics_la['referral_rate']:.2%}")
    print(f"FN no derivados: {metrics_la['false_negatives_non_referred']}")

    print_metrics_table({
        'Deterministic': metrics_det,
        'MC Dropout': metrics_mc,
        'Laplace': metrics_la
    })

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

        if '_epistemic_uncertainty' not in model_results or '_y_true' not in model_results:
            continue

        uncertainties = model_results['_epistemic_uncertainty']
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
