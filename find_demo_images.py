"""
find_demo_images.py - Búsqueda de imágenes representativas para la demo
========================================================================

Recorre train/val/test con AMBOS modelos (MC Dropout y Laplace) y aplica
triage_decision a cada imagen con cada uno. La decisión "ganadora" que
determina en qué carpeta se guarda la imagen es la de MC Dropout (modelo
principal de incertidumbre), pero el nombre del fichero refleja qué modelos
coinciden en esa decisión mediante los sufijos _mc, _la o _mc_la.

Lógica del nombre de archivo:
    - _mc_la  → ambos modelos toman la misma decisión
    - _mc     → solo MC Dropout toma esa decisión (Laplace difiere)
    - _la     → solo Laplace toma esa decisión (MC Dropout difiere)
      (este caso no ocurre porque MC Dropout es el modelo guía, pero
       puede ocurrir en batches donde Laplace "vota" distinto)

Uso:
    python find_demo_images.py

Requisitos:
    - Modelos entrenados en models/  (deterministic_model.pt,
      mc_dropout_model.pt, laplace_fitted.pkl)
    - Splits guardados en results/data_splits/
    - Dataset en data/train/
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, DATA_DIR,
    BATCH_SIZE, CONFIDENCE_THRESHOLD
)
from models import load_model, LaplaceWrapper
from data import get_dataloaders
from metrics import triage_decision

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
DEMO_DIR        = Path("demo_images")
IMAGES_PER_CLASS = 5    # Imágenes a guardar por categoría
MC_SAMPLES       = 50   # Forward passes (para ambos modelos)

# Umbrales de respaldo si no se encuentra results/uncertainty_thresholds.json
UNCERTAINTY_THRESHOLD_MC_FALLBACK = 0.00010697950347093865
UNCERTAINTY_THRESHOLD_LA_FALLBACK = 5.562150818150258e-06


# =============================================================================
# CARGA DE MODELOS
# =============================================================================

def load_mc_dropout_model():
    path = MODELS_DIR / "mc_dropout_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    model = load_model(path, model_type="mc_dropout")
    model.eval()
    return model


def load_laplace_model(train_loader):
    """
    Carga el modelo determinista y fittea Laplace sobre él.
    No intenta cargar desde pickle porque el objeto Laplace no es
    serializable de forma fiable entre entornos.
    """
    det_path = MODELS_DIR / "deterministic_model.pt"
    if not det_path.exists():
        raise FileNotFoundError(f"Modelo determinista no encontrado: {det_path}")

    print("  Cargando pesos del modelo determinista...")
    det_model = load_model(det_path, model_type="deterministic")

    print("  Fiteando Laplace Approximation (puede tardar unos minutos)...")
    laplace_model = LaplaceWrapper(det_model)
    laplace_model.fit(train_loader)

    return laplace_model


# =============================================================================
# DATALOADERS
# =============================================================================

def load_all_dataloaders():
    splits_dir = RESULTS_DIR / "data_splits"
    train_csv  = splits_dir / "train_split.csv"
    val_csv    = splits_dir / "val_split.csv"
    test_csv   = splits_dir / "test_split.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        print("✓ Usando splits guardados desde main.py")
        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)
        test_df  = pd.read_csv(test_csv)
    else:
        print("⚠️  Splits no encontrados — generando con create_data_splits()")
        from data import create_data_splits
        train_df, val_df, test_df = create_data_splits()

    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, batch_size=BATCH_SIZE
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# HELPERS
# =============================================================================

def denormalize_to_pil(tensor: torch.Tensor) -> Image.Image:
    mean   = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std    = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img    = (tensor.cpu() * std + mean).clamp(0, 1)
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(img_np)


def load_model_uncertainty_thresholds() -> dict[str, float]:
    """
    Carga umbrales de incertidumbre calibrados por modelo.

    Prioriza results/uncertainty_thresholds.json y usa valores de respaldo
    si el archivo no existe o no tiene el formato esperado.
    """
    thresholds_path = RESULTS_DIR / "uncertainty_thresholds.json"

    if not thresholds_path.exists():
        print(
            "⚠️  No se encontró results/uncertainty_thresholds.json; "
            "usando umbrales de respaldo."
        )
        return {
            "mc_dropout": UNCERTAINTY_THRESHOLD_MC_FALLBACK,
            "laplace": UNCERTAINTY_THRESHOLD_LA_FALLBACK,
        }

    with open(thresholds_path, "r") as f:
        thresholds_data = json.load(f)

    try:
        mc_thr = float(thresholds_data["mc_dropout"]["epistemic_threshold"])
        la_thr = float(thresholds_data["laplace"]["epistemic_threshold"])
    except (KeyError, TypeError, ValueError):
        print(
            "⚠️  Formato inválido en uncertainty_thresholds.json; "
            "usando umbrales de respaldo."
        )
        return {
            "mc_dropout": UNCERTAINTY_THRESHOLD_MC_FALLBACK,
            "laplace": UNCERTAINTY_THRESHOLD_LA_FALLBACK,
        }

    return {"mc_dropout": mc_thr, "laplace": la_thr}


def decisions_for_batch(
    model,
    images: torch.Tensor,
    uncertainty_threshold: float,
) -> list[dict]:
    """Devuelve lista de dicts triage_decision para cada imagen del batch."""
    images = images.to(DEVICE)
    mean_probs, epistemic, _, _, _ = model.predict_with_uncertainty(
        images, n_samples=MC_SAMPLES
    )
    return [
        triage_decision(
            mean_probs[i, 1].item(),
            epistemic[i].item(),
            uncertainty_threshold=uncertainty_threshold,
        )
        for i in range(len(images))
    ]


def model_suffix(dec_mc: str, dec_la: str) -> str:
    """
    Construye el sufijo del nombre de archivo en función de qué modelo(s)
    toman la decisión que se está guardando (siempre guiada por MC Dropout).

    dec_mc: decisión de MC Dropout ("BENIGN" / "CANCER" / "UNCERTAIN")
    dec_la: decisión de Laplace

    Retorna: "_mc_la" si coinciden, "_mc" si solo MC toma esa decisión.
    """
    if dec_mc == dec_la:
        return "_mc_la"
    return "_mc"


# =============================================================================
# MAIN
# =============================================================================

def main():
    # model_thresholds = load_model_uncertainty_thresholds()
    model_thresholds = {
        "mc_dropout": UNCERTAINTY_THRESHOLD_MC_FALLBACK,
        "laplace": UNCERTAINTY_THRESHOLD_LA_FALLBACK,
    }
    uncertainty_threshold_mc = model_thresholds["mc_dropout"]
    uncertainty_threshold_la = model_thresholds["laplace"]

    print("\n" + "="*65)
    print("  BÚSQUEDA DE IMÁGENES DEMO — MC DROPOUT + LAPLACE")
    print("="*65)
    print(f"  MC samples:            {MC_SAMPLES}")
    print(f"  Uncertainty threshold (MC): {uncertainty_threshold_mc:.12g}")
    print(f"  Uncertainty threshold (LA): {uncertainty_threshold_la:.12g}")
    print(f"  Confidence threshold:  {CONFIDENCE_THRESHOLD}")
    print(f"  Imágenes por clase:    {IMAGES_PER_CLASS}")
    print("="*65 + "\n")

    # ------------------------------------------------------------------
    # Carpetas de salida
    # ------------------------------------------------------------------
    for cat in ["benign", "cancer", "uncertain"]:
        (DEMO_DIR / cat).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cargar modelos
    # ------------------------------------------------------------------
    print("Cargando MC Dropout...")
    mc_model = load_mc_dropout_model()

    print("Cargando dataloaders...")  # necesario antes de Laplace
    train_loader, val_loader, test_loader = load_all_dataloaders()

    print("Cargando Laplace (fiteando sobre train — puede tardar)...")
    la_model = load_laplace_model(train_loader)

    all_loaders = [
        ("train", train_loader),
        ("val",   val_loader),
        ("test",  test_loader),
    ]

    # ------------------------------------------------------------------
    # Contadores y colecciones
    # ------------------------------------------------------------------
    # Contadores por modelo y categoría
    counters_mc = Counter()
    counters_la = Counter()

    # Imágenes ya guardadas por categoría (guiadas por MC Dropout)
    collected = {"BENIGN": [], "CANCER": [], "UNCERTAIN": []}
    needed    = {cat: IMAGES_PER_CLASS for cat in collected}

    # Estadísticas agregadas (guiadas por MC Dropout)
    stats = {cat: {"probs_mc": [], "epis_mc": [], "probs_la": [], "epis_la": []}
             for cat in collected}

    # ------------------------------------------------------------------
    # Recorrido
    # ------------------------------------------------------------------
    # for split_name, loader in all_loaders:
    #     print(f"Procesando split: {split_name} ({len(loader.dataset):,} imágenes)...")

    # Solo procesar el loader de test
    split_name, loader = all_loaders[2]  # ("test", test_loader)
    print(f"Procesando split: {split_name} ({len(loader.dataset):,} imágenes)...")
    for images, labels in tqdm(loader, desc=f"  {split_name}", leave=False):
        # Inferencia con ambos modelos
        decs_mc = decisions_for_batch(
            mc_model,
            images,
            uncertainty_threshold=uncertainty_threshold_mc,
            )
        decs_la = decisions_for_batch(
            la_model,
            images,
            uncertainty_threshold=uncertainty_threshold_la,
        )

        for i in range(len(images)):
            dec_mc = decs_mc[i]["decision"]   # Guía
            dec_la = decs_la[i]["decision"]

            counters_mc[dec_mc] += 1
            counters_la[dec_la] += 1

            # Acumular estadísticas según la categoría de MC Dropout
            stats[dec_mc]["probs_mc"].append(decs_mc[i]["confidence"])
            stats[dec_mc]["epis_mc"].append(
                # epistemic no está en el dict de triage_decision;
                # usamos confidence como proxy (ya calculado dentro)
                # → guardamos solo lo que ya tenemos
                0.0  # placeholder; ver nota abajo (*)
            )
            stats[dec_mc]["probs_la"].append(decs_la[i]["confidence"])

            # Guardar imagen si aún necesitamos más de esta categoría
            if len(collected[dec_mc]) < needed[dec_mc]:
                pil_img = denormalize_to_pil(images[i])
                suffix  = model_suffix(dec_mc, dec_la)
                gt      = labels[i].item()
                idx     = len(collected[dec_mc]) + 1
                fname   = f"{dec_mc.lower()}_{idx:02d}_gt{gt}{suffix}.png"
                pil_img.save(DEMO_DIR / dec_mc.lower() / fname)
                collected[dec_mc].append(fname)

    # ------------------------------------------------------------------
    # Resumen de acuerdo entre modelos
    # ------------------------------------------------------------------
    # Reconstruir contadores de acuerdo (requeriría guardar pares durante el
    # loop; aquí lo aproximamos con lo que tenemos en 'collected')
    agreement_counts = {cat: 0 for cat in collected}
    for cat, fnames in collected.items():
        agreement_counts[cat] = sum(1 for f in fnames if "_mc_la" in f)

    # ------------------------------------------------------------------
    # JSON de metadatos
    # ------------------------------------------------------------------
    total_mc = sum(counters_mc.values())
    total_la = sum(counters_la.values())

    summary = {
        "mc_samples": MC_SAMPLES,
        "uncertainty_thresholds": {
            "mc_dropout": uncertainty_threshold_mc,
            "laplace": uncertainty_threshold_la,
        },
        "confidence_threshold":  CONFIDENCE_THRESHOLD,
        "total_images_processed": total_mc,
        "mc_dropout": {
            "counts": dict(counters_mc),
            "percentages": {
                cat: round(counters_mc.get(cat, 0) / total_mc * 100, 2)
                for cat in ["BENIGN", "CANCER", "UNCERTAIN"]
            },
        },
        "laplace": {
            "counts": dict(counters_la),
            "percentages": {
                cat: round(counters_la.get(cat, 0) / total_la * 100, 2)
                for cat in ["BENIGN", "CANCER", "UNCERTAIN"]
            },
        },
        "saved_images": collected,
        "agreement_in_saved": agreement_counts,
    }

    json_path = DEMO_DIR / "demo_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Imprimir resumen
    # ------------------------------------------------------------------
    print("\n" + "="*65)
    print("  RESUMEN DE DECISIONES DE TRIAGE")
    print("="*65)
    print(f"  Total imágenes procesadas: {total_mc:,}\n")

    print(f"  {'Categoría':<12} {'MC Dropout':>12} {'Laplace':>12}")
    print("  " + "-"*38)
    for cat in ["BENIGN", "CANCER", "UNCERTAIN"]:
        n_mc = counters_mc.get(cat, 0)
        n_la = counters_la.get(cat, 0)
        pct_mc = n_mc / total_mc * 100 if total_mc > 0 else 0
        pct_la = n_la / total_la * 100 if total_la > 0 else 0
        print(f"  {cat:<12} {n_mc:>6,} ({pct_mc:4.1f}%)  {n_la:>6,} ({pct_la:4.1f}%)")

    print("\n  Acuerdo MC+Laplace en imágenes guardadas:")
    for cat, fnames in collected.items():
        n_agree = sum(1 for f in fnames if "_mc_la" in f)
        print(f"    {cat:<10}: {n_agree}/{len(fnames)} coinciden")

    print(f"\n  Imágenes guardadas en: {DEMO_DIR.resolve()}/")
    print(f"  Metadatos:             {json_path.resolve()}")
    print("="*65 + "\n")

    # (*) Nota: triage_decision no devuelve la incertidumbre epistémica cruda,
    # solo la decisión y la confianza. Si necesitas las epis en el JSON de
    # estadísticas, extrae epistemic directamente de predict_with_uncertainty
    # en decisions_for_batch y pásala junto al dict de triage.


if __name__ == "__main__":
    main()