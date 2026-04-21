#!/usr/bin/env python3
"""
benchmark_tiempos.py
===================

Script para medir tiempos de inferencia de forma más justa que en evaluate.py.

Qué corrige respecto a la medición actual:
1. Excluye el coste del DataLoader y del acceso a disco.
2. Usa warm-up antes de cronometrar.
3. Sincroniza CUDA antes y después de cada medición.
4. Mide sobre los mismos batches ya cargados en GPU para todos los modelos.
5. Separa explícitamente:
   - Determinista: forward + softmax
   - MC Dropout: predict_with_uncertainty()
   - Laplace (solo probabilidad): predict_proba()  [GLM + probit]
   - Laplace completo: predict_with_uncertainty()

Además, opcionalmente cuenta cuántas veces se ejecutan layer4 y fc dentro de
Laplace para comprobar si laplace-torch reutiliza el backbone y solo muestrea
la última capa.

Uso recomendado:
    python benchmark_tiempos.py --max-batches 20 --repeats 10

Notas:
- Mide solo compute del modelo, no I/O.
- Si no existe un Laplace serializado, reconstruye y hace fit usando train_loader.
- Reporta tiempos por lote y por imagen, que suelen ser más comparables que el
  tiempo total sobre un test completo.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import DEVICE, MODELS_DIR, RESULTS_DIR, RANDOM_SEED
from data import create_data_splits, get_dataloaders
from models import LaplaceWrapper, load_model


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------

def set_seed(seed: int = RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class BenchResult:
    name: str
    mean_sec: float
    std_sec: float
    min_sec: float
    max_sec: float
    mean_ms_per_batch: float
    mean_ms_per_image: float
    repeats: int
    num_batches: int
    batch_size: int
    num_images: int


class Counter:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self, module, inp, out) -> None:
        self.n += 1


# -----------------------------------------------------------------------------
# Carga de datos y modelos
# -----------------------------------------------------------------------------

def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reutiliza los splits guardados si existen; si no, los vuelve a crear.
    """
    split_dir = RESULTS_DIR / "data_splits"
    train_csv = split_dir / "train_split.csv"
    val_csv = split_dir / "val_split.csv"
    test_csv = split_dir / "test_split.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        print("✓ Using saved data splits from results/data_splits")
        return train_df, val_df, test_df

    print("⚠️  Saved splits not found. Recreating splits from seed...")
    return create_data_splits()


def load_models_and_laplace(train_loader):
    det_model_path = MODELS_DIR / "deterministic_model.pt"
    mc_model_path = MODELS_DIR / "mc_dropout_model.pt"

    if not det_model_path.exists() or not mc_model_path.exists():
        raise FileNotFoundError(
            "No se encontraron deterministic_model.pt o mc_dropout_model.pt en MODELS_DIR"
        )

    det_model = load_model(det_model_path, model_type="deterministic")
    mc_model = load_model(mc_model_path, model_type="mc_dropout")

    # Laplace se construye sobre el modelo determinista entrenado
    laplace_model = LaplaceWrapper(det_model)

    # Si no hay serialización utilizable, hacemos fit.
    # En vuestro proyecto el .pkl normalmente no existe, así que lo dejamos simple.
    print("⚠️  Rebuilding Laplace and fitting on train_loader for benchmarking...")
    laplace_model.fit(train_loader)

    det_model.eval()
    mc_model.eval()

    return det_model, mc_model, laplace_model


# -----------------------------------------------------------------------------
# Cache de batches para eliminar I/O del cronómetro
# -----------------------------------------------------------------------------

def cache_batches_on_device(loader, device: torch.device, max_batches: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Carga un número fijo de batches y los deja ya en GPU/DEVICE.
    Así la medición excluye DataLoader, acceso a disco y transforms.
    """
    cached: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i, (images, labels) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        cached.append((images, labels))

    if not cached:
        raise RuntimeError("No se pudieron cachear batches del test_loader")

    total_images = sum(batch[0].shape[0] for batch in cached)
    print(f"✓ Cached {len(cached)} batches on {device} ({total_images} images)")
    return cached


# -----------------------------------------------------------------------------
# Funciones a cronometrar
# -----------------------------------------------------------------------------

@torch.no_grad()
def run_det_batch(det_model, images: torch.Tensor) -> torch.Tensor:
    logits = det_model(images)
    return F.softmax(logits, dim=1)


@torch.no_grad()
def run_mc_batch(mc_model, images: torch.Tensor, n_samples: int) -> torch.Tensor:
    mean_probs, _, _, _, _ = mc_model.predict_with_uncertainty(images, n_samples=n_samples)
    return mean_probs


@torch.no_grad()
def run_laplace_proba_batch(laplace_model, images: torch.Tensor) -> torch.Tensor:
    return laplace_model.predict_proba(images)


@torch.no_grad()
def run_laplace_full_batch(laplace_model, images: torch.Tensor, n_samples: int) -> torch.Tensor:
    mean_probs, _, _, _, _ = laplace_model.predict_with_uncertainty(images, n_samples=n_samples)
    return mean_probs


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------

def bench_batches(
    name: str,
    fn: Callable[..., torch.Tensor],
    cached_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    repeats: int,
    warmup: int,
    *fn_args,
    **fn_kwargs,
) -> BenchResult:
    """
    Cronometra una función sobre los mismos batches ya cargados en DEVICE.
    """
    # Warm-up
    for _ in range(warmup):
        for images, _ in cached_batches:
            _ = fn(*fn_args, images, **fn_kwargs)
        cuda_sync()

    times: List[float] = []
    num_images = sum(images.shape[0] for images, _ in cached_batches)
    batch_size = cached_batches[0][0].shape[0]

    for _ in range(repeats):
        cuda_sync()
        t0 = time.perf_counter()
        for images, _ in cached_batches:
            _ = fn(*fn_args, images, **fn_kwargs)
        cuda_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_sec = float(np.mean(times))
    std_sec = float(np.std(times))
    return BenchResult(
        name=name,
        mean_sec=mean_sec,
        std_sec=std_sec,
        min_sec=float(np.min(times)),
        max_sec=float(np.max(times)),
        mean_ms_per_batch=1000.0 * mean_sec / len(cached_batches),
        mean_ms_per_image=1000.0 * mean_sec / num_images,
        repeats=repeats,
        num_batches=len(cached_batches),
        batch_size=int(batch_size),
        num_images=int(num_images),
    )


def count_laplace_backbone_calls(laplace_model, images: torch.Tensor, n_samples: int) -> Dict[str, int]:
    """
    Comprueba cuántas veces se ejecutan layer4 y fc en Laplace.
    Sirve para inferir si se reutiliza el backbone y solo se muestrea la última capa.
    """
    c_layer4 = Counter()
    c_fc = Counter()

    h1 = laplace_model.model.backbone.layer4.register_forward_hook(c_layer4)
    h2 = laplace_model.model.backbone.fc.register_forward_hook(c_fc)

    try:
        c_layer4.n = 0
        c_fc.n = 0
        _ = laplace_model.predict_proba(images)
        cuda_sync()
        proba_counts = {"layer4": c_layer4.n, "fc": c_fc.n}

        c_layer4.n = 0
        c_fc.n = 0
        _ = laplace_model.predict_with_uncertainty(images, n_samples=n_samples)
        cuda_sync()
        full_counts = {"layer4": c_layer4.n, "fc": c_fc.n}
    finally:
        h1.remove()
        h2.remove()

    return {
        "predict_proba_layer4_calls": int(proba_counts["layer4"]),
        "predict_proba_fc_calls": int(proba_counts["fc"]),
        "predict_with_uncertainty_layer4_calls": int(full_counts["layer4"]),
        "predict_with_uncertainty_fc_calls": int(full_counts["fc"]),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark justo de tiempos de inferencia")
    parser.add_argument("--max-batches", type=int, default=20,
                        help="Número de batches de test a cachear y usar en el benchmark")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Número de repeticiones del cronómetro")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Número de warm-up runs antes de medir")
    parser.add_argument("--mc-samples", type=int, default=50,
                        help="Número de muestras para MC Dropout")
    parser.add_argument("--laplace-samples", type=int, default=100,
                        help="Número de muestras para Laplace predict_with_uncertainty")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Semilla aleatoria")
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print("BENCHMARK JUSTO DE TIEMPOS DE INFERENCIA")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"max_batches={args.max_batches} | repeats={args.repeats} | warmup={args.warmup}")

    train_df, val_df, test_df = load_splits()
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)

    det_model, mc_model, laplace_model = load_models_and_laplace(train_loader)

    cached_test = cache_batches_on_device(test_loader, DEVICE, args.max_batches)
    example_images, _ = cached_test[0]

    print("\nTiming deterministic forward + softmax...")
    det_res = bench_batches(
        "deterministic",
        run_det_batch,
        cached_test,
        args.repeats,
        args.warmup,
        det_model,
    )

    print("Timing MC Dropout full predict_with_uncertainty...")
    mc_res = bench_batches(
        "mc_dropout_full",
        run_mc_batch,
        cached_test,
        args.repeats,
        args.warmup,
        mc_model,
        n_samples=args.mc_samples,
    )

    print("Timing Laplace predict_proba (GLM + probit)...")
    la_proba_res = bench_batches(
        "laplace_predict_proba",
        run_laplace_proba_batch,
        cached_test,
        args.repeats,
        args.warmup,
        laplace_model,
    )

    print("Timing Laplace full predict_with_uncertainty...")
    la_full_res = bench_batches(
        "laplace_full",
        run_laplace_full_batch,
        cached_test,
        args.repeats,
        args.warmup,
        laplace_model,
        n_samples=args.laplace_samples,
    )

    print("\nCounting backbone calls inside Laplace...")
    hook_info = count_laplace_backbone_calls(laplace_model, example_images, args.laplace_samples)

    results = {
        "device": str(DEVICE),
        "max_batches": args.max_batches,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "mc_samples": args.mc_samples,
        "laplace_samples": args.laplace_samples,
        "deterministic": asdict(det_res),
        "mc_dropout_full": asdict(mc_res),
        "laplace_predict_proba": asdict(la_proba_res),
        "laplace_full": asdict(la_full_res),
        "laplace_hook_info": hook_info,
        "interpretation_hint": {
            "expected_pattern": (
                "laplace_predict_proba debería ser del mismo orden que deterministic; "
                "laplace_full debería ser >= laplace_predict_proba y normalmente >= deterministic. "
                "Si predict_with_uncertainty ejecuta layer4 solo una vez y fc muchas veces, "
                "laplace-torch probablemente reutiliza el backbone y muestrea solo la última capa."
            )
        }
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "timing_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "-" * 80)
    for key in ["deterministic", "mc_dropout_full", "laplace_predict_proba", "laplace_full"]:
        r = results[key]
        print(
            f"{r['name']:<24} "
            f"mean={r['mean_sec']:.4f}s ± {r['std_sec']:.4f}s | "
            f"{r['mean_ms_per_batch']:.2f} ms/batch | "
            f"{r['mean_ms_per_image']:.4f} ms/image"
        )

    print("\nLaplace hook info:")
    for k, v in hook_info.items():
        print(f"  {k}: {v}")

    print(f"\n✓ Results saved to: {out_path}")
    print("-" * 80)


if __name__ == "__main__":
    main()
