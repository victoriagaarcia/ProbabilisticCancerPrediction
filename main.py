#!/usr/bin/env python3
# Latest version
"""
=============================================================================
BAYESIAN NEURAL NETWORKS FOR HISTOPATHOLOGIC CANCER DETECTION
=============================================================================
Main execution script for the complete pipeline.

This script orchestrates:
1. Data preparation and loading
2. Training of deterministic CNN (MAP estimate)
3. Laplace Approximation on the trained model
4. MC Dropout training and evaluation
5. Comprehensive evaluation and comparison
6. Generation of figures for the report

Usage:
    python main.py                    # Run complete pipeline
    python main.py --train-only       # Only train models
    python main.py --eval-only        # Only evaluate (requires trained models)
    python main.py --quick            # Quick test with reduced epochs

Authors: [Your names here]
Course: Probabilistic AI - Master's Program
Date: April 2026
=============================================================================
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import pickle

# Import project modules
from config import (
    DEVICE, DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
    NUM_EPOCHS, BATCH_SIZE, RANDOM_SEED,
    MC_SAMPLES, LAPLACE_SAMPLES, DROPOUT_RATE
)
from data import create_data_splits, get_dataloaders
from models import (
    create_deterministic_model, 
    create_mc_dropout_model,
    LaplaceWrapper,
    load_model
)
from train import train_deterministic_model, train_mc_dropout_model
from evaluate import full_evaluation, generate_uncertainty_report
from visualize import generate_all_figures, setup_plotting_style


def set_seed(seed: int = RANDOM_SEED):
    """
    Set random seeds for reproducibility.
    
    This ensures that results are reproducible across runs by fixing
    the random state of all relevant libraries.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # deterministic=True garantiza reproducibilidad en GPU
        # benchmark=False evita que cuDNN elija kernels no deterministas
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_header():
    """Print a nice header for the experiment."""
    print("\n" + "="*75)
    print("   BAYESIAN NEURAL NETWORKS FOR HISTOPATHOLOGIC CANCER DETECTION")
    print("="*75)
    print(f"   Device: {DEVICE}")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*75 + "\n")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'─'*75}")
    print(f"  {title}")
    print(f"{'─'*75}\n")


def check_data_exists() -> bool:
    """
    Check if the dataset exists in the expected location.
    
    Returns:
        bool: True if data exists, False otherwise.
    """
    train_dir = DATA_DIR / "train"
    if not train_dir.exists():
        print(f"⚠️  Dataset not found at {DATA_DIR}")
        print("\nPlease download the dataset from Kaggle:")
        print("  https://www.kaggle.com/c/histopathologic-cancer-detection/data")
        print(f"\nAnd extract it to: {DATA_DIR}")
        print("Expected structure:")
        print("  data/")
        print("  ├── train/")
        print("  │   ├── image1.tif")
        print("  │   └── ...")
        print("  └── train_labels.csv")
        return False
    return True

def save_data_splits(train_df: pd.DataFrame,
                     val_df: pd.DataFrame,
                     test_df: pd.DataFrame,
                     save_dir: Path = RESULTS_DIR / "data_splits",
                     seed: int = RANDOM_SEED) -> None:
    """
    Save the exact data splits used in training/evaluation so they can be
    reused later (e.g. in the demo) without relying on random seeds.

    Args:
        train_df: Training split dataframe
        val_df: Validation split dataframe
        test_df: Test split dataframe
        save_dir: Directory where CSV files and metadata will be stored
        seed: Random seed used for the current run
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    train_path = save_dir / "train_split.csv"
    val_path = save_dir / "val_split.csv"
    test_path = save_dir / "test_split.csv"
    meta_path = save_dir / "split_metadata.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    split_metadata = {
        "seed": seed,
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
    }

    with open(meta_path, "w") as f:
        json.dump(split_metadata, f, indent=2)

    print("\n✓ Exact data splits saved for later reuse")
    print(f"  Train split: {train_path}")
    print(f"  Val split:   {val_path}")
    print(f"  Test split:  {test_path}")
    print(f"  Metadata:    {meta_path}")


def train_pipeline(args, train_loader, val_loader):
    """
    Execute the training pipeline for all models.
    
    Args:
        args: Command line arguments
        train_loader: Training data loader
        val_loader: Validation data loader
        
    Returns:
        tuple: (deterministic_model, mc_dropout_model, laplace_model, histories)
    """
    histories = {}
    num_epochs = args.epochs if args.epochs else NUM_EPOCHS
    
    # =========================================================================
    # STEP 1: Train Deterministic CNN (MAP Estimate)
    # =========================================================================
    print_section("STEP 1: Training Deterministic CNN (MAP Estimate)")
    
    print("This model serves as:")
    print("  • Baseline for performance comparison")
    print("  • MAP estimate (ω*) for Laplace Approximation")
    print(f"\nTraining for {num_epochs} epochs...")

    # Medir tiempo de entrenamiento para el modelo determinista y uso de memoria
    start_time = time.perf_counter()
    mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0 # Memoria antes del entrenamiento
    
    det_model, det_history = train_deterministic_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_path=MODELS_DIR / "deterministic_model.pt"
    )

    end_time = time.perf_counter()
    mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    det_history['training_time_sec'] = end_time - start_time
    det_history['training_memory_delta_mb'] = (mem_after - mem_before) / 1e6  # Convertir a MB

    histories['deterministic'] = det_history
    # Guardar JSON con el historial de entrenamiento del modelo determinista
    with open(RESULTS_DIR / "deterministic_training_history.json", 'w') as f:
        json.dump(det_history, f, indent=2)

    print(f"\n✓ Deterministic model trained successfully")
    print(f"  Best validation AUC: {max(det_history['val_auc']):.4f}")
    
    # ========================================================================
    # STEP 2: Apply Laplace Approximation
    # =========================================================================
    print_section("STEP 2: Applying Laplace Approximation")
    # Prueba de texto para explicar el proceso de Laplace Approximation
    
    print("Laplace approximation constructs a Gaussian posterior:")
    print("  q(ω) = N(ω | ω*, Σ)")
    print("  where Σ = (H + λI)⁻¹ and H is the Hessian")
    print("\nApplying to last layer only (tractable computation)...")
    
    laplace_model = LaplaceWrapper(det_model)
    
    # Medir tiempo de ajuste para el modelo Laplace y uso de memoria
    start_time = time.time()
    mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0 # Memoria antes del ajuste

    # Fit Laplace (compute Hessian)
    laplace_model.fit(train_loader)

    end_time = time.time()
    mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    la_history = {
        'fit_time_sec': end_time - start_time,
        'fit_memory_delta_mb': (mem_after - mem_before) / 1e6  # Convertir a MB
    }
    histories['laplace'] = la_history
    # Guardar JSON con el historial de ajuste del modelo Laplace
    with open(RESULTS_DIR / "laplace_fit_history.json", 'w') as f:
        json.dump(la_history, f, indent=2)

    
    print(f"\n✓ Laplace approximation fitted in {la_history['fit_time_sec']:.2f}s")
    print(f"  Prior precision optimized via marginal likelihood")
    
    # Save Laplace model (with pickle since it contains non-serializable objects)
    # Guardamos el objeto Laplace fitted completo (contiene la Hessiana/covarianza)
    # with open(MODELS_DIR / "laplace_fitted.pkl", "wb") as f:
    #     pickle.dump(laplace_model.la, f) # No es serializable!
    
    # torch.save({
    #     'base_model_state': det_model.state_dict(),
    #     'laplace_fitted': laplace_model.fitted,
    # }, MODELS_DIR / "laplace_model.pt")
    torch.save({
        'base_model_state': det_model.state_dict(),
        # 'laplace_obj': laplace_model.la, # esto no es serializable
        'laplace_fitted': laplace_model.fitted,
        'model_type': 'laplace'
    }, MODELS_DIR / "laplace_model.pt")
    
    # =========================================================================
    # STEP 3: Train MC Dropout Model
    # =========================================================================
    print_section("STEP 3: Training MC Dropout Model")
    
    print("MC Dropout approximates Bayesian inference by:")
    print("  • Keeping dropout active at test time")
    print("  • Each forward pass samples from approximate posterior")
    print(f"  • Dropout rate: {DROPOUT_RATE}")
    print(f"\nTraining for {num_epochs} epochs...")

    # Medir tiempo de entrenamiento para el modelo MC Dropout y uso de memoria
    start_time = time.perf_counter()
    mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    mc_model, mc_history = train_mc_dropout_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_path=MODELS_DIR / "mc_dropout_model.pt"
    )

    end_time = time.perf_counter()
    mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    mc_history['training_time_sec'] = end_time - start_time
    mc_history['training_memory_delta_mb'] = (mem_after - mem_before) / 1e6  # Convertir a MB

    histories['mc_dropout'] = mc_history
    # Guardar JSON con el historial de entrenamiento del modelo MC Dropout
    with open(RESULTS_DIR / "mc_dropout_training_history.json", 'w') as f:
        json.dump(mc_history, f, indent=2)

    print(f"\n✓ MC Dropout model trained successfully")
    print(f"  Best validation AUC: {max(mc_history['val_auc']):.4f}")
    
    return det_model, mc_model, laplace_model, histories


def evaluation_pipeline(det_model, mc_model, laplace_model, 
                        test_loader, train_loader, histories=None):
    """
    Execute the complete evaluation pipeline.
    
    Args:
        det_model: Trained deterministic model
        mc_model: Trained MC Dropout model
        laplace_model: Fitted Laplace model
        test_loader: Test data loader
        train_loader: Training loader (for refitting Laplace if needed)
        histories: Training histories (optional)
        
    Returns:
        dict: Complete evaluation results
    """
    print_section("STEP 4: Comprehensive Evaluation")
    
    print("Evaluating all models on test set...")
    print(f"  • Deterministic CNN (point estimate)")
    print(f"  • Laplace Approximation ({LAPLACE_SAMPLES} posterior samples)")
    print(f"  • MC Dropout ({MC_SAMPLES} forward passes)")
    
    # Run full evaluation
    results = full_evaluation(
        det_model=det_model,
        mc_model=mc_model,
        laplace_model=laplace_model,
        test_loader=test_loader,
        save_dir=RESULTS_DIR
    )
    
    # =========================================================================
    # Print Results Summary
    # =========================================================================
    print_section("RESULTS SUMMARY")
    
    # Classification metrics
    print("Classification Performance (AUC-ROC):")
    print(f"  Deterministic:  {results['deterministic']['auc']:.4f}")
    print(f"  Laplace:        {results['laplace']['auc']:.4f}")
    print(f"  MC Dropout:     {results['mc_dropout']['auc']:.4f}")
    
    print("\nCalibration Metrics (ECE - lower is better):")
    print(f"  Deterministic:  {results['deterministic']['ece']:.4f}")
    print(f"  Laplace:        {results['laplace']['ece']:.4f}")
    print(f"  MC Dropout:     {results['mc_dropout']['ece']:.4f}")
    
    print("\nBrier Score (lower is better):")
    print(f"  Deterministic:  {results['deterministic']['brier']:.4f}")
    print(f"  Laplace:        {results['laplace']['brier']:.4f}")
    print(f"  MC Dropout:     {results['mc_dropout']['brier']:.4f}")
    
    # New metrics
    print("\nUncertainty Decomposition & Triage (Laplace / MC Dropout):")
    for model_name in ['laplace', 'mc_dropout']:
        r = results[model_name]
        print(f"\n  {model_name}:")
        print(f"    Referral rate (uncertain → human): {r.get('referral_rate', 0):.1%}")
        print(f"    Coverage (auto-resolved):          {r.get('coverage', 0):.1%}")
        print(f"    Accuracy on confident predictions: {r.get('accuracy_on_confident', 0):.4f}")

    print("\nInference Profiling:")
    for model_name in ['deterministic', 'laplace', 'mc_dropout']:
        r = results[model_name]
        print(f"  {model_name}: {r.get('inference_time_sec', 0) * 1000:.1f} ms | "
              f"{r.get('inference_memory_delta_mb', 0):.1f} MB")

    # =========================================================================
    # Generate Uncertainty Report
    # =========================================================================
    print_section("STEP 5: Uncertainty Analysis")
    
    report = generate_uncertainty_report(results, save_path=RESULTS_DIR / "uncertainty_report.txt")
    print(report)
    
    # =========================================================================
    # Generate Figures
    # =========================================================================
    print_section("STEP 6: Generating Figures")
    
    setup_plotting_style()
    generate_all_figures(
        results=results,
        histories=histories,
        test_loader=test_loader,
        save_dir=FIGURES_DIR
    )
    
    print(f"\n✓ All figures saved to {FIGURES_DIR}")
    
    return results


def main():
    """Main entry point for the experiment."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="BNN for Histopathologic Cancer Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Full pipeline
  python main.py --train-only       # Train models only
  python main.py --eval-only        # Evaluate existing models
  python main.py --quick            # Quick test (5 epochs)
  python main.py --epochs 50        # Custom number of epochs
        """
    )
    parser.add_argument('--train-only', action='store_true',
                        help='Only train models, skip evaluation')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate existing models')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 5 epochs')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.epochs = 5
    
    # Set random seed
    set_seed(args.seed)

    # Keep track of execution time
    print(f"Execution started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print header
    print_header()
    
    # Check if data exists
    if not check_data_exists():
        sys.exit(1)
    
    # Create output directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Data Preparation
    # =========================================================================
    print_section("DATA PREPARATION")
    
    print("Loading and splitting dataset...")
    train_df, val_df, test_df = create_data_splits()
    
    print(f"  Training samples:   {len(train_df):,}")
    print(f"  Validation samples: {len(val_df):,}")
    print(f"  Test samples:       {len(test_df):,}")

    # Save splits used in training
    save_data_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        save_dir=RESULTS_DIR / "data_splits",
        seed=args.seed
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df,
        batch_size=BATCH_SIZE
    )
    
    print(f"\n✓ Data loaders created (batch size: {BATCH_SIZE})")
    print(f"Data loaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =========================================================================
    # Training or Loading
    # =========================================================================
    histories = None
    
    if args.eval_only:
        # Load existing models
        print_section("LOADING EXISTING MODELS")
        
        det_model_path = MODELS_DIR / "deterministic_model.pt"
        mc_model_path = MODELS_DIR / "mc_dropout_model.pt"
        
        if not det_model_path.exists() or not mc_model_path.exists():
            print("❌ Trained models not found. Run training first.")
            sys.exit(1)
        
        det_model = load_model(det_model_path, model_type='deterministic')
        mc_model = load_model(mc_model_path, model_type='mc_dropout')
        
        # Create and fit Laplace model
        # laplace_model = LaplaceWrapper(det_model)
        # laplace_model.fit(train_loader)

        # Rebuild and fit Laplace model
        # laplace_model_path = MODELS_DIR / "laplace_model.pt"
        # laplace_model = LaplaceWrapper(det_model)
        # laplace_model.fit(train_loader)

        # Load Laplace fitted object (.pkl)
        laplace_pkl_path = MODELS_DIR / "laplace_model.pkl"
        laplace_model = LaplaceWrapper(det_model)
        if laplace_pkl_path.exists(): # Este path no va a existir nunca porque no sepuede guardar
            with open(laplace_pkl_path, "rb") as f:
                laplace_model.la = pickle.load(f)
                print("✓ Laplace model loaded successfully from pickle")
        else:
            print("⚠️  Laplace fitted model not found. Fitting now...")
            laplace_model.fit(train_loader)
        
        print("✓ Models loaded successfully")
        print(f"Models loaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Try to load histories
        history_path = RESULTS_DIR / "training_histories.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                histories = json.load(f)
    else:
        # Train models
        det_model, mc_model, laplace_model, histories = train_pipeline(
            args, train_loader, val_loader
        )
        print(f"Models trained at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save training histories
        with open(RESULTS_DIR / "training_histories.json", 'w') as f:
            json.dump(histories, f, indent=2)
    
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    if not args.train_only:
        results = evaluation_pipeline(
            det_model=det_model,
            mc_model=mc_model,
            laplace_model=laplace_model,
            test_loader=test_loader,
            train_loader=train_loader,
            histories=histories
        )

        print(f"Evaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save final results
        def _to_serializable(v):
            """Convierte tipos numpy a tipos Python nativos para JSON."""
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            if isinstance(v, dict):
                return {dk: _to_serializable(dv) for dk, dv in v.items()}
            return v

        with open(RESULTS_DIR / "final_results.json", 'w') as f:
            serializable_results = {}
            for model_name, model_results in results.items():
                serializable_results[model_name] = {
                    k: _to_serializable(v)
                    for k, v in model_results.items()
                    if not k.startswith('_')  # Skip per-sample arrays (_y_true, etc.)
                }
            json.dump(serializable_results, f, indent=2)
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*75)
    print("   EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*75)
    print(f"\n  Models saved to:     {MODELS_DIR}")
    print(f"  Results saved to:    {RESULTS_DIR}")
    print(f"  Figures saved to:    {FIGURES_DIR}")
    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
