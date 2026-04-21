# Bayesian Neural Networks for Histopathologic Cancer Detection

## 🎯 Project Overview

This project implements **Bayesian Neural Networks (BNNs)** for uncertainty-aware classification of histopathology images, comparing three approaches:

1. **Deterministic CNN** (ResNet18) - Baseline/MAP estimate
2. **Laplace Approximation** - Gaussian posterior approximation
3. **MC Dropout** - Variational approximation via dropout

The clinical motivation is to identify not just *what* the model predicts, but *how confident* it is—enabling prioritization of ambiguous cases for expert review.

---

## 📁 Project Structure

```
bnn_cancer_detection/
├── config.py          # Configuration and hyperparameters
├── data.py            # Dataset loading and preprocessing
├── models.py          # Model architectures (Deterministic, MC Dropout, Laplace)
├── train.py           # Training loop with early stopping
├── metrics.py         # Evaluation metrics (AUC, ECE, Brier Score)
├── evaluate.py        # Model evaluation and uncertainty analysis
├── visualize.py       # Figure generation for report
├── main.py            # Main execution script
├── app.py             # Streamlit interactive demo
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection/data) dataset from Kaggle and extract to `./data/`:

```
data/
├── train/
│   ├── 0000d563....tif
│   └── ...
└── train_labels.csv
```

### 3. Run Complete Pipeline

```bash
# Full training + evaluation (recommended)
python main.py

# Quick test (5 epochs, for debugging)
python main.py --quick

# Train only (skip evaluation)
python main.py --train-only

# Evaluate existing models
python main.py --eval-only
```

### 4. Launch Interactive Demo

```bash
streamlit run app.py
```

---

## 🧠 Theoretical Background

### Bayesian Neural Networks

Instead of learning point estimates of weights $\omega^{*}$, BNNs learn a **posterior distribution** $p(\omega | \mathcal{D})$. Predictions integrate over this posterior:

$$p(y^{*} | x^{*}, \mathcal{D}) = \int p(y^{*} | x^{*}, \omega) \, p(\omega | \mathcal{D}) \, d\omega$$

This integral is intractable for neural networks, so we use **approximate inference**.

### Laplace Approximation

The key insight: once we have a trained model (MAP estimate $\omega^{*}$), we can approximate the posterior as a Gaussian centered at that point:

$$q(\omega) = \mathcal{N}(\omega \mid \omega^{*}, \Sigma), \quad \Sigma = \left( \mathbf{H} + \lambda \mathbf{I} \right)^{-1}$$

where $\mathbf{H}$ is the Hessian of the loss at $\omega^{*}$.

**Advantages:**
- Post-hoc: no retraining required
- Principled uncertainty from curvature
- Fast with last-layer approximation

**Limitations:**
- Local approximation (misses multimodal posteriors)
- Gaussian assumption may be restrictive

### MC Dropout

Dropout at test time can be interpreted as approximate variational inference:

$$p(y^{*} | x^{*}) \approx \frac{1}{T} \sum_{t=1}^{T} p(y^{*} | x^{*}, \omega_t), \quad \omega_t \sim q(\omega)$$

Each forward pass with a different dropout mask samples from an implicit posterior.

**Advantages:**
- Simple implementation (just keep dropout active)
- No modification to training procedure

**Limitations:**
- Dropout rate is a hyperparameter, not learned
- May underestimate uncertainty

---

## 📊 Metrics

### Classification Metrics
- **AUC-ROC**: Discrimination ability
- **F1 Score**: Harmonic mean of precision/recall
- **Accuracy**: Overall correctness

### Calibration Metrics

#### Expected Calibration Error (ECE)

Measures alignment between predicted confidence and actual accuracy:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

A perfectly calibrated model has ECE = 0.

#### Brier Score

Mean squared error between predicted probabilities and true labels:

$$\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2$$

Lower is better. Captures both calibration and discrimination.

---

## 📈 Expected Results

| Model | AUC-ROC | ECE (↓) | Brier (↓) |
|-------|---------|---------|-----------|
| Deterministic | ~0.95 | ~0.08 | ~0.12 |
| Laplace | ~0.95 | ~0.04 | ~0.10 |
| MC Dropout | ~0.94 | ~0.05 | ~0.11 |

*Results may vary based on random seed and training configuration.*

---

## 🔬 Key Code Components

### Configuration (`config.py`)

```python
# Training hyperparameters
NUM_EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4  # Corresponds to Gaussian prior p(ω) = N(0, λ⁻¹I)

# Bayesian inference
MC_SAMPLES = 50           # Forward passes for MC Dropout
LAPLACE_SAMPLES = 100     # Posterior samples for Laplace
DROPOUT_RATE = 0.3        # Dropout probability
```

### Laplace Wrapper (`models.py`)

```python
from laplace import Laplace

class LaplaceWrapper:
    def fit(self, train_loader):
        # Fit Laplace to last layer only (tractable)
        self.la = Laplace(
            self.model, 
            'classification',
            subset_of_weights='last_layer',
            hessian_structure='kron'
        )
        self.la.fit(train_loader)
        self.la.optimize_prior_precision(method='marglik')
```

### MC Dropout Prediction (`models.py`)

```python
def predict_with_uncertainty(self, x, n_samples=50):
    # Set eval mode to freeze BatchNorm, then re-enable only Dropout layers
    self.model.eval()
    for m in self.model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    preds = torch.stack([
        torch.softmax(self.model(x), dim=1)
        for _ in range(n_samples)
    ])

    mean_pred = preds.mean(dim=0)
    epistemic_uncertainty = preds.var(dim=0)

    return mean_pred, epistemic_uncertainty
```

---

## 📂 Output Files

After running `main.py`:

```
models/
├── deterministic_model.pt    # Trained deterministic CNN
├── mc_dropout_model.pt       # Trained MC Dropout CNN
└── laplace_model.pt          # Laplace approximation

results/
├── final_results.json        # All metrics
├── training_histories.json   # Training curves
└── uncertainty_report.txt    # Clinical analysis

figures/
├── training_curves.pdf       # Loss and AUC over epochs
├── reliability_diagram.pdf   # Calibration visualization
├── roc_curves.pdf           # ROC comparison
├── uncertainty_histogram.pdf # Uncertainty distributions
└── high_uncertainty_samples.pdf  # Ambiguous cases
```

---

## 📝 Report Sections

The generated figures and metrics support these report sections:

1. **Introduction**: Clinical motivation for uncertainty quantification
2. **Methods**: 
   - Deterministic CNN architecture
   - Laplace approximation theory
   - MC Dropout interpretation
3. **Experiments**:
   - Dataset description (70/15/15 split)
   - Training procedure (Adam, weight decay as prior)
   - Evaluation protocol
4. **Results**:
   - Classification performance comparison
   - Calibration analysis (reliability diagrams)
   - Uncertainty analysis (high-uncertainty cases)
5. **Discussion**:
   - Laplace vs MC Dropout trade-offs
   - Clinical applicability
   - Limitations

---

## ⚠️ Known Limitations

1. **Last-layer Laplace**: We only apply Laplace to the final layer for tractability. Full-network Laplace would be more principled but O(P³) in parameters.

2. **Prior sensitivity**: The weight decay (= prior precision) significantly affects uncertainty estimates. We use marginal likelihood optimization to mitigate this.

3. **Out-of-distribution detection**: These methods are designed for in-distribution uncertainty. For OOD detection, consider adding input preprocessing or dedicated OOD methods.

---

## 📚 References

- Daxberger et al. (2021). "Laplace Redux: Effortless Bayesian Deep Learning"
- Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation"
- Lakshminarayanan et al. (2017). "Simple and Scalable Predictive Uncertainty"
- Guo et al. (2017). "On Calibration of Modern Neural Networks"

---

## 👥 Authors

Elena Ardura
Victoria García

*Probabilistic AI - Master's Program, April 2026*