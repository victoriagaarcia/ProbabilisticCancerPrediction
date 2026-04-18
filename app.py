# Latest version (?)
"""
=============================================================================
STREAMLIT DEMO: BAYESIAN NEURAL NETWORKS FOR CANCER DETECTION
=============================================================================
Interactive demonstration of uncertainty quantification in medical imaging.

Run with: streamlit run app.py

Features:
- Upload histopathology images for prediction
- Compare Deterministic vs. Laplace vs. MC Dropout
- Visualize predictive uncertainty
- Explore pre-loaded example images
=============================================================================
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import io
import json
import pickle

# Import project modules
from config import (
    DEVICE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    MODELS_DIR, RESULTS_DIR, BATCH_SIZE
)
from models import load_model, LaplaceWrapper #, create_deterministic_model
# from models import load_model, load_laplace_model
from data import get_dataloaders
from data import get_transforms

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="BNN Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .uncertainty-low { color: #28a745; }
    .uncertainty-medium { color: #ffc107; }
    .uncertainty-high { color: #dc3545; }
    .prediction-positive { color: #dc3545; font-weight: bold; }
    .prediction-negative { color: #28a745; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Load training data for Laplace fitting
# =============================================================================
@st.cache_resource
def load_train_loader():
    """
    Load the exact train split used during training from saved CSV files,
    then rebuild the train dataloader for Laplace fitting.
    """
    splits_dir = RESULTS_DIR / "data_splits"
    train_csv = splits_dir / "train_split.csv"
    val_csv = splits_dir / "val_split.csv"
    test_csv = splits_dir / "test_split.csv"
    metadata_json = splits_dir / "split_metadata.json"

    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Saved split files not found in {splits_dir}. "
            "Run main.py after adding save_data_splits(...) first."
        )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Optional metadata check
    if metadata_json.exists():
        with open(metadata_json, "r") as f:
            metadata = json.load(f)
        expected_seed = metadata.get("seed", None)
        st.sidebar.caption(f"Loaded saved data splits (seed={expected_seed})")

    train_loader, _, _ = get_dataloaders(
        train_df, val_df, test_df,
        batch_size=BATCH_SIZE
    )

    return train_loader

# =============================================================================
# Model Loading (Cached)
# =============================================================================
@st.cache_resource
def load_models():
    """Load all trained models (cached for performance)."""
    models = {}
    
    # Check if models exist
    det_path = MODELS_DIR / "deterministic_model.pt"
    mc_path = MODELS_DIR / "mc_dropout_model.pt"
    # lap_path = MODELS_DIR / "laplace_model.pt"
    
    if not det_path.exists():
        return None
    
    # Load deterministic model
    models['deterministic'] = load_model(det_path, model_type='deterministic')
    models['deterministic'].eval()
    
    # Load MC Dropout model
    if mc_path.exists():
        models['mc_dropout'] = load_model(mc_path, model_type='mc_dropout')
    
    # Create Laplace wrapper (will need to be fitted)
    # models['laplace'] = LaplaceWrapper(models['deterministic'])

    # Load Laplace model if exists
    # if lap_path.exists():
    #     models['laplace'] = load_laplace_model(lap_path)

    # Load .pkl Laplace model, otherwise fit it
    laplace_pkl_path = MODELS_DIR / "laplace_model.pkl"
    try:
        laplace_model = LaplaceWrapper(models['deterministic'])
        if laplace_pkl_path.exists():
            with open(laplace_pkl_path, "rb") as f:
                laplace_model.la = pickle.load(f)
            laplace_model.fitted = True
            st.sidebar.success("✓ Laplace loaded from disk")
        else:
            st.sidebar.warning("Laplace pickle not found — fitting now (slow)...")
            train_loader = load_train_loader()
            laplace_model.fit(train_loader)
        models['laplace'] = laplace_model
    except Exception as e:
        st.warning(f"Laplace model could not be initialized: {e}")
    return models


# =============================================================================
# Prediction Functions
# =============================================================================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess an image for model input."""
    # Resize to expected size
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to tensor
    img_array = np.array(image) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)
    
    # Normalize with ImageNet stats
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor.unsqueeze(0).to(DEVICE)


def predict_deterministic(model, image_tensor):
    """Get prediction from deterministic model."""
    model.eval()
    with torch.no_grad():
        # logits = model(image_tensor)
        # prob = torch.sigmoid(logits).item()
        # prob = torch.softmax(logits, dim=1)[0, 1].item()  # NUM_CLASSES=2, get cancer class probability
        probs = model.predict_proba(image_tensor)
        prob = probs[0, 1].item()  # Get probability of cancer class
    return prob, 0.0  # No uncertainty for deterministic


def predict_mc_dropout(model, image_tensor, n_samples=50):
    """Get prediction with MC Dropout uncertainty."""
    # model.train()  # Enable dropout
    # probs = []
    # with torch.no_grad():
    #     for _ in range(n_samples):
    #         logits = model(image_tensor)
    #         prob = torch.sigmoid(logits)
    #         probs.append(prob.item())
    # probs = np.array(probs)
    # mean_prob = probs.mean()
    # uncertainty = probs.std()
    # return mean_prob, uncertainty
    model.train()  # Enable dropout at test time
    mean_probs, epistemic, aleatoric, total_unc, _ = model.predict_with_uncertainty(
        image_tensor, n_samples=n_samples
    )
    prob = mean_probs[0, 1].item()
    epis_unc = epistemic[0].item()
    ale_unc = aleatoric[0].item()
    return prob, epis_unc, ale_unc



def predict_laplace(laplace_model, image_tensor, n_samples=100):
    """Get prediction with Laplace approximation uncertainty."""
    mean_probs, epistemic, aleatoric, total_unc, _ = laplace_model.predict_with_uncertainty(
        image_tensor, n_samples=n_samples
    )
    # return mean_prob.item(), uncertainty.item()
    prob = mean_probs[0, 1].item()
    epis_unc = epistemic[0].item()
    ale_unc = aleatoric[0].item()
    return prob, epis_unc, ale_unc


# =============================================================================
# Visualization Functions
# =============================================================================
def create_probability_gauge(prob, title="Probability"):
    """Create a gauge chart for probability visualization."""
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Create horizontal bar
    ax.barh([0], [prob], color='#dc3545' if prob > 0.5 else '#28a745', height=0.5)
    ax.barh([0], [1-prob], left=[prob], color='#e9ecef', height=0.5)
    
    # Add threshold line
    ax.axvline(x=0.5, color='#333', linestyle='--', linewidth=2)
    
    # Styling
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add probability text
    ax.text(prob, 0, f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_uncertainty_visualization(results):
    """Create comparison visualization of model uncertainties."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for ax, (model_name, result), color in zip(axes, results.items(), colors):
        prob = result['probability']
        uncertainty = result['uncertainty']

        ax.bar([0], [prob], color=color, alpha=0.7, width=0.5)
        if uncertainty > 0:
            ax.errorbar([0], [prob], yerr=[uncertainty], color='black',
                        capsize=10, capthick=2, linewidth=2)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel('Cancer Probability')
        ax.set_title(f'{model_name}\nP={prob:.3f} ± {uncertainty:.3f}', fontsize=12)

    plt.tight_layout()
    return fig

def uncertainty_color(uncertainty):
    """Get color class based on uncertainty level."""
    if uncertainty < 0.1:
        return "uncertainty-low"
    elif uncertainty < 0.2:
        return "uncertainty-medium"
    else:
        return "uncertainty-high"


def uncertainty_interpretation(uncertainty):
    """Get human-readable interpretation of uncertainty."""
    if uncertainty < 0.05:
        return "Very confident prediction"
    elif uncertainty < 0.1:
        return "Confident prediction"
    elif uncertainty < 0.15:
        return "Moderate uncertainty - review recommended"
    elif uncertainty < 0.2:
        return "High uncertainty - expert review needed"
    else:
        return "Very high uncertainty - requires careful examination"


# =============================================================================
# Main Application
# =============================================================================
def main():
    # Header
    st.markdown('<p class="main-header">🔬 Bayesian Neural Networks for Cancer Detection</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Uncertainty-Aware Histopathologic Image Classification</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    mc_samples = st.sidebar.slider(
        "MC Dropout Samples",
        min_value=10, max_value=100, value=50, step=10,
        help="Number of forward passes for MC Dropout"
    )
    
    laplace_samples = st.sidebar.slider(
        "Laplace Samples",
        min_value=50, max_value=200, value=100, step=25,
        help="Number of posterior samples for Laplace"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📚 About")
    st.sidebar.markdown("""
    This demo compares three approaches to neural network predictions:
    
    **Deterministic CNN**: Standard point estimate without uncertainty.
    
    **Laplace Approximation**: Gaussian approximation to the posterior 
    distribution of weights, centered at the MAP estimate.
    
    **MC Dropout**: Uses dropout at test time to approximate Bayesian inference.
    """)
    
    # Load models
    models = load_models()
    
    if models is None:
        st.error("""
        ⚠️ **Trained models not found!**
        
        Please train the models first by running:
        ```bash
        python main.py
        ```
        
        This will train the deterministic, Laplace, and MC Dropout models.
        """)
        return
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a histopathology image",
            type=['png', 'jpg', 'jpeg', 'tif'],
            help="Upload a 96x96 histopathology patch"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess
            image_tensor = preprocess_image(image)
            
            # Run predictions button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Running inference..."):
                    from metrics import triage_decision
                    results = {}

                    # Deterministic — sin incertidumbre
                    prob_det, _ = predict_deterministic(models['deterministic'], image_tensor)
                    results['Deterministic'] = {
                        'probability': prob_det,
                        'epistemic': 0.0,
                        'aleatoric': 0.0,
                    }

                    # Laplace
                    if 'laplace' in models:
                        prob_lap, epi_lap, ale_lap = predict_laplace(
                            models['laplace'], image_tensor, n_samples=laplace_samples
                        )
                        results['Laplace'] = {
                            'probability': prob_lap,
                            'epistemic': epi_lap,
                            'aleatoric': ale_lap,
                            'triage': triage_decision(prob_lap, epi_lap),
                        }

                    # MC Dropout
                    if 'mc_dropout' in models:
                        prob_mc, epi_mc, ale_mc = predict_mc_dropout(
                            models['mc_dropout'], image_tensor, n_samples=mc_samples
                        )
                        results['MC Dropout'] = {
                            'probability': prob_mc,
                            'epistemic': epi_mc,
                            'aleatoric': ale_mc,
                            'triage': triage_decision(prob_mc, epi_mc),
                        }

                    st.session_state['results'] = results
        else:
            st.info("👆 Upload an image to get started")
    
    with col2:
        st.header("📊 Results")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            
            # Show comparison visualization
            fig = create_uncertainty_visualization(results)
            st.pyplot(fig)
            plt.close()
            
            # Detailed results
            st.markdown("### Detailed Predictions")
            
            cols = st.columns(len(results))
            
            for col, (model_name, result) in zip(cols, results.items()):
                with col:
                    prob = result['probability']
                    epi  = result.get('epistemic', 0.0)
                    ale  = result.get('aleatoric', 0.0)

                    st.markdown(f"**{model_name}**")

                    prediction = "Cancer" if prob > 0.5 else "Normal"
                    pred_class = "prediction-positive" if prob > 0.5 else "prediction-negative"
                    st.markdown(f"Prediction: <span class='{pred_class}'>{prediction}</span>",
                                unsafe_allow_html=True)

                    st.metric("Probability", f"{prob:.1%}")

                    if epi > 0:
                        st.markdown(f"**Epistemic uncertainty:** {epi:.4f}")
                        st.markdown(f"**Aleatoric uncertainty:** {ale:.4f}")

                    # Mostrar decisión de triaje si existe
                    if 'triage' in result:
                        t = result['triage']
                        st.markdown(f"**Clinical decision:**")
                        st.markdown(t['action'])
            
            # Clinical interpretation
            st.markdown("### 🏥 Clinical Interpretation")
            
            # Use Laplace uncertainty for interpretation
            # laplace_unc = results['Laplace']['uncertainty']
            # ref_prob = results['Laplace']['probability']
            # interpretation = uncertainty_interpretation(laplace_unc)

            # Use best available model for interpretation (prefer Laplace, then MC Dropout)
            if 'Laplace' in results:
                ref_model = 'Laplace'
            elif 'MC Dropout' in results:
                ref_model = 'MC Dropout'
            else:
                ref_model = 'Deterministic'

            ref_unc = results[ref_model].get('epistemic', 0.0) + results[ref_model].get('aleatoric', 0.0)
            ref_prob = results[ref_model]['probability']

            interpretation = uncertainty_interpretation(ref_unc)
            
            if ref_prob > 0.5:
                st.warning(f"""
                **Potential Malignancy Detected** (P = {ref_prob:.1%})
                
                {interpretation}
                
                {"⚠️ High uncertainty suggests this case should be reviewed by a pathologist." 
                 if ref_unc > 0.15 else ""}
                """)
            else:
                st.success(f"""
                **No Malignancy Detected** (P = {ref_prob:.1%})
                
                {interpretation}
                
                {"ℹ️ Consider additional review due to uncertainty in prediction." 
                 if ref_unc > 0.15 else ""}
                """)
        else:
            st.info("Upload an image and click 'Analyze' to see predictions")
    
    # Educational section
    st.markdown("---")
    st.header("📖 Understanding Uncertainty")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Epistemic Uncertainty
        *"What the model doesn't know"*
        
        Arises from limited training data or out-of-distribution inputs.
        Can be reduced with more data.
        
        **Clinical relevance**: High epistemic uncertainty suggests the 
        model hasn't seen similar cases before.
        """)
    
    with col2:
        st.markdown("""
        ### Aleatoric Uncertainty
        *"Inherent noise in the data"*
        
        Arises from ambiguous tissue patterns or imaging artifacts.
        Cannot be reduced with more data.
        
        **Clinical relevance**: High aleatoric uncertainty may indicate 
        genuinely ambiguous pathology.
        """)
    
    with col3:
        st.markdown("""
        ### Why It Matters
        
        Traditional neural networks give overconfident predictions.
        Bayesian methods provide calibrated uncertainties.
        
        **Clinical application**: 
        - Prioritize high-uncertainty cases for expert review
        - Improve diagnostic workflow efficiency
        - Reduce false negatives in screening
        """)


if __name__ == "__main__":
    main()
