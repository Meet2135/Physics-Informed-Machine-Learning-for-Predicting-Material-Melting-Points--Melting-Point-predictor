import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# --- Checks ---
try:
    from pymatgen.core.composition import Composition
    from matminer.featurizers.composition import ElementProperty
except ImportError:
    st.error("❌ Critical Libraries Missing! Please run: pip install matminer pymatgen")
    st.stop()

st.set_page_config(page_title="Melting Point Predictor", page_icon="🔥", layout="wide")

# --- Configuration ---
BASE_DIR = "."
MODEL_PATH = os.path.join(BASE_DIR, "ultimate_model.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "final_imputer.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "final_feature_names.pkl")
PLOT_PATH = os.path.join(BASE_DIR, "Final_Parity_Plot_R2_Only.png")

# --- Load Model ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        feats = joblib.load(FEATURES_PATH)
        return model, imputer, feats
    except FileNotFoundError:
        return None, None, None

model, imputer, feature_names = load_artifacts()

# --- Logic ---
def get_prediction(formula, model, imputer, feature_names):
    # 1. Parse
    try:
        f_clean = str(formula).replace(" ", "")
        comp = Composition(f_clean)
        if len(comp.elements) == 0: return None, None, "Invalid Formula"
    except:
        return None, None, "Parsing Error"

    # 2. Featurize (Magpie)
    try:
        ep_feat = ElementProperty.from_preset("magpie")
        X_mag = ep_feat.featurize_many([comp], ignore_errors=True, pbar=False)
        X_mag = pd.DataFrame(X_mag, columns=ep_feat.feature_labels())
    except Exception as e:
        return None, None, f"Featurization Error: {str(e)}"

    # 3. Simple Features
    X_simp = pd.DataFrame()
    for col in feature_names:
        if col.startswith("has_"):
            elem = col.replace("has_", "")
            X_simp[col] = [1 if elem in [e.symbol for e in comp.elements] else 0]
        elif col == "_num_distinct_elements":
            X_simp[col] = [len(comp.elements)]
        elif col == "_total_atoms":
            X_simp[col] = [comp.num_atoms]

    # 4. Combine & Align
    X_combined = pd.concat([X_mag, X_simp], axis=1)
    X_final = pd.DataFrame(0.0, index=[0], columns=feature_names)
    common_cols = X_combined.columns.intersection(feature_names)
    X_final.update(X_combined[common_cols])

    # 5. Predict
    X_imp = pd.DataFrame(imputer.transform(X_final), columns=feature_names)
    pred = model.predict(X_imp)[0]
    return pred, f_clean, None

# --- UI ---
st.title("🔥 Material Melting Point Predictor")
st.markdown("**Model:** Stacked Ensemble (XGBoost + LightGBM + Random Forest)")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("1. Enter Material")
    formula_input = st.text_input("Chemical Formula", value="NaCl", placeholder="e.g. NaCl, SiO2")
    
    if st.button("Predict Melting Point", type="primary"):
        if model is None:
            st.error(f"❌ Model files not found in {os.getcwd()}")
            st.info("Download 'ultimate_model.pkl', 'final_imputer.pkl', 'final_feature_names.pkl'")
        else:
            with st.spinner("Calculating Physics Descriptors..."):
                pred_val, clean_name, error_msg = get_prediction(formula_input, model, imputer, feature_names)
            
            if error_msg:
                st.error(f"⚠️ Error: {error_msg}")
            elif pred_val is not None:
                # Logic: If < 0, it means features failed silently.
                if pred_val < 0:
                     st.error(f"⚠️ Prediction Failed (Result: {pred_val:.2f} K).")
                     st.warning("This means your computer failed to calculate the physics properties (Magpie). Try running: 'pip install --upgrade matminer'")
                else:
                    st.success("Analysis Complete!")
                    st.metric(label=f"Predicted Tm ({clean_name})", value=f"{pred_val:.2f} K")
                    st.progress(min(pred_val / 4500, 1.0), text="Temperature Scale (0 - 4500 K)")

with col2:
    st.subheader("2. Model Accuracy")
    if os.path.exists(PLOT_PATH):
        # FIXED: Use 'use_container_width' to fix the warning
        st.image(PLOT_PATH, caption="Test Set Performance", use_container_width=True)
    else:
        st.warning(f"Plot not found at {PLOT_PATH}")