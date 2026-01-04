# Physics-Informed Machine Learning for Predicting Material Melting Points

## 📌 Project Overview
**Melting Point Prediction** is a machine learning project designed to estimate the melting points of materials based on their fundamental **chemical and physical properties**. Unlike standard "black box" models, this project is **Physics-Informed**, meaning it explicitly leverages domain knowledge from material science.

The model integrates periodic trends—such as **electronegativity, atomic radius, valence electrons, and atomic mass**—to improve prediction accuracy. By understanding the underlying physics (e.g., stronger metallic bonding correlates with higher melting points), the model achieves better generalization and interpretability.

## 🚀 Key Features
- **Physics-Informed Feature Engineering**: Extracts elemental properties (e.g., atomic number, specific heat, density) using `matminer` and `pymatgen`.
- **Hybrid Ensemble Model**: Combines **XGBoost, LightGBM, and Random Forest** for robust regression performance.
- **Interactive Web Interface**: A **Streamlit**-based UI allows users to input chemical formulas (e.g., `NaCl`, `SiO2`) and get real-time predictions.
- **Visualizations**: Includes parity plots and feature importance charts to explain *why* a melting point was predicted.

## 📂 Project Structure
```
melting-point-prediction/
│
├── final_minor_folder/       # Contains the deployed model and application
│   ├── app.py                # Streamlit Web Application entry point
│   ├── ultimate_model.pkl    # Trained Hybrid Model
│   ├── final_imputer.pkl     # Data pre-processing imputer
│   ├── final_feature_names.pkl # Feature definitions
│   └── ... (Visualizations & artifacts)
│
├── main_project_minor.ipynb  # Jupyter Notebook containing the full training pipeline
│                             # (Data Loading -> Feature Engineering -> Training -> Evaluation)
│
├── dataset/                  # folder containing raw data files
│   ├── pnas.2209630119.sapp.pdf
│   ├── pnas.2209630119.sd01.xlsx
│   └── ...
│
└── README.md                 # Project Documentation
```

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`
- **Material Science**: `pymatgen`, `matminer`
- **Web UI**: `streamlit`
- **Data Manipulation**: `pandas`, `numpy`

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/melting-point-prediction.git
    cd melting-point-prediction
    ```

2.  **Install Dependencies**
    Ensure you have Python installed. Then, install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn xgboost lightgbm streamlit pymatgen matminer joblib
    ```

    *Note: `matminer` and `pymatgen` are critical for fetching elemental properties.*

## 🏃‍♂️ How to Run

### 1. Run the Web Application
To start the interactive Melting Point Predictor:
```bash
cd final_minor_folder
python -m streamlit run app.py
```
This will open the interface in your browser (usually at `http://localhost:8501`).

### 2. Retrain or Explore the Model
If you want to see how the model was trained, investigate the data analysis, or retrain it:
1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook main_project_minor.ipynb
    ```
2.  Run the cells sequentially to reproduce the feature extraction and model training steps.

## 📊 Theory & Methodology
The project builds upon the hypothesis that **macroscopic properties** (like melting point) are governed by **microscopic atomic interactions**.
- **Data Source**: We use experimentally measured melting points from validated datasets.
- **Feature Extraction**:
    - **Magpie Presets**: Uses `matminer` to calculate stats (mean, range, deviation) for properties like *Number of d-shell valence electrons*, *Covalent Radius*, etc.
    - **Composition**: Parses chemical formulas to determine stoichiometry.
- **Modeling**: A stacked regression model learns the non-linear mapping between these physical descriptors and the melting temperature ($T_m$).

## 📈 Results
- **R² Score**: High correlation between predicted and actual values (see `Final_Parity_Plot.png` in the app folder).

## 📸 Project Screenshots

### 1. Material Melting Point Predictor
<img width="100%" alt="Material Melting Point Predictor" src="https://github.com/user-attachments/assets/ec6939ad-71f8-4894-a99f-9732a387422f" />

### 2. Material Analysis & Atomic Fingerprint
<img width="100%" alt="Material Analysis" src="https://github.com/user-attachments/assets/f8d31fcf-413f-499d-94e5-c1d433f69a4a" />

### 3. Reliability Score & Model Performance
<img width="100%" alt="Reliability Score" src="https://github.com/user-attachments/assets/e67274ea-4dd4-43c6-86e5-6e4fb0e86cc3" />

### 4. Feature Importance
<img width="100%" alt="Feature Importance" src="https://github.com/user-attachments/assets/9565ada0-244b-43b2-af00-84f6c7bacc6b" />





