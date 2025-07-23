# Libraries
import os, joblib, warnings, sys
import streamlit as st
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

# Configure the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ML functions
@st.cache_resource
def load_ml_utils(model_path: str, scaler_path: str, label_encoder_path: str):
    """Load ML models (cached to avoid reloading)"""
    try:
        # Suppress warnings during model loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            label_encoder = joblib.load(label_encoder_path)
        
        # Show info message about potential version differences
        st.info("ℹ️ Models loaded successfully.\nNote: Using models trained with a different sklearn version.")
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"❌ Error loading ML model: {str(e)}")
        return None, None, None

def validate_fighter_data(fighterA_data, fighterB_data):
    """Validate fighter data efficiently"""
    errors = []
    if not fighterA_data["name"] or not fighterB_data["name"]:
        errors.append("Please provide names for both fighters.")
    if fighterA_data["wins"] < fighterA_data["kos"]:
        errors.append("Fighter A: Wins cannot be less than KOs.")
    if fighterB_data["wins"] < fighterB_data["kos"]:
        errors.append("Fighter B: Wins cannot be less than KOs.")
    return errors

def predict_outcome(fighterA, fighterB, model, scaler, label_encoder):
    """Predict the outcome of the match using the ML model"""
    wins_diff = fighterA["wins"] - fighterB["wins"]
    losses_diff = fighterA["losses"] - fighterB["losses"]
    drawn_diff = fighterA["draws"] - fighterB["draws"]
    ko_diff = fighterA["kos"] - fighterB["kos"]

    totalA = fighterA["wins"] + fighterA["losses"] + fighterA["draws"]
    totalB = fighterB["wins"] + fighterB["losses"] + fighterB["draws"]
    total_fight_diff = totalA - totalB

    win_rate_A = fighterA["wins"] / totalA if totalA > 0 else 0
    win_rate_B = fighterB["wins"] / totalB if totalB > 0 else 0
    win_rate_diff = win_rate_A - win_rate_B

    ko_rate_A = fighterA["kos"] / totalA if totalA > 0 else 0
    ko_rate_B = fighterB["kos"] / totalB if totalB > 0 else 0
    ko_rate_diff = ko_rate_A - ko_rate_B

    features = pd.DataFrame(
        [[
            wins_diff, losses_diff, drawn_diff,
            ko_diff, total_fight_diff,
            win_rate_diff, ko_rate_diff
        ]],
        columns=[
            'wins_diff', 'losses_diff', 'drawn_diff',
            'ko_diff', 'total_fights_diff', 'win_rate_diff', 'ko_rate_diff'
        ]
    )

    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)
    return label_encoder.inverse_transform(pred)[0]

# Streamlit page configuration
st.set_page_config(
    page_title="Boxing Match Prediction",
    page_icon="🥊",
    layout="wide"
)

st.title("🥊 Boxing Match Prediction")
st.caption("🚀 Predict the outcome of a boxing match based on fighters' statistics.")

# Fighter information input
with st.expander("Fighters Information", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fighter A")
        fighterA = {
            "name": st.text_input("Name", key="fighterA_name"),
            "wins": st.number_input("Wins", min_value=0, key="fighterA_wins"),
            "draws": st.number_input("Draws", min_value=0, key="fighterA_draws"),
            "losses": st.number_input("Losses", min_value=0, key="fighterA_losses"),
            "kos": st.number_input("KOs", min_value=0, key="fighterA_kos")
        }

    with col2:
        st.subheader("Fighter B")
        fighterB = {
            "name": st.text_input("Name", key="fighterB_name"),
            "wins": st.number_input("Wins", min_value=0, key="fighterB_wins"),
            "draws": st.number_input("Draws", min_value=0, key="fighterB_draws"),
            "losses": st.number_input("Losses", min_value=0, key="fighterB_losses"),
            "kos": st.number_input("KOs", min_value=0, key="fighterB_kos")
        }

# Load ML model
model, scaler, label_encoder = load_ml_utils(
    'model/ML/boxing_model.pkl',
    'model/ML/scaler.pkl',
    'model/ML/label_encoder.pkl'
)

# Predict button
if st.button("Predict Outcome"):
    validation_errors = validate_fighter_data(fighterA, fighterB)
    if validation_errors:
        for error in validation_errors:
            st.warning(f"⚠️ {error}")
    elif model and scaler and label_encoder:
        try:
            result = predict_outcome(fighterA, fighterB, model, scaler, label_encoder)
            if result == "draw":
                st.info("🤝 The match is predicted to end in a draw.")
            elif result == "win_A":
                st.success(f"🥊 The match is predicted to be won by Fighter A :  **{fighterA['name']}**")
            elif result == "win_B":
                st.success(f"🥊 The match is predicted to be won by Fighter B :  **{fighterB['name']}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}. This might be due to model version incompatibility.")
    else:
        st.error("❌ Error loading the ML model. Please check the model files.")
