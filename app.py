"""Streamlit demo for the fixed Polymind reference models."""

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.polymind_predict import predict_polymind

st.set_page_config(
    page_title="Polymind",
    page_icon="🧪",
    layout="wide",
)

PROPERTIES = [
    ("Tg", "Glass transition temperature", "°C", 2),
    ("FFV", "Fractional free volume", "dimensionless", 4),
    ("Tc", "Thermal conductivity", "W/(m·K)", 4),
    ("Density", "Density", "g/cm³", 4),
    ("Rg", "Radius of gyration", "Å", 2),
]

st.title("🧪 Polymind")
st.write("Explore five polymer properties from a repeat-unit SMILES.")
st.caption("Reference models · Version 1 · Mohamed Elbadry & Suhmi Hwang")

with st.form("prediction_form"):
    smiles_input = st.text_input(
        "Polymer repeat-unit SMILES",
        value="*CC*",
        help="Use * to represent polymer attachment points.",
    )
    submitted = st.form_submit_button("Predict properties", type="primary")

if submitted:
    st.session_state.pop("prediction_result", None)
    smiles = smiles_input.strip()
    mol = Chem.MolFromSmiles(smiles) if smiles else None

    if mol is None or mol.GetNumAtoms() == 0:
        st.error("Please enter a valid, non-empty SMILES.")
        st.stop()

    try:
        with st.spinner("Calculating polymer properties..."):
            predictions = predict_polymind([smiles])
    except ValueError as error:
        st.error(f"Prediction could not be completed: {error}")
        st.stop()
    except FileNotFoundError:
        st.error("The reference model bundle is missing from models/polymind_v1.")
        st.stop()
    except Exception as error:
        st.error("Prediction failed. See the technical details below.")
        with st.expander("Technical details"):
            st.exception(error)
        st.stop()

    st.session_state["prediction_result"] = (smiles, predictions)

if "prediction_result" in st.session_state:
    result_smiles, predictions = st.session_state["prediction_result"]
    result_mol = Chem.MolFromSmiles(result_smiles)

    st.divider()
    st.subheader("Polymer structure")
    st.code(result_smiles, language=None)
    st.image(
        Draw.MolToImage(result_mol, size=(700, 260)),
        caption="Represented repeat unit; * marks attachment points.",
    )

    st.subheader("Predicted properties")
    columns = st.columns(5)

    for column, (target, name, unit, decimals) in zip(columns, PROPERTIES):
        value = float(predictions.iloc[0][target])
        column.metric(target, f"{value:.{decimals}f}")
        column.caption(f"{name} · {unit}")

    export = pd.DataFrame({"SMILES": [result_smiles]})
    for target, _, unit, _ in PROPERTIES:
        export[f"{target} ({unit})"] = predictions[target].to_numpy()

    st.download_button(
        "Download predictions as CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="polymind_predictions.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Predictions are model estimates, not experimental measurements. "
    "Performance on unfamiliar polymer families has not been established."
)

with st.expander("About this reference version"):
    st.write(
        "Five separate models predict the five properties. "
        "Each model was fitted only on observed labels in the shared "
        "training partition."
    )
    st.write(
        "This version uses Morgan fingerprints and/or 21 descriptors. "
        "Additional descriptors, Uni-Mol features and external Tg data "
        "are being investigated separately."
    )
