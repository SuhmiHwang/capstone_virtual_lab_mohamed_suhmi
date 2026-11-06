"""Prototype: draw a repeat unit (or type a SMILES) and predict five properties with the v2 models.

Run: uv run streamlit run app_v2.py
"""

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from streamlit_ketcher import st_ketcher

from src.polymer_input import to_repeat_unit
from src.polymind_predict_v2 import predict_polymind_v2

st.set_page_config(page_title="Polymind v2", page_icon="🧪", layout="wide")

PROPERTIES = [
    ("Tg", "Glass transition temperature", "°C", 1),
    ("FFV", "Fractional free volume", "", 3),
    ("Tc", "Thermal conductivity", "W/(m·K)", 3),
    ("Density", "Density", "g/cm³", 3),
    ("Rg", "Radius of gyration", "Å", 2),
]
EXAMPLES = {
    "Polystyrene": "*CC(*)c1ccccc1",
    "PMMA": "*CC(*)(C)C(=O)OC",
    "Nylon-6": "*CCCCCC(=O)N*",
    "Polycarbonate (bisphenol A)": "*Oc1ccc(C(C)(C)c2ccc(OC(*)=O)cc2)cc1",
    "Polyethylene": "*CC*",
}

st.title("🧪 Polymind")
st.write("Draw a polymer repeat unit, or enter its SMILES, and predict five properties.")
st.caption("Prototype · v2 models · Mohamed Elbadry & Suhmi Hwang")

start = st.selectbox("Start from an example (optional)", ["(empty)"] + list(EXAMPLES))
start_smiles = EXAMPLES.get(start, "")

draw_tab, smiles_tab = st.tabs(["✏️ Draw structure", "⌨️ Enter SMILES"])
with draw_tab:
    st.info(
        "Draw **one repeat unit** and mark the two points where the chain continues with a "
        "**\\*** atom: click an atom tool, type `*` and press Enter, or choose *Generic groups → \\**. "
        "Then click **Apply** below the editor."
    )
    # MOLFILE keeps the * attachment points; the editor's SMILES export drops them.
    start_block = Chem.MolToMolBlock(Chem.MolFromSmiles(start_smiles)) if start_smiles else ""
    drawn = st_ketcher(start_block, height=450, molecule_format="MOLFILE", key=f"ketcher_{start}")with smiles_tab:
    typed = st.text_input("Repeat-unit SMILES", value=start_smiles, key=f"smiles_{start}",
                          help="Use * for the two attachment points, e.g. *CC(*)c1ccccc1")
    use_typed = st.button("Use this SMILES")

source = typed if use_typed else drawn
if source is not None and source != st.session_state.get("last_source"):
    st.session_state["last_source"] = source
    smiles, message = to_repeat_unit(source)
    st.session_state["result"] = None
    if message:
        st.session_state["message"] = message
    else:
        st.session_state["message"] = None
        with st.spinner("Calculating descriptors and predicting..."):
            try:
                st.session_state["result"] = (smiles, predict_polymind_v2([smiles]))
            except Exception as error:  # show the reason instead of crashing the demo
                st.session_state["message"] = f"Prediction could not be completed: {error}"

if st.session_state.get("message"):
    st.warning(st.session_state["message"])

if st.session_state.get("result"):
    smiles, predictions = st.session_state["result"]
    st.divider()
    left, right = st.columns([1, 2])
    left.subheader("Repeat unit")
    left.image(Draw.MolToImage(Chem.MolFromSmiles(smiles), size=(420, 260)))
    left.code(smiles, language=None)
    right.subheader("Predicted properties")
    for column, (target, name, unit, decimals) in zip(right.columns(5), PROPERTIES):
        column.metric(target, f"{float(predictions.iloc[0][target]):.{decimals}f}")
        column.caption(f"{name}{' · ' + unit if unit else ''}")
    export = pd.DataFrame({"SMILES": [smiles]})
    for target, _, unit, _ in PROPERTIES:
        export[f"{target} ({unit})" if unit else target] = predictions[target].to_numpy()
    right.download_button("Download as CSV", export.to_csv(index=False).encode("utf-8"),
                          "polymind_prediction.csv", "text/csv")

st.divider()
st.caption("Predictions are model estimates, not experimental measurements. "
           "Accuracy is lower for polymer families that are rare in the training data.")
