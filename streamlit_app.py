import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

ART_DIR = Path("webapp_artifacts")
MODEL_PATH = ART_DIR / "best_model.joblib"
LE_PATH    = ART_DIR / "label_encoder.joblib"
SCHEMA_PATH= ART_DIR / "schema.json"

# ---- Load artifacts ----
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)         # sklearn Pipeline(preprocess, selector, clf)
    le    = joblib.load(LE_PATH)            # sklearn LabelEncoder for target names
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return model, le, schema

best_model, le, schema = load_artifacts()

st.set_page_config(page_title="Farmona", page_icon="🌱", layout="centered")
st.title("🌱 Farmona")
st.write("Enter feature values (or pick a subset) and press **Predict**.")

# ---- Feature selection ----
all_features = list(schema.keys())
selected = st.multiselect("Features to provide explicitly",
                          options=all_features,
                          default=all_features[: min(6, len(all_features))])

# ---- Dynamic inputs: build a dict of values (selected from user, others from defaults) ----
values = {}
cols = st.columns(2) if len(selected) > 1 else [st]
for i, feat in enumerate(selected):
    spec = schema[feat]
    with cols[i % len(cols)]:
        if spec["type"] == "numeric":
            lo, hi = spec.get("range", [None, None])
            default = spec.get("default", 0.0)
            values[feat] = st.number_input(feat, value=float(default), step=0.1,
                                           help=f"Default≈{default:.3f}" + (f" • typical range [{lo:.2f}, {hi:.2f}]" if lo is not None else ""))
        else:
            choices = spec.get("choices") or []
            default = spec.get("default", choices[0] if choices else "")
            if choices:
                values[feat] = st.selectbox(feat, choices, index=max(0, choices.index(default)) if default in choices else 0)
            else:
                values[feat] = st.text_input(feat, value=str(default))

# Fill remaining features with defaults
for feat in all_features:
    if feat not in values:
        spec = schema[feat]
        values[feat] = spec.get("default", 0.0 if spec["type"] == "numeric" else "")

# ---- Predict button ----
if st.button("🔮 Predict"):
    # Build single-row DataFrame in the original column order
    X_input = pd.DataFrame([{k: values[k] for k in all_features}])

    # Coerce types: numerics -> float, cats -> string
    for f, spec in schema.items():
        if spec["type"] == "numeric":
            X_input[f] = pd.to_numeric(X_input[f], errors="coerce")
        else:
            X_input[f] = X_input[f].astype(str)

    try:
        y_pred = best_model.predict(X_input)[0]
        label  = le.inverse_transform([y_pred])[0]
        st.success(f"**Prediction:** {label}")

        # Optional: show top probabilities if supported
        try:
            proba = best_model.predict_proba(X_input)[0]
            class_idx = getattr(best_model.named_steps.get("clf", None), "classes_", None)
            if class_idx is not None and len(proba) == len(class_idx):
                # Map encoded -> label names
                names = le.inverse_transform(class_idx)
                top = np.argsort(-proba)[:5]
                st.subheader("Top probabilities")
                for i in top:
                    st.write(f"- {names[i]}: {proba[i]:.3f}")
        except Exception:
            pass

        with st.expander("Show input row the model received"):
            st.dataframe(X_input)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Model: sklearn Pipeline (preprocess → selector → classifier)")
