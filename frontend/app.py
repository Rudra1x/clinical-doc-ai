import streamlit as st
import requests

API_URL = "http://127.0.0.1:8001/infer"

st.set_page_config(
    page_title="Clinical Document Intelligence",
    layout="wide",
)

st.title(" AI-Powered Clinical Document Intelligence")

st.markdown(
    """
This tool demonstrates:
- Automated ICD-10 coding
- Clinical entity extraction (NER)
- Explainable medical NLP
"""
)

# Input
text = st.text_area(
    "Enter clinical document text:",
    height=200,
    placeholder="Patient admitted with chest pain and hypertension.",
)

explain = st.checkbox("Enable explainability (slower)")

if st.button("Run Analysis") and text.strip():
    payload = {
        "document_text": text,
        "explain": explain,
    }

    with st.spinner("Analyzing document..."):
        response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        st.error("API error")
    else:
        result = response.json()

        
        # ICD Predictions
        
        st.subheader(" ICD-10 Predictions")

        for icd in result["icd_predictions"]:
            st.markdown(
                f"**{icd['code']}** — {icd['description']} "
                f"({icd['confidence'] * 100:.1f}%)"
            )
            st.progress(icd["confidence"])

        
        # Named Entities
        
        st.subheader(" Extracted Clinical Entities")

        if result["ner_entities"]:
            for label, values in result["ner_entities"].items():
                st.markdown(f"**{label}**: {', '.join(values)}")
        else:
            st.write("No entities detected.")

        
        # Explainability
        
        if explain and result["explanations"]:
            st.subheader("🔍 Explainability (Top ICD Code)")

            for code, tokens in result["explanations"].items():
                st.markdown(f"**ICD Code: {code}**")
                for tok in tokens:
                    st.write(
                        f"{tok['token']} → {tok['importance']:.4f}"
                    )