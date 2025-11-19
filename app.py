import streamlit as st
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import re

st.set_page_config(page_title="AI Prescription Verification", layout="centered")
st.title("💊 AI Medical Prescription Verification")
st.write("Upload a handwritten prescription. This AI extracts text, detects medicines, and checks dose safety.")

# -----------------------------
# LOAD LIGHTWEIGHT STRONG OCR MODEL
# -----------------------------
@st.cache_resource
def load_ocr_model():
    processor = AutoProcessor.from_pretrained("holoviz/vila-ocr")
    model = AutoModelForVision2Seq.from_pretrained("holoviz/vila-ocr")
    return processor, model

processor, model = load_ocr_model()

# -----------------------------
# SAFE DOSE DATABASE
# -----------------------------
SAFE_DOSES = {
    "paracetamol": {"max_mg": 4000, "per_dose": 1000},
    "dolo": {"max_mg": 4000, "per_dose": 1000},
    "ibuprofen": {"max_mg": 2400, "per_dose": 400},
    "azithromycin": {"max_mg": 500, "per_dose": 500},
    "montair": {"max_mg": 10, "per_dose": 10},
    "cetirizine": {"max_mg": 10, "per_dose": 10},
    "pantoprazole": {"max_mg": 40, "per_dose": 40},
}

def extract_medicines(text):
    meds_found = {}
    for med in SAFE_DOSES:
        if med.lower() in text.lower():
            meds_found[med] = SAFE_DOSES[med]
    return meds_found

def extract_dose(text):
    found = re.findall(r'(\d+)\s*mg', text.lower())
    return [int(x) for x in found] if found else None

def check_safety(meds_found, doses):
    out = []
    if not meds_found:
        return ["⚠ No known medicine detected."]
    for med, limits in meds_found.items():
        if not doses:
            out.append(f"⚠ Dose for **{med}** not detected.")
        else:
            for d in doses:
                if d > limits["per_dose"]:
                    out.append(f"❌ **Unsafe for {med}**: {d}mg (limit {limits['per_dose']}mg)")
                else:
                    out.append(f"✅ Safe dose: {med} ({d}mg)")
    return out

# -----------------------------
# OCR FUNCTION
# -----------------------------
def run_ocr(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

# -----------------------------
# STREAMLIT UI
# -----------------------------
uploaded = st.file_uploader("Upload prescription", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="📸 Uploaded Prescription", use_column_width=True)

    st.write("🔍 Extracting text...")
    extracted_text = run_ocr(image)

    st.subheader("📄 Extracted Text")
    st.code(extracted_text)

    meds_found = extract_medicines(extracted_text)
    doses = extract_dose(extracted_text)
    results = check_safety(meds_found, doses)

    st.subheader("🩺 Validation Result")
    for r in results:
        st.write(r)

    st.success("✔ Analysis Completed")
