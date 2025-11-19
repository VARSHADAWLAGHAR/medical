import streamlit as st
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import re

st.set_page_config(page_title="AI Prescription Verification", layout="centered")

st.title("💊 AI Medical Prescription Verification")
st.write("Upload a handwritten prescription. The AI will extract text, identify medicines, and check dose safety.")

# -----------------------------
# LOAD STRONG HUGGING FACE MODEL
# -----------------------------
@st.cache_resource
def load_ocr_model():
    processor = AutoProcessor.from_pretrained("google/pix2text-large-handwriting")
    model = AutoModelForVision2Seq.from_pretrained("google/pix2text-large-handwriting")
    return processor, model

processor, model = load_ocr_model()

# -----------------------------
# COMMON MEDICINES & SAFE DOSE
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
    for med in SAFE_DOSES.keys():
        if med.lower() in text.lower():
            meds_found[med] = SAFE_DOSES[med]
    return meds_found

def extract_dose(text):
    # Find numbers like 250mg, 500mg, 1g
    dose_match = re.findall(r'(\d+)\s*mg', text.lower())
    doses = [int(d) for d in dose_match]
    return doses if doses else None

def check_safety(meds_found, doses):
    msg = []
    if not meds_found:
        return ["⚠ No known medicine detected."]
    for med, limits in meds_found.items():
        if doses:
            for d in doses:
                if d > limits["per_dose"]:
                    msg.append(f"❌ **Unsafe dose for {med}**: {d}mg (limit {limits['per_dose']}mg)")
                else:
                    msg.append(f"✅ Safe dose: {med} ({d}mg)")
        else:
            msg.append(f"⚠ Dose for **{med}** not detected.")
    return msg

# -----------------------------
# OCR FUNCTION
# -----------------------------
def run_ocr(image):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(**inputs)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return text

# -----------------------------
# STREAMLIT UI
# -----------------------------
uploaded = st.file_uploader("Upload prescription image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="📸 Uploaded Prescription", use_column_width=True)

    st.write("🔍 Extracting text from prescription... please wait 5–10 sec.")

    extracted_text = run_ocr(image)

    st.subheader("📄 Extracted Text")
    st.code(extracted_text)

    meds_found = extract_medicines(extracted_text)
    doses = extract_dose(extracted_text)
    results = check_safety(meds_found, doses)

    st.subheader("🩺 Medicine Validation Result")
    for r in results:
        st.write(r)

    st.success("✔ Analysis Completed")
