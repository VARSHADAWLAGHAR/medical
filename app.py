import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re
import requests
from streamlit_lottie import st_lottie

st.set_page_config(page_title="AI Prescription Verification", layout="centered")

# LOTTIE
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_med = load_lottie("https://assets1.lottiefiles.com/packages/lf20_q5pk6p1k.json")

# HEADER
st.markdown(
    """
    <h1 style='text-align:center; color:#6C63FF;'>💊 AI Prescription Verification</h1>
    <p style='text-align:center; font-size:18px'>
        Upload a handwritten prescription. The AI extracts medication and checks safety.
    </p>
    """,
    unsafe_allow_html=True
)

if lottie_med:
    st_lottie(lottie_med, height=200)

# LOAD MODEL
@st.cache_resource
def load_ocr_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return processor, model

processor, model = load_ocr_model()

# DATABASE
SAFE_DOSES = {
    "paracetamol": {"max_mg": 4000, "per_dose": 1000},
    "dolo": {"max_mg": 4000, "per_dose": 1000},
    "ibuprofen": {"max_mg": 2400, "per_dose": 400},
    "azithromycin": {"max_mg": 500, "per_dose": 500},
    "cetirizine": {"max_mg": 10, "per_dose": 10},
    "pantoprazole": {"max_mg": 40, "per_dose": 40},
}

def extract_medicines(text):
    return {med: SAFE_DOSES[med] for med in SAFE_DOSES if med.lower() in text.lower()}

def extract_dose(text):
    doses = re.findall(r'(\d+)\s*mg', text.lower())
    return [int(d) for d in doses] if doses else None

def check_safety(meds, doses):
    results = []
    for med, limits in meds.items():
        if not doses:
            results.append(f"⚠ No dose detected for **{med}**.")
            continue
        for d in doses:
            if d > limits["per_dose"]:
                results.append(f"❌ Unsafe dose for {med}: {d}mg (limit {limits['per_dose']}mg)")
            else:
                results.append(f"✅ Safe dose for {med}: {d}mg")
    return results

def run_ocr(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    ids = model.generate(pixel_values)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

# MAIN UI
st.subheader("📤 Upload Prescription Image")
uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)

    st.image(image, caption="Uploaded Prescription", use_column_width=True)
    st.info("🔍 Extracting text...")

    text = run_ocr(image)

    st.subheader("📄 Extracted Text")
    st.code(text)

    meds = extract_medicines(text)
    doses = extract_dose(text)
    results = check_safety(meds, doses)

    st.subheader("🩺 Analysis Result")
    for r in results:
        st.write(r)

    st.success("✔ Completed!")
