# app.py
import streamlit as st
from PIL import Image
import io
import re
import json

# transformers imports
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# image processing
import cv2
import numpy as np

# OCR model names (small by default to reduce memory)
TROCR_MODEL = "microsoft/trocr-small-handwritten"   # good balance for handwriting
NER_MODEL = "d4data/biomedical-ner-all"             # medical NER (may download on first run)

st.set_page_config(page_title="Prescription OCR & Verification", page_icon="🩺", layout="centered")
st.title("🩺 Prescription OCR & Simple Verification")

# -------------------------
# Utility: Preprocess image
# -------------------------
def preprocess_image_bytes(img_bytes):
    """Take image bytes, return preprocessed PIL image (RGB) and a debug cv2 image."""
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image bytes.")
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # upscale to help OCR
    h, w = gray.shape
    scale = 2.0 if max(h,w) < 2000 else 1.2
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # denoise / blur small
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    # adaptive thresholding to improve contrast
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    # convert to PIL RGB
    pil = Image.fromarray(cv2.cvtColor(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))
    return pil

# -------------------------
# Load models (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_ocr_model():
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
    return processor, model

@st.cache_resource(show_spinner=False)
def load_ner_pipeline():
    try:
        tok = AutoTokenizer.from_pretrained(NER_MODEL)
        ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
        ner_pipe = pipeline("ner", model=ner_model, tokenizer=tok, aggregation_strategy="simple")
        return ner_pipe
    except Exception as e:
        st.warning("Could not load medical NER model (will skip NER).")
        return None

processor, trocr_model = load_ocr_model()
ner_pipeline = load_ner_pipeline()

# -------------------------
# OCR helpers
# -------------------------
def ocr_with_trocr(pil_image):
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values, max_length=512)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

def extract_dosages(text):
    """Find dosage-like tokens (e.g., 500 mg, 1 tablet, 5 ml)"""
    dosages = []
    for m in re.finditer(r'(\d{1,4}\s*(?:mg|g|ml|mcg|µg|tablet|tab|capsule|caps|tds|bd|od))', text, flags=re.IGNORECASE):
        dosages.append(m.group(0).strip())
    # also find numeric + mg pattern
    for m in re.finditer(r'(\d{2,4})\s*mg', text, flags=re.IGNORECASE):
        dosages.append(m.group(0).strip())
    return list(dict.fromkeys(dosages))  # dedupe while preserving order

def extract_dates(text):
    # very simple date patterns
    dates = re.findall(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', text)
    return dates

def simple_rules_check(meds_and_doses):
    """A tiny rules engine. Replace or expand with real medical checks."""
    issues = []
    # example: paracetamol single dose > 1000 mg flagged
    for item in meds_and_doses:
        med = item.get("medicine", "").lower()
        dose_text = item.get("dose", "")
        num = None
        num_match = re.search(r'(\d{2,5})', dose_text)
        if num_match:
            num = int(num_match.group(1))
        if "paracetamol" in med or "acetaminophen" in med:
            if num and num > 1000:
                issues.append(f"High paracetamol dose detected: {dose_text}")
        if "amoxicillin" in med:
            if num and num > 1000:
                issues.append(f"Suspicious amoxicillin dose: {dose_text}")
    return issues

# -------------------------
# UI: upload and actions
# -------------------------
st.markdown("Upload a prescription image (photo or scan). The app will preprocess the image, run OCR (TrOCR), run a medical NER (if available), and run basic rule checks.")

uploaded = st.file_uploader("Prescription image", type=["jpg","jpeg","png","tiff"])

col1, col2 = st.columns([2,1])

with col1:
    if uploaded:
        # show preview
        image_bytes = uploaded.read()
        try:
            preprocessed = preprocess_image_bytes(image_bytes)
        except Exception as e:
            st.error(f"Image preprocessing failed: {e}")
            st.stop()

        st.image(preprocessed, caption="Preprocessed image (used for OCR)", use_column_width=True)

        # OCR
        st.info("Running OCR (TrOCR). This may take a few seconds on first run.")
        try:
            text = ocr_with_trocr(preprocessed)
        except Exception as e:
            st.warning("TrOCR failed — falling back to basic pytesseract OCR.")
            try:
                import pytesseract
                text = pytesseract.image_to_string(preprocessed)
            except Exception as e2:
                st.error("Both TrOCR and pytesseract failed.")
                st.stop()

        st.subheader("Extracted Text")
        st.write(text)

        # NER
        ner_results = []
        if ner_pipeline is not None:
            try:
                st.info("Running medical NER...")
                ner_results = ner_pipeline(text)
                st.subheader("Named Entities (NER)")
                st.write(ner_results)
            except Exception as e:
                st.warning("NER failed or timed out.")
                ner_results = []

        # Simple extraction of medicine names and dosages
        meds = []
        # build simple medicines list from NER if available
        if ner_results:
            for ent in ner_results:
                label = ent.get("entity_group", ent.get("entity", "")).upper()
                word = ent.get("word") if "word" in ent else ent.get("text", "")
                if "MED" in label or "DRUG" in label or "CHEM" in label or "MEDICINE" in label or label == "MEDICATION":
                    meds.append({"medicine": word, "span": (ent.get("start"), ent.get("end"))})
        # fallback: try heuristics: look for capitalized words followed by mg
        if not meds:
            for m in re.finditer(r'([A-Z][a-zA-Z0-9\-\s]{2,40}?)\s+(\d{1,4}\s*(?:mg|ml|g))', text):
                meds.append({"medicine": m.group(1).strip(), "dose": m.group(2).strip()})

        # attach dosages found by regex to meds where possible
        dosages = extract_dosages(text)
        for i, med in enumerate(meds):
            if "dose" not in med:
                med["dose"] = dosages[i] if i < len(dosages) else ""

        # Build final structured output
        output = {
            "raw_text": text,
            "medicines": meds,
            "dosages_found": dosages,
            "dates_found": extract_dates(text),
        }

        # rules
        issues = simple_rules_check([{"medicine": m.get("medicine",""), "dose": m.get("dose","")} for m in meds])
        output["issues"] = issues

        st.subheader("Structured Output")
        st.json(output)

        # allow download of JSON report
        st.download_button("Download JSON report", data=json.dumps(output, indent=2), file_name="prescription_report.json", mime="application/json")

with col2:
    st.info("Tips:")
    st.write("- Use clear photos (good lighting, no glare).")
    st.write("- Try to crop tightly around prescription.")
    st.write("- For messy handwriting, try multiple photos or typed prescription.")
    st.write("- The first run downloads models and may take longer.")
    st.write("---")
    st.write("Optional: If you have IBM Watson credentials and want extra checks (drug interactions), add them in a secure backend and call Watson here. I can help if you want to integrate it later.")

# Optional: show small footer
st.markdown("---")
st.caption("Built with TrOCR (Hugging Face) + optional medical NER. This tool is for demo/hackathon only and not a medical device. Always have a pharmacist/doctor confirm.")
