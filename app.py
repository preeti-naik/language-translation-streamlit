import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Multilingual AI Translation System",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Multilingual Translation System")
st.caption("Transformer-Based Multilingual Translation System Supporting 200+ Languages")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# --------------------------------------------------
# LANGUAGE MAP (User-friendly names → Model codes)
# --------------------------------------------------
LANG_NAME_TO_CODE = {
    # Common
    "english": "eng_Latn",
    "hindi": "hin_Deva",
    "odia": "ory_Orya",
    "oriya": "ory_Orya",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "spanish": "spa_Latn",

    # Extra (for "Other")
    "bengali": "ben_Beng",
    "tamil": "tam_Taml",
    "telugu": "tel_Telu",
    "marathi": "mar_Deva",
    "urdu": "urd_Arab",
    "italian": "ita_Latn",
    "portuguese": "por_Latn",
    "russian": "rus_Cyrl",
    "japanese": "jpn_Jpan",
    "korean": "kor_Hang",
    "arabic": "arb_Arab",
    "chinese": "zho_Hans"
}

# --------------------------------------------------
# DROPDOWN OPTIONS
# --------------------------------------------------
DROPDOWN_LANGUAGES = [
    "English",
    "Hindi",
    "Odia",
    "French",
    "German",
    "Spanish",
    "Other"
]

# --------------------------------------------------
# SOURCE LANGUAGE DETECTION MAP
# --------------------------------------------------
DETECT_TO_MODEL_LANG = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "ur": "urd_Arab"
}

# --------------------------------------------------
# TRANSLATION FUNCTION
# --------------------------------------------------
def translate_text(text, target_lang_code):
    try:
        detected_lang = detect(text)
    except:
        detected_lang = "en"

    if detected_lang not in DETECT_TO_MODEL_LANG:
        detected_lang = "en"

    tokenizer.src_lang = DETECT_TO_MODEL_LANG[detected_lang]

    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)

    output_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code),
        max_length=256
    )

    return tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

# --------------------------------------------------
# UI
# --------------------------------------------------
input_text = st.text_area(
    "Enter text in any language",
    height=120,
    placeholder="Example: Hello, how are you?"
)

selected_language = st.selectbox("Target Language", DROPDOWN_LANGUAGES)

target_lang_code = None

if selected_language == "Other":
    other_lang = st.text_input(
        "Enter target language name (e.g. Japanese, Arabic, Russian)"
    )
    if other_lang:
        target_lang_code = LANG_NAME_TO_CODE.get(other_lang.strip().lower())
else:
    target_lang_code = LANG_NAME_TO_CODE.get(selected_language.lower())

if st.button("Translate"):
    if not input_text.strip():
        st.warning("Please enter text")
        st.stop()

    if not target_lang_code:
        st.error("Language not supported yet. Please try another language.")
        st.stop()

    with st.spinner("Translating..."):
        result = translate_text(input_text, target_lang_code)

    st.success("Translation Completed")
    st.text_area("Output", result, height=120)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("AI-Driven Language Translation System with Global Language Support")
