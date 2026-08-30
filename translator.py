import streamlit as st
from deep_translator import GoogleTranslator
from gTTS import gTTS
import io

# Page configuration
st.set_page_config(page_title="Language Translation Tool", layout="centered")

st.title("🌐 Language Translation Tool")
st.write("Translate text instantly across multiple languages.")

# Supported languages mapping (Display Name: Code)
LANGUAGE_OPTIONS = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja",
    "Russian": "ru",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Italian": "it"
}

# UI Layout: Language Selection
col1, col2 = st.columns(2)
with col1:
    source_lang_name = st.selectbox("Source Language", ["Auto Detect"] + list(LANGUAGE_OPTIONS.keys()))
with col2:
    target_lang_name = st.selectbox("Target Language", list(LANGUAGE_OPTIONS.keys()), index=0)

# Input text box
input_text = st.text_area("Enter text to translate:", height=150, placeholder="Type your text here...")

# Translate action
if st.button("Translate", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text to translate.")
    else:
        try:
            # Map selected names to language codes
            src = "auto" if source_lang_name == "Auto Detect" else LANGUAGE_OPTIONS[source_lang_name]
            tgt = LANGUAGE_OPTIONS[target_lang_name]

            # Process translation via Google Translator API
            translated_text = GoogleTranslator(source=src, target=tgt).translate(input_text)

            # Store result in session state
            st.session_state["translated_text"] = translated_text
            st.session_state["target_code"] = tgt

        except Exception as e:
            st.error(f"Translation failed: {e}")

# Display Translated Output
if "translated_text" in st.session_state:
    st.subheader("Translated Output:")
    st.code(st.session_state["translated_text"], language="text")

    # Optional Feature: Text-to-Speech (TTS)
    try:
        tts = gTTS(text=st.session_state["translated_text"], lang=st.session_state["target_code"])
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        st.audio(audio_bytes, format="audio/mp3")
    except Exception:
        st.info("Audio pronunciation not available for this language/text.")
