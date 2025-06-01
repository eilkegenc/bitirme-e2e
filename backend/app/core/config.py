import os

class Settings:
    APP_NAME: str = "Pronunciation Assessment API"
    # Assuming 'assets' is a sibling to 'core', 'services' etc. inside 'app'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")

    PHONEME_CLASSIFIER_PATH: str = os.path.join(ASSETS_DIR, "phoneme_classifier.joblib")
    PROCESSED_DATA_CSV_PATH: str = os.path.join(ASSETS_DIR, "processed_with_alignments.csv")

    # Model names from Hugging Face / local paths if downloaded
    CRISPER_WHISPER_MODEL_NAME: str = "nyrahealth/CrisperWhisper"
    WAV2VEC2_PHONEME_MODEL_NAME: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"

    # Phonemizer settings
    PHONEMIZER_LANG: str = "en-us"
    PHONEMIZER_BACKEND: str = "espeak" # Ensure espeak is installed in the environment

    # TTS settings
    TTS_LANG: str = "en"

settings = Settings()

# Ensure assets directory exists
if not os.path.exists(settings.ASSETS_DIR):
    os.makedirs(settings.ASSETS_DIR)
    print(f"Created assets directory at {settings.ASSETS_DIR}. Ensure your model files are placed here.")