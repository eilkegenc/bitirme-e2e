# backend/app/services/classification_service.py
import joblib
import pandas as pd
from Levenshtein import distance as levenshtein_distance
import logging
import os

# Attempt to import settings. If this script is run directly for testing
# and `app` is not in PYTHONPATH, this might fail.
# For FastAPI, `..core.config` should work.
try:
    from ..core.config import settings
except ImportError:
    # Fallback for standalone execution or if relative import fails in some contexts
    # This assumes a specific directory structure if run standalone for testing.
    # You might need to adjust this if you plan to run this file directly often.
    import sys
    # Add the parent directory of 'app' to sys.path to find 'core.config'
    # This assumes classification_service.py is in app/services/
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.core.config import settings
    print("Warning: classification_service.py using fallback import for settings. This is okay for direct script testing.")


logger = logging.getLogger(__name__)

# --- Classifier Model Loading ---
# Initialize the variable at the module level
classifier_model_instance = None

# Check if the model file exists and try to load it
# This code runs when the module is first imported.
if not hasattr(settings, 'PHONEME_CLASSIFIER_PATH'):
    logger.error("settings.PHONEME_CLASSIFIER_PATH is not defined in config. Cannot load classifier model.")
elif not os.path.exists(settings.PHONEME_CLASSIFIER_PATH):
    logger.warning(f"Phoneme classifier model not found at {settings.PHONEME_CLASSIFIER_PATH}. "
                   "Model-based classification will be disabled. Ensure 'scripts/train_classifier.py' has been run successfully.")
    # classifier_model_instance remains None
else:
    try:
        classifier_model_instance = joblib.load(settings.PHONEME_CLASSIFIER_PATH)
        logger.info(f"Successfully loaded phoneme classifier model from {settings.PHONEME_CLASSIFIER_PATH}")
    except Exception as e:
        logger.error(f"Error loading phoneme classifier model from {settings.PHONEME_CLASSIFIER_PATH}: {e}. "
                     "Model-based classification will be disabled.")
        classifier_model_instance = None # Ensure it's None if loading fails

# Stopwords from notebook Cell 14 (converted to lowercase for case-insensitive check)
# Used to skip classification for common words where minor mispronunciations are often ignored.
STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "so", "to", "in", "on", "at", "for", "by", "of",
    "with", "up", "off", "as", "i", "you", "he", "she", "it", "we", "they", "is", "are",
    "was", "were", "be", "been", "do", "does", "did", "has", "have", "had", "will", "shall",
    "can", "would", "may", "must", "that", "this", "there", "not"
}

def classify_word_pronunciation(
    word_text: str,
    expected_phonemes_word: str,
    predicted_phonemes_word: str,
    use_model: bool = True,
    levenshtein_fallback_threshold: int = 2
    ) -> dict:
    """
    Classifies a single word's pronunciation based on expected and predicted phonemes.

    Args:
        word_text (str): The text of the word.
        expected_phonemes_word (str): The expected phonetic transcription of the word.
        predicted_phonemes_word (str): The predicted phonetic transcription from user's audio.
        use_model (bool): If True and model is loaded, use the ML classifier. Otherwise, use Levenshtein.
        levenshtein_fallback_threshold (int): Threshold for Levenshtein distance if used as fallback or primary.

    Returns:
        dict: A dictionary containing classification details including:
              "word", "expected_phonemes_word", "predicted_phonemes_word",
              "distance", "normalized_distance", "label" ("correct", "incorrect", "skipped (stopword)"),
              and "method" (e.g., "model_based", "levenshtein_threshold_X").
    """
    result = {
        "word": word_text,
        "expected_phonemes_word": expected_phonemes_word,
        "predicted_phonemes_word": predicted_phonemes_word,
        "distance": None,
        "normalized_distance": None,
        "label": "unknown", # Default label
        "method": ""       # How the label was determined
    }

    # Clean the word text for stopword checking (remove common trailing punctuation)
    cleaned_word_text_for_stopword_check = word_text.lower().strip(".,?!;")
    if cleaned_word_text_for_stopword_check in STOPWORDS:
        result["label"] = "skipped (stopword)"
        result["method"] = "stopword_check"
        # Optionally calculate distance even for stopwords for informational purposes
        if expected_phonemes_word and predicted_phonemes_word: # Check if phonemes are not empty
             result["distance"] = levenshtein_distance(expected_phonemes_word, predicted_phonemes_word)
             len_exp = len(expected_phonemes_word)
             result["normalized_distance"] = result["distance"] / max(1, len_exp)
        return result

    # Ensure phoneme strings are not None before Levenshtein calculation
    # (Though they should ideally be empty strings if no phonemes, not None)
    safe_expected_phonemes = expected_phonemes_word if expected_phonemes_word is not None else ""
    safe_predicted_phonemes = predicted_phonemes_word if predicted_phonemes_word is not None else ""

    dist = levenshtein_distance(safe_expected_phonemes, safe_predicted_phonemes)
    len_exp = len(safe_expected_phonemes)
    norm_dist = dist / max(1, len_exp) # Avoid division by zero if len_exp is 0

    result["distance"] = dist
    result["normalized_distance"] = norm_dist

    # Determine label using model or fallback
    can_use_model = use_model and classifier_model_instance is not None
    
    if can_use_model:
        result["method"] = "model_based"
        try:
            # The model was trained on features: ["distance", "len_expected", "normalized_distance"]
            features_df = pd.DataFrame([[dist, len_exp, norm_dist]],
                                     columns=["distance", "len_expected", "normalized_distance"])
            
            prediction = classifier_model_instance.predict(features_df)[0]
            # Assuming your training script labels: 1 for 'correct', 0 for 'incorrect'.
            # Adjust if your model's label encoding is different.
            result["label"] = "correct" if prediction == 1 else "incorrect"
            logger.debug(f"Model prediction for '{word_text}': {result['label']} (features: d={dist},l={len_exp},n={norm_dist:.2f})")

        except Exception as e:
            logger.error(f"Error during model prediction for word '{word_text}': {e}. Falling back to Levenshtein.")
            # Fallback to Levenshtein if model prediction fails for some reason
            result["label"] = "correct" if dist <= levenshtein_fallback_threshold else "incorrect"
            result["method"] += f"_fallback_levenshtein_{levenshtein_fallback_threshold}"
            
    else: # Fallback to Levenshtein threshold if model not available or use_model is False
        result["label"] = "correct" if dist <= levenshtein_fallback_threshold else "incorrect"
        result["method"] = f"levenshtein_threshold_{levenshtein_fallback_threshold}"
        if use_model and classifier_model_instance is None:
             result["method"] += "_model_unavailable"
        logger.debug(f"Levenshtein classification for '{word_text}': {result['label']} (dist: {dist}, threshold: {levenshtein_fallback_threshold})")

    return result

# Example of how you might test this file standalone (optional)
if __name__ == '__main__':
    print("Testing classification_service.py standalone...")
    if classifier_model_instance:
        print("Classifier model IS LOADED.")
    else:
        print("Classifier model IS NOT LOADED. Classification will use fallback.")

    # Test with dummy data
    test_word = "example"
    # Simulate phonemes - in reality, these would come from your ASR and phoneme prediction services
    exp_ph = "ɪɡzˈæmpəl" # Correct IPA (example)
    pred_ph_correct = "ɪɡzæmpəl" # Very close prediction
    pred_ph_minor_error = "ɛɡzæmpəl" # Minor error
    pred_ph_major_error = "ɛksmpl"   # Major error
    pred_ph_stopword_exp = "ðə"      # Phonemes for "the"
    pred_ph_stopword_pred = "də"     # Predicted for "the"


    print("\n--- Test Case: Correct Pronunciation (Model or Threshold) ---")
    res_correct = classify_word_pronunciation(test_word, exp_ph, pred_ph_correct)
    print(res_correct)

    print("\n--- Test Case: Minor Error ---")
    res_minor = classify_word_pronunciation(test_word, exp_ph, pred_ph_minor_error)
    print(res_minor)

    print("\n--- Test Case: Major Error ---")
    res_major = classify_word_pronunciation(test_word, exp_ph, pred_ph_major_error)
    print(res_major)

    print("\n--- Test Case: Stopword ('the') ---")
    res_stopword = classify_word_pronunciation("the", pred_ph_stopword_exp, pred_ph_stopword_pred)
    print(res_stopword)

    print("\n--- Test Case: Empty Phonemes (should not crash) ---")
    res_empty = classify_word_pronunciation("testempty", "", "")
    print(res_empty)
    res_empty_exp = classify_word_pronunciation("testemptyexp", "abc", "")
    print(res_empty_exp)
    res_empty_pred = classify_word_pronunciation("testemptypred", "", "xyz")
    print(res_empty_pred)