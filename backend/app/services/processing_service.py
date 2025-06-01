import torch
import torchaudio
from transformers import pipeline, AutoModelForCTC, Wav2Vec2Processor as Wav2Vec2PhonemeProcessor
import phonemizer as global_phonemizer
from pydub import AudioSegment
import os
import tempfile
import logging
import shutil # For rmtree
import uuid

from ..core.config import settings

logger = logging.getLogger(__name__)
TEMP_DIR_FOR_SESSIONS = tempfile.mkdtemp(prefix="pronunciation_sessions_")

# --- Model Loading (should happen once on startup) ---
# ASR Pipeline (CrisperWhisper)
# IMPORTANT: The notebook uses `!python transcribe.py`. This implies `transcribe.py` from
# CrisperWhisper has specific logic. You MUST adapt that logic into a callable Python function.
# For this example, I'm using a standard Hugging Face pipeline structure, which might need
# adjustments for CrisperWhisper or you'll import a function from the CrisperWhisper repo.
# This is a CRITICAL integration point.
try:
    # Ensure the nyrahealth/transformers fork is correctly installed and used.
    # The `device_map="auto"` is good for multi-GPU or CPU fallback.
    # `torch_dtype` for performance on CUDA.
    asr_pipeline_instance = pipeline(
        "automatic-speech-recognition",
        model=settings.CRISPER_WHISPER_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto", # Handles GPU/CPU automatically
        # return_timestamps="word" # This should be supported by CrisperWhisper
    )
    logger.info(f"ASR Pipeline loaded with {settings.CRISPER_WHISPER_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load ASR pipeline ({settings.CRISPER_WHISPER_MODEL_NAME}): {e}. ASR functionality will be limited.")
    asr_pipeline_instance = None

# Wav2Vec2 Phoneme Model
try:
    w2v_phoneme_processor = Wav2Vec2PhonemeProcessor.from_pretrained(settings.WAV2VEC2_PHONEME_MODEL_NAME)
    w2v_phoneme_model = AutoModelForCTC.from_pretrained(settings.WAV2VEC2_PHONEME_MODEL_NAME)
    if torch.cuda.is_available():
        w2v_phoneme_model = w2v_phoneme_model.to("cuda") # Manually move to GPU if device_map isn't used here
    logger.info(f"Wav2Vec2 Phoneme model loaded: {settings.WAV2VEC2_PHONEME_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load Wav2Vec2 Phoneme model ({settings.WAV2VEC2_PHONEME_MODEL_NAME}): {e}. Word audio phonemization will fail.")
    w2v_phoneme_processor = None
    w2v_phoneme_model = None

def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribes audio using the loaded ASR pipeline.
    Returns a dictionary with "text" and "chunks" (with "text" and "timestamp" tuples).
    """
    if not asr_pipeline_instance:
        raise RuntimeError("ASR pipeline is not available.")
    try:
        # The pipeline should return timestamps if `return_timestamps="word"` is effective.
        # You might need to pass `generate_kwargs={"return_timestamps": "word"}` if model expects it.
        result = asr_pipeline_instance(audio_path, return_timestamps="word")
        # Ensure the output format matches: {"text": "full sentence", "chunks": [{"text": "word", "timestamp": (start, end)}, ...]}
        if "chunks" not in result or "text" not in result:
             logger.warning(f"ASR output for {audio_path} missing 'text' or 'chunks'. Result: {result}")
             # Attempt to reconstruct if possible or raise error
             # This might happen if CrisperWhisper has a different output format than standard HF pipeline
             raise ValueError("ASR output format incorrect. Needs 'text' and 'chunks'.")
        logger.info(f"Transcription successful for: {audio_path}")
        return result
    except Exception as e:
        logger.error(f"Error during transcription of {audio_path}: {e}")
        raise

def get_phonemes_from_text_for_word(text: str) -> str:
    """ Phonemizes a single word or short phrase for 'expected' phonemes. """
    if not text.strip(): return ""
    try:
        phonemes = global_phonemizer.phonemize(
            text,
            language=settings.PHONEMIZER_LANG,
            backend=settings.PHONEMIZER_BACKEND,
            strip=True, # Remove leading/trailing whitespace from phonemes
            preserve_punctuation=False, # Usually False for individual words
            with_stress=True
        )
        return phonemes.strip()
    except Exception as e:
        logger.error(f"Error phonemizing text '{text}': {e}")
        return "[PHONEMIZER_ERROR]"

def get_phonemes_from_text_sentence(text: str) -> str:
    """ Phonemizes a full sentence. """
    if not text.strip(): return ""
    try:
        # For sentences, preserve_punctuation might be useful if your model handles it.
        # Here, matching the notebook's likely behavior of stripping.
        phonemes = global_phonemizer.phonemize(
            text,
            language=settings.PHONEMIZER_LANG,
            backend=settings.PHONEMIZER_BACKEND,
            strip=True,
            preserve_punctuation=True, # Keep punctuation to see how espeak handles it
            with_stress=True
        )
        return phonemes.strip()
    except Exception as e:
        logger.error(f"Error phonemizing sentence '{text}': {e}")
        return "[PHONEMIZER_ERROR_SENTENCE]"


def segment_audio_into_words(original_audio_path: str, transcription_chunks: list, session_id: str) -> list[dict]:
    """
    Segments the original audio into word clips based on timestamps.
    Saves clips into a session-specific directory.
    Returns a list of dicts, each with word_text and the path to its audio segment.
    """
    session_clips_dir = os.path.join(TEMP_DIR_FOR_SESSIONS, session_id, "word_clips")
    os.makedirs(session_clips_dir, exist_ok=True)
    logger.info(f"Word clips for session {session_id} will be saved to: {session_clips_dir}")

    try:
        audio = AudioSegment.from_file(original_audio_path)
    except Exception as e:
        logger.error(f"Could not load audio file {original_audio_path} with pydub: {e}")
        raise

    word_segments_info = []
    PADDING_SEC = 0.05  # 50 ms padding from notebook

    for i, chunk in enumerate(transcription_chunks):
        word_text_original = chunk["text"].strip()
        timestamp = chunk["timestamp"] # Should be (start_sec, end_sec)

        if not word_text_original or timestamp is None or timestamp[0] is None or timestamp[1] is None:
            logger.warning(f"Skipping chunk due to missing text or timestamp: {chunk}")
            continue
        
        start_sec, end_sec = timestamp
        
        # Basic validation for timestamps
        if start_sec > end_sec:
            logger.warning(f"Invalid timestamp for '{word_text_original}': start {start_sec} > end {end_sec}. Attempting to fix or skip.")
            # Option 1: Skip
            # continue
            # Option 2: Try to make a small segment if possible, or swap
            if end_sec < start_sec - 0.1 : # if significantly different
                start_sec, end_sec = end_sec, start_sec # swap
            else: # if very close, make a tiny clip around start_sec
                end_sec = start_sec + 0.1


        # Sanitize word_text for filename
        word_text_clean_for_filename = "".join(c if c.isalnum() else "_" for c in word_text_original)
        if len(word_text_clean_for_filename) > 30: word_text_clean_for_filename = word_text_clean_for_filename[:30]


        clip_filename = f"word_{i}_{word_text_clean_for_filename}_{uuid.uuid4().hex[:6]}.wav"
        clip_path = os.path.join(session_clips_dir, clip_filename)

        # Apply padding
        padded_start_ms = int(max(0.0, start_sec - PADDING_SEC) * 1000)
        padded_end_ms = int((end_sec + PADDING_SEC) * 1000)
        padded_end_ms = min(padded_end_ms, len(audio)) # Ensure not beyond audio length

        if padded_start_ms >= padded_end_ms : # if segment is zero or negative length after padding
             # try to create a minimal length segment around the original start time
            center_ms = int(start_sec * 1000)
            padded_start_ms = max(0, center_ms - int(PADDING_SEC * 1000 / 2 * 5)) # 250ms minimal clip
            padded_end_ms = min(len(audio), center_ms + int(PADDING_SEC * 1000 / 2 * 5))
            if padded_start_ms >= padded_end_ms:
                logger.warning(f"Could not create valid segment for '{word_text_original}' at {start_sec}-{end_sec}s. Skipping.")
                continue


        try:
            word_audio_segment = audio[padded_start_ms:padded_end_ms]
            word_audio_segment.export(clip_path, format="wav")
            word_segments_info.append({
                "word_text": word_text_original, # The original text from ASR for this chunk
                "audio_segment_path": clip_path,
                "original_timestamp": timestamp
            })
        except Exception as e:
            logger.error(f"Failed to export or segment for '{word_text_original}' at {clip_path}: {e}")

    return word_segments_info


def get_phonemes_from_word_audio_segment(word_audio_path: str) -> str:
    """ Predicts phonemes from a single word audio clip using Wav2Vec2. """
    if not w2v_phoneme_model or not w2v_phoneme_processor:
        raise RuntimeError("Wav2Vec2 phoneme model/processor is not available.")
    try:
        waveform, sr = torchaudio.load(word_audio_path)
        if sr != 16000: # Wav2Vec2 typically expects 16kHz
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        input_values = w2v_phoneme_processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values
        if torch.cuda.is_available():
            input_values = input_values.to("cuda")

        with torch.no_grad():
            logits = w2v_phoneme_model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        phonemes_str_raw = w2v_phoneme_processor.batch_decode(predicted_ids)[0] # batch_decode returns a list

        # Process raw phonemes: remove spaces and convert to lowercase to match espeak output style
        processed_phonemes = "".join(phonemes_str_raw.split()).lower()
        return processed_phonemes
    except Exception as e:
        logger.error(f"Error predicting phonemes for word audio {word_audio_path}: {e}")
        return "[W2V_PHONEME_ERROR]"

def cleanup_session_files(session_id: str):
    session_dir = os.path.join(TEMP_DIR_FOR_SESSIONS, session_id)
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir)
            logger.info(f"Cleaned up session directory: {session_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up session directory {session_dir}: {e}")