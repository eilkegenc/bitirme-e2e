from gtts import gTTS
from pydub import AudioSegment
import os
import tempfile
import logging

from ..core.config import settings

logger = logging.getLogger(__name__)

def _generate_tts_mp3_temp(text: str, lang: str = settings.TTS_LANG, slow: bool = False) -> str:
    """Generates an MP3 TTS file and returns its temporary path."""
    try:
        fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd) # gTTS will open and write to it

        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(temp_path)
        logger.info(f"Saved TTS audio to temporary file: {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Error generating TTS for '{text}': {e}")
        # Clean up if file was created before error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise # Re-raise to be handled by the caller

def generate_feedback_audio(user_word_audio_segment_path: str, correct_word_text: str) -> bytes:
    """
    Creates feedback audio: "You said [user_audio] but you should have said [correct_word_tts]."
    Returns the audio content as bytes (MP3 format).
    """
    temp_files_to_clean = []
    try:
        if not os.path.exists(user_word_audio_segment_path):
            logger.error(f"User word audio segment path not found: {user_word_audio_segment_path}")
            raise FileNotFoundError(f"User word audio segment not found: {user_word_audio_segment_path}")

        # 1. TTS for "You said"
        prefix_text = "You said"
        prefix_tts_path = _generate_tts_mp3_temp(prefix_text)
        temp_files_to_clean.append(prefix_tts_path)
        prefix_audio = AudioSegment.from_mp3(prefix_tts_path)

        # 2. User's spoken word audio segment (already a WAV file)
        user_audio_segment = AudioSegment.from_wav(user_word_audio_segment_path)

        # 3. TTS for "but you should have said [correct_word_text]."
        suffix_text = f"but you should have said {correct_word_text}."
        suffix_tts_path = _generate_tts_mp3_temp(suffix_text)
        temp_files_to_clean.append(suffix_tts_path)
        suffix_audio = AudioSegment.from_mp3(suffix_tts_path)

        # 4. Concatenate audio segments
        # Optional: Add a short silence between segments for clarity
        silence = AudioSegment.silent(duration=300) # 300ms
        final_audio = prefix_audio + silence + user_audio_segment + silence + suffix_audio

        # Export to an in-memory bytes buffer
        # To do this, we first export to a temporary file, then read it.
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_final_output_mp3:
            final_audio.export(tmp_final_output_mp3.name, format="mp3")
            final_output_mp3_path = tmp_final_output_mp3.name
        temp_files_to_clean.append(final_output_mp3_path) # Add to cleanup list

        with open(final_output_mp3_path, "rb") as f:
            audio_bytes = f.read()

        logger.info(f"Successfully generated feedback audio for word '{correct_word_text}'.")
        return audio_bytes

    except Exception as e:
        logger.error(f"Error in generate_feedback_audio for '{correct_word_text}': {e}")
        raise # Propagate the error
    finally:
        # Cleanup all temporary files created during this process
        for path in temp_files_to_clean:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.debug(f"Cleaned up temp file: {path}")
                except Exception as e_remove:
                    logger.warning(f"Could not remove temp file {path}: {e_remove}")