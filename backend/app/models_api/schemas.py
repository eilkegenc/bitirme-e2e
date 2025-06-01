from pydantic import BaseModel
from typing import List, Tuple, Optional

class WordTimestamp(BaseModel):
    text: str
    timestamp: Tuple[Optional[float], Optional[float]]

class TranscriptionData(BaseModel):
    full_text: str
    words_with_timestamps: List[WordTimestamp]
    expected_phonemes_full_text: str # Phonemes for the whole transcribed sentence

class WordAnalysisResult(BaseModel):
    word: str
    expected_phonemes_word: str # Phonemes from ASR text for this specific word
    predicted_phonemes_word: str # Phonemes from word audio segment for this specific word
    distance: Optional[int] = None
    normalized_distance: Optional[float] = None
    label: str # e.g., "correct", "incorrect", "skipped (stopword)"
    method: str # e.g., "model_based", "levenshtein_threshold"
    # We need a way to identify the audio segment for feedback
    word_audio_segment_id: Optional[str] = None # e.g., the filename of the segmented clip

class FullAnalysisResponse(BaseModel):
    transcription_data: TranscriptionData
    word_analysis_results: List[WordAnalysisResult]
    # Could also include a session_id or unique_id for the full analysis
    analysis_id: str