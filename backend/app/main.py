from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import tempfile
import os
import io
import uuid # For generating unique session IDs
import logging

from .core.config import settings
from .models_api import schemas # Pydantic models
from .services import processing_service, classification_service, feedback_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_NAME, version="0.1.0")

# CORS (Cross-Origin Resource Sharing)
# Allows your Streamlit frontend (on a different port) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Or specify origins: ["http://localhost", "http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    # Warm up models or perform initial checks if necessary
    # The models in services are loaded at import time.
    # You can add checks here to ensure they loaded correctly.
    if not processing_service.asr_pipeline_instance:
        logger.critical("ASR Pipeline is NOT loaded. /analyze endpoint will fail.")
    if not processing_service.w2v_phoneme_model:
        logger.critical("Wav2Vec2 Phoneme Model is NOT loaded. Word audio phonemization will fail.")
    if not classification_service.classifier_model_instance:
        logger.warning("Phoneme Classifier Model is NOT loaded. Classification will use fallback.")
    logger.info("Startup complete.")


def _cleanup_temp_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Cleaned up temporary file: {path}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary file {path}: {e}")


@app.post("/analyze_full_audio/", response_model=schemas.FullAnalysisResponse)
async def analyze_full_audio_endpoint(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...)
):
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    # Save uploaded file temporarily
    try:
        # Use a temporary file that gets a unique name
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as tmp_audio:
            shutil.copyfileobj(audio_file.file, tmp_audio)
            tmp_audio_path = tmp_audio.name
        logger.info(f"Audio file saved temporarily to {tmp_audio_path}")
        background_tasks.add_task(_cleanup_temp_file, tmp_audio_path)
    except Exception as e:
        logger.error(f"Failed to save uploaded audio file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {str(e)}")
    finally:
        audio_file.file.close()


    session_id = str(uuid.uuid4()) # Unique ID for this analysis run / session
    word_analysis_results_list = []

    try:
        # 1. Transcribe audio to get text and word chunks (timestamps)
        transcription_output = processing_service.transcribe_audio(tmp_audio_path)
        full_transcribed_text = transcription_output["text"]
        word_chunks_from_asr = transcription_output["chunks"] # List of {'text': 'word', 'timestamp': (start, end)}

        # 2. Get 'expected' phonemes for the full transcribed sentence
        expected_phonemes_full_sentence = processing_service.get_phonemes_from_text_sentence(full_transcribed_text)

        transcription_data_response = schemas.TranscriptionData(
            full_text=full_transcribed_text,
            words_with_timestamps=[schemas.WordTimestamp(**chunk) for chunk in word_chunks_from_asr],
            expected_phonemes_full_text=expected_phonemes_full_sentence
        )

        # 3. Segment original audio into word clips
        # word_segments_info: List of {'word_text': str, 'audio_segment_path': str, 'original_timestamp': tuple}
        word_segments_info = processing_service.segment_audio_into_words(
            tmp_audio_path,
            word_chunks_from_asr,
            session_id  # Pass session_id to create session-specific directories for clips
        )
        # Schedule cleanup of the entire session directory for word clips
        session_clips_dir = os.path.join(processing_service.TEMP_DIR_FOR_SESSIONS, session_id)
        # background_tasks.add_task(processing_service.cleanup_session_files, session_id)


        # 4. For each word segment:
        for segment_info in word_segments_info:
            word_text = segment_info["word_text"]
            word_audio_path = segment_info["audio_segment_path"]

            # 4a. Get 'expected' phonemes for this specific word (from ASR text)
            expected_phonemes_for_word = processing_service.get_phonemes_from_text_for_word(word_text)

            # 4b. Get 'predicted' phonemes from the word's audio segment
            predicted_phonemes_for_word_audio = processing_service.get_phonemes_from_word_audio_segment(word_audio_path)

            # 4c. Classify pronunciation
            classification_result_dict = classification_service.classify_word_pronunciation(
                word_text=word_text,
                expected_phonemes_word=expected_phonemes_for_word,
                predicted_phonemes_word=predicted_phonemes_for_word_audio,
                use_model=True # You can make this a query parameter if desired
            )
            
            # Add word_audio_segment_id (filename can serve as ID for feedback)
            classification_result_dict["word_audio_segment_id"] = os.path.basename(word_audio_path)

            word_analysis_results_list.append(schemas.WordAnalysisResult(**classification_result_dict))

        return schemas.FullAnalysisResponse(
            analysis_id=session_id,
            transcription_data=transcription_data_response,
            word_analysis_results=word_analysis_results_list
        )

    except RuntimeError as e: # Catch specific operational errors like model not loaded
        logger.error(f"Runtime error during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Service temporarily unavailable: {str(e)}")
    except FileNotFoundError as e:
        logger.error(f"File not found error during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"A required file was not found: {str(e)}")
    except ValueError as e: # Catch data validation or format errors
        logger.error(f"Value error during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid data or format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during full audio analysis: {e}", exc_info=True)
        # Also cleanup session files if an error occurs mid-processing before background task runs
        if 'session_id' in locals() and session_id:
            processing_service.cleanup_session_files(session_id)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/get_word_feedback_audio/")
async def get_word_feedback_audio_endpoint(
    session_id: str = Query(..., description="The analysis session ID."),
    word_audio_segment_id: str = Query(..., description="The ID (filename) of the user's spoken word audio segment."),
    correct_word_text: str = Query(..., description="The text of the correctly pronounced word.")
):
    # Construct the path to the user's specific word audio segment
    user_word_audio_path = os.path.join(processing_service.TEMP_DIR_FOR_SESSIONS, session_id, "word_clips", word_audio_segment_id)

    if not os.path.exists(user_word_audio_path):
        logger.error(f"Requested word audio segment not found: {user_word_audio_path}")
        raise HTTPException(status_code=404, detail="Word audio segment not found. It might have been cleaned up or the ID is incorrect.")

    try:
        feedback_audio_bytes = feedback_service.generate_feedback_audio(
            user_word_audio_segment_path=user_word_audio_path,
            correct_word_text=correct_word_text
        )
        # StreamingResponse is good for binary data like audio
        return StreamingResponse(io.BytesIO(feedback_audio_bytes), media_type="audio/mpeg")

    except FileNotFoundError as e:
        logger.error(f"File not found during feedback audio generation: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating feedback audio for word '{correct_word_text}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate feedback audio: {str(e)}")