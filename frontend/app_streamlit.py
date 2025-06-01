import streamlit as st
import requests
import os
import io # For BytesIO

# Configuration
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000") # Use environment variable or default
ANALYZE_ENDPOINT = f"{BACKEND_URL}/analyze_full_audio/"
FEEDBACK_ENDPOINT = f"{BACKEND_URL}/get_word_feedback_audio/"

st.set_page_config(layout="wide", page_title="Pronunciation Practice Tool")

st.title("🗣️ Pronunciation Practice")
st.markdown("Upload an audio file of your speech. The system will analyze it and provide feedback.")

# --- Session State Initialization ---
if "analysis_id" not in st.session_state:
    st.session_state.analysis_id = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None


# --- File Uploader ---
uploaded_audio_file = st.file_uploader(
    "Choose an audio file (WAV, MP3)",
    type=["wav", "mp3", "ogg", "m4a", "flac"]
)

if uploaded_audio_file is not None:
    st.audio(uploaded_audio_file, format=uploaded_audio_file.type)

    if st.button("Analyze My Speech", type="primary", key="analyze_btn"):
        st.session_state.analysis_results = None # Clear previous results
        st.session_state.analysis_id = None
        st.session_state.error_message = None
        
        files_to_upload = {"audio_file": (uploaded_audio_file.name, uploaded_audio_file.getvalue(), uploaded_audio_file.type)}
        with st.spinner("Analyzing your speech... This might take a moment 🧠"):
            try:
                response = requests.post(ANALYZE_ENDPOINT, files=files_to_upload, timeout=300) # 5 min timeout
                response.raise_for_status() # Will raise an HTTPError for bad responses (4XX or 5XX)
                st.session_state.analysis_results = response.json()
                st.session_state.analysis_id = st.session_state.analysis_results.get("analysis_id")
                st.success("Analysis complete! See results below.")
            except requests.exceptions.HTTPError as http_err:
                try:
                    error_detail = response.json().get("detail", response.text)
                    st.session_state.error_message = f"Analysis failed (HTTP {response.status_code}): {error_detail}"
                except ValueError: # If response is not JSON
                     st.session_state.error_message = f"Analysis failed (HTTP {response.status_code}): {response.text}"
                st.error(st.session_state.error_message)
            except requests.exceptions.RequestException as req_err:
                st.session_state.error_message = f"Analysis failed: Could not connect to the backend or network error. ({req_err})"
                st.error(st.session_state.error_message)
            except Exception as e:
                st.session_state.error_message = f"An unexpected error occurred during analysis: {e}"
                st.error(st.session_state.error_message)

# --- Display Analysis Results ---
if st.session_state.error_message:
    st.error(st.session_state.error_message) # Persist error message

if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    transcription_data = results.get("transcription_data", {})
    word_analysis = results.get("word_analysis_results", [])

    st.divider()
    st.subheader("📜 Transcription")
    st.markdown(f"**Recognized Text:** {transcription_data.get('full_text', 'N/A')}")
    st.markdown(f"**Expected Phonemes (Full Sentence):** `{transcription_data.get('expected_phonemes_full_text', 'N/A')}`")

    st.divider()
    st.subheader("🔎 Word-by-Word Pronunciation Analysis")

    if not word_analysis:
        st.info("No word-level analysis available. The audio might have been too short or analysis encountered an issue.")
    else:
        # Determine number of columns dynamically, e.g., 3 or 4 words per row
        num_words = len(word_analysis)
        cols_per_row = 4 if num_words > 3 else num_words if num_words > 0 else 1
        
        row_containers = [st.columns(cols_per_row) for _ in range((num_words + cols_per_row - 1) // cols_per_row)]
        
        word_idx = 0
        for row in row_containers:
            for col in row:
                if word_idx < num_words:
                    with col:
                        word_data = word_analysis[word_idx]
                        word_text = word_data.get("word", "N/A")
                        label = word_data.get("label", "unknown").lower()
                        method = word_data.get("method", "")
                        word_audio_id = word_data.get("word_audio_segment_id") # For feedback

                        # Card styling
                        card_style = "padding:15px; border-radius:10px; margin:5px; box-shadow: 0 2px 5px 0 rgba(0,0,0,0.1);"
                        if "correct" in label:
                            card_style += "background-color:#e6ffe6; border-left: 5px solid #4CAF50;" # Greenish
                            emoji = "✅"
                        elif "incorrect" in label:
                            card_style += "background-color:#ffe6e6; border-left: 5px solid #f44336;" # Reddish
                            emoji = "❌"
                        elif "skipped" in label:
                            card_style += "background-color:#f0f0f0; border-left: 5px solid #9E9E9E;" # Greyish
                            emoji = "⏭️"
                        else:
                            card_style += "background-color:#fff3e0; border-left: 5px solid #ff9800;" # Orangish for unknown/error
                            emoji = "❓"
                        
                        st.markdown(f"<div style='{card_style}'>", unsafe_allow_html=True)
                        st.markdown(f"<h5>{word_text} {emoji}</h5>", unsafe_allow_html=True)
                        st.caption(f"Status: {label.capitalize()} ({method})")
                        st.markdown(f"<small>🗣️ Expected: `{word_data.get('expected_phonemes_word', 'N/A')}`</small>", unsafe_allow_html=True)
                        st.markdown(f"<small>🎤 You Said: `{word_data.get('predicted_phonemes_word', 'N/A')}`</small>", unsafe_allow_html=True)
                        if word_data.get("distance") is not None:
                            st.caption(f"📏 Distance: {word_data.get('distance')}")

                        # Feedback button for incorrect words
                        if "incorrect" in label and st.session_state.analysis_id and word_audio_id:
                            feedback_key = f"feedback_btn_{st.session_state.analysis_id}_{word_audio_id}"
                            if st.button("Get Audio Feedback", key=feedback_key, help=f"Get feedback for '{word_text}'"):
                                with st.spinner(f"Generating feedback for '{word_text}'..."):
                                    try:
                                        params = {
                                            "session_id": st.session_state.analysis_id,
                                            "word_audio_segment_id": word_audio_id,
                                            "correct_word_text": word_text # Assuming word_text is the correct form
                                        }
                                        feedback_response = requests.get(FEEDBACK_ENDPOINT, params=params, timeout=60)
                                        feedback_response.raise_for_status()
                                        # Play audio directly in the card
                                        st.audio(feedback_response.content, format="audio/mpeg")
                                        st.success("Feedback generated!")
                                    except requests.exceptions.HTTPError as http_err_fb:
                                        try:
                                            err_detail_fb = feedback_response.json().get("detail", feedback_response.text)
                                        except ValueError:
                                            err_detail_fb = feedback_response.text
                                        st.error(f"Feedback error (HTTP {feedback_response.status_code}): {err_detail_fb}")
                                    except requests.exceptions.RequestException as req_err_fb:
                                        st.error(f"Feedback error: Connection/Network issue. ({req_err_fb})")
                                    except Exception as e_fb:
                                        st.error(f"Feedback error: Unexpected issue. ({e_fb})")
                        st.markdown("</div>", unsafe_allow_html=True)
                word_idx += 1


st.sidebar.header("How to Use")
st.sidebar.markdown("""
1.  **Upload Your Audio:** Click 'Browse files' and select an audio recording of your speech.
2.  **Analyze Speech:** Click the 'Analyze My Speech' button.
3.  **View Results:**
    * The transcription of your speech will appear.
    * Each word will be analyzed for pronunciation accuracy.
    * Expected vs. your phonemes will be shown.
4.  **Get Feedback:** For words marked 'incorrect', click 'Get Audio Feedback' to hear how it should sound combined with your attempt.
""")
st.sidebar.markdown("---")
st.sidebar.info("This tool uses AI to help improve your pronunciation. Ensure your microphone is clear for best results.")