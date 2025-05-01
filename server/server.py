from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
import logging # Use logging for better feedback

# --- Configuration ---
# Adjust these paths based on your actual StyleTTS2 installation and model location
STYLE_TTS_BASE_DIR = os.getenv("STYLE_TTS_BASE_DIR", "/opt/StyleTTS2") # Use env var or default
INFERENCE_SCRIPT = os.path.join(STYLE_TTS_BASE_DIR, "synthesize.py") # Adjust script name if needed
# Define model and config paths (use Env Vars ideally)
MODEL_PATH = os.getenv("STYLE_TTS_MODEL_PATH", os.path.join(STYLE_TTS_BASE_DIR, "Models/LibriTTS/epoch_2nd_00100.pth")) # Example path
CONFIG_PATH = os.getenv("STYLE_TTS_CONFIG_PATH", os.path.join(STYLE_TTS_BASE_DIR, "Configs/LibriTTS/config.yml")) # Example path
OUTPUT_DIR = os.getenv("STYLE_TTS_OUTPUT_DIR", os.path.join(STYLE_TTS_BASE_DIR, "results"))
REFERENCE_AUDIO_DIR = os.getenv("STYLE_TTS_REFERENCE_DIR", os.path.join(STYLE_TTS_BASE_DIR, "assets", "references"))

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Pydantic Models ---
class StyleTTS2Request(BaseModel):
    reference_speaker_name: str # e.g., "adeline_question" - used to find the reference wav
    text_input: str
    narration_id: str       # Unique ID for the output file name
    # Optional parameters matching synthesize.py script arguments
    diffusion_steps: int = 5 # Example default, adjust based on script
    embedding_scale: float = 1.0 # Example (often called alpha or embedding_scale) adjust based on script

# --- FastAPI Endpoints ---

@app.post("/synthesize-speech")
def synthesize_speech(request: StyleTTS2Request):
    """
    Synthesizes speech using StyleTTS 2 based on input text and a short reference audio.
    """
    logger.info(f"Received synthesis request for narration_id: {request.narration_id}")

    # Construct the path to the short reference audio file
    # Assumes reference files are named like {reference_speaker_name}.wav
    reference_audio_path = os.path.join(REFERENCE_AUDIO_DIR, f"{request.reference_speaker_name}.wav")
    logger.info(f"Looking for reference audio at: {reference_audio_path}")

    # --- Input Validation ---
    if not os.path.exists(reference_audio_path):
        logger.error(f"Reference audio not found: {reference_audio_path}")
        raise HTTPException(status_code=404, detail=f"Reference audio not found at {reference_audio_path}")

    if not os.path.exists(INFERENCE_SCRIPT):
         logger.error(f"Inference script not found: {INFERENCE_SCRIPT}")
         raise HTTPException(status_code=500, detail=f"Server configuration error: Inference script not found at {INFERENCE_SCRIPT}")

    if not os.path.exists(MODEL_PATH):
         logger.error(f"Model checkpoint not found: {MODEL_PATH}")
         raise HTTPException(status_code=500, detail=f"Server configuration error: Model checkpoint not found at {MODEL_PATH}")

    if not os.path.exists(CONFIG_PATH):
         logger.error(f"Config file not found: {CONFIG_PATH}")
         raise HTTPException(status_code=500, detail=f"Server configuration error: Config file not found at {CONFIG_PATH}")

    if not request.text_input.strip():
        logger.error("Text input is empty.")
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    # Construct the output file path
    output_wav = os.path.join(OUTPUT_DIR, f"speech-{request.narration_id}-output.wav")
    logger.info(f"Output audio will be saved to: {output_wav}")

    # --- Construct the Subprocess Command ---
    # Adjust arguments based on the EXACT signature of your synthesize.py script
    cmd = [
        "python", INFERENCE_SCRIPT,
        "--text", request.text_input,
        "--reference_audio", reference_audio_path, # Changed argument name based on common usage
        "--model_path", MODEL_PATH,
        "--config_path", CONFIG_PATH,
        "--output_wav", output_wav,
        # --- Add optional parameters ---
        # Check your script for exact names (e.g., --steps, --alpha, --beta, --scale)
        "--diffusion_steps", str(request.diffusion_steps),
        "--embedding_scale", str(request.embedding_scale),
        # Add other relevant args like --device if needed (often handled by the script)
        # "--device", "cuda" # Uncomment if needed
    ]
    logger.info(f"Executing command: {' '.join(cmd)}")

    # --- Run the Command ---
    try:
        # Using capture_output=True to get stdout/stderr
        result = subprocess.run(cmd, capture_output=True, text=True, check=False) # check=False to handle error manually

        # Check return code
        if result.returncode != 0:
            error_message = f"StyleTTS 2 inference script failed with code {result.returncode}.\nStderr:\n{result.stderr}\nStdout:\n{result.stdout}"
            logger.error(error_message)
            raise HTTPException(status_code=500, detail=error_message)

        # --- Verify Output File ---
        if not os.path.exists(output_wav):
             error_message = f"Inference script finished successfully but output file was not found at {output_wav}.\nStderr:\n{result.stderr}\nStdout:\n{result.stdout}"
             logger.error(error_message)
             raise HTTPException(status_code=500, detail=error_message)

        logger.info(f"Successfully generated speech: {output_wav}")
        # Optionally log stdout from the script if needed for debugging success cases
        # logger.debug(f"Inference script stdout:\n{result.stdout}")

        return {
            "status": "success",
            "message": "StyleTTS 2 speech synthesis completed.",
            "output_audio_path": output_wav
        }

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (like validation errors)
        raise http_exc
    except Exception as e:
        error_message = f"An unexpected error occurred during synthesis: {str(e)}"
        logger.exception(error_message) # Log full traceback
        raise HTTPException(status_code=500, detail=error_message)

# --- Example Usage ---
# Save this script as style_tts_server.py (or similar)
# Run with: uvicorn style_tts_server:app --host 0.0.0.0 --port 8081 (or your desired port)

# Example curl command:
# Assume you have a reference file at /opt/StyleTTS2/assets/references/adeline_question.wav
# curl -X POST http://localhost:8081/synthesize-speech \
#   -H "Content-Type: application/json" \
#   -d '{
#     "reference_speaker_name": "adeline_question",
#     "text_input": "Is this going to sound like a genuine question?",
#     "narration_id": "question-test-01",
#     "diffusion_steps": 5,
#     "embedding_scale": 1.2
#   }'