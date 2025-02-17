import requests
import torch
import whisper
import subprocess
import tempfile
import logging
import json
from flask import Request
import functions_framework
import faster_whisper


# Configure logging
logging.basicConfig(level=logging.INFO)
model = faster_whisper.WhisperModel('ivrit-ai/faster-whisper-v2-d4')


@functions_framework.http
def transcribe_audio(request: Request):
    # Set CORS headers for the preflight request
    if request.method == "OPTIONS":
        # Allows GET requests from any origin with the Content-Type
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600"
        }
        return ("", 204, headers)

    # Set CORS headers for the main request
    headers = {"Access-Control-Allow-Origin": "*"}

    try:
        request_json = request.get_json()
        if not request_json or "wav" not in request_json:
            return ({"error": "Invalid request, 'wav' field is required"}, 400, headers)
        if not request_json or "language" not in request_json:
            return ({"error": "Invalid request, 'language' field is required, try en, he or fr"}, 400, headers)

        wav_url = request_json["wav"]
        language_input = request_json.get("language")  # Optional parameter

        logging.info(f"Downloading audio from: {wav_url}")
        
        # Download the audio file
        response = requests.get(wav_url, timeout=10, allow_redirects=True)
        response.raise_for_status()

        # Write the downloaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(response.content)
            temp_audio_path = temp_audio.name

        # Convert to WAV format using ffmpeg
        temp_wav_path = temp_audio_path.replace(".mp3", ".wav")
        subprocess.run(
            ["ffmpeg", "-i", temp_audio_path, "-ar", "16000", "-ac", "1", "-f", "wav", temp_wav_path],
            check=True
        )

        logging.info("Running Whisper transcription...")



        segs, _ = model.transcribe(temp_wav_path, language=language_input)

        texts = [s.text for s in segs]

        transcribed_text = ' '.join(texts)
        
        return ({"transcription": transcribed_text}, 200, headers)

    except Exception as e:
        logging.error(f"Error processing audio: {str(e)}")
        return ({"error": str(e)}, 500, headers)