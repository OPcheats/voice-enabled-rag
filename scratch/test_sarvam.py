import os
import httpx
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("SARVAM_API_KEY", "")
print(f"SARVAM_API_KEY configured: {bool(api_key)}")

wav_path = "scratch/speech_question.wav"
with open(wav_path, "rb") as f:
    audio_bytes = f.read()

print(f"Testing Sarvam STT API with real spoken audio file ({len(audio_bytes)} bytes)...")

url = "https://api.sarvam.ai/speech-to-text"
headers = {"api-subscription-key": api_key}
files = {"file": ("speech_question.wav", audio_bytes, "audio/wav")}
data = {
    "model": "saaras:v3",
    "mode": "transcribe"
}

with httpx.Client(timeout=15.0) as client:
    response = client.post(url, headers=headers, files=files, data=data)
    print("HTTP Status Code:", response.status_code)
    if response.status_code == 200:
        res_json = response.json()
        print("Response JSON:", res_json)
        print("Extracted Transcript:", repr(res_json.get("transcript")))
    else:
        print("Response Error:", response.text)
