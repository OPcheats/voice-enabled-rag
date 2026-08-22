import httpx

api_url = "http://localhost:8000/ask"
wav_path = "scratch/speech_question.wav"

with open(wav_path, "rb") as f:
    audio_bytes = f.read()

print(f"Sending audio file ({len(audio_bytes)} bytes) to POST /ask...")

with httpx.Client(timeout=30.0) as client:
    response = client.post(
        api_url,
        files={"audio": ("speech_question.wav", audio_bytes, "audio/wav")}
    )
    print("HTTP Status Code:", response.status_code)
    print("Response JSON:")
    import json
    print(json.dumps(response.json(), indent=2))
