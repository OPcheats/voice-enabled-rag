import os
import win32com.client

os.makedirs("scratch", exist_ok=True)
wav_path = os.path.abspath("scratch/speech_question.wav")

speaker = win32com.client.Dispatch("SAPI.SpVoice")
stream = win32com.client.Dispatch("SAPI.SpFileStream")

# 3 = SSFMCreateForWrite
stream.Open(wav_path, 3, False)
speaker.AudioOutputStream = stream
speaker.Speak("What is artificial intelligence?")
stream.Close()

print(f"Generated WAV file at {wav_path}, size: {os.path.getsize(wav_path)} bytes")
