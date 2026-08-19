from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


SAMPLE_RATE = 48000
CHANNELS = 2
DEFAULT_DURATION = 5
DEFAULT_DEVICE = 5


def record_audio(
    output_path: str | Path = "voice/test.wav",
    duration: int = DEFAULT_DURATION,
) -> Path:
    """Record microphone audio and save it as a mono WAV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording for {duration} seconds...")
    print("Speak now.")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=DEFAULT_DEVICE,
    )

    sd.wait()

    # Use the microphone-array channels together.
    mono = np.mean(audio, axis=1)

    mono = np.clip(mono, -1.0, 1.0)
    audio_int16 = (mono * 32767).astype(np.int16)

    write(
        str(output_path),
        SAMPLE_RATE,
        audio_int16,
    )

    print(f"Recording saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    record_audio()