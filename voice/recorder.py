from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


SAMPLE_RATE = 48000
CHANNELS = 2
DEFAULT_DURATION = 5
DEFAULT_DEVICE = 9
GAIN = 5.0


def record_audio(
    output_path: str | Path = "voice/test.wav",
    duration: int = DEFAULT_DURATION,
) -> Path:
    """Record microphone audio using the previously working configuration."""

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

    channel_1_rms = np.sqrt(np.mean(audio[:, 0] ** 2))
    channel_2_rms = np.sqrt(np.mean(audio[:, 1] ** 2))

    print(f"Channel 1 RMS: {channel_1_rms:.6f}")
    print(f"Channel 2 RMS: {channel_2_rms:.6f}")

    if channel_1_rms >= channel_2_rms:
        mono = audio[:, 0]
        selected_channel = 1
    else:
        mono = audio[:, 1]
        selected_channel = 2

    print(f"Using channel: {selected_channel}")

    mono = mono * GAIN
    mono = np.clip(mono, -1.0, 1.0)

    audio_int16 = (mono * 32767).astype(np.int16)

    print(f"Gain: {GAIN}x")
    print(f"Output peak: {np.max(np.abs(audio_int16))}")

    write(
        str(output_path),
        SAMPLE_RATE,
        audio_int16,
    )

    print(f"Recording saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    record_audio()