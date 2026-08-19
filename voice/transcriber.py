from faster_whisper import WhisperModel


MODEL_NAME = "small"


class Transcriber:
    """Convert recorded audio into text."""

    def __init__(self, model_name: str = MODEL_NAME):
        print("Loading speech recognition model...")

        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
        )

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file into text."""
        segments, _ = self.model.transcribe(audio_path)

        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        )

        return text.strip()


if __name__ == "__main__":
    transcriber = Transcriber()

    print("Transcriber loaded successfully.")