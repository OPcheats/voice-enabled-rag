from faster_whisper import WhisperModel


MODEL_NAME = "medium"


class Transcriber:
    """Convert recorded audio into English text."""

    def __init__(self, model_name: str = MODEL_NAME):
        print("Loading speech recognition model...")

        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
        )

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file into English text."""

        segments, _ = self.model.transcribe(
            audio_path,
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=(
                "This is a history question. "
                "The question may contain terms such as "
                "Manhattan Project, atomic bomb, atomic researchers, "
                "World War II, and nuclear weapons."
            ),
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        )

        return text.strip()


if __name__ == "__main__":
    transcriber = Transcriber()

    print("Transcriber loaded successfully.")