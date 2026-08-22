import pyttsx3


class Speaker:
    """Convert text to speech using the Windows speech engine."""

    def __init__(self):
        self.engine = pyttsx3.init()

        self.engine.setProperty("rate", 170)
        self.engine.setProperty("volume", 1.0)

    def speak(self, text: str) -> None:
        """Speak the provided text."""

        if not text or not text.strip():
            return

        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as exc:
            print(f"TTS warning: {exc}")

    def close(self) -> None:
        """Stop the speech engine cleanly."""

        try:
            self.engine.stop()
        except Exception:
            pass


if __name__ == "__main__":
    speaker = Speaker()

    try:
        speaker.speak(
            "Hello. This is a test of the voice enabled RAG assistant."
        )
    finally:
        speaker.close()