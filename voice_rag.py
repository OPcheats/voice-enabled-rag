from voice.recorder import record_audio
from voice.transcriber import Transcriber
from voice.speaker import Speaker

from retrieval.retriever import Retriever
from retrieval.context import build_context
from rag.generator import RAGGenerator


class VoiceRAG:
    """Complete voice-enabled RAG pipeline."""

    def __init__(self):
        print("Initializing Voice RAG system...")

        print("Loading retriever...")
        self.retriever = Retriever()

        print("Loading answer generator...")
        self.generator = RAGGenerator()

        print("Loading speech recognition model...")
        self.transcriber = Transcriber()

        print("Loading speaker...")
        self.speaker = Speaker()

        print("Voice RAG system ready.")

    def run(self):
        """Run one complete voice-question cycle."""

        input(
            "\nPress Enter to start recording..."
        )

        # --------------------------------------------------
        # 1. Record voice
        # --------------------------------------------------
        audio_path = record_audio(
            output_path="voice/test.wav",
            duration=5,
        )

        # --------------------------------------------------
        # 2. Speech-to-text
        # --------------------------------------------------
        print("\nTranscribing...")

        question = self.transcriber.transcribe(
            str(audio_path)
        )

        print(f"Transcribed question: {question}")

        if not question:
            answer = "I could not understand the question."
            print(f"\nAnswer: {answer}")
            self.speaker.speak(answer)
            return

        # --------------------------------------------------
        # 3. Retrieve relevant information
        # --------------------------------------------------
        print("\nRetrieving relevant information...")

        results = self.retriever.retrieve(
            question,
            top_k=3,
        )

        if not results:
            answer = (
                "I could not find relevant information "
                "for that question."
            )

            print(f"\nAnswer: {answer}")
            self.speaker.speak(answer)
            return

        # --------------------------------------------------
        # 4. Build context
        # --------------------------------------------------
        context = build_context(results)

        # --------------------------------------------------
        # 5. Generate answer
        # --------------------------------------------------
        answer = self.generator.generate(
            question,
            context,
        )

        print(f"\nAnswer: {answer}")

        # --------------------------------------------------
        # 6. Text-to-speech
        # --------------------------------------------------
        self.speaker.speak(answer)


if __name__ == "__main__":
    app = VoiceRAG()
    app.run()