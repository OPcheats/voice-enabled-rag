from rag.generator import RAGGenerator
from retrieval.retriever import Retriever
from retrieval.context import build_context

from voice.recorder import record_audio
from voice.transcriber import Transcriber
from voice.speaker import Speaker


class RAGApp:
    """Voice-enabled Retrieval-Augmented Generation application."""

    def __init__(self):
        print("Initializing RAG system...")

        self.retriever = Retriever()
        self.generator = RAGGenerator()

        self.transcriber = Transcriber()
        self.speaker = Speaker()

    def ask(self, question: str) -> str:
        """Retrieve relevant information and generate an answer."""

        question = question.strip()

        if not question:
            return "Please provide a question."

        print()
        print("Retrieving relevant information...")

        results = self.retriever.retrieve(
            question,
            top_k=5,
        )

        if not results:
            return "I could not find relevant information in the retrieved documents."

        context = build_context(results)

        answer = self.generator.generate(
            question,
            context,
        )

        return answer

    def ask_by_voice(self) -> str:
        """Record speech, transcribe it, retrieve an answer, and speak it."""

        audio_path = record_audio(
            output_path="voice/test.wav",
            duration=5,
        )

        print()
        print("Transcribing...")

        question = self.transcriber.transcribe(
            str(audio_path)
        )

        print(f"Transcribed question: {question}")

        if not question:
            answer = "I could not understand the question."
            print(f"Answer: {answer}")
            self.speaker.speak(answer)
            return answer

        answer = self.ask(question)

        print()
        print(f"Answer: {answer}")

        self.speaker.speak(answer)

        return answer


if __name__ == "__main__":
    app = RAGApp()

    print()
    print("=" * 60)
    print("VOICE-ENABLED RAG ASSISTANT")
    print("=" * 60)
    print()
    print("Press Enter to start recording.")
    input()

    app.ask_by_voice()