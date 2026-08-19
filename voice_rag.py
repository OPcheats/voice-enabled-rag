from rag.app import RAGApp
from voice.recorder import record_audio
from voice.transcriber import Transcriber


def main() -> None:
    print("=" * 60)
    print("VOICE RAG")
    print("=" * 60)

    transcriber = Transcriber()
    rag = RAGApp()

    audio_path = record_audio(
        output_path="voice/query.wav",
        duration=5,
    )

    question = transcriber.transcribe(str(audio_path))

    print()
    print("QUESTION")
    print("-" * 60)
    print(question)

    if not question:
        print("No speech detected.")
        return

    answer = rag.ask(question)

    print()
    print("ANSWER")
    print("-" * 60)
    print(answer)


if __name__ == "__main__":
    main()