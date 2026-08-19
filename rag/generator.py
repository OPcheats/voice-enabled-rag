from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from retrieval.context import build_context
from retrieval.retriever import Retriever


MODEL_NAME = "google/flan-t5-small"


class RAGGenerator:
    """Retrieve context and generate a grounded answer."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        model_name: str = MODEL_NAME,
    ):
        self.retriever = retriever or Retriever()

        print("Loading generation model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(
        self,
        question: str,
        top_k: int = 3,
        max_new_tokens: int = 80,
        min_score: float = 0.30,
    ) -> str:
        """Retrieve relevant context and generate a grounded answer."""

        results = self.retriever.retrieve(
            question,
            top_k=top_k,
        )

        if not results:
            return "The information is not available in the provided context."

        best_score = results[0][1]

        if best_score < min_score:
            return "The information is not available in the provided context."

        context = build_context(results)

        prompt = f"""You are a question-answering assistant.

Answer the question using only the information in the context.

Rules:
- Give a concise answer in 1-2 sentences.
- Do not copy the context word-for-word.
- Do not add facts that are not supported by the context.
- If the context does not answer the question, say:
  "The information is not available in the provided context."

Context:
{context}

Question:
{question}

Answer:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )


if __name__ == "__main__":
    generator = RAGGenerator()

    question = (
        "What was the immediate impact "
        "of the success of the Manhattan Project?"
    )

    answer = generator.generate(question)

    print()
    print("=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question)

    print()
    print("=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(answer)