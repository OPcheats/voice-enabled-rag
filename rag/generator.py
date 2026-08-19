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

    def build_prompt(
        self,
        question: str,
        top_k: int = 3,
    ) -> str:
        """Build a grounded prompt from retrieved context."""
        results = self.retriever.retrieve(
            question,
            top_k=top_k,
        )

        context = build_context(results)

        return f"""Answer the question using only the provided context.

If the context does not contain enough information to answer,
say that the information is not available in the provided context.

Context:
{context}

Question:
{question}

Answer:"""

    def generate(
        self,
        question: str,
        top_k: int = 3,
        max_new_tokens: int = 80,
    ) -> str:
        """Retrieve context and generate an answer."""
        prompt = self.build_prompt(
            question,
            top_k=top_k,
        )

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