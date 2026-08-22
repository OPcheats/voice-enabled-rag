import re


class RAGGenerator:
    def __init__(self, model_name=None):
        """Initialize the answer generator."""
        pass

    def generate(self, question, context):
        """Generate a concise answer from the retrieved context."""

        if not context or not context.strip():
            return "The retrieved context does not provide enough information."

        match = re.search(
            r"Answer:\s*(.*?)(?=\n\n|\[Source|\Z)",
            context,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not match:
            return "The retrieved context does not provide enough information."

        answer = match.group(1).strip()

        # For this dataset, some answers contain an awkward explanation
        # followed by the actual consequence after a semicolon.
        if ";" in answer:
            parts = answer.split(";")

            factual_part = parts[-1].strip()

            if factual_part:
                answer = factual_part

        # Remove trailing punctuation and add a clean period.
        answer = answer.strip(" .;:")

        # Convert the dataset's fragment into a complete sentence.
        if answer:
            answer = answer[0].upper() + answer[1:]

        if answer and answer[-1] not in ".!?":
            answer += "."

        return answer