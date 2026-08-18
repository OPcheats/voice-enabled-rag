from collections.abc import Iterator
from typing import Any

from datasets import load_dataset


DATASET_NAME = "ai4bharat/MSMARCO-XI"


def get_dataset() -> Iterator[dict[str, Any]]:
    """Return the training split as a streaming dataset."""
    return load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )


def get_sample_records(
    dataset: Iterator[dict[str, Any]],
    count: int = 2,
) -> list[dict[str, Any]]:
    """Read only a small number of records from a streaming dataset."""
    records = []

    for index, record in enumerate(dataset):
        records.append(record)

        if index + 1 >= count:
            break

    return records


def main():
    print(f"Inspecting dataset: {DATASET_NAME}")

    dataset = get_dataset()

    print("\n" + "=" * 60)
    print("DATASET STREAM")
    print("=" * 60)
    print(dataset)

    print("\n" + "=" * 60)
    print("COLUMNS")
    print("=" * 60)
    print(dataset.column_names)

    print("\n" + "=" * 60)
    print("FIRST RECORD")
    print("=" * 60)

    records = get_sample_records(dataset, count=1)

    if records:
        print(records[0])


if __name__ == "__main__":
    main()