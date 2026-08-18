from datasets import load_dataset


DATASET_NAME = "ai4bharat/MSMARCO-XI"


def main():
    print(f"Inspecting dataset: {DATASET_NAME}")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

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

    first_record = next(iter(dataset))

    for key, value in first_record.items():
        print(f"\n--- {key} ---")
        print(type(value))
        print(value)


if __name__ == "__main__":
    main()