from datasets import load_dataset


DATASET_NAME = "ai4bharat/MSMARCO-XI"


def main():
    print(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME)

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(dataset)

    for split_name, split in dataset.items():
        print("\n" + "=" * 60)
        print(f"SPLIT: {split_name}")
        print("=" * 60)

        print(f"Number of rows: {len(split)}")
        print(f"Columns: {split.column_names}")
        print(f"Features: {split.features}")

        if len(split) > 0:
            print("\nFirst record:")
            print(split[0])


if __name__ == "__main__":
    main()