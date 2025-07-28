import datasets

try:
    loaded_dataset = datasets.load_from_disk("processed_dataset")
    print("Dataset loaded!")
    print(loaded_dataset)
except Exception as e:
    print(f"Failed to load dataset. Error: {e}")