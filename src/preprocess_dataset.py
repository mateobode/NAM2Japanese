from prepare_dataset import get_dataset, prepare_dataset
from model import get_feature_extractor_and_tokenizer

try:
    print("Starting dataset preprocessing ...")
    feature_extractor, tokenizer, _ = get_feature_extractor_and_tokenizer()
    raw_dataset = get_dataset()

    processed_dataset = raw_dataset.map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
        remove_columns = raw_dataset.column_names["train"],
        num_proc=5
    )
    
    processed_dataset.save_to_disk("processed_large_dataset")
    print("Dataset preprocessing complete. Saved to 'processed_large_dataset'")
except Exception as e:
    print(f"Couldn't process dataset. Error:{e}")