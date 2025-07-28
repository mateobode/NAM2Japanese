import os
from datasets import Dataset, DatasetDict, Audio


def create_split_data(prefix_list, audio_dir, transcription_dir):
    data = {"audio": [], "sentence": []}
    for prefix in prefix_list:
        lab_path = os.path.join(transcription_dir, f"{prefix}_transcription.lab")
        audio_path = os.path.join(audio_dir, f"{prefix}_audio.wav")

        # Save the transcription string instead of the file path
        with open(lab_path, 'r', encoding='utf-8') as f:
            transcription_text = f.read().strip()
            
        data["audio"].append(audio_path)
        data["sentence"].append(transcription_text)
    return data


def get_dataset():
    data_dir = "data"
    audio_dir = os.path.join(data_dir, "nam")
    transcription_dir = os.path.join(data_dir, "transcription")

    all_transcriptions = os.listdir(transcription_dir)
    prefixes = sorted([f.replace("_transcription.lab", "") for f in all_transcriptions if f.endswith("_transcription.lab")])

    # Define split boundaries
    train_size = int(0.8 * len(prefixes))
    val_size = int(0.1 * len(prefixes))

    # Create lists of prefixes for each split P.S: I renamed the data as follows -> {nr}_audio.wav and {nr}_transcription.lab
    train_prefixes = prefixes[:train_size]
    val_prefixes = prefixes[train_size : train_size + val_size]
    test_prefixes = prefixes[train_size + val_size :]

    print(f"Total files: {len(prefixes)}")
    print(f"Train split size: {len(train_prefixes)}")
    print(f"Validation split size: {len(val_prefixes)}")
    print(f"Test split size: {len(test_prefixes)}")

    # Create the datasets for each split
    train_data = create_split_data(train_prefixes, audio_dir, transcription_dir)
    val_data = create_split_data(val_prefixes, audio_dir, transcription_dir)
    test_data = create_split_data(test_prefixes, audio_dir, transcription_dir)

    # Build the DatasetDict
    final_dataset = DatasetDict({
        "train": Dataset.from_dict(train_data),
        "validation": Dataset.from_dict(val_data),
        "test": Dataset.from_dict(test_data)
    })

    # Cast the audio column to 16kHz sampling rate for Whisper
    final_dataset = final_dataset.cast_column("audio", Audio(sampling_rate=16000))
    return final_dataset


def prepare_dataset(batch, feature_extractor, tokenizer):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
