import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor


@dataclass
class DataCollatorNAMSeq2SeqPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", return_attention_mask=True)

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def get_model_and_processor():
    base_model_id = "openai/whisper-small"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model_id, language="ja", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(base_model_id, language="ja", task="transcribe")
    processor = WhisperProcessor.from_pretrained(base_model_id, language="ja", task="transcribe", return_attention_mask=True)

    model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    model.generation_config.language = "japanese"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Freezing the base model layers to not update them during training
    model = prepare_model_for_kbit_training(model)
    
    for param in model.get_encoder().conv1.parameters():
        param.requires_grad = True
    for param in model.get_encoder().conv2.parameters():
        param.requires_grad = True

    #for param in model.get_encoder().embed_positions.parameters():
    #    param.requires_grad = True

    #def make_input_require_grad(module, input, output):
    #    output.requires_grad_(True)
    #
    #model.get_encoder().conv1.register_forward_hook(make_input_require_grad)
    #model.get_encoder().embed_positions.weight.requires_grad = True

    # Configuring LoRA for the model
    config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", 'k_proj', "o_proj", "fc1", "fc2", "encoder.embed_positions"],
        lora_dropout=0.1,
        bias="none",
        layers_to_transform=[0, 1, 2, 3, 4, 5],
        layers_pattern="encoder.layers",
        use_rslora= True,
        init_lora_weights="eva",
        eva_config={
            "rho": 0.6,
            "num_singular_values": 64,
            "eva_gamma": 0.8
        }
    )

    model = get_peft_model(model, config)

    return model, feature_extractor, tokenizer, processor


def get_base_model_and_processor_for_inference():
    """Load base model and processor for inference without training setup."""
    base_model_id = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(base_model_id, language="ja", task="transcribe", return_attention_mask=True)
    model = WhisperForConditionalGeneration.from_pretrained(base_model_id, device_map="auto")
    
    # Set generation config for Japanese transcription
    model.generation_config.language = "japanese"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="japanese", task="transcribe")
    
    return model, processor
