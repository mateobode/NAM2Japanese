import torch
import argparse
import os
from tqdm import tqdm
from peft import PeftModel
from data_preprocessing import get_dataset
from model import get_base_model_and_processor_for_inference

def run_inference(args):
    print("Loading base model and processor...")
    model, processor = get_base_model_and_processor_for_inference()
    
    print(f"Loading fine-tuned model from checkpoint: {args.checkpoint_path}")
    model = PeftModel.from_pretrained(model, args.checkpoint_path)
    model.merge_and_unload()
    model.eval()

    print("Model loaded successfully.")

    print("Loading dataset...")
    dataset = get_dataset()
    test_data = dataset[args.split]
    
    print(f"Found {len(test_data)} samples to transcribe from {args.split} split.")

    os.makedirs(args.log_dir, exist_ok=True)
    output_file_path = os.path.join(args.log_dir, f"inference_results_{args.split}.txt")

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write("INFERENCE RESULTS\n")
        output_file.write("=" * 80 + "\n")
        output_file.write(f"Dataset Split: {args.split}\n")
        output_file.write(f"Checkpoint: {args.checkpoint_path}\n")
        output_file.write("=" * 80 + "\n\n")
        
        for i, sample in enumerate(tqdm(test_data, desc="Transcribing samples")):
            try:
                audio_data = sample["audio"]
                real_transcription = sample["sentence"]
                audio_filename = os.path.basename(audio_data["path"]) if "path" in audio_data else f"sample_{i}"
                
                inputs = processor(
                    audio_data["array"], 
                    sampling_rate=audio_data["sampling_rate"], 
                    return_tensors="pt",
                    return_attention_mask=True
                )
                
                input_features = inputs.input_features.to(model.device)
                attention_mask = inputs.attention_mask.to(model.device)

                generated_ids = model.generate(
                    input_features=input_features, 
                    attention_mask=attention_mask,
                    max_length=128
                )
                
                inference_transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                output_file.write(f"Audio File: {audio_filename}\n")
                output_file.write(f"Real:       {real_transcription}\n")
                output_file.write(f"Inference:  {inference_transcription}\n")
                output_file.write("-" * 80 + "\n\n")
                
            except Exception as e:
                print(f"\nCould not process sample {i}. Error: {e}")
                continue

    print(f"\nInference complete. Results saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on dataset with real vs inference transcription comparison.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to use for inference (default: test).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="output/checkpoint-5000/adapter_model",
        help="Path to the fine-tuned adapter model checkpoint.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save the inference results.",
    )
    args = parser.parse_args()
    run_inference(args)
