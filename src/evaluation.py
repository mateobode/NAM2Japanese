import torch
import evaluate
import argparse
import os
from tqdm import tqdm
from peft import PeftModel
from prepare_dataset import get_dataset
from model import get_base_model_and_processor_for_inference

def main(args):
    device = "cuda:3"
    print(f"Using device: {device}")

    print("Loading base model and processor...")
    model, processor = get_base_model_and_processor_for_inference(device_map=device)
    
    print(f"Loading fine-tuned model from checkpoint: {args.checkpoint_path}")
    model = PeftModel.from_pretrained(model, args.checkpoint_path)
    model.merge_and_unload()
    model.eval()  # Set the model to evaluation mode

    print("Model loaded successfully.")

    print(f"Loading {args.split} dataset...")
    full_dataset = get_dataset()
    eval_dataset = full_dataset[args.split]

    print(f"Running evaluation on the {args.split} set...")
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    os.makedirs(args.log_dir, exist_ok=True)
    output_file_path = os.path.join(args.log_dir, f"evaluation_{args.split}.txt")

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write("EVALUATION RESULTS\n")
        output_file.write("=" * 80 + "\n")
        output_file.write(f"Dataset Split: {args.split}\n")
        output_file.write(f"Checkpoint: {args.checkpoint_path}\n")
        output_file.write("=" * 80 + "\n\n")
        
        for i, item in enumerate(tqdm(eval_dataset, desc="Evaluating samples")):
            audio_sample = item["audio"]
            reference_text = item["sentence"]
            audio_filename = os.path.basename(audio_sample["path"]) if "path" in audio_sample else f"sample_{i}"

            inputs = processor(
                audio_sample["array"], 
                sampling_rate=audio_sample["sampling_rate"], 
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
            
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            wer_metric.add(predictions=transcription, references=reference_text)
            cer_metric.add(predictions=transcription, references=reference_text)
            
            individual_cer = evaluate.load("cer").compute(predictions=[transcription], references=[reference_text])
            
            output_file.write(f"Audio File: {audio_filename}\n")
            output_file.write(f"Reference:  {reference_text}\n")
            output_file.write(f"Prediction: {transcription}\n")
            output_file.write(f"CER:        {individual_cer * 100:.2f}%\n")
            output_file.write("-" * 80 + "\n\n")

    final_wer = wer_metric.compute()
    final_cer = cer_metric.compute()
    print(f"\nFinal {args.split.capitalize()} WER: {final_wer * 100:.2f}%")
    print(f"Final {args.split.capitalize()} CER: {final_cer * 100:.2f}%")
    
    with open(output_file_path, "a", encoding="utf-8") as output_file:
        output_file.write(f"\n" + "=" * 80 + "\n")
        output_file.write(f"FINAL {args.split.upper()} WER: {final_wer * 100:.2f}%\n")
        output_file.write(f"FINAL {args.split.upper()} CER: {final_cer * 100:.2f}%\n")
        output_file.write(f"=" * 80 + "\n")

    print(f"Evaluation complete. Results saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Whisper model.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to use for evaluation (default: test).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="output/checkpoint-14500/adapter_model",
        help="Path to the fine-tuned adapter model checkpoint.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save the evaluation results.",
    )
    args = parser.parse_args()
    main(args)
