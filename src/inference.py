import torch
import gradio as gr
import librosa
import numpy as np

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from pathlib import Path


model_path = "model/checkpoint-11000"
device = "cuda" if torch.cuda.is_available() else "mps"

print(f"Loading model from: {model_path}")
print(f"Using device: {device}")

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-large-v3",
    language="ja",
    task="transcribe",
    return_attention_mask=True
)

print("Loading base model...")
base_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3",
    torch_dtype=torch.float32
)

print("Loading LoRA adapters...")
model_with_lora = PeftModel.from_pretrained(base_model, model_path)

print("Merging LoRA adapters...")
model = model_with_lora.merge_and_unload()
model.to(device)
model.eval()

print("Model loaded and ready!")


def transcribe(audio):
    if audio is None:
        return "Please provide an audio file."
    
    try:
        audio_data, sr = librosa.load(audio, sr=16000, mono=True)
        
        # Simple normalization
        if np.abs(audio_data).max() > 0:
            audio_data = audio_data / np.abs(audio_data).max() * 0.95
        
        inputs = processor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        input_features = inputs.input_features.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                language="ja",
                task="transcribe",
                num_beams=5,
                temperature=0.0,
                max_length=448,
                attention_mask=attention_mask
            )
        
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return text if text else "No speech detected."
        
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="NAM Japanese Transcription") as iface:
    gr.Markdown("# NAM Japanese Transcription")
    gr.Markdown("NAM to Japanese text using fine-tuned Whisper Large v3.")
    
    with gr.Tabs():
        with gr.TabItem("Record & Transcribe"):
            gr.Markdown("### Record with NAM Mic and get transcription")
            
            record_input = gr.Audio(
                sources=["microphone"], 
                type="filepath", 
                label="Record with NAM Mic"
            )
            record_output = gr.Textbox(
                label="Transcription", 
                lines=3,
                placeholder="Transcription will appear here..."
            )
            
            record_input.change(
                fn=transcribe,
                inputs=[record_input],
                outputs=[record_output]
            )
        
        with gr.TabItem("Upload File"):
            gr.Markdown("### Upload a pre-recorded NAM audio file")
            
            file_input = gr.Audio(
                sources=["upload"], 
                type="filepath", 
                label="Upload NAM audio file"
            )
            file_output = gr.Textbox(
                label="Transcription", 
                lines=3,
                placeholder="Transcription will appear here..."
            )
            
            file_input.change(
                fn=transcribe,
                inputs=[file_input],
                outputs=[file_output]
            )

iface.launch() 