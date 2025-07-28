# NAM2Text: Japanese Speech Recognition with Whisper

A Japanese speech recognition system that fine-tunes OpenAI's Whisper model on Non Audible Mic (NAM) data recordings using Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters.

## Overview

This project adapts the pre-trained Whisper base model for Japanese speech transcription by fine-tuning it on a custom NAM dataset. Instead of full model fine-tuning, we use LoRA (Low-Rank Adaptation) to efficiently adapt the model while keeping most parameters frozen, making training faster and more memory-efficient.

## Key Features

- **Efficient Fine-tuning**: Uses LoRA adapters instead of full model fine-tuning
- **Japanese Language Support**: Configured specifically for Japanese transcription tasks
- **Complete Pipeline**: Training, inference, evaluation, and visualization tools
- **Robust Evaluation**: WER (Word Error Rate) and CER (Character Error Rate) metrics
- **Checkpoint Management**: Saves only adapter weights, not full model checkpoints

## Project Structure

```
NAM2Text/
├── src/                          # Source code
│   ├── train.py                  # Training script
│   ├── model.py                  # Model setup and data collator
│   ├── data_preprocessing.py     # Dataset loading and preprocessing
│   ├── inference.py              # Inference script
│   ├── evaluation.py             # Evaluation with metrics
│   └── plot_training_data.py     # Training visualization
├── data/                         # Dataset (not tracked in git)
│   ├── nam/                      # Audio files ({id}_audio.wav)
│   └── transcription/            # Transcription files ({id}_transcription.lab)
├── output/                       # Training checkpoints
│   └── checkpoint-{step}/        # Model checkpoints
│       └── adapter_model/        # LoRA adapter weights
├── logs/                         # Evaluation and inference logs
├── train_data/                   # Training metrics (CSV files)
├── plots/                        # Training visualization plots
├── exp.ipynb                     # Experiment notebook for audio files
└── pipeline.ipynb                # Complete training pipeline
```

## Setup and Installation

### Prerequisites

```bash
# Core dependencies
pip install torch transformers datasets gradio evaluate
pip install peft accelerate tensorboard
pip install librosa scipy matplotlib tqdm
```

### Data Preparation

The dataset should be organized as follows:

```
data/
├── nam/
│   ├── 1_audio.wav
│   ├── 2_audio.wav
│   └── ...
└── transcription/
    ├── 1_transcription.lab
    ├── 2_transcription.lab
    └── ...
```

- **Audio files**: 48kHz WAV format, named `{id}_audio.wav`
- **Transcription files**: UTF-8 encoded text files, named `{id}_transcription.lab`
- **Data split**: Automatically splits into 80% train, 10% validation, 10% test

## Usage

### 1. Training

Train the model with LoRA fine-tuning:

```bash
python src/train.py
```

**Training Configuration:**
- **Epochs**: 30
- **Batch Size**: 4 (with gradient accumulation steps of 4)
- **Learning Rate**: 2e-4 with cosine scheduler
- **LoRA Config**: r=64, alpha=128, targeting embeddings layers, q_proj, v_proj, k_proj, o_proj of WhisperAttention layers alongside fc1, fc2 Linear layers.
- **EVA Config**: rho=0.95, num_singular_values=64, eva_gamma=0.8
- **Checkpoints**: Saved every 500 steps in `output/checkpoint-{step}/`

### 2. Inference

Run inference using gradio for near real-time performance:

```bash
python src/inference.py 
```

### 3. Evaluation

Evaluate model performance with WER/CER metrics:

```bash
python src/evaluation.py --checkpoint_path output/checkpoint-{step}/adapter_model --split test --log_dir logs
```

**Metrics Computed:**
- **WER (Word Error Rate)**: Measures word-level transcription accuracy
- **CER (Character Error Rate)**: Measures character-level transcription accuracy

### 4. Visualization

Plot training progress and metrics:

```bash
python src/plot_training_data.py
```

Generates plots for:
- Training and validation loss curves
- WER progress over training
- Learning rate schedule
- Gradient norms

## Technical Details

### Model Architecture

- **Base Model**: OpenAI Whisper Large V3
- **Fine-tuning Method**: LoRA adapters
- **Target Modules**: All layers in attention mechanism, embeddings and Linear layers
- **Language Configuration**: Japanese transcription task

### Training Process

1. **Data Loading**: Loads audio-transcription pairs and creates the dataset
2. **Preprocessing**: Converts audio to 16Khz and creates log-mel spectrograms (80 mel bins)
3. **Tokenization**: Uses Whisper's Japanese tokenizer
4. **Training**: Seq2SeqTrainer with custom PEFT callback for adapter saving
5. **Evaluation**: Periodic evaluation on validation set with WER metric

### Key Implementation Details

- **Audio Processing**: 16kHz sampling rate, log-mel spectrogram features
- **Sequence Length**: Maximum generation length of 128 tokens
- **Mixed Precision**: FP16 training for memory efficiency
- **Gradient Clipping**: Max gradient norm of 0.5
- **Best Model Selection**: Based on lowest validation CER

## Expected Outcomes

### Training Results
- **Training Loss**: Progressive decrease over 10 epochs
- **Validation WER**: Improvement in word error rate on validation set
- **Validation CER**: Improvement in characterr error rate on validation set
- **Checkpoints**: Saved every 500 steps with best model selection

### Evaluation Results
- **Evaluation Files**: Generated evaluations saved in logs directory
- **Performance Metrics**: WER and CER scores on test set
- **Comparison**: Side-by-side comparison of predicted vs. real transcriptions

### Visualization Outputs
- **Loss Curves**: Training and validation loss over time
- **WER Progress**: Word error rate improvement during training
- **Learning Rate Schedule**: Cosine learning rate decay visualization
- **Training Overview**: Comprehensive training metrics dashboard

## Performance Expectations

With the LoRA fine-tuning approach:
- **Training Time**: Significantly reduced compared to full fine-tuning
- **Memory Usage**: Lower GPU memory requirements due to frozen base model
- **Model Size**: Only adapter weights need to be saved and loaded
- **Inference Speed**: Similar to base Whisper model performance

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient checkpointing

### Monitoring Training

- **TensorBoard**: Logs saved to `output/runs/` for monitoring
- **Console Output**: Progress bars and metrics printed during training
- **CSV Files**: Download from TensorBoard and save to `train_data/` directory
