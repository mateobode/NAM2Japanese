import os
import evaluate
from functools import partial
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import EarlyStoppingCallback
from data_preprocessing import get_dataset, prepare_dataset
from model import get_model_and_processor, DataCollatorNAMSeq2SeqPadding


# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control
    

def compute_metrics(pred, tokenizer, wer_metric, cer_metric):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer, "cer": cer}


def main():
    # Load the dataset
    final_dataset = get_dataset()

    # Get the model and processor
    model, feature_extractor, tokenizer, processor = get_model_and_processor()

    # Prepare the dataset
    final_dataset = final_dataset.map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
        remove_columns=final_dataset.column_names["train"],
    )

    # Define the data collator
    data_collator = DataCollatorNAMSeq2SeqPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    compute_metrics_fn = partial(
            compute_metrics, 
            tokenizer=tokenizer, 
            wer_metric=wer_metric, 
            cer_metric=cer_metric
        )

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
        learning_rate=5e-5,
        warmup_ratio=0.1,
        num_train_epochs=20,
        lr_scheduler_type="cosine",
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=128,
        max_grad_norm= 1.0,
        save_steps=300,
        eval_steps=300,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        weight_decay=0.1,
        save_total_limit=5,
    )

    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=final_dataset["train"],
        eval_dataset=final_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        processing_class=processor.feature_extractor,
        callbacks=[
            SavePeftModelCallback,
            EarlyStoppingCallback(
                early_stopping_patience=7, 
                early_stopping_threshold=0.001
            )
        ],
    )
    model.config.use_cache = False  # Disable cache to avoid issues with gradient checkpointing

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
