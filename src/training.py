"""
Training utilities for LoRA and QLoRA experiments using Unsloth
Refactored to use Unsloth's optimized SFTTrainer
"""

import torch
from datasets import load_dataset
import time
import pandas as pd
import numpy as np
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from unsloth import is_bfloat16_supported


def prepare_alpaca_dataset(tokenizer, max_length=512, num_samples=None):
    """
    Load and prepare Alpaca instruction-following dataset
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        num_samples: Number of samples to use (None = all)
    
    Returns:
        train_dataset, eval_dataset
    """
    print("Loading Alpaca dataset...")
    
    # Load Alpaca dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    if num_samples:
        dataset = dataset.select(range(num_samples))
        print(f"Using {num_samples} samples for faster training")
    
    # Unsloth uses a specific prompt format
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    
    # Define EOS token
    EOS_TOKEN = tokenizer.eos_token
    
    def formatting_prompts_func(examples):
        """Format examples for Unsloth SFTTrainer"""
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            # Format with or without input
            if input_text:
                text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
            else:
                text = alpaca_prompt.format(instruction, "", output) + EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}
    
    # Apply formatting
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )
    
    # Split train/eval
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"✓ Dataset prepared:")
    print(f"  Training samples: {len(split_dataset['train'])}")
    print(f"  Evaluation samples: {len(split_dataset['test'])}")
    
    return split_dataset["train"], split_dataset["test"]


class MemoryTrackingCallback:
    """Callback to track memory usage during training"""
    
    def __init__(self):
        self.memory_log = []
        self.peak_memory = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """Track memory after each step"""
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, peak_mem)
            self.memory_log.append(peak_mem)
    
    def get_memory_stats(self):
        """Return memory statistics"""
        if not self.memory_log:
            return {}
        
        return {
            "peak_memory_mb": self.peak_memory,
            "mean_memory_mb": np.mean(self.memory_log),
            "std_memory_mb": np.std(self.memory_log)
        }


def train_model_unsloth(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir="./results",
    num_epochs=1,
    batch_size=4,
    learning_rate=2e-4,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    max_steps=200,
):
    """
    Train model using Unsloth's optimized SFTTrainer
    
    Args:
        model: PEFT model with LoRA/QLoRA
        tokenizer: Tokenizer
        train_dataset: Training dataset (formatted)
        eval_dataset: Evaluation dataset (formatted)
        output_dir: Directory to save results
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        logging_steps: Log every N steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        max_steps: Maximum training steps
    
    Returns:
        trainer, training_results
    """
    
    # Detect optimal settings
    fp16_supported = not is_bfloat16_supported()
    bf16_supported = is_bfloat16_supported()
    
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max steps: {max_steps}")
    print(f"Epochs: {num_epochs}")
    print(f"FP16: {fp16_supported}, BF16: {bf16_supported}")
    print(f"{'='*60}\n")
    
    # Training arguments optimized for Unsloth
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        max_steps=max_steps,
        fp16=fp16_supported,
        bf16=bf16_supported,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        optim="adamw_8bit",  # Unsloth optimization
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
    )
    
    # Memory tracking callback
    memory_callback = MemoryTrackingCallback()
    
    # Unsloth's SFTTrainer (optimized for instruction tuning)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
        packing=False,  # Can be True for efficiency
        callbacks=[memory_callback],
    )
    
    # Train
    print("Starting training...")
    start_time = time.time()
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    # Get memory stats
    memory_stats = memory_callback.get_memory_stats()
    
    # Compile results
    results = {
        "training_loss": train_result.training_loss,
        "training_time_seconds": training_time,
        "steps": train_result.global_step,
        "time_per_step": training_time / train_result.global_step if train_result.global_step > 0 else 0,
        **memory_stats
    }
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {training_time:.2f}s")
    print(f"Time per step: {results['time_per_step']:.3f}s")
    print(f"Peak memory: {memory_stats.get('peak_memory_mb', 0):.2f} MB")
    print(f"Final loss: {train_result.training_loss:.4f}")
    print(f"{'='*60}\n")
    
    return trainer, results


def run_experiment_unsloth(
    model_name="gpt2-medium",
    load_in_4bit=False,  # True for QLoRA, False for LoRA
    rank=8,
    num_samples=1000,
    max_steps=200,
    batch_size=4,
    learning_rate=2e-4,
    output_dir="./results",
):
    """
    Run a complete experiment using Unsloth
    
    Args:
        model_name: Base model to use
        load_in_4bit: True for QLoRA (4-bit), False for LoRA (16-bit)
        rank: LoRA rank
        num_samples: Number of training samples
        max_steps: Training steps
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Save directory
    
    Returns:
        Dictionary with experiment results
    """
    from model_utils_unsloth import load_gpt2_unsloth, setup_gpt2_lora, clear_memory
    
    clear_memory()
    
    quantization = "4bit" if load_in_4bit else "16bit"
    experiment_name = f"{quantization}_r{rank}"
    
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: {experiment_name} (Unsloth)")
    print(f"{'#'*70}\n")
    
    # Load model using Unsloth
    model, tokenizer = load_gpt2_unsloth(load_in_4bit=load_in_4bit)
    
    # Setup LoRA using Unsloth
    model = setup_gpt2_lora(model, rank=rank)
    
    # Prepare data
    train_dataset, eval_dataset = prepare_alpaca_dataset(
        tokenizer,
        num_samples=num_samples
    )
    
    # Train using Unsloth
    trainer, results = train_model_unsloth(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=f"{output_dir}/{experiment_name}",
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    # Save model
    model.save_pretrained(f"{output_dir}/{experiment_name}/final_model")
    
    # Add metadata
    results["experiment_name"] = experiment_name
    results["quantization"] = quantization
    results["rank"] = rank
    results["model_name"] = model_name
    results["num_samples"] = num_samples
    results["library"] = "unsloth"
    
    clear_memory()
    
    return results, model, tokenizer