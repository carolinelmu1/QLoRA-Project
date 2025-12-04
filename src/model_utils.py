"""
Model utilities for LoRA and QLoRA configuration using Unsloth
Refactored to use Unsloth library as recommended by TA
"""

import torch
import gc
from unsloth import FastLanguageModel


def get_model_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def clear_memory():
    """Clear GPU and system memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_unsloth_model(
    model_name="unsloth/mistral-7b",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
):
    """
    Load model using Unsloth's optimized loader
    
    Args:
        model_name: Model identifier (Unsloth supports many models)
        max_seq_length: Maximum sequence length
        load_in_4bit: If True, use QLoRA (4-bit). If False, use standard LoRA (16-bit)
        dtype: Data type (None = auto-detect)
    
    Returns:
        model, tokenizer
    """
    quantization_type = "4-bit QLoRA" if load_in_4bit else "16-bit LoRA"
    print(f"Loading {model_name} with {quantization_type} using Unsloth...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,  # Auto-detect, or use torch.float16, torch.bfloat16
        load_in_4bit=load_in_4bit,  # Use 4-bit quantization for QLoRA
        # trust_remote_code=True,  # Uncomment if needed
    )
    
    mem_usage = get_model_memory_usage()
    print(f"✓ Model loaded. Memory usage: {mem_usage:.2f} MB")
    
    return model, tokenizer


def setup_lora_unsloth(
    model,
    rank=8,
    target_modules=None,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
):
    """
    Setup LoRA/QLoRA using Unsloth's optimized PEFT configuration
    
    Args:
        model: Base model from load_unsloth_model
        rank: LoRA rank (r)
        target_modules: Which modules to adapt (None = auto-detect optimal)
        lora_alpha: Scaling parameter
        lora_dropout: Dropout for LoRA layers
        bias: Bias configuration
        use_gradient_checkpointing: Gradient checkpointing mode
        random_state: Random seed
    
    Returns:
        Model with LoRA adapters configured
    """
    
    # Unsloth auto-detects optimal target modules if None
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
    
    print(f"Configuring LoRA with Unsloth:")
    print(f"  Rank: {rank}")
    print(f"  LoRA alpha: {lora_alpha}")
    print(f"  Target modules: {target_modules}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=False,  # Rank stabilized LoRA
        loftq_config=None,  # LoftQ quantization
    )
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ LoRA configured:")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def print_model_architecture(model):
    """Print trainable vs frozen parameters by layer"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"✓ TRAINABLE: {name:50s} {param.numel():>12,}")
    
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print("="*60 + "\n")


# GPT-2 specific functions for compatibility
def load_gpt2_unsloth(load_in_4bit=False):
    """
    Load GPT-2 Medium using Unsloth
    
    Note: Unsloth is optimized for newer models. For GPT-2, we'll use
    a compatible configuration.
    
    Args:
        load_in_4bit: If True, QLoRA (4-bit). If False, standard LoRA (16-bit)
    
    Returns:
        model, tokenizer
    """
    # For GPT-2, we use HuggingFace model with Unsloth's optimization
    model_name = "gpt2-medium"  # 355M parameters
    
    print(f"Loading {model_name} ({'4-bit' if load_in_4bit else '16-bit'})...")
    
    try:
        # Try Unsloth's loader first
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1024,  # GPT-2 context length
            dtype=None,
            load_in_4bit=load_in_4bit,
        )
    except Exception as e:
        print(f"⚠️  Unsloth loader not available for GPT-2: {e}")
        print("Falling back to manual configuration...")
        
        # Fallback: Manual configuration
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        if load_in_4bit:
            # QLoRA: 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            # Standard LoRA: 16-bit
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
    
    mem_usage = get_model_memory_usage()
    print(f"✓ Model loaded. Memory usage: {mem_usage:.2f} MB")
    
    return model, tokenizer


def setup_gpt2_lora(model, rank=8, lora_alpha=16, lora_dropout=0.05):
    """
    Setup LoRA for GPT-2 using Unsloth or PEFT
    
    Args:
        model: GPT-2 model
        rank: LoRA rank
        lora_alpha: Scaling parameter
        lora_dropout: Dropout
    
    Returns:
        Model with LoRA adapters
    """
    # GPT-2 specific target modules
    target_modules = ["c_attn"]  # GPT-2 uses c_attn for Q,K,V
    
    try:
        # Try Unsloth's PEFT setup
        model = FastLanguageModel.get_peft_model(
            model,
            r=rank,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    except Exception as e:
        print(f"⚠️  Using standard PEFT: {e}")
        
        # Fallback: Standard PEFT
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        
        # Prepare for k-bit training if quantized
        if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ LoRA configured for GPT-2:")
    print(f"  Rank: {rank}")
    print(f"  Target modules: {target_modules}")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model