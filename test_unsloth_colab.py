"""
Unsloth Quick Test Script for Google Colab
Run this to verify Unsloth works before running full experiments
"""

# ============================================================================
# UNSLOTH QUICK TEST - Run this in Google Colab
# ============================================================================

print("="*70)
print(" "*20 + "UNSLOTH QUICK TEST")
print("="*70)

# Step 1: Check GPU
print("\n[1/6] Checking GPU...")
import torch
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("❌ No GPU detected! Go to Runtime → Change runtime type → GPU")
    exit(1)

# Step 2: Install Unsloth
print("\n[2/6] Installing Unsloth (this may take 2-3 minutes)...")
import subprocess
import sys

try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    ])
    print("✓ Unsloth installed successfully!")
except Exception as e:
    print(f"❌ Error installing Unsloth: {e}")
    exit(1)

# Step 3: Import Unsloth
print("\n[3/6] Importing Unsloth...")
try:
    from unsloth import FastLanguageModel
    print("✓ Unsloth imported successfully!")
except Exception as e:
    print(f"❌ Error importing Unsloth: {e}")
    exit(1)

# Step 4: Load model with 16-bit (LoRA)
print("\n[4/6] Testing 16-bit LoRA...")
try:
    model_16bit, tokenizer = FastLanguageModel.from_pretrained(
        model_name="gpt2",  # Use small GPT-2 for quick test
        max_seq_length=512,
        dtype=None,
        load_in_4bit=False,  # 16-bit
    )
    print("✓ 16-bit model loaded!")
    
    # Setup LoRA
    model_16bit = FastLanguageModel.get_peft_model(
        model_16bit,
        r=8,
        target_modules=["c_attn"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    print("✓ LoRA configured!")
    
    # Count parameters
    trainable = sum(p.numel() for p in model_16bit.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_16bit.parameters())
    print(f"✓ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Clean up
    del model_16bit
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"❌ Error with 16-bit model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Load model with 4-bit (QLoRA)
print("\n[5/6] Testing 4-bit QLoRA...")
try:
    model_4bit, tokenizer = FastLanguageModel.from_pretrained(
        model_name="gpt2",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,  # 4-bit NF4
    )
    print("✓ 4-bit quantized model loaded!")
    
    # Setup QLoRA
    model_4bit = FastLanguageModel.get_peft_model(
        model_4bit,
        r=8,
        target_modules=["c_attn"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    print("✓ QLoRA configured!")
    
    # Count parameters
    trainable = sum(p.numel() for p in model_4bit.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_4bit.parameters())
    print(f"✓ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Clean up
    del model_4bit
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"❌ Error with 4-bit model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 6: Quick inference test
print("\n[6/6] Testing inference...")
try:
    # Reload a model for quick test
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="gpt2",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["c_attn"],
        lora_alpha=16,
    )
    
    # Enable fast inference
    FastLanguageModel.for_inference(model)
    
    # Test generation
    inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=20, do_sample=False)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"✓ Generated text: '{generated_text}'")
    
    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"❌ Error during inference: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Final summary
print("\n" + "="*70)
print(" "*20 + "✅ ALL TESTS PASSED!")
print("="*70)
print("\n📊 Summary:")
print("  ✓ GPU detected and working")
print("  ✓ Unsloth installed correctly")
print("  ✓ 16-bit LoRA works")
print("  ✓ 4-bit QLoRA works")
print("  ✓ Inference works")
print("\n🚀 You're ready to run the full experiments!")
print("\nNext steps:")
print("  1. Upload your project files to Colab")
print("  2. Run notebooks/01_baseline_lora_unsloth.ipynb")
print("  3. Run notebooks/02_qlora_implementation_unsloth.ipynb")
print("  4. Run notebooks/03_diagnostic_analysis_unsloth.ipynb")
print("\n" + "="*70)