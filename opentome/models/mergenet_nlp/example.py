"""
Example usage of MergeNet NLP model.

This script demonstrates how to:
1. Create a MergeNet model
2. Forward pass for training
3. Generate text
"""

import torch
from opentome.models.mergenet_nlp import MergeNetConfig, MergeNetForCausalLM


def create_model_example():
    """Create a small MergeNet model for testing."""
    config = MergeNetConfig(
        vocab_size=320,  # 256 bytes + 64 offset
        hidden_size=384,  # Small model
        num_local_layers=4,  # LoT layers
        num_encoder_layers=4,  # LoE DTEM layers (local_depth)
        num_latent_layers=8,  # LaM layers
        num_heads=6,
        num_kv_heads=6,
        intermediate_size=1536,
        max_position_embeddings=2048,
        lambda_local=4.0,
        dtem_window_size=4,
        dtem_t=1,
        use_softkmax=False,
        phase="phase2",  # Prediction mode
    )
    
    model = MergeNetForCausalLM(config)
    return model, config


def training_example():
    """Example of training forward pass."""
    print("=" * 60)
    print("Training Example")
    print("=" * 60)
    
    model, config = create_model_example()
    model.eval()  # For this example
    
    # Move to GPU if available and convert to bfloat16 (FlashAttention requirement)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using device: {device}, dtype: {dtype}")
    model = model.to(device=device, dtype=dtype)
    
    # Create dummy input (batch_size=2, seq_len=128)
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
    print()


def generation_example():
    """Example of text generation."""
    print("=" * 60)
    print("Generation Example")
    print("=" * 60)
    
    model, config = create_model_example()
    model.eval()
    
    # Move to GPU if available and convert to bfloat16 (FlashAttention requirement)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using device: {device}, dtype: {dtype}")
    model = model.to(device=device, dtype=dtype)
    
    # Create prompt
    prompt_text = "Hello, world!"
    # Simulate byte tokenization (replace with actual BltTokenizer in practice)
    prompt_bytes = bytes(prompt_text, encoding="utf-8")
    # Add offset (assuming offset=64 like BLT)
    offset = 64
    prompt_ids = [b + offset for b in prompt_bytes]
    # Add BOS token
    bos_id = config.bos_token_id
    prompt_ids = [bos_id] + prompt_ids
    
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    print(f"Prompt: '{prompt_text}'")
    print(f"Prompt IDs: {input_ids.shape}")
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_length=min(input_ids.shape[1] + 50, 256),  # Generate 50 more tokens
            temperature=0.8,
            top_p=0.9,
            top_k=50,
        )
    
    print(f"Generated IDs shape: {generated.shape}")
    print(f"Generated: {generated[0, :20].tolist()}")  # Show first 20 tokens
    print()


def model_architecture_summary():
    """Print model architecture summary."""
    print("=" * 60)
    print("Model Architecture Summary")
    print("=" * 60)
    
    model, config = create_model_example()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Configuration:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Local layers (LoT): {config.num_local_layers}")
    print(f"  - Encoder layers (LoE): {config.num_encoder_layers}")
    print(f"  - Latent layers (LaM): {config.num_latent_layers}")
    print(f"  - Compression ratio (lambda): {config.lambda_local}")
    print(f"  - DTEM window size: {config.dtem_window_size}")
    print()
    print(f"Model Components:")
    print(f"  1. Shared Local Transformer (LoT)")
    print(f"     - Extracts local byte-level context (no DTEM)")
    print(f"     - Layers: {config.num_local_layers}")
    print(f"  2. Local Encoder (LoE)")
    print(f"     - Soft token merging with DTEM")
    print(f"     - DTEM layers: {config.num_encoder_layers} (local_depth)")
    print(f"     - Compresses L tokens to ~L/{config.lambda_local} tokens")
    print(f"  3. Latent Model (LaM)")
    print(f"     - Pure GPT transformer on latent words")
    print(f"     - Layers: {config.num_latent_layers}")
    print(f"  4. Local Decoder (LoD)")
    print(f"     - Decodes latent words back to bytes")
    print(f"     - Uses cross-attention with band mask")
    print()
    print(f"Parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    print(f"  - Size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    print()


if __name__ == "__main__":
    # Run examples
    model_architecture_summary()
    # training_example()
    generation_example()
    
    print("=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)

