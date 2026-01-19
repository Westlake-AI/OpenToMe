# MergeNet NLP

**MergeNet**: A Hierarchical Hybrid Transformer with Differentiable Tokenization for byte-level language modeling.

## Architecture Overview

MergeNet consists of four key components:

```
Input Bytes → [LoT] → [LoE] → [LaM] → [LoD] → Output Logits
              (H)     (Z)     (O)     (H')
```

### 1. **Shared Local Transformer (LoT)**
- Extracts local context from byte sequences
- Standard Transformer with causal attention + RoPE
- Output: Local context representations **H** (B, L, d)

### 2. **Local Encoder (LoE)** 
- Differentiable soft tokenization using DTEM
- Performs soft token merging with source matrix tracking
- Compresses L tokens to ~L/λ tokens
- Random partitioning + weighted soft merge
- Output: Latent words **Z** (B, N, d) where N ≈ L/λ

### 3. **Latent Model (LaM)**
- Pure GPT-style transformer (no ToMe)
- Operates on compressed latent word sequences
- Standard causal self-attention
- Output: Predicted latent representations **O** (B, N, d)

### 4. **Local Decoder (LoD)**
- Decodes latent words back to byte level
- Cross-attention with band mask + grid bias
- Query: Original H from LoT
- Key/Value: O from LaM
- Output: Byte-level representations **H'** (B, L, d)

## Key Features

- **No Fixed Patching**: Unlike MegaByte/BLT, MergeNet learns dynamic tokenization via soft merging
- **Source Matrix Tracking**: Maintains differentiable token provenance through compression
- **Band Mask Decoding**: Efficient cross-attention with local window constraints
- **Perceiver Refinement**: TopK selection + cross-attention for better latent representations
- **Two-Phase Training**: Phase 1 (reconstruction) → Phase 2 (prediction)

## Configuration

```python
from opentome.models.mergenet_nlp import MergeNetConfig, MergeNetForCausalLM

config = MergeNetConfig(
    vocab_size=320,           # 256 bytes + offset
    hidden_size=768,          # Model dimension
    num_local_layers=4,       # LoT layers (no DTEM)
    num_encoder_layers=4,     # LoE DTEM layers (local_depth)
    num_latent_layers=8,      # LaM layers
    num_heads=12,
    lambda_local=4.0,         # Compression ratio
    dtem_window_size=16,      # DTEM local window
    max_position_embeddings=2048,
    phase="phase2",           # Training phase
)

model = MergeNetForCausalLM(config)
```

## Usage

### Training

```python
import torch
from opentome.models.mergenet_nlp import MergeNetForCausalLM

model = MergeNetForCausalLM.from_pretrained("path/to/checkpoint")
model.train()

# Prepare data
input_ids = torch.randint(0, 320, (batch_size, seq_len))
labels = input_ids.clone()

# Forward pass
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
loss.backward()
```

### Generation

```python
model.eval()

# Prepare prompt
prompt_ids = torch.tensor([[1, 72, 101, 108, 108, 111]])  # Example: "Hello"

# Generate with efficient sliding window
generated = model.generate(
    input_ids=prompt_ids,
    max_length=256,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    use_sliding_window=True,  # Default: efficient generation
)
```

**Sliding Window Mechanism:**
- **Key insight**: LoE is just a tokenizer (bytes→latents), only used once at initialization!
- **Initialization**: Prompt → LoT → LoE (tokenizer) → LaM → O_queue
- **Per step**: New byte → LoT → H, then H (query) + O_queue (KV) → LocalDecoder → Prediction
- **Every λ steps**: LaM autoregressively generates next latent word, O_queue slides (FIFO)
- **Complexity**: O(L·d² + N·d²/λ) per step vs O(L²·d²) full recompute (~4x faster when λ=4)

## Model Sizes

| Model | Hidden | Layers (LoT+LoE+LaM) | Params | Description |
|-------|--------|----------------------|--------|-------------|
| Small | 384 | 4+4+8 | ~50M | Fast, lightweight |
| Base | 768 | 4+4+8 | ~200M | Balanced performance |
| Large | 1024 | 6+6+12 | ~500M | High capacity |

## Implementation Details

### Generation Algorithm Details

The sliding window generation leverages the hierarchical nature of MergeNet:

**Stage 1: Initialization (once)**
```
input_bytes → LoT → H_buffer
H_buffer → LoE (differentiable tokenizer) → Z_merged
Z_merged → LaM (GPT) → O_queue (N latent words)
```

**Stage 2: Byte-by-byte generation loop**
```
For each new byte:
  1. Update byte_buffer (sliding window, size L)
  2. LoT(byte_buffer) → H_buffer
  3. H_last + O_queue → LocalDecoder → next_byte_logits
  4. Sample next byte
  
  Every λ bytes:
    5. LaM(O_queue) → O_new  (autoregressive generation)
    6. O_queue = O_queue[1:] + O_new[-1:]  (sliding window)
```

**Why this works:**
- LoE acts as a learned tokenizer: bytes → latent words (differentiable compression)
- Once initialized, we don't need to re-tokenize; LaM generates new latent words directly
- LaM is a pure GPT decoder: given N latent words, it predicts the (N+1)-th latent word
- LocalDecoder bridges latent words (semantic) to bytes (surface form)

**Complexity benefits:**
- Without sliding window: O(L² · d²) per byte
- With sliding window: O((L + N) · d²) ≈ O(L · d²) per byte (since N = L/λ << L)
- Speedup factor: ~λ (4x when λ=4)

### Source Matrix Tracking
- Shape: (B, N, width) where width = 2 × window_size × local_depth + 1
- `local_depth` = `num_encoder_layers` (number of DTEM blocks in LoE)
- Sparse band matrix storing byte-to-token provenance
- Used for Perceiver bias: `log(source_matrix)`

### Band Mask Formula
For byte position t attending to latent position j:
- Grid bias: `-γ × |t/λ - j|`
- Causal constraint: `j ≤ floor(t/λ)`
- Window constraint: `|j - floor(t/λ)| ≤ W_infer`

### TopK Selection
1. Compute center of mass for each merged token
2. Select top-k by token size (strength)
3. Sort by center of mass (spatial order)
4. Refine via Perceiver cross-attention

## Files

- `configuration_mergenet.py`: Model configuration
- `model.py`: Main model implementation
  - `SharedLocalTransformer`: LoT module
  - `LocalEncoderNLP`: LoE module with DTEM
  - `LatentModel`: LaM module (pure GPT)
  - `LocalDecoder`: LoD module with band mask
  - `MergeNetForCausalLM`: Complete language model
- `example.py`: Usage examples
- `__init__.py`: Package exports

## Dependencies

- `torch >= 2.0`
- `transformers >= 4.30`
- `fla` (Flash Linear Attention)
- `opentome.timm.dtem` (DTEM implementation)

## Training Tips

1. **Two-Phase Training**:
   - Phase 1: Set `phase="phase1"` for reconstruction
   - Phase 2: Set `phase="phase2"` for prediction
   
2. **Hyperparameters**:
   - Start with `lambda_local=4.0` (4x compression)
   - `dtem_window_size=16` for local merging
   - `W_infer` is auto-computed from lambda and window size

3. **Memory Optimization**:
   - Use gradient checkpointing: `model.gradient_checkpointing_enable()`
   - Reduce `lambda_local` for lower compression (more memory)
   - Enable Flash Attention if available

## Citation

```bibtex
@article{mergenet2024,
  title={MergeNet: A Hierarchical Hybrid Transformer with Differentiable Tokenization},
  author={[Authors]},
  year={2024}
}
```

## Differences from CV Version

| Aspect | CV (Image) | NLP (Text) |
|--------|------------|------------|
| Input | Patches | Bytes |
| Position Encoding | 2D absolute | 1D RoPE |
| CLS Token | Yes (preserved) | No |
| LoE Attention | Can see future in Phase 1 | Always causal |
| LaM | Had ToMe option | Pure GPT (no ToMe) |
| Output | Classification logits | LM logits |

## Known Limitations

1. **Memory**: Source matrix tracking adds ~L × width overhead
   - Mitigated by sparse band structure
   
2. **Context Length**: Limited by max_position_embeddings (2048 default)
   - Can be extended with proper position encoding

## Future Work

- [x] Optimize generation with proper sliding window queue ✅ (Implemented)
- [ ] Support for KV caching in LoT for further speedup
- [ ] Flash Attention integration in all modules
- [ ] Multi-GPU training recipes
- [ ] Pretrained checkpoints

---

**Status**: Initial implementation complete ✅

For questions or issues, please refer to the main repository.

