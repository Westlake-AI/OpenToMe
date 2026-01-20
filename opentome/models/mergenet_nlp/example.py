"""Minimal examples for MergeNet NLP checks."""

import torch
from opentome.models.mergenet_nlp import MergeNetConfig, MergeNetForCausalLM


def create_model_example():
    """Create a small MergeNet model for testing."""
    config = MergeNetConfig(
        vocab_size=320,  # 256 bytes + 64 offset
        hidden_size=384,  # Small model
        num_local_layers=4,  # LoT layers
        local_depth=4,  # LoE DTEM layers
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


if False:
    def training_example():
        model, config = create_model_example()
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        model = model.to(device=device, dtype=dtype)
        batch_size = 2
        seq_len = 20
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
        print(outputs.loss.item(), outputs.logits.shape)


def generation_example():
    print("=" * 60)
    print("Generation Example")
    print("=" * 60)
    
    model, config = create_model_example()
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using device: {device}, dtype: {dtype}")
    model = model.to(device=device, dtype=dtype)
    
    prompt_text = "Hello, world!"
    prompt_bytes = bytes(prompt_text, encoding="utf-8")
    offset = 64
    prompt_ids = [b + offset for b in prompt_bytes]
    bos_id = config.bos_token_id
    prompt_ids = [bos_id] + prompt_ids
    
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    print(f"Prompt: '{prompt_text}'")
    print(f"Prompt IDs: {input_ids.shape}")
    
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_length=min(input_ids.shape[1] + 50, 256),
            temperature=0.8,
            top_p=0.9,
            top_k=50,
        )
    
    print(f"Generated IDs shape: {generated.shape}")
    print(f"Generated: {generated[0, :20].tolist()}")
    print()


if False:
    def model_architecture_summary():
        model, config = create_model_example()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(total_params, trainable_params)


def causal_mask_sanity_check():
    print("=" * 60)
    print("Causal Mask Sanity Check")
    print("=" * 60)

    model, config = create_model_example()
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    print(f"Using device: {device}, dtype: {dtype}")
    model = model.to(device=device, dtype=dtype)

    torch.manual_seed(0)
    batch_size = 1
    seq_len = 24
    prefix_len = 12

    base_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    alt_ids = base_ids.clone()
    alt_ids[:, prefix_len:] = torch.randint(
        0, config.vocab_size, (batch_size, seq_len - prefix_len), device=device
    )

    with torch.no_grad():
        logits_base = model(input_ids=base_ids).logits
        logits_alt = model(input_ids=alt_ids).logits

    compare_len = max(prefix_len - 1, 1)
    diff = (logits_base[:, :compare_len, :] - logits_alt[:, :compare_len, :]).abs().max().item()
    tol = 1e-3 if dtype == torch.float32 else 1e-2
    print(f"Max abs diff (prefix logits): {diff:.6f} (tol={tol})")
    print("PASS" if diff <= tol else "WARN: possible non-causal influence")

    with torch.no_grad():
        h_base = model.model.shared_local_transformer(base_ids, attention_mask=None)
        h_alt = model.model.shared_local_transformer(alt_ids, attention_mask=None)
    h_diff = (h_base[:, :compare_len, :] - h_alt[:, :compare_len, :]).abs().max().item()
    h_tol = 1e-4 if dtype == torch.float32 else 5e-3
    print(f"Max abs diff (LoT prefix): {h_diff:.6f} (tol={h_tol})")
    if diff > tol and h_diff <= h_tol:
        print("Hint: leakage likely introduced after LoT (e.g., LoE/LoD/LaM).")

    with torch.no_grad():
        z_base, _, _, _ = model.model.local_encoder(h_base, phase=model.config.phase)
        z_alt, _, _, _ = model.model.local_encoder(h_alt, phase=model.config.phase)
    z_finite = torch.isfinite(z_base).all() and torch.isfinite(z_alt).all()
    if not z_finite:
        nan_base = torch.isnan(z_base).float().mean().item()
        nan_alt = torch.isnan(z_alt).float().mean().item()
        inf_base = torch.isinf(z_base).float().mean().item()
        inf_alt = torch.isinf(z_alt).float().mean().item()
        print(f"LoE merged tokens contain non-finite values.")
        print(f"  - z_base nan%={nan_base:.4f}, inf%={inf_base:.4f}")
        print(f"  - z_alt  nan%={nan_alt:.4f}, inf%={inf_alt:.4f}")
    else:
        z_diff = (z_base - z_alt).abs().max().item()
        z_tol = 1e-4 if dtype == torch.float32 else 5e-3
        print(f"Max abs diff (LoE merged tokens): {z_diff:.6f} (tol={z_tol})")
        if z_diff > z_tol:
            print("Note: LoE is sensitive to future tokens; strict causality will not hold.")
    print()


def sliding_window_alignment_check():
    print("=" * 60)
    print("Sliding Window Alignment Check")
    print("=" * 60)

    model, config = create_model_example()
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    print(f"Using device: {device}, dtype: {dtype}")
    model = model.to(device=device, dtype=dtype)

    torch.manual_seed(0)
    prompt_len = 10
    input_ids = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

    with torch.no_grad():
        out_full = model.generate(
            input_ids=input_ids,
            max_length=prompt_len + 16,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            use_sliding_window=False,
        )
        out_sw = model.generate(
            input_ids=input_ids,
            max_length=prompt_len + 16,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            use_sliding_window=True,
        )

    same = torch.equal(out_full, out_sw)
    print(f"Alignment: {'PASS' if same else 'WARN: mismatch'}")
    if not same:
        mismatch = (out_full != out_sw).nonzero(as_tuple=False)
        if mismatch.numel() > 0:
            first_pos = mismatch[0, 1].item()
            print(f"First mismatch position: {first_pos}")
        print(f"Full: {out_full[0].tolist()}")
        print(f"SW  : {out_sw[0].tolist()}")
    print()


def gradient_flow_check():
    print("=" * 60)
    print("Gradient Flow Check")
    print("=" * 60)

    model, config = create_model_example()
    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    print(f"Using device: {device}, dtype: {dtype}")
    model = model.to(device=device, dtype=dtype)

    torch.manual_seed(0)
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    captured = {}
    def _lot_hook(module, inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(out):
            out.retain_grad()
            captured["lot_out"] = out
    hook_handle = model.model.shared_local_transformer.register_forward_hook(_lot_hook)

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    print(f"Loss: {loss.item():.4f}")
    loss.backward()
    hook_handle.remove()

    prefixes = [
        "model.shared_local_transformer",
        "model.local_encoder",
        "model.latent_model",
        "model.local_decoder",
        "lm_head",
    ]

    def summarize_prefix(prefix):
        total = 0
        with_grad = 0
        nonzero = 0
        max_abs = 0.0
        for name, param in model.named_parameters():
            if not name.startswith(prefix) or not param.requires_grad:
                continue
            total += 1
            if param.grad is None:
                continue
            with_grad += 1
            grad = param.grad
            if not torch.isfinite(grad).all():
                continue
            gmax = grad.abs().max().item()
            max_abs = max(max_abs, gmax)
            if gmax > 0:
                nonzero += 1
        return total, with_grad, nonzero, max_abs

    for prefix in prefixes:
        total, with_grad, nonzero, max_abs = summarize_prefix(prefix)
        if with_grad == 0 or nonzero == 0:
            print(f"{prefix}: WARN grad missing (total={total}, with_grad={with_grad}, nonzero={nonzero})")
        else:
            print(f"{prefix}: grad ok (total={total}, with_grad={with_grad}, nonzero={nonzero}, max_abs={max_abs:.3e})")

    lot_prefix = "model.shared_local_transformer"
    total, with_grad, nonzero, _ = summarize_prefix(lot_prefix)
    if with_grad == 0 or nonzero == 0:
        print("LoT grads missing on LM loss; running aux loss check...")
        model.zero_grad(set_to_none=True)
        h_aux = model.model.shared_local_transformer(input_ids, attention_mask=None)
        aux_loss = h_aux.float().mean()
        aux_loss.backward()
        total, with_grad, nonzero, max_abs = summarize_prefix(lot_prefix)
        if with_grad == 0 or nonzero == 0:
            print(f"LoT aux grad: FAIL (total={total}, with_grad={with_grad}, nonzero={nonzero})")
        else:
            print(f"LoT aux grad: OK (total={total}, with_grad={with_grad}, nonzero={nonzero}, max_abs={max_abs:.3e})")

    if "lot_out" in captured and captured["lot_out"].grad is not None:
        g = captured["lot_out"].grad
        g_ok = torch.isfinite(g).all() and g.abs().sum().item() > 0
        print(f"LoT output grad: {'OK' if g_ok else 'ZERO/NON-FINITE'} (max_abs={g.abs().max().item():.3e})")
    else:
        print("LoT output grad: NOT AVAILABLE")
    print()


if __name__ == "__main__":
    generation_example()
    causal_mask_sanity_check()
    sliding_window_alignment_check()
    gradient_flow_check()
    
    print("=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)

