"""Test that size and source_matrix.sum(dim=-1) are consistent after DTEM."""

import torch
from opentome.models.mergenet_nlp import MergeNetConfig, MergeNetForCausalLM


def test_size_source_consistency():
    """Test size and source_matrix consistency."""
    print("=" * 80)
    print("Testing Size and Source Matrix Consistency")
    print("=" * 80)
    
    config = MergeNetConfig(
        vocab_size=320,
        hidden_size=384,
        num_local_layers=4,
        num_encoder_layers=4,
        num_latent_layers=8,
        num_heads=6,
        num_kv_heads=6,
        intermediate_size=1536,
        lambda_local=4.0,
        dtem_window_size=16,
        dtem_t=1,
    )
    
    model = MergeNetForCausalLM(config)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = model.to(device=device, dtype=dtype)
    
    # Test input
    batch_size = 2
    seq_len = 20
    input_ids = torch.randint(64, 320, (batch_size, seq_len), device=device)
    
    print(f"\nInput shape: {input_ids.shape}")
    
    with torch.no_grad():
        # Forward through LoT
        H = model.model.shared_local_transformer(input_ids, use_cache=False)
        print(f"LoT output shape: {H.shape}")
        
        # Forward through LoE (DTEM)
        Z_merged, size_merged, source_matrix, info = model.model.local_encoder(H, phase="phase2")
        print(f"LoE output shape: {Z_merged.shape}")
        print(f"Size shape: {size_merged.shape}")
        print(f"Source matrix shape: {source_matrix.shape}")
        
        # Check consistency
        source_sum = source_matrix.sum(dim=-1, keepdim=True)  # (B, N, 1)
        
        print(f"\n{'='*80}")
        print("Checking Consistency:")
        print('='*80)
        
        # Compare size vs source_sum
        diff = (size_merged - source_sum).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"\nAbsolute difference (size - source_sum):")
        print(f"  Max:  {max_diff:.8f}")
        print(f"  Mean: {mean_diff:.8f}")
        
        # Relative difference
        rel_diff = diff / (size_merged.abs() + 1e-8)
        max_rel = rel_diff.max().item()
        mean_rel = rel_diff.mean().item()
        
        print(f"\nRelative difference:")
        print(f"  Max:  {max_rel:.8f}")
        print(f"  Mean: {mean_rel:.8f}")
        
        # Show sample values
        print(f"\nSample comparison (first batch, first 10 tokens):")
        print(f"  size:        {size_merged[0, :10, 0].tolist()}")
        print(f"  source_sum:  {source_sum[0, :10, 0].tolist()}")
        
        # Check for zeros
        zero_size = (size_merged == 0).sum().item()
        zero_source = (source_sum == 0).sum().item()
        print(f"\nZero values:")
        print(f"  size has {zero_size} zeros")
        print(f"  source_sum has {zero_source} zeros")
        
        # Tolerance check
        tolerance = 1e-3 if dtype == torch.float32 else 1e-2  # Relaxed for bfloat16
        is_consistent = max_diff < tolerance
        
        if is_consistent:
            print(f"\n✅ PASS: size and source_matrix are consistent (max_diff < {tolerance})")
        else:
            print(f"\n❌ FAIL: size and source_matrix differ (max_diff = {max_diff:.8f})")
            # Show positions with large differences
            large_diff_mask = diff[:, :, 0] > tolerance
            if large_diff_mask.any():
                large_diff_pos = large_diff_mask.nonzero()
                print(f"\nPositions with large differences (>{tolerance}):")
                for pos in large_diff_pos[:5]:  # Show first 5
                    b, n = pos[0].item(), pos[1].item()
                    print(f"  [{b}, {n}]: size={size_merged[b, n, 0].item():.6f}, "
                          f"source_sum={source_sum[b, n, 0].item():.6f}, "
                          f"diff={diff[b, n, 0].item():.6f}")
        
        return is_consistent


if __name__ == "__main__":
    test_size_source_consistency()

