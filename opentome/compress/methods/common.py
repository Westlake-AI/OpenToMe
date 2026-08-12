import math

import torch
import torch.nn.functional as F

from ..base import TensorPair


def gather_tokens(states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    indices = indices.unsqueeze(-1).expand(*indices.shape, states.shape[-1])
    return states.gather(dim=2, index=indices)


def group_queries(query_states: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    batch, num_query_heads, query_length, head_dim = query_states.shape
    if num_query_heads % num_kv_heads:
        raise ValueError("The number of query heads must be divisible by the number of KV heads")
    groups = num_query_heads // num_kv_heads
    return query_states.reshape(batch, num_kv_heads, groups, query_length, head_dim)


def observation_scores(
    key_states: torch.Tensor,
    query_states: torch.Tensor,
    window_size: int,
    kernel_size: int,
    pooling: str,
) -> torch.Tensor:
    """Compute SnapKV observation-window scores for prefix tokens."""
    sequence_length = key_states.shape[-2]
    observation_length = min(window_size, sequence_length)
    candidate_length = sequence_length - observation_length
    if candidate_length <= 0:
        return key_states.new_empty((*key_states.shape[:2], 0), dtype=torch.float32)

    queries = group_queries(query_states[..., -observation_length:, :], key_states.shape[1])
    logits = torch.einsum("bkgqd,bksd->bkgqs", queries, key_states)
    logits = logits / math.sqrt(key_states.shape[-1])
    row = torch.arange(observation_length, device=logits.device).view(-1, 1)
    col = torch.arange(observation_length, device=logits.device).view(1, -1)
    logits[..., candidate_length:].masked_fill_(col > row, torch.finfo(logits.dtype).min)

    scores = torch.softmax(logits.float(), dim=-1)[..., :candidate_length].sum(dim=-2).mean(dim=2)
    if kernel_size > 1:
        pool = F.avg_pool1d if pooling == "avgpool" else F.max_pool1d
        scores = pool(scores, kernel_size, stride=1, padding=kernel_size // 2)
    return scores


def snap_compress(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    query_states: torch.Tensor,
    capacity: int,
    window_size: int,
    kernel_size: int,
    pooling: str,
) -> TensorPair:
    sequence_length = key_states.shape[-2]
    if sequence_length <= capacity:
        return key_states, value_states
    window_size = min(window_size, capacity - 1, sequence_length - 1)
    prefix_budget = capacity - window_size
    scores = observation_scores(key_states, query_states, window_size, kernel_size, pooling)
    indices = scores.topk(prefix_budget, dim=-1).indices.sort(dim=-1).values
    return (
        torch.cat((gather_tokens(key_states[..., :-window_size, :], indices), key_states[..., -window_size:, :]), dim=-2),
        torch.cat((gather_tokens(value_states[..., :-window_size, :], indices), value_states[..., -window_size:, :]), dim=-2),
    )
