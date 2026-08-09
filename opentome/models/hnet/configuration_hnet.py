"""HNet model configuration

H-Net is a hierarchical byte-level language model that uses a two-stage architecture:
  - Outer stage: byte-level encoder/decoder with Mamba2 SSM layers
  - Inner stage: chunk-level (latent) Mamba2/MHA transformer

Reference: https://github.com/goombalab/hnet
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class HNetConfig(PretrainedConfig):
    r"""
    Configuration class for the H-Net hierarchical byte-level language model.

    H-Net operates at the byte level (vocab_size=256) and learns to dynamically
    segment bytes into variable-length chunks using a cosine-similarity based
    routing module. The architecture comprises:

      Stage 0 (Outer):
        - encoder:      a stack of SSM/MHA layers at the byte level
        - routing:      RoutingModule predicts chunk boundaries
        - main_network: an inner HNet or isotropic stack at the chunk level
        - dechunk:      EMA-based de-aggregation from chunk → byte
        - decoder:      a stack of SSM/MHA layers at the byte level

      Stage N (Innermost):
        - main_network: a plain isotropic SSM/MHA stack

    Architecture is described by ``arch_layout``, a nested list of strings:
      - Each string is a sequence of ``(m|M|t|T)<n>`` tokens:
          - ``m<n>``: n Mamba2 layers **without** MLP
          - ``M<n>``: n Mamba2 layers **with** SwiGLU MLP
          - ``t<n>``: n MHA layers **without** MLP
          - ``T<n>``: n MHA layers **with** SwiGLU MLP
      - Nesting: a 3-element list ``[encoder_spec, [inner_layout...], decoder_spec]``
        denotes an outer non-innermost stage; a 1-element list ``[spec]`` denotes
        the innermost stage.

    Example 2-stage L architecture (from hnet/configs/hnet_2stage_L.json)::

        arch_layout = ["m4", ["T1m4", ["T26"], "m4T1"], "m4"]
        d_model     = [1024, 1024, 1536]   # per-stage hidden dims
        d_intermediate = [0, 2816, 4096]   # 0 = no MLP at that stage

    Args:
        arch_layout (list):
            Nested list describing the encoder/decoder/inner architecture at each stage.
            See above for format.
        d_model (List[int]):
            Hidden dimension at each stage.  ``d_model[0]`` is the byte-level dim.
        d_intermediate (List[int]):
            Intermediate dim for SwiGLU MLP at each stage. 0 means the MLP is
            disabled for that stage (lowercase m/t blocks).
        vocab_size (int, optional, defaults to 256):
            Byte-level vocabulary size.  Should stay at 256 for UTF-8 byte tokens.
        ssm_cfg (dict, optional):
            Hyperparameters for Mamba2:
                - ``d_state`` (int, 128): SSM state size.
                - ``d_conv``  (int, 4):   convolution kernel size.
                - ``expand``  (int, 2):   expansion factor.
                - ``chunk_size`` (int, 256): chunk size for the Mamba2 SSD kernel.
        attn_cfg (dict, optional):
            Hyperparameters for MHA:
                - ``num_heads``      (List[int]): number of heads per stage.
                - ``rotary_emb_dim`` (List[int]): RoPE dim per stage (0 = no RoPE).
                - ``window_size``    (List[int]): local attention window per stage
                  (-1 = global).
        tie_embeddings (bool, optional, defaults to False):
            Whether to tie the input embedding and the LM head weights.
        initializer_range (float, optional, defaults to 0.02):
            Stddev of the normal initializer for weight matrices.
        fuse_cross_entropy (bool, optional, defaults to True):
            Use FusedCrossEntropyLoss from fla for faster training.
        fuse_linear_cross_entropy (bool, optional, defaults to False):
            Fuse the lm_head linear + cross entropy into one kernel (saves
            activation memory).  Cannot be used together with ``fuse_cross_entropy``.
        use_l2warp (bool, optional, defaults to False):
            Wrap the cross-entropy loss with an L2 regularisation term.
        lb_loss_weight (float, optional, defaults to 0.001):
            Weight of the load-balancing loss that encourages a target chunk
            compression ratio.
        lb_loss_N (float, optional, defaults to 4.0):
            Target compression ratio N used in the load-balancing loss.
            The router is encouraged to select 1/N tokens as chunk boundaries.
    """

    model_type = "hnet"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # Core architecture
        arch_layout: Optional[List] = None,
        d_model: Optional[List[int]] = None,
        d_intermediate: Optional[List[int]] = None,
        vocab_size: int = 256,
        # SSM (Mamba2) configuration
        ssm_cfg: Optional[dict] = None,
        # Attention (MHA) configuration
        attn_cfg: Optional[dict] = None,
        # Embedding
        tie_embeddings: bool = False,
        # Weight init
        initializer_range: float = 0.02,
        # Loss configuration
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        # Load-balancing loss
        lb_loss_weight: float = 0.001,
        lb_loss_N: float = 4.0,
        # DeChunk kernel parameters
        dechunk_headdim: int = 32,
        dechunk_block_size: int = 256,
        **kwargs,
    ):
        # ----- defaults -----
        if arch_layout is None:
            # single-stage 1-layer Mamba2 (smallest possible model for testing)
            arch_layout = [["m1"]]
        if d_model is None:
            d_model = [512]
        if d_intermediate is None:
            d_intermediate = [0] * len(d_model)
        if ssm_cfg is None:
            ssm_cfg = {}
        if attn_cfg is None:
            attn_cfg = {}

        self.arch_layout = arch_layout
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.vocab_size = vocab_size

        # SSM hyperparams (per-stage lists are handled inside the model)
        _ssm_defaults = dict(d_state=128, d_conv=4, expand=2, chunk_size=256)
        _ssm_defaults.update(ssm_cfg)
        self.ssm_cfg = _ssm_defaults

        # Attention hyperparams
        n_stages = len(d_model)
        _attn_defaults = dict(
            num_heads=[16] * n_stages,
            rotary_emb_dim=[32] * n_stages,
            window_size=[-1] * n_stages,
        )
        _attn_defaults.update(attn_cfg)
        self.attn_cfg = _attn_defaults

        self.tie_embeddings = tie_embeddings
        self.initializer_range = initializer_range

        # Loss
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.lb_loss_weight = lb_loss_weight
        self.lb_loss_N = lb_loss_N

        # DeChunk
        self.dechunk_headdim = dechunk_headdim
        self.dechunk_block_size = dechunk_block_size

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot both be True."
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improve memory "
                "efficiency at the potential cost of reduced precision. "
                "If you observe loss divergence, consider disabling this."
            )

        # Remove tie_word_embeddings from kwargs to avoid duplicate
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(
            tie_word_embeddings=tie_embeddings,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience helpers used by the model
    # ------------------------------------------------------------------

    def get_stage_ssm_cfg(self, stage_idx: int) -> dict:
        """Return SSM config for the requested stage (scalar for all stages)."""
        return dict(self.ssm_cfg)  # same for all stages

    def get_stage_attn_cfg(self, stage_idx: int) -> dict:
        """Return attention config dict for the requested stage."""
        return {
            k: (v[stage_idx] if isinstance(v, (list, tuple)) else v)
            for k, v in self.attn_cfg.items()
        }

    def num_stages(self) -> int:
        return len(self.d_model)


__all__ = ["HNetConfig"]
