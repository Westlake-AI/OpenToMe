"""Blt model configuration"""

from typing import Optional
from transformers.configuration_utils import PretrainedConfig
# from transformers.modeling_rope_utils import RopeParameters
from opentome.models.modeling_rope_utils import RopeParameters
from transformers.utils import logging


logger = logging.get_logger(__name__)


class BltLocalEncoderConfig(PretrainedConfig):
    """
    Configuration class for the Blt Local Encoder component.
    """

    model_type = "blt_local_encoder"
    default_theta = 500000.0

    def __init__(
        self,
        vocab_size: Optional[int] = 260,
        cross_attn_all_layers: Optional[bool] = False,
        cross_attn_k: Optional[int] = 2,
        hidden_size_global: Optional[int] = 2048,
        hidden_size: Optional[int] = 1024,
        num_attention_heads: Optional[int] = 16,
        num_key_value_heads: Optional[int] = None,
        num_hidden_layers: Optional[int] = 1,
        rms_norm_eps: Optional[float] = 1e-5,
        dropout: Optional[float] = 0.0,
        max_position_embeddings: Optional[int] = 24576,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        hidden_act: Optional[str] = "silu",
        intermediate_size: Optional[int] = 2816,
        initializer_range: Optional[float] = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.cross_attn_all_layers = cross_attn_all_layers
        self.cross_attn_k = cross_attn_k
        self.hidden_size_global = hidden_size_global
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size or int(8 * hidden_size / 3)
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rope_parameters = rope_parameters

        # Remove tie_word_embeddings from kwargs to avoid duplicate parameter error
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(**kwargs, tie_word_embeddings=False)


class BltLocalDecoderConfig(PretrainedConfig):
    """
    Configuration class for the Blt Local Decoder component.
    """

    model_type = "blt_local_decoder"
    default_theta = 500000.0

    def __init__(
        self,
        vocab_size: Optional[int] = 260,
        cross_attn_all_layers: Optional[bool] = True,
        cross_attn_k: Optional[int] = 2,
        hidden_size_global: Optional[int] = 2048,
        hidden_size: Optional[int] = 1024,
        num_attention_heads: Optional[int] = 16,
        num_key_value_heads: Optional[int] = None,
        num_hidden_layers: Optional[int] = 9,
        rms_norm_eps: Optional[float] = 1e-5,
        dropout: Optional[float] = 0.0,
        max_position_embeddings: Optional[int] = 24576,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        hidden_act: Optional[str] = "silu",
        intermediate_size: Optional[int] = 2816,
        initializer_range: Optional[float] = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.cross_attn_all_layers = cross_attn_all_layers
        self.cross_attn_k = cross_attn_k
        self.hidden_size_global = hidden_size_global
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size or int(8 * hidden_size / 3)
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rope_parameters = rope_parameters

        # Remove tie_word_embeddings from kwargs to avoid duplicate parameter error
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(**kwargs, tie_word_embeddings=False)


class BltGlobalTransformerConfig(PretrainedConfig):
    """
    Configuration class for the Blt Global Transformer component.
    """

    model_type = "blt_global_transformer"
    default_theta = 500000.0

    def __init__(
        self,
        hidden_size: Optional[int] = 2048,
        num_attention_heads: Optional[int] = 16,
        num_key_value_heads: Optional[int] = None,
        num_hidden_layers: Optional[int] = 25,
        rms_norm_eps: Optional[float] = 1e-5,
        dropout: Optional[float] = 0.0,
        max_position_embeddings: Optional[int] = 4096,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        hidden_act: Optional[str] = "silu",
        intermediate_size: Optional[int] = 5632,
        initializer_range: Optional[float] = 0.02,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size or int(8 * hidden_size / 3)
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rope_parameters = rope_parameters

        # Remove tie_word_embeddings from kwargs to avoid duplicate parameter error
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(**kwargs, tie_word_embeddings=False)


class BltPatcherConfig(PretrainedConfig):
    r"""
    Configuration class for the Blt Patcher/Entropy model component.

    Args:
        vocab_size (`int`, *optional*, defaults to 260):
            Vocabulary size of the Blt patcher model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling the patcher model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 14):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MLP representations.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "blt_patcher"

    def __init__(
        self,
        vocab_size: Optional[int] = 260,
        hidden_size: Optional[int] = 768,
        num_hidden_layers: Optional[int] = 14,
        num_attention_heads: Optional[int] = 12,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: Optional[int] = 8192,
        rms_norm_eps: Optional[float] = 1e-5,
        dropout: Optional[float] = 0.0,
        intermediate_size: Optional[int] = 2048,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        initializer_range: Optional[float] = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout
        self.hidden_act = "silu"  # Blt uses silu activation
        self.intermediate_size = intermediate_size or int(8 * self.hidden_size / 3)
        self.initializer_range = initializer_range
        self.rope_parameters = rope_parameters

        # Remove tie_word_embeddings from kwargs to avoid duplicate parameter error
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(**kwargs, tie_word_embeddings=False)


class BltConfig(PretrainedConfig):
    
    model_type = "blt"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0
    sub_configs = {
        "patcher_config": BltPatcherConfig,
        "encoder_config": BltLocalEncoderConfig,
        "decoder_config": BltLocalDecoderConfig,
        "global_config": BltGlobalTransformerConfig,
    }
    

    def __init__(
        self,
        vocab_size: Optional[int] = 260,
        max_position_embeddings: Optional[int] = 4096,
        patch_in_forward: Optional[bool] = True,
        patch_size: Optional[int] = 4,
        patching_mode: Optional[str] = "entropy",
        patching_threshold: Optional[float] = 1.335442066192627,
        patching_batch_size: Optional[int] = 1,
        max_patch_length: Optional[int] = None,
        cross_attn_k: Optional[int] = 2,
        encoder_hash_byte_group_size: Optional[int] = None,
        encoder_hash_byte_group_vocab: Optional[int] = 500002,
        encoder_hash_byte_group_nb_functions: Optional[int] = 1,
        patcher_config: Optional[dict] = None,
        encoder_config: Optional[dict] = None,
        decoder_config: Optional[dict] = None,
        global_config: Optional[dict] = None,
        tie_word_embeddings: Optional[bool] = False,
        initializer_range: Optional[float] = 0.02,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        **kwargs,
    ):
        # Basic model configuration
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range

        # Patching configuration
        self.patch_in_forward = patch_in_forward
        self.patch_size = patch_size
        self.patching_mode = patching_mode
        self.patching_threshold = patching_threshold
        self.patching_batch_size = patching_batch_size
        self.max_patch_length = max_patch_length
        self.patching_device = kwargs.get("patching_device", "cuda")
        self.realtime_patching = kwargs.get("realtime_patching", True)
        self.patching_threshold_add = kwargs.get("patching_threshold_add")
        self.monotonicity = kwargs.get("monotonicity", False)

        # Cross attention configurations
        self.cross_attn_k = cross_attn_k

        # Encoder configurations
        self.encoder_hash_byte_group_size = encoder_hash_byte_group_size or [3, 4, 5, 6, 7, 8]
        self.encoder_hash_byte_group_vocab = encoder_hash_byte_group_vocab
        self.encoder_hash_byte_group_nb_functions = encoder_hash_byte_group_nb_functions

        # FLA configurations
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        # Initialize component configurations
        if patcher_config is None:
            self.patcher_config = BltPatcherConfig(initializer_range=initializer_range)
            logger.info("patcher_config is None, using default Blt patcher config")
        elif isinstance(patcher_config, dict):
            patcher_config.setdefault("initializer_range", initializer_range)
            self.patcher_config = BltPatcherConfig(**patcher_config)
        elif isinstance(patcher_config, BltPatcherConfig):
            self.patcher_config = patcher_config

        if encoder_config is None:
            self.encoder_config = BltLocalEncoderConfig(initializer_range=initializer_range)
            logger.info("encoder_config is None, using default Blt encoder config")
        elif isinstance(encoder_config, dict):
            encoder_config.setdefault("initializer_range", initializer_range)
            self.encoder_config = BltLocalEncoderConfig(**encoder_config)
        elif isinstance(encoder_config, BltLocalEncoderConfig):
            self.encoder_config = encoder_config

        if decoder_config is None:
            self.decoder_config = BltLocalDecoderConfig(initializer_range=initializer_range)
            logger.info("decoder_config is None, using default Blt decoder config")
        elif isinstance(decoder_config, dict):
            decoder_config.setdefault("initializer_range", initializer_range)
            self.decoder_config = BltLocalDecoderConfig(**decoder_config)
        elif isinstance(decoder_config, BltLocalDecoderConfig):
            self.decoder_config = decoder_config

        if global_config is None:
            self.global_config = BltGlobalTransformerConfig(initializer_range=initializer_range)
            logger.info("global_config is None, using default Blt global config")
        elif isinstance(global_config, dict):
            global_config.setdefault("initializer_range", initializer_range)
            self.global_config = BltGlobalTransformerConfig(**global_config)
        elif isinstance(global_config, BltGlobalTransformerConfig):
            self.global_config = global_config

        # Determine if token embedding projection is needed based on dimension mismatch (7b)
        encoder_cross_output_size = self.encoder_config.hidden_size * self.cross_attn_k
        self.global_config.encoder_cross_output_size = (
            encoder_cross_output_size if encoder_cross_output_size != self.global_config.hidden_size else None
        )

        self.rope_parameters = rope_parameters

        # Remove tie_word_embeddings from kwargs to avoid duplicate parameter error
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = [
    "BltConfig",
    "BltPatcherConfig",
    "BltLocalEncoderConfig",
    "BltLocalDecoderConfig",
    "BltGlobalTransformerConfig",
]
