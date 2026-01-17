from transformers.configuration_utils import PretrainedConfig


class MergeNetConfig(PretrainedConfig):
    """
    Configuration class for MergeNet NLP models.
    
    MergeNet is a hierarchical hybrid transformer with differentiable tokenization
    designed for byte-level language modeling.
    
    Args:
        vocab_size (`int`, *optional*, defaults to 320):
            Vocabulary size (256 bytes + special tokens offset).
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_local_layers (`int`, *optional*, defaults to 4):
            Number of layers in Shared Local Transformer (LoT).
        num_latent_layers (`int`, *optional*, defaults to 8):
            Number of layers in Latent Model (LaM).
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer.
        num_kv_heads (`int`, *optional*):
            Number of key-value heads for Grouped Query Attention. If None, defaults to num_heads.
        intermediate_size (`int`, *optional*):
            Dimensionality of the "intermediate" (feed-forward) layer. If None, defaults to 4 * hidden_size.
        hidden_act (`str` or `function`, *optional*, defaults to `"swish"`):
            The non-linear activation function in the feed-forward layer.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        lambda_local (`float`, *optional*, defaults to 4.0):
            Compression ratio for Local Encoder (target: L -> L/lambda).
        dtem_window_size (`int`, *optional*, defaults to 16):
            Local window size for DTEM soft merging in Local Encoder.
        dtem_t (`int`, *optional*, defaults to 1):
            Merge granularity alignment factor for DTEM.
        dtem_feat_dim (`int`, *optional*):
            Feature dimension for DTEM metric. If None, auto-determined based on hidden_size.
        use_softkmax (`bool`, *optional*, defaults to False):
            Whether to use soft k-max selection in DTEM.
        grid_bias_gamma (`float`, *optional*, defaults to 1.0):
            Coefficient for grid bias in Local Decoder: -gamma * |t/lambda - j|.
        W_infer (`int`, *optional*):
            Sliding window size for inference. If None, defaults to ceil(4 * dtem_window_size / lambda_local).
        qkv_bias (`bool`, *optional*, defaults to True):
            Whether to add bias to QKV projections.
        qk_norm (`bool`, *optional*, defaults to False):
            Whether to apply layer norm to Q and K.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing weight matrices.
        use_cache (`bool`, *optional*, defaults to True):
            Whether or not the model should return the last key/values attentions.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to False):
            Whether to tie input and output embeddings.
        phase (`str`, *optional*, defaults to "phase2"):
            Training phase: "phase1" (reconstruction) or "phase2" (prediction).
        drop_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate.
        attn_drop_rate (`float`, *optional*, defaults to 0.0):
            Attention dropout rate.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Stochastic depth rate.
    """

    model_type = "mergenet"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 320,  # 256 bytes + 64 special tokens offset (BLT default)
        hidden_size: int = 768,
        num_local_layers: int = 4,
        num_latent_layers: int = 8,
        num_heads: int = 12,
        num_kv_heads: int | None = None,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        lambda_local: float = 4.0,
        dtem_window_size: int = 16,
        dtem_t: int = 1,
        dtem_feat_dim: int | None = None,
        use_softkmax: bool = False,
        grid_bias_gamma: float = 1.0,
        W_infer: int | None = None,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        phase: str = "phase2",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_local_layers = num_local_layers
        self.num_latent_layers = num_latent_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        
        # MergeNet specific
        self.lambda_local = lambda_local
        self.dtem_window_size = dtem_window_size
        self.dtem_t = dtem_t
        self.dtem_feat_dim = dtem_feat_dim
        self.use_softkmax = use_softkmax
        self.grid_bias_gamma = grid_bias_gamma
        
        # Compute W_infer if not provided
        if W_infer is None:
            import math
            self.W_infer = math.ceil(num_local_layers * dtem_window_size / lambda_local)
        else:
            self.W_infer = W_infer
        
        # Transformer parameters
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        
        # Training
        self.phase = phase
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

