# Copyright 2025 Jan Ploski and Pathway Technology, Inc.
# Licensed under the Apache License, Version 2.0

from transformers.configuration_utils import PretrainedConfig


class BDHConfig(PretrainedConfig):
    model_type = "bdh"

    def __init__(
        self,
        num_hidden_layers: int = 6,
        hidden_size: int = 256,
        dropout: float = 0.1,
        num_attention_heads: int = 4,
        mlp_internal_dim_multiplier: int = 128,
        vocab_size: int = 256,
        sliding_window: int = None,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=1,
        use_cache=True,
        max_position_embeddings: int = 4096,
        attn_implementation=None,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads
        self.mlp_internal_dim_multiplier = mlp_internal_dim_multiplier
        self.vocab_size = vocab_size
        self.sliding_window = sliding_window
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings

        # Store BDH-specific attention implementation separately
        # because transformers validates attn_implementation values
        if attn_implementation is None:
            attn_implementation = kwargs.pop("_attn_implementation", "bdh_recurrent")
        else:
            kwargs.pop("_attn_implementation", None)

        # Store our custom implementation type
        self.bdh_attn_implementation = attn_implementation

        # Remove any attn_implementation from kwargs to avoid conflict
        kwargs.pop("attn_implementation", None)

        # Pass "eager" to parent class to pass validation
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            attn_implementation="eager",  # Must be valid HF value
            **kwargs,
        )
